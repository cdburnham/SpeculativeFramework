#include "osdzu3_cli.h"
#include "osdzu3_board.h"
#include "osdzu3_config.h"
#include "osdzu3_dataset.h"
#include "osdzu3_network.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#else
#include <direct.h>
#include <process.h>
#endif

#define OSDZU3_MAX_BENCHMARK_CONFIGS 32
#define OSDZU3_MAX_BENCHMARK_THRESHOLDS 16
#define OSDZU3_MAX_BENCHMARK_RUNS 256
#define OSDZU3_MAX_CSV_INPUTS 256

typedef struct {
    const char *config_path;
    osdzu3_train_options_t train_options;
} osdzu3_runtime_options_t;

typedef struct {
    char csv_path[OSDZU3_MAX_PATH];
    char meta_path[OSDZU3_MAX_PATH];
    char checkpoint_path[OSDZU3_MAX_PATH];
    char config_path[OSDZU3_MAX_PATH];
    char run_id[OSDZU3_MAX_RUN_ID];
    char benchmark_group[OSDZU3_MAX_RUN_ID];
    float threshold;
    uint32_t epochs;
} osdzu3_benchmark_run_t;

typedef struct {
    char csv_path[OSDZU3_MAX_PATH];
    char framework_variant[OSDZU3_MAX_VARIANT];
    char config_path[OSDZU3_MAX_PATH];
    char run_id[OSDZU3_MAX_RUN_ID];
    char benchmark_group[OSDZU3_MAX_RUN_ID];
    float threshold;
    double final_train_accuracy;
    double final_eval_accuracy;
    double mean_epoch_ms;
    double total_epoch_ms;
    double mean_sample_us;
    unsigned long spec_updates;
    unsigned long fallback_updates;
    unsigned long epochs;
} osdzu3_compiled_summary_t;

static void osdzu3_print_usage(void) {
    osdzu3_board_printf("Usage:\n");
    osdzu3_board_printf("  speculative_framework validate <config.json>\n");
    osdzu3_board_printf("  speculative_framework describe <config.json>\n");
    osdzu3_board_printf("  speculative_framework train <config.json> [--epochs N] [--metrics path] [--checkpoint path] [--threshold X]\n");
    osdzu3_board_printf("  speculative_framework infer <config.json> --split train|test --index N\n");
    osdzu3_board_printf("  speculative_framework shell <config.json>\n");
    osdzu3_board_printf("  speculative_framework benchmark <output_dir> <config1.json> [config2.json ...] [--thresholds a,b,c] [--repeat N] [--epochs N] [--max-parallel N]\n");
    osdzu3_board_printf("  speculative_framework compile-results <csv-or-dir> [more ...] [--output summary.csv]\n");
}

static void osdzu3_set_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list args;
    if (error == NULL || error_size == 0) {
        return;
    }
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static bool osdzu3_path_ends_with(const char *path, const char *suffix) {
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);
    return path_len >= suffix_len && strcmp(path + path_len - suffix_len, suffix) == 0;
}

static const char *osdzu3_path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
#ifdef _WIN32
    const char *backslash = strrchr(path, '\\');
    if (backslash != NULL && (slash == NULL || backslash > slash)) {
        slash = backslash;
    }
#endif
    return slash ? slash + 1 : path;
}

static void osdzu3_sanitize_token(const char *src, char *dst, size_t dst_size) {
    size_t i = 0;
    if (dst_size == 0) {
        return;
    }
    while (*src != '\0' && i + 1 < dst_size) {
        char ch = *src++;
        if (isalnum((unsigned char) ch)) {
            dst[i++] = ch;
        } else {
            dst[i++] = '_';
        }
    }
    dst[i] = '\0';
}

static bool osdzu3_parse_u32(const char *text, uint32_t *value_out) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == NULL || *end != '\0') {
        return false;
    }
    *value_out = (uint32_t) value;
    return true;
}

static bool osdzu3_parse_float_value(const char *text, float *value_out) {
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == NULL || *end != '\0') {
        return false;
    }
    *value_out = (float) value;
    return true;
}

static bool osdzu3_validate_threshold_override(float threshold, char *error, size_t error_size) {
    if (threshold < 0.10f || threshold > 0.35f) {
        osdzu3_set_error(error, error_size, "threshold override must be between 0.10 and 0.35");
        return false;
    }
    return true;
}

static bool osdzu3_apply_runtime_overrides(osdzu3_app_config_t *config,
                                           const osdzu3_train_options_t *train_options,
                                           char *error,
                                           size_t error_size) {
    if (train_options == NULL) {
        return true;
    }
    if (train_options->threshold_override_enabled) {
        if (!config->network.speculative_enabled) {
            osdzu3_set_error(error, error_size, "threshold override is only valid for speculative-enabled configs");
            return false;
        }
        if (!osdzu3_validate_threshold_override(train_options->threshold_override, error, error_size)) {
            return false;
        }
        config->network.speculative_threshold = train_options->threshold_override;
    }
    return osdzu3_validate_config(config, error, error_size);
}

static bool osdzu3_open_runtime(const osdzu3_runtime_options_t *options,
                                osdzu3_app_config_t *config,
                                osdzu3_network_t *network,
                                osdzu3_dataset_t *dataset,
                                char *error,
                                size_t error_size) {
    if (!osdzu3_load_config(options->config_path, config, error, error_size)) {
        return false;
    }
    if (!osdzu3_apply_runtime_overrides(config, &options->train_options, error, error_size)) {
        return false;
    }
    if (!osdzu3_network_init(network, config, error, error_size)) {
        return false;
    }
    if (!osdzu3_dataset_open(dataset,
                             &config->dataset,
                             config->network.input_size,
                             config->network.class_count,
                             error,
                             error_size)) {
        osdzu3_network_free(network);
        return false;
    }
    if (!osdzu3_network_load_checkpoint(network,
                                        options->train_options.checkpoint_override_path != NULL ?
                                            options->train_options.checkpoint_override_path :
                                            config->logging.checkpoint_bin,
                                        error,
                                        error_size)) {
        osdzu3_dataset_close(dataset);
        osdzu3_network_free(network);
        return false;
    }
    return true;
}

static void osdzu3_close_runtime(osdzu3_network_t *network, osdzu3_dataset_t *dataset) {
    osdzu3_dataset_close(dataset);
    osdzu3_network_free(network);
}

static int osdzu3_run_shell(const char *config_path) {
    osdzu3_app_config_t config;
    osdzu3_network_t network;
    osdzu3_dataset_t dataset;
    char error[256];
    char line[OSDZU3_MAX_CMD];
    osdzu3_runtime_options_t options;

    memset(&options, 0, sizeof(options));
    options.config_path = config_path;
    options.train_options.framework_variant = OSDZU3_FRAMEWORK_VARIANT;
    options.train_options.config_path = config_path;

    if (!osdzu3_open_runtime(&options, &config, &network, &dataset, error, sizeof(error))) {
        osdzu3_board_printf("error: %s\n", error);
        return 1;
    }

    osdzu3_board_printf("speculative_framework shell ready. type help for commands.\n");
    for (;;) {
        osdzu3_board_printf("speculative_framework> ");
        osdzu3_board_flush();
        if (fgets(line, sizeof(line), stdin) == NULL) {
            break;
        }
        if (strncmp(line, "quit", 4) == 0 || strncmp(line, "exit", 4) == 0) {
            break;
        }
        if (strncmp(line, "help", 4) == 0) {
            osdzu3_board_printf("help | describe | train | infer train <idx> | infer test <idx> | quit\n");
            continue;
        }
        if (strncmp(line, "describe", 8) == 0) {
            osdzu3_print_config_summary(stdout, &config);
            osdzu3_network_describe(stdout, &network);
            continue;
        }
        if (strncmp(line, "train", 5) == 0) {
            if (!osdzu3_network_train(&network, &dataset, &options.train_options, error, sizeof(error))) {
                osdzu3_board_printf("error: %s\n", error);
            }
            continue;
        }
        if (strncmp(line, "infer", 5) == 0) {
            char split_name[16];
            unsigned long index;
            uint32_t prediction;
            float confidence;
            if (sscanf(line, "infer %15s %lu", split_name, &index) == 2) {
                osdzu3_dataset_split_t split =
                    (strcmp(split_name, "train") == 0) ? OSDZU3_SPLIT_TRAIN : OSDZU3_SPLIT_TEST;
                if (!osdzu3_network_infer(&network, &dataset, split, (uint32_t) index, &prediction, &confidence,
                                          error, sizeof(error))) {
                    osdzu3_board_printf("error: %s\n", error);
                } else {
                    osdzu3_board_printf("prediction=%u confidence=%.5f\n", prediction, confidence);
                }
            } else {
                osdzu3_board_printf("usage: infer train|test <index>\n");
            }
            continue;
        }
        osdzu3_board_printf("unknown command\n");
    }

    osdzu3_close_runtime(&network, &dataset);
    return 0;
}

static bool osdzu3_ensure_directory(const char *path, char *error, size_t error_size) {
    char buffer[OSDZU3_MAX_PATH];
    size_t len = strlen(path);
    size_t i;

    if (len >= sizeof(buffer)) {
        osdzu3_set_error(error, error_size, "directory path too long: %s", path);
        return false;
    }
    strcpy(buffer, path);

    for (i = 1; i < len; i++) {
        if (buffer[i] == '/') {
            buffer[i] = '\0';
            if (strlen(buffer) > 0) {
#ifdef _WIN32
                _mkdir(buffer);
#else
                mkdir(buffer, 0777);
#endif
            }
            buffer[i] = '/';
        }
    }
#ifdef _WIN32
    if (_mkdir(buffer) != 0 && errno != EEXIST) {
#else
    if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
#endif
        osdzu3_set_error(error, error_size, "failed to create directory: %s", path);
        return false;
    }
    return true;
}

static int osdzu3_csv_split(char *line, char **fields, int max_fields) {
    int count = 0;
    char *cursor = line;
    while (count < max_fields) {
        fields[count++] = cursor;
        while (*cursor != '\0' && *cursor != ',' && *cursor != '\n' && *cursor != '\r') {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '\r') {
            *cursor = '\0';
            break;
        }
        *cursor = '\0';
        cursor++;
    }
    return count;
}

static bool osdzu3_compile_single_csv(const char *csv_path,
                                      osdzu3_compiled_summary_t *summary,
                                      char *error,
                                      size_t error_size) {
    FILE *stream = fopen(csv_path, "r");
    char line[2048];
    unsigned long epochs = 0;
    double total_epoch_ms = 0.0;
    double total_sample_us = 0.0;
    unsigned long total_spec_updates = 0;
    unsigned long total_fallback_updates = 0;

    if (stream == NULL) {
        osdzu3_set_error(error, error_size, "could not open csv: %s", csv_path);
        return false;
    }

    memset(summary, 0, sizeof(*summary));
    strncpy(summary->csv_path, csv_path, sizeof(summary->csv_path) - 1U);

    if (fgets(line, sizeof(line), stream) == NULL) {
        fclose(stream);
        osdzu3_set_error(error, error_size, "csv is empty: %s", csv_path);
        return false;
    }

    while (fgets(line, sizeof(line), stream) != NULL) {
        char *fields[16];
        int count = osdzu3_csv_split(line, fields, 16);
        if (count < 10) {
            continue;
        }
        epochs++;
        summary->final_train_accuracy = atof(fields[6]);
        summary->final_eval_accuracy = atof(fields[7]);
        total_epoch_ms += atof(fields[8]);
        total_sample_us += atof(fields[9]);
        total_spec_updates += (unsigned long) strtoul(fields[3], NULL, 10);
        total_fallback_updates += (unsigned long) strtoul(fields[4], NULL, 10);
        if (count > 10) {
            summary->threshold = (float) atof(fields[10]);
        }
        if (count > 11) {
            strncpy(summary->framework_variant, fields[11], sizeof(summary->framework_variant) - 1U);
        }
        if (count > 12) {
            strncpy(summary->config_path, fields[12], sizeof(summary->config_path) - 1U);
        }
        if (count > 13) {
            strncpy(summary->run_id, fields[13], sizeof(summary->run_id) - 1U);
        }
        if (count > 14) {
            strncpy(summary->benchmark_group, fields[14], sizeof(summary->benchmark_group) - 1U);
        }
    }

    fclose(stream);
    if (epochs == 0UL) {
        osdzu3_set_error(error, error_size, "csv has no data rows: %s", csv_path);
        return false;
    }

    summary->epochs = epochs;
    summary->total_epoch_ms = total_epoch_ms;
    summary->mean_epoch_ms = total_epoch_ms / (double) epochs;
    summary->mean_sample_us = total_sample_us / (double) epochs;
    summary->spec_updates = total_spec_updates;
    summary->fallback_updates = total_fallback_updates;
    return true;
}

static bool osdzu3_is_directory(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool osdzu3_collect_csv_inputs(const char *path,
                                      char inputs[][OSDZU3_MAX_PATH],
                                      int *count,
                                      char *error,
                                      size_t error_size) {
    if (*count >= OSDZU3_MAX_CSV_INPUTS) {
        osdzu3_set_error(error, error_size, "too many csv inputs");
        return false;
    }
    if (!osdzu3_is_directory(path)) {
        if (osdzu3_path_ends_with(path, ".csv")) {
            strncpy(inputs[*count], path, OSDZU3_MAX_PATH - 1U);
            (*count)++;
        }
        return true;
    }

    {
        DIR *dir = opendir(path);
        struct dirent *entry;
        if (dir == NULL) {
            osdzu3_set_error(error, error_size, "could not open directory: %s", path);
            return false;
        }
        while ((entry = readdir(dir)) != NULL) {
            char full_path[OSDZU3_MAX_PATH];
            if (entry->d_name[0] == '.') {
                continue;
            }
            snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
            if (osdzu3_is_directory(full_path)) {
                continue;
            }
            if (!osdzu3_path_ends_with(entry->d_name, ".csv")) {
                continue;
            }
            if (strcmp(entry->d_name, "benchmark_summary.csv") == 0 ||
                strcmp(entry->d_name, "compile_results.csv") == 0) {
                continue;
            }
            strncpy(inputs[*count], full_path, OSDZU3_MAX_PATH - 1U);
            (*count)++;
            if (*count >= OSDZU3_MAX_CSV_INPUTS) {
                break;
            }
        }
        closedir(dir);
    }
    return true;
}

static void osdzu3_print_compiled_summary(const osdzu3_compiled_summary_t *summary) {
    osdzu3_board_printf("%s\n", summary->csv_path);
    osdzu3_board_printf("  variant: %s\n", summary->framework_variant);
    osdzu3_board_printf("  config: %s\n", summary->config_path);
    osdzu3_board_printf("  run_id: %s\n", summary->run_id);
    osdzu3_board_printf("  benchmark_group: %s\n", summary->benchmark_group);
    osdzu3_board_printf("  threshold: %.6f\n", summary->threshold);
    osdzu3_board_printf("  epochs: %lu\n", summary->epochs);
    osdzu3_board_printf("  final_train_accuracy: %.6f\n", summary->final_train_accuracy);
    osdzu3_board_printf("  final_eval_accuracy: %.6f\n", summary->final_eval_accuracy);
    osdzu3_board_printf("  mean_epoch_ms: %.6f\n", summary->mean_epoch_ms);
    osdzu3_board_printf("  total_epoch_ms: %.6f\n", summary->total_epoch_ms);
    osdzu3_board_printf("  mean_sample_us: %.6f\n", summary->mean_sample_us);
    osdzu3_board_printf("  spec_updates: %lu\n", summary->spec_updates);
    osdzu3_board_printf("  fallback_updates: %lu\n", summary->fallback_updates);
}

static int osdzu3_run_compile_results(int argc, char **argv) {
    char csv_inputs[OSDZU3_MAX_CSV_INPUTS][OSDZU3_MAX_PATH];
    osdzu3_compiled_summary_t summaries[OSDZU3_MAX_CSV_INPUTS];
    int input_count = 0;
    int summary_count = 0;
    int i;
    const char *output_path = NULL;
    char error[256];

    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && (i + 1) < argc) {
            output_path = argv[++i];
            continue;
        }
        if (!osdzu3_collect_csv_inputs(argv[i], csv_inputs, &input_count, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
    }

    if (input_count == 0) {
        osdzu3_board_printf("error: no csv inputs found\n");
        return 1;
    }

    for (i = 0; i < input_count; i++) {
        if (osdzu3_compile_single_csv(csv_inputs[i], &summaries[summary_count], error, sizeof(error))) {
            summary_count++;
        } else {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
    }

    for (i = 0; i < summary_count; i++) {
        osdzu3_print_compiled_summary(&summaries[i]);
    }

    if (summary_count == 2) {
        double speedup = summaries[0].total_epoch_ms > 0.0 ?
            (summaries[1].total_epoch_ms / summaries[0].total_epoch_ms) : 0.0;
        osdzu3_board_printf("Comparison\n");
        osdzu3_board_printf("  eval_accuracy_delta: %.6f\n",
                            summaries[0].final_eval_accuracy - summaries[1].final_eval_accuracy);
        osdzu3_board_printf("  total_time_speedup: %.6fx\n", speedup);
        osdzu3_board_printf("  mean_epoch_ms_delta: %.6f\n",
                            summaries[0].mean_epoch_ms - summaries[1].mean_epoch_ms);
        osdzu3_board_printf("  mean_sample_us_delta: %.6f\n",
                            summaries[0].mean_sample_us - summaries[1].mean_sample_us);
    }

    if (output_path != NULL) {
        FILE *stream = fopen(output_path, "w");
        if (stream == NULL) {
            osdzu3_board_printf("error: could not open summary output path\n");
            return 1;
        }
        fprintf(stream,
                "csv_path,framework_variant,config_path,run_id,benchmark_group,threshold,epochs,final_train_accuracy,final_eval_accuracy,mean_epoch_ms,total_epoch_ms,mean_sample_us,spec_updates,fallback_updates\n");
        for (i = 0; i < summary_count; i++) {
            fprintf(stream,
                    "%s,%s,%s,%s,%s,%.6f,%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%lu,%lu\n",
                    summaries[i].csv_path,
                    summaries[i].framework_variant,
                    summaries[i].config_path,
                    summaries[i].run_id,
                    summaries[i].benchmark_group,
                    summaries[i].threshold,
                    summaries[i].epochs,
                    summaries[i].final_train_accuracy,
                    summaries[i].final_eval_accuracy,
                    summaries[i].mean_epoch_ms,
                    summaries[i].total_epoch_ms,
                    summaries[i].mean_sample_us,
                    summaries[i].spec_updates,
                    summaries[i].fallback_updates);
        }
        fclose(stream);
    }

    return 0;
}

static int osdzu3_write_benchmark_meta(const osdzu3_benchmark_run_t *run) {
    FILE *stream = fopen(run->meta_path, "w");
    if (stream == NULL) {
        return 0;
    }
    fprintf(stream, "framework_variant=%s\n", OSDZU3_FRAMEWORK_VARIANT);
    fprintf(stream, "config_path=%s\n", run->config_path);
    fprintf(stream, "run_id=%s\n", run->run_id);
    fprintf(stream, "benchmark_group=%s\n", run->benchmark_group);
    fprintf(stream, "threshold=%.6f\n", run->threshold);
    fprintf(stream, "epochs=%u\n", run->epochs);
    fprintf(stream, "metrics_csv=%s\n", run->csv_path);
    fprintf(stream, "checkpoint_path=%s\n", run->checkpoint_path);
    fclose(stream);
    return 1;
}

static int osdzu3_spawn_train_process(const char *self_path,
                                      const osdzu3_benchmark_run_t *run,
                                      uint32_t epochs) {
#ifdef _WIN32
    char threshold_str[32];
    char epochs_str[32];
    _snprintf(threshold_str, sizeof(threshold_str), "%.2f", run->threshold);
    _snprintf(epochs_str, sizeof(epochs_str), "%u", epochs);
    return _spawnl(_P_NOWAIT,
                   self_path,
                   self_path,
                   "train",
                   run->config_path,
                   "--epochs", epochs_str,
                   "--metrics", run->csv_path,
                   "--checkpoint", run->checkpoint_path,
                   "--threshold", threshold_str,
                   "--run-id", run->run_id,
                   "--benchmark-group", run->benchmark_group,
                   NULL);
#else
    pid_t pid = fork();
    if (pid == 0) {
        char threshold_str[32];
        char epochs_str[32];
        snprintf(threshold_str, sizeof(threshold_str), "%.2f", run->threshold);
        snprintf(epochs_str, sizeof(epochs_str), "%u", epochs);
        execl(self_path,
              self_path,
              "train",
              run->config_path,
              "--epochs", epochs_str,
              "--metrics", run->csv_path,
              "--checkpoint", run->checkpoint_path,
              "--threshold", threshold_str,
              "--run-id", run->run_id,
              "--benchmark-group", run->benchmark_group,
              (char *) NULL);
        _exit(127);
    }
    return (int) pid;
#endif
}

static int osdzu3_wait_for_one_process(void) {
#ifdef _WIN32
    return 0;
#else
    int status = 0;
    pid_t pid = wait(&status);
    if (pid < 0) {
        return -1;
    }
    return status;
#endif
}

static bool osdzu3_parse_threshold_csv(const char *text, float *thresholds, int *count, char *error, size_t error_size) {
    char buffer[256];
    char *token;
    char *saveptr = NULL;
    *count = 0;
    strncpy(buffer, text, sizeof(buffer) - 1U);
    buffer[sizeof(buffer) - 1U] = '\0';
    token = strtok_r(buffer, ",", &saveptr);
    while (token != NULL) {
        float value;
        while (isspace((unsigned char) *token)) {
            token++;
        }
        if (!osdzu3_parse_float_value(token, &value) ||
            !osdzu3_validate_threshold_override(value, error, error_size)) {
            return false;
        }
        thresholds[(*count)++] = value;
        token = strtok_r(NULL, ",", &saveptr);
    }
    return *count > 0;
}

static int osdzu3_run_benchmark(int argc, char **argv) {
    const char *output_dir;
    const char *configs[OSDZU3_MAX_BENCHMARK_CONFIGS];
    float thresholds[OSDZU3_MAX_BENCHMARK_THRESHOLDS];
    osdzu3_benchmark_run_t runs[OSDZU3_MAX_BENCHMARK_RUNS];
    int config_count = 0;
    int threshold_count = 0;
    uint32_t repeat = 1U;
    uint32_t epochs_override = 0U;
    uint32_t max_parallel = 2U;
    int run_count = 0;
    int i;
    int active = 0;
    char error[256];

    if (argc < 4) {
        osdzu3_board_printf("error: benchmark requires an output directory and at least one config\n");
        return 1;
    }

    output_dir = argv[2];
    if (!osdzu3_ensure_directory(output_dir, error, sizeof(error))) {
        osdzu3_board_printf("error: %s\n", error);
        return 1;
    }

    for (i = 3; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) == 0) {
            break;
        }
        configs[config_count++] = argv[i];
    }
    if (config_count == 0) {
        osdzu3_board_printf("error: benchmark needs at least one config path\n");
        return 1;
    }

    while (i < argc) {
        if (strcmp(argv[i], "--thresholds") == 0 && (i + 1) < argc) {
            if (!osdzu3_parse_threshold_csv(argv[++i], thresholds, &threshold_count, error, sizeof(error))) {
                osdzu3_board_printf("error: %s\n", error);
                return 1;
            }
        } else if (strcmp(argv[i], "--repeat") == 0 && (i + 1) < argc) {
            if (!osdzu3_parse_u32(argv[++i], &repeat)) {
                osdzu3_board_printf("error: invalid repeat count\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--epochs") == 0 && (i + 1) < argc) {
            if (!osdzu3_parse_u32(argv[++i], &epochs_override)) {
                osdzu3_board_printf("error: invalid epoch override\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--max-parallel") == 0 && (i + 1) < argc) {
            if (!osdzu3_parse_u32(argv[++i], &max_parallel) || max_parallel == 0U) {
                osdzu3_board_printf("error: invalid max-parallel value\n");
                return 1;
            }
        } else {
            osdzu3_board_printf("error: unknown benchmark option %s\n", argv[i]);
            return 1;
        }
        i++;
    }

    if (threshold_count == 0) {
        thresholds[threshold_count++] = 0.25f;
    }

    for (i = 0; i < config_count; i++) {
        int repeat_index;
        int threshold_index;
        char config_label[OSDZU3_MAX_RUN_ID];
        osdzu3_sanitize_token(osdzu3_path_basename(configs[i]), config_label, sizeof(config_label));
        for (repeat_index = 0; repeat_index < (int) repeat; repeat_index++) {
            for (threshold_index = 0; threshold_index < threshold_count; threshold_index++) {
                osdzu3_benchmark_run_t *run = &runs[run_count];
                char threshold_token[16];
                snprintf(threshold_token, sizeof(threshold_token), "%.2f", thresholds[threshold_index]);
                for (char *p = threshold_token; *p != '\0'; ++p) {
                    if (*p == '.') {
                        *p = '_';
                    }
                }
                snprintf(run->run_id, sizeof(run->run_id), "%s_t%s_r%02d", config_label, threshold_token, repeat_index + 1);
                snprintf(run->benchmark_group, sizeof(run->benchmark_group), "%s", config_label);
                snprintf(run->csv_path, sizeof(run->csv_path), "%s/%s_metrics.csv", output_dir, run->run_id);
                snprintf(run->meta_path, sizeof(run->meta_path), "%s/%s.meta.txt", output_dir, run->run_id);
                snprintf(run->checkpoint_path, sizeof(run->checkpoint_path), "%s/%s_checkpoint.bin", output_dir, run->run_id);
                strncpy(run->config_path, configs[i], sizeof(run->config_path) - 1U);
                run->threshold = thresholds[threshold_index];
                run->epochs = epochs_override;
                if (!osdzu3_write_benchmark_meta(run)) {
                    osdzu3_board_printf("error: could not write metadata for %s\n", run->run_id);
                    return 1;
                }
                run_count++;
            }
        }
    }

    for (i = 0; i < run_count; i++) {
        int pid_or_status;
        while (active >= (int) max_parallel) {
            if (osdzu3_wait_for_one_process() != 0) {
                osdzu3_board_printf("error: benchmark child process failed\n");
                return 1;
            }
            active--;
        }
        pid_or_status = osdzu3_spawn_train_process(argv[0], &runs[i], epochs_override);
        if (pid_or_status < 0) {
            osdzu3_board_printf("error: failed to spawn training process for %s\n", runs[i].run_id);
            return 1;
        }
#ifdef _WIN32
        OSDZU3_UNUSED(pid_or_status);
#endif
        active++;
    }

    while (active > 0) {
        if (osdzu3_wait_for_one_process() != 0) {
            osdzu3_board_printf("error: benchmark child process failed\n");
            return 1;
        }
        active--;
    }

    {
        char summary_path[OSDZU3_MAX_PATH];
        char *compile_args[OSDZU3_MAX_CSV_INPUTS + 5];
        int compile_argc = 0;
        compile_args[compile_argc++] = argv[0];
        compile_args[compile_argc++] = "compile-results";
        compile_args[compile_argc++] = (char *) output_dir;
        compile_args[compile_argc++] = "--output";
        snprintf(summary_path, sizeof(summary_path), "%s/benchmark_summary.csv", output_dir);
        compile_args[compile_argc++] = summary_path;
        return osdzu3_run_compile_results(compile_argc, compile_args);
    }
}

int osdzu3_cli_run(int argc, char **argv) {
    char error[256];

    if (argc < 2) {
        osdzu3_print_usage();
        return 1;
    }

    if (strcmp(argv[1], "validate") == 0) {
        osdzu3_app_config_t config;
        if (argc < 3) {
            osdzu3_print_usage();
            return 1;
        }
        if (!osdzu3_load_config(argv[2], &config, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
        osdzu3_board_printf("config is valid\n");
        return 0;
    }

    if (strcmp(argv[1], "describe") == 0) {
        osdzu3_app_config_t config;
        osdzu3_network_t network;
        if (argc < 3) {
            osdzu3_print_usage();
            return 1;
        }
        if (!osdzu3_load_config(argv[2], &config, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
        if (!osdzu3_network_init(&network, &config, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
        osdzu3_print_config_summary(stdout, &config);
        osdzu3_network_describe(stdout, &network);
        osdzu3_network_free(&network);
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        osdzu3_app_config_t config;
        osdzu3_network_t network;
        osdzu3_dataset_t dataset;
        osdzu3_runtime_options_t options;
        int i;

        if (argc < 3) {
            osdzu3_print_usage();
            return 1;
        }

        memset(&options, 0, sizeof(options));
        options.config_path = argv[2];
        options.train_options.framework_variant = OSDZU3_FRAMEWORK_VARIANT;
        options.train_options.config_path = argv[2];

        for (i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--epochs") == 0 && (i + 1) < argc) {
                if (!osdzu3_parse_u32(argv[++i], &options.train_options.epoch_override)) {
                    osdzu3_board_printf("error: invalid epoch override\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--metrics") == 0 && (i + 1) < argc) {
                options.train_options.metrics_override_path = argv[++i];
            } else if (strcmp(argv[i], "--checkpoint") == 0 && (i + 1) < argc) {
                options.train_options.checkpoint_override_path = argv[++i];
            } else if (strcmp(argv[i], "--threshold") == 0 && (i + 1) < argc) {
                options.train_options.threshold_override_enabled = true;
                if (!osdzu3_parse_float_value(argv[++i], &options.train_options.threshold_override) ||
                    !osdzu3_validate_threshold_override(options.train_options.threshold_override, error, sizeof(error))) {
                    osdzu3_board_printf("error: %s\n", error);
                    return 1;
                }
            } else if (strcmp(argv[i], "--run-id") == 0 && (i + 1) < argc) {
                options.train_options.run_id = argv[++i];
            } else if (strcmp(argv[i], "--benchmark-group") == 0 && (i + 1) < argc) {
                options.train_options.benchmark_group = argv[++i];
            }
        }

        if (!osdzu3_open_runtime(&options, &config, &network, &dataset, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
        if (!osdzu3_network_train(&network, &dataset, &options.train_options, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            osdzu3_close_runtime(&network, &dataset);
            return 1;
        }
        osdzu3_close_runtime(&network, &dataset);
        return 0;
    }

    if (strcmp(argv[1], "infer") == 0) {
        osdzu3_app_config_t config;
        osdzu3_network_t network;
        osdzu3_dataset_t dataset;
        osdzu3_runtime_options_t options;
        osdzu3_dataset_split_t split = OSDZU3_SPLIT_TEST;
        uint32_t index = 0;
        uint32_t prediction;
        float confidence;
        int i;

        if (argc < 3) {
            osdzu3_print_usage();
            return 1;
        }

        memset(&options, 0, sizeof(options));
        options.config_path = argv[2];

        for (i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--split") == 0 && (i + 1) < argc) {
                split = (strcmp(argv[++i], "train") == 0) ? OSDZU3_SPLIT_TRAIN : OSDZU3_SPLIT_TEST;
            } else if (strcmp(argv[i], "--index") == 0 && (i + 1) < argc) {
                if (!osdzu3_parse_u32(argv[++i], &index)) {
                    osdzu3_board_printf("error: invalid inference index\n");
                    return 1;
                }
            }
        }
        if (!osdzu3_open_runtime(&options, &config, &network, &dataset, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            return 1;
        }
        if (!osdzu3_network_infer(&network, &dataset, split, index, &prediction, &confidence, error, sizeof(error))) {
            osdzu3_board_printf("error: %s\n", error);
            osdzu3_close_runtime(&network, &dataset);
            return 1;
        }
        osdzu3_board_printf("prediction=%u confidence=%.5f\n", prediction, confidence);
        osdzu3_close_runtime(&network, &dataset);
        return 0;
    }

    if (strcmp(argv[1], "shell") == 0) {
        if (argc < 3) {
            osdzu3_print_usage();
            return 1;
        }
        return osdzu3_run_shell(argv[2]);
    }

    if (strcmp(argv[1], "benchmark") == 0) {
        return osdzu3_run_benchmark(argc, argv);
    }

    if (strcmp(argv[1], "compile-results") == 0) {
        return osdzu3_run_compile_results(argc, argv);
    }

    osdzu3_print_usage();
    return 1;
}
