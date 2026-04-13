#include "osdzu3_config.h"
#include "osdzu3_json.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static void osdzu3_set_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list args;
    if (error == NULL || error_size == 0) {
        return;
    }
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static bool osdzu3_read_text_file(const char *path, char **buffer_out, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    long size;
    char *buffer;

    if (file == NULL) {
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }
    size = ftell(file);
    if (size < 0) {
        fclose(file);
        return false;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }

    buffer = (char *) malloc((size_t) size + 1U);
    if (buffer == NULL) {
        fclose(file);
        return false;
    }

    if (fread(buffer, 1, (size_t) size, file) != (size_t) size) {
        free(buffer);
        fclose(file);
        return false;
    }
    buffer[size] = '\0';
    fclose(file);

    *buffer_out = buffer;
    *size_out = (size_t) size;
    return true;
}

static bool osdzu3_parse_activation(const char *text, osdzu3_activation_t *activation) {
    if (strcmp(text, "linear") == 0) {
        *activation = OSDZU3_ACT_LINEAR;
        return true;
    }
    if (strcmp(text, "relu") == 0) {
        *activation = OSDZU3_ACT_RELU;
        return true;
    }
    if (strcmp(text, "leaky_relu") == 0) {
        *activation = OSDZU3_ACT_LEAKY_RELU;
        return true;
    }
    if (strcmp(text, "sigmoid") == 0) {
        *activation = OSDZU3_ACT_SIGMOID;
        return true;
    }
    if (strcmp(text, "softmax") == 0) {
        *activation = OSDZU3_ACT_SOFTMAX;
        return true;
    }
    return false;
}

static bool osdzu3_parse_dataset_format(const char *text, osdzu3_dataset_format_t *format) {
    if (strcmp(text, "mnist_idx") == 0) {
        *format = OSDZU3_DATASET_MNIST_IDX;
        return true;
    }
    if (strcmp(text, "synthetic_xor") == 0) {
        *format = OSDZU3_DATASET_SYNTHETIC_XOR;
        return true;
    }
    if (strcmp(text, "dense_bin") == 0) {
        *format = OSDZU3_DATASET_DENSE_BIN;
        return true;
    }
    return false;
}

static void osdzu3_default_config(osdzu3_app_config_t *config) {
    memset(config, 0, sizeof(*config));
    config->network.speculative_enabled = true;
    config->network.speculative_threshold = 0.02f;
    config->network.weight_init_scale = 0.10f;
    config->training.epochs = 5;
    config->training.batch_size = 8;
    config->training.learning_rate = 0.01f;
    config->training.gradient_clip = 5.0f;
    config->training.seed = 7;
    strncpy(config->logging.metrics_csv, "logs/train_metrics.csv", sizeof(config->logging.metrics_csv) - 1U);
    strncpy(config->logging.predictions_csv, "logs/predictions.csv", sizeof(config->logging.predictions_csv) - 1U);
    strncpy(config->logging.checkpoint_bin, "logs/model_checkpoint.bin", sizeof(config->logging.checkpoint_bin) - 1U);
}

bool osdzu3_load_config(const char *path, osdzu3_app_config_t *config, char *error, size_t error_size) {
    char *json = NULL;
    size_t json_size = 0;
    osdzu3_json_parser_t parser;
    osdzu3_json_token_t tokens[OSDZU3_MAX_JSON_TOKENS];
    int count;
    int network_idx;
    int training_idx;
    int dataset_idx;
    int logging_idx;
    int layers_idx;
    int input_size_idx;
    int class_count_idx;
    int i;

    osdzu3_default_config(config);

    if (!osdzu3_read_text_file(path, &json, &json_size)) {
        osdzu3_set_error(error, error_size, "could not read config file: %s", path);
        return false;
    }

    osdzu3_json_init(&parser);
    count = osdzu3_json_tokenize(&parser, json, json_size, tokens, OSDZU3_MAX_JSON_TOKENS);
    if (count < 1 || tokens[0].type != OSDZU3_JSON_OBJECT) {
        free(json);
        osdzu3_set_error(error, error_size, "invalid JSON in %s", path);
        return false;
    }

    network_idx = osdzu3_json_object_get(json, tokens, count, 0, "network");
    training_idx = osdzu3_json_object_get(json, tokens, count, 0, "training");
    dataset_idx = osdzu3_json_object_get(json, tokens, count, 0, "dataset");
    logging_idx = osdzu3_json_object_get(json, tokens, count, 0, "logging");

    if (network_idx < 0 || training_idx < 0 || dataset_idx < 0) {
        free(json);
        osdzu3_set_error(error, error_size, "config must include network, training, and dataset objects");
        return false;
    }

    input_size_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "input_size");
    class_count_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "class_count");
    if (input_size_idx < 0 || class_count_idx < 0 ||
        !osdzu3_json_token_to_u32(json, &tokens[input_size_idx], &config->network.input_size) ||
        !osdzu3_json_token_to_u32(json, &tokens[class_count_idx], &config->network.class_count)) {
        free(json);
        osdzu3_set_error(error, error_size, "network.input_size and network.class_count must be unsigned integers");
        return false;
    }

    {
        int threshold_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "speculative_threshold");
        int enabled_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "speculative_enabled");
        int init_scale_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "weight_init_scale");
        if (threshold_idx >= 0) {
            if (!osdzu3_json_token_to_float(json, &tokens[threshold_idx], &config->network.speculative_threshold)) {
                free(json);
                osdzu3_set_error(error, error_size, "network.speculative_threshold must be numeric");
                return false;
            }
        }
        if (enabled_idx >= 0) {
            if (!osdzu3_json_token_to_bool(json, &tokens[enabled_idx], &config->network.speculative_enabled)) {
                free(json);
                osdzu3_set_error(error, error_size, "network.speculative_enabled must be boolean");
                return false;
            }
        }
        if (init_scale_idx >= 0) {
            if (!osdzu3_json_token_to_float(json, &tokens[init_scale_idx], &config->network.weight_init_scale)) {
                free(json);
                osdzu3_set_error(error, error_size, "network.weight_init_scale must be numeric");
                return false;
            }
        }
    }

    layers_idx = osdzu3_json_object_get(json, tokens, count, network_idx, "layers");
    if (layers_idx < 0 || tokens[layers_idx].type != OSDZU3_JSON_ARRAY) {
        free(json);
        osdzu3_set_error(error, error_size, "network.layers must be an array");
        return false;
    }

    config->network.layer_count = (uint32_t) tokens[layers_idx].size;
    for (i = 0; i < tokens[layers_idx].size; i++) {
        int layer_idx = osdzu3_json_array_get(tokens, count, layers_idx, i);
        int units_idx = osdzu3_json_object_get(json, tokens, count, layer_idx, "units");
        int activation_idx = osdzu3_json_object_get(json, tokens, count, layer_idx, "activation");
        char activation_text[32];
        if (layer_idx < 0 || units_idx < 0 || activation_idx < 0) {
            free(json);
            osdzu3_set_error(error, error_size, "each network layer must contain units and activation");
            return false;
        }
        if (!osdzu3_json_token_to_u32(json, &tokens[units_idx], &config->network.layers[i].units)) {
            free(json);
            osdzu3_set_error(error, error_size, "layer %d units must be an unsigned integer", i);
            return false;
        }
        if (osdzu3_json_token_to_string(json, &tokens[activation_idx], activation_text, sizeof(activation_text)) != 0 ||
            !osdzu3_parse_activation(activation_text, &config->network.layers[i].activation)) {
            free(json);
            osdzu3_set_error(error, error_size, "layer %d activation is not supported", i);
            return false;
        }
    }

    {
        int epochs_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "epochs");
        int batch_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "batch_size");
        int lr_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "learning_rate");
        int clip_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "gradient_clip");
        int seed_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "seed");
        int train_limit_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "train_limit");
        int test_limit_idx = osdzu3_json_object_get(json, tokens, count, training_idx, "test_limit");
        if (epochs_idx < 0 || batch_idx < 0 || lr_idx < 0 || clip_idx < 0) {
            free(json);
            osdzu3_set_error(error, error_size, "training requires epochs, batch_size, learning_rate, and gradient_clip");
            return false;
        }
        if (!osdzu3_json_token_to_u32(json, &tokens[epochs_idx], &config->training.epochs) ||
            !osdzu3_json_token_to_u32(json, &tokens[batch_idx], &config->training.batch_size) ||
            !osdzu3_json_token_to_float(json, &tokens[lr_idx], &config->training.learning_rate) ||
            !osdzu3_json_token_to_float(json, &tokens[clip_idx], &config->training.gradient_clip)) {
            free(json);
            osdzu3_set_error(error, error_size, "training fields have invalid types");
            return false;
        }
        if (seed_idx >= 0) {
            if (!osdzu3_json_token_to_u32(json, &tokens[seed_idx], &config->training.seed)) {
                free(json);
                osdzu3_set_error(error, error_size, "training.seed must be an unsigned integer");
                return false;
            }
        }
        if (train_limit_idx >= 0 &&
            !osdzu3_json_token_to_u32(json, &tokens[train_limit_idx], &config->training.train_limit)) {
            free(json);
            osdzu3_set_error(error, error_size, "training.train_limit must be an unsigned integer");
            return false;
        }
        if (test_limit_idx >= 0 &&
            !osdzu3_json_token_to_u32(json, &tokens[test_limit_idx], &config->training.test_limit)) {
            free(json);
            osdzu3_set_error(error, error_size, "training.test_limit must be an unsigned integer");
            return false;
        }
    }

    {
        int format_idx = osdzu3_json_object_get(json, tokens, count, dataset_idx, "format");
        char format_text[32];
        if (format_idx < 0 ||
            osdzu3_json_token_to_string(json, &tokens[format_idx], format_text, sizeof(format_text)) != 0 ||
            !osdzu3_parse_dataset_format(format_text, &config->dataset.format)) {
            free(json);
            osdzu3_set_error(error, error_size, "dataset.format must be mnist_idx, synthetic_xor, or dense_bin");
            return false;
        }
        if (config->dataset.format == OSDZU3_DATASET_MNIST_IDX ||
            config->dataset.format == OSDZU3_DATASET_DENSE_BIN) {
            int train_features_idx = osdzu3_json_object_get(json, tokens, count, dataset_idx, "train_features");
            int train_labels_idx = osdzu3_json_object_get(json, tokens, count, dataset_idx, "train_labels");
            int test_features_idx = osdzu3_json_object_get(json, tokens, count, dataset_idx, "test_features");
            int test_labels_idx = osdzu3_json_object_get(json, tokens, count, dataset_idx, "test_labels");
            if (train_features_idx < 0 || train_labels_idx < 0 || test_features_idx < 0 || test_labels_idx < 0) {
                free(json);
                osdzu3_set_error(error, error_size, "datasets with file-backed features require all feature and label paths");
                return false;
            }
            if (osdzu3_json_token_to_string(json, &tokens[train_features_idx], config->dataset.train_features,
                                            sizeof(config->dataset.train_features)) != 0 ||
                osdzu3_json_token_to_string(json, &tokens[train_labels_idx], config->dataset.train_labels,
                                            sizeof(config->dataset.train_labels)) != 0 ||
                osdzu3_json_token_to_string(json, &tokens[test_features_idx], config->dataset.test_features,
                                            sizeof(config->dataset.test_features)) != 0 ||
                osdzu3_json_token_to_string(json, &tokens[test_labels_idx], config->dataset.test_labels,
                                            sizeof(config->dataset.test_labels)) != 0) {
                free(json);
                osdzu3_set_error(error, error_size, "dataset paths exceed supported length");
                return false;
            }
        }
    }

    if (logging_idx >= 0) {
        int metrics_idx = osdzu3_json_object_get(json, tokens, count, logging_idx, "metrics_csv");
        int predictions_idx = osdzu3_json_object_get(json, tokens, count, logging_idx, "predictions_csv");
        int checkpoint_idx = osdzu3_json_object_get(json, tokens, count, logging_idx, "checkpoint_bin");
        if (metrics_idx >= 0) {
            if (osdzu3_json_token_to_string(json, &tokens[metrics_idx], config->logging.metrics_csv,
                                            sizeof(config->logging.metrics_csv)) != 0) {
                free(json);
                osdzu3_set_error(error, error_size, "logging.metrics_csv path exceeds supported length");
                return false;
            }
        }
        if (predictions_idx >= 0) {
            if (osdzu3_json_token_to_string(json, &tokens[predictions_idx], config->logging.predictions_csv,
                                            sizeof(config->logging.predictions_csv)) != 0) {
                free(json);
                osdzu3_set_error(error, error_size, "logging.predictions_csv path exceeds supported length");
                return false;
            }
        }
        if (checkpoint_idx >= 0) {
            if (osdzu3_json_token_to_string(json, &tokens[checkpoint_idx], config->logging.checkpoint_bin,
                                            sizeof(config->logging.checkpoint_bin)) != 0) {
                free(json);
                osdzu3_set_error(error, error_size, "logging.checkpoint_bin path exceeds supported length");
                return false;
            }
        }
    }

    free(json);
    return osdzu3_validate_config(config, error, error_size);
}

bool osdzu3_validate_config(const osdzu3_app_config_t *config, char *error, size_t error_size) {
    uint32_t i;

    if (config->network.input_size == 0 || config->network.input_size > OSDZU3_MAX_INPUTS) {
        osdzu3_set_error(error, error_size, "network.input_size must be between 1 and %d", OSDZU3_MAX_INPUTS);
        return false;
    }
    if (config->network.class_count == 0 || config->network.class_count > OSDZU3_MAX_CLASSES) {
        osdzu3_set_error(error, error_size, "network.class_count must be between 1 and %d", OSDZU3_MAX_CLASSES);
        return false;
    }
    if (config->network.layer_count == 0 || config->network.layer_count > OSDZU3_MAX_LAYERS) {
        osdzu3_set_error(error, error_size, "network.layer_count must be between 1 and %d", OSDZU3_MAX_LAYERS);
        return false;
    }
    if (config->network.layers[config->network.layer_count - 1U].units != config->network.class_count) {
        osdzu3_set_error(error, error_size, "final layer size must match network.class_count");
        return false;
    }
    if (config->network.layers[config->network.layer_count - 1U].activation != OSDZU3_ACT_SOFTMAX) {
        osdzu3_set_error(error, error_size, "final layer activation must be softmax");
        return false;
    }
    for (i = 0; i < config->network.layer_count; i++) {
        if (config->network.layers[i].units == 0 || config->network.layers[i].units > OSDZU3_MAX_NEURONS) {
            osdzu3_set_error(error, error_size, "layer %u units must be between 1 and %d", i, OSDZU3_MAX_NEURONS);
            return false;
        }
    }
    if (config->training.batch_size == 0) {
        osdzu3_set_error(error, error_size, "training.batch_size must be greater than zero");
        return false;
    }
    if (config->training.learning_rate <= 0.0f) {
        osdzu3_set_error(error, error_size, "training.learning_rate must be positive");
        return false;
    }
    if (config->training.gradient_clip <= 0.0f) {
        osdzu3_set_error(error, error_size, "training.gradient_clip must be positive");
        return false;
    }
    if (config->network.weight_init_scale <= 0.0f) {
        osdzu3_set_error(error, error_size, "network.weight_init_scale must be positive");
        return false;
    }
    if (config->network.speculative_enabled &&
        (config->network.speculative_threshold < 0.10f || config->network.speculative_threshold > 0.35f)) {
        osdzu3_set_error(error, error_size, "speculative threshold must be between 0.10 and 0.35");
        return false;
    }
    return true;
}

const char *osdzu3_activation_name(osdzu3_activation_t activation) {
    switch (activation) {
        case OSDZU3_ACT_LINEAR: return "linear";
        case OSDZU3_ACT_RELU: return "relu";
        case OSDZU3_ACT_LEAKY_RELU: return "leaky_relu";
        case OSDZU3_ACT_SIGMOID: return "sigmoid";
        case OSDZU3_ACT_SOFTMAX: return "softmax";
        default: return "unknown";
    }
}

const char *osdzu3_dataset_format_name(osdzu3_dataset_format_t format) {
    switch (format) {
        case OSDZU3_DATASET_MNIST_IDX: return "mnist_idx";
        case OSDZU3_DATASET_SYNTHETIC_XOR: return "synthetic_xor";
        case OSDZU3_DATASET_DENSE_BIN: return "dense_bin";
        default: return "unknown";
    }
}

void osdzu3_print_config_summary(FILE *stream, const osdzu3_app_config_t *config) {
    uint32_t i;
    fprintf(stream, "Network input size: %u\n", config->network.input_size);
    fprintf(stream, "Class count: %u\n", config->network.class_count);
    fprintf(stream, "Layer count: %u\n", config->network.layer_count);
    for (i = 0; i < config->network.layer_count; i++) {
        fprintf(stream, "  Layer %u: units=%u activation=%s\n",
                i,
                config->network.layers[i].units,
                osdzu3_activation_name(config->network.layers[i].activation));
    }
    fprintf(stream, "Speculative backprop: %s\n", config->network.speculative_enabled ? "enabled" : "disabled");
    fprintf(stream, "Speculative threshold: %.6f\n", config->network.speculative_threshold);
    fprintf(stream, "Weight init scale: %.6f\n", config->network.weight_init_scale);
    fprintf(stream, "Epochs: %u\n", config->training.epochs);
    fprintf(stream, "Batch size: %u\n", config->training.batch_size);
    fprintf(stream, "Learning rate: %.6f\n", config->training.learning_rate);
    fprintf(stream, "Gradient clip: %.6f\n", config->training.gradient_clip);
    fprintf(stream, "Dataset format: %s\n", osdzu3_dataset_format_name(config->dataset.format));
    fprintf(stream, "Metrics CSV: %s\n", config->logging.metrics_csv);
    fprintf(stream, "Checkpoint BIN: %s\n", config->logging.checkpoint_bin);
}
