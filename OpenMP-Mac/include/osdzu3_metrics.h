#ifndef OSDZU3_METRICS_H
#define OSDZU3_METRICS_H

#include "osdzu3_common.h"

typedef struct {
    uint32_t epoch;
    uint32_t samples;
    uint32_t batches;
    uint32_t speculative_updates;
    uint32_t fallback_updates;
    float average_loss;
    float train_accuracy;
    float eval_accuracy;
    double epoch_ms;
    double average_sample_us;
    float threshold;
    char framework_variant[OSDZU3_MAX_VARIANT];
    char config_path[OSDZU3_MAX_PATH];
    char run_id[OSDZU3_MAX_RUN_ID];
    char benchmark_group[OSDZU3_MAX_RUN_ID];
} osdzu3_epoch_metrics_t;

typedef struct {
    FILE *stream;
    bool enabled;
} osdzu3_metrics_logger_t;

bool osdzu3_metrics_open(osdzu3_metrics_logger_t *logger, const char *path);
void osdzu3_metrics_write(osdzu3_metrics_logger_t *logger, const osdzu3_epoch_metrics_t *metrics);
void osdzu3_metrics_close(osdzu3_metrics_logger_t *logger);

#endif
