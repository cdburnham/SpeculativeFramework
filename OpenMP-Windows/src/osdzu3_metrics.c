#include "osdzu3_metrics.h"

#include <string.h>

bool osdzu3_metrics_open(osdzu3_metrics_logger_t *logger, const char *path) {
    logger->enabled = false;
    logger->stream = NULL;

    if (path == NULL || path[0] == '\0') {
        return true;
    }

    logger->stream = fopen(path, "w");
    if (logger->stream == NULL) {
        return false;
    }

    logger->enabled = true;
    fprintf(logger->stream,
            "epoch,samples,batches,speculative_updates,fallback_updates,average_loss,train_accuracy,eval_accuracy,epoch_ms,average_sample_us,threshold,framework_variant,config_path,run_id,benchmark_group\n");
    return true;
}

void osdzu3_metrics_write(osdzu3_metrics_logger_t *logger, const osdzu3_epoch_metrics_t *metrics) {
    if (!logger->enabled || logger->stream == NULL) {
        return;
    }

    fprintf(logger->stream,
            "%u,%u,%u,%u,%u,%.6f,%.6f,%.6f,%.3f,%.3f,%.6f,%s,%s,%s,%s\n",
            metrics->epoch,
            metrics->samples,
            metrics->batches,
            metrics->speculative_updates,
            metrics->fallback_updates,
            metrics->average_loss,
            metrics->train_accuracy,
            metrics->eval_accuracy,
            metrics->epoch_ms,
            metrics->average_sample_us,
            metrics->threshold,
            metrics->framework_variant,
            metrics->config_path,
            metrics->run_id,
            metrics->benchmark_group);
    fflush(logger->stream);
}

void osdzu3_metrics_close(osdzu3_metrics_logger_t *logger) {
    if (logger->stream != NULL) {
        fclose(logger->stream);
        logger->stream = NULL;
    }
    logger->enabled = false;
}
