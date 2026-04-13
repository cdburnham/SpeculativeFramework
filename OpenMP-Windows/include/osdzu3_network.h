#ifndef OSDZU3_NETWORK_H
#define OSDZU3_NETWORK_H

#include "osdzu3_dataset.h"
#include "osdzu3_metrics.h"

typedef struct {
    osdzu3_app_config_t config;
    uint32_t sizes[OSDZU3_MAX_LAYERS + 1];
    float *weights[OSDZU3_MAX_LAYERS];
    float *biases[OSDZU3_MAX_LAYERS];
    float *grad_weights[OSDZU3_MAX_LAYERS];
    float *grad_biases[OSDZU3_MAX_LAYERS];
    float *spec_grad_weights[OSDZU3_MAX_LAYERS];
    float *spec_grad_biases[OSDZU3_MAX_LAYERS];
    float *activations[OSDZU3_MAX_LAYERS + 1];
    float *preactivations[OSDZU3_MAX_LAYERS];
    float *deltas[OSDZU3_MAX_LAYERS];
    float *cached_activations[OSDZU3_MAX_LAYERS + 1];
    float *cached_preactivations[OSDZU3_MAX_LAYERS];
    bool cache_valid[OSDZU3_MAX_CLASSES];
} osdzu3_network_t;

typedef struct {
    uint32_t epoch_override;
    const char *metrics_override_path;
    const char *checkpoint_override_path;
    bool threshold_override_enabled;
    float threshold_override;
    const char *framework_variant;
    const char *config_path;
    const char *run_id;
    const char *benchmark_group;
} osdzu3_train_options_t;

bool osdzu3_network_init(osdzu3_network_t *network,
                         const osdzu3_app_config_t *config,
                         char *error,
                         size_t error_size);
void osdzu3_network_free(osdzu3_network_t *network);
void osdzu3_network_describe(FILE *stream, const osdzu3_network_t *network);
bool osdzu3_network_train(osdzu3_network_t *network,
                          const osdzu3_dataset_t *dataset,
                          const osdzu3_train_options_t *options,
                          char *error,
                          size_t error_size);
bool osdzu3_network_infer(const osdzu3_network_t *network,
                          const osdzu3_dataset_t *dataset,
                          osdzu3_dataset_split_t split,
                          uint32_t index,
                          uint32_t *prediction_out,
                          float *confidence_out,
                          char *error,
                          size_t error_size);
bool osdzu3_network_save_checkpoint(const osdzu3_network_t *network,
                                    const char *path,
                                    char *error,
                                    size_t error_size);
bool osdzu3_network_load_checkpoint(osdzu3_network_t *network,
                                    const char *path,
                                    char *error,
                                    size_t error_size);

#endif
