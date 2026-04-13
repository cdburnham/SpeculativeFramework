#ifndef OSDZU3_CONFIG_H
#define OSDZU3_CONFIG_H

#include "osdzu3_common.h"

typedef enum {
    OSDZU3_ACT_LINEAR = 0,
    OSDZU3_ACT_RELU,
    OSDZU3_ACT_LEAKY_RELU,
    OSDZU3_ACT_SIGMOID,
    OSDZU3_ACT_SOFTMAX
} osdzu3_activation_t;

typedef enum {
    OSDZU3_DATASET_MNIST_IDX = 0,
    OSDZU3_DATASET_SYNTHETIC_XOR,
    OSDZU3_DATASET_DENSE_BIN
} osdzu3_dataset_format_t;

typedef struct {
    uint32_t units;
    osdzu3_activation_t activation;
} osdzu3_layer_config_t;

typedef struct {
    uint32_t input_size;
    uint32_t class_count;
    uint32_t layer_count;
    osdzu3_layer_config_t layers[OSDZU3_MAX_LAYERS];
    bool speculative_enabled;
    float speculative_threshold;
    float weight_init_scale;
} osdzu3_network_config_t;

typedef struct {
    uint32_t epochs;
    uint32_t batch_size;
    uint32_t train_limit;
    uint32_t test_limit;
    uint32_t seed;
    float learning_rate;
    float gradient_clip;
} osdzu3_training_config_t;

typedef struct {
    osdzu3_dataset_format_t format;
    char train_features[OSDZU3_MAX_PATH];
    char train_labels[OSDZU3_MAX_PATH];
    char test_features[OSDZU3_MAX_PATH];
    char test_labels[OSDZU3_MAX_PATH];
} osdzu3_dataset_config_t;

typedef struct {
    char metrics_csv[OSDZU3_MAX_PATH];
    char predictions_csv[OSDZU3_MAX_PATH];
    char checkpoint_bin[OSDZU3_MAX_PATH];
} osdzu3_logging_config_t;

typedef struct {
    osdzu3_network_config_t network;
    osdzu3_training_config_t training;
    osdzu3_dataset_config_t dataset;
    osdzu3_logging_config_t logging;
} osdzu3_app_config_t;

bool osdzu3_load_config(const char *path, osdzu3_app_config_t *config, char *error, size_t error_size);
bool osdzu3_validate_config(const osdzu3_app_config_t *config, char *error, size_t error_size);
void osdzu3_print_config_summary(FILE *stream, const osdzu3_app_config_t *config);
const char *osdzu3_activation_name(osdzu3_activation_t activation);
const char *osdzu3_dataset_format_name(osdzu3_dataset_format_t format);

#endif
