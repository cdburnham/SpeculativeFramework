#ifndef OSDZU3_DATASET_H
#define OSDZU3_DATASET_H

#include "osdzu3_config.h"

typedef enum {
    OSDZU3_SPLIT_TRAIN = 0,
    OSDZU3_SPLIT_TEST
} osdzu3_dataset_split_t;

typedef struct {
    osdzu3_dataset_format_t format;
    uint32_t feature_count;
    uint32_t class_count;
    uint32_t train_samples;
    uint32_t test_samples;
    long train_feature_data_offset;
    long train_label_data_offset;
    long test_feature_data_offset;
    long test_label_data_offset;
    FILE *train_features;
    FILE *train_labels;
    FILE *test_features;
    FILE *test_labels;
} osdzu3_dataset_t;

bool osdzu3_dataset_open(osdzu3_dataset_t *dataset,
                         const osdzu3_dataset_config_t *config,
                         uint32_t expected_feature_count,
                         uint32_t expected_class_count,
                         char *error,
                         size_t error_size);
void osdzu3_dataset_close(osdzu3_dataset_t *dataset);
uint32_t osdzu3_dataset_sample_count(const osdzu3_dataset_t *dataset, osdzu3_dataset_split_t split);
bool osdzu3_dataset_read_sample(const osdzu3_dataset_t *dataset,
                                osdzu3_dataset_split_t split,
                                uint32_t index,
                                float *features_out,
                                uint32_t *label_out,
                                char *error,
                                size_t error_size);

#endif
