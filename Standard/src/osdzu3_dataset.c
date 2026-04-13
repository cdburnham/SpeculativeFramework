#include "osdzu3_dataset.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define OSDZU3_DENSE_FEATURE_MAGIC 0x4F534446U
#define OSDZU3_DENSE_LABEL_MAGIC 0x4F53444CU

static void osdzu3_set_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list args;
    if (error == NULL || error_size == 0) {
        return;
    }
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static uint32_t osdzu3_read_u32_be(FILE *stream) {
    uint8_t buffer[4];
    if (fread(buffer, 1, 4, stream) != 4) {
        return 0;
    }
    return ((uint32_t) buffer[0] << 24U) |
           ((uint32_t) buffer[1] << 16U) |
           ((uint32_t) buffer[2] << 8U) |
           (uint32_t) buffer[3];
}

static bool osdzu3_open_mnist_file_pair(const char *feature_path,
                                        const char *label_path,
                                        FILE **feature_file,
                                        FILE **label_file,
                                        uint32_t expected_feature_count,
                                        uint32_t *sample_count,
                                        char *error,
                                        size_t error_size) {
    uint32_t image_magic;
    uint32_t label_magic;
    uint32_t images;
    uint32_t rows;
    uint32_t cols;
    uint32_t labels;

    *feature_file = fopen(feature_path, "rb");
    *label_file = fopen(label_path, "rb");

    if (*feature_file == NULL || *label_file == NULL) {
        osdzu3_set_error(error, error_size, "could not open MNIST files");
        return false;
    }

    image_magic = osdzu3_read_u32_be(*feature_file);
    images = osdzu3_read_u32_be(*feature_file);
    rows = osdzu3_read_u32_be(*feature_file);
    cols = osdzu3_read_u32_be(*feature_file);

    label_magic = osdzu3_read_u32_be(*label_file);
    labels = osdzu3_read_u32_be(*label_file);

    if (image_magic != 2051U || label_magic != 2049U) {
        osdzu3_set_error(error, error_size, "MNIST magic headers are invalid");
        return false;
    }
    if (rows * cols != expected_feature_count) {
        osdzu3_set_error(error, error_size, "MNIST image size %ux%u does not match expected input size %u",
                         rows, cols, expected_feature_count);
        return false;
    }
    if (images != labels) {
        osdzu3_set_error(error, error_size, "MNIST feature and label counts do not match");
        return false;
    }

    *sample_count = images;
    return true;
}

static bool osdzu3_open_dense_file_pair(const char *feature_path,
                                        const char *label_path,
                                        FILE **feature_file,
                                        FILE **label_file,
                                        long *feature_data_offset,
                                        long *label_data_offset,
                                        uint32_t expected_feature_count,
                                        uint32_t expected_class_count,
                                        uint32_t *sample_count,
                                        char *error,
                                        size_t error_size) {
    uint32_t feature_magic;
    uint32_t feature_samples;
    uint32_t feature_count;
    uint32_t label_magic;
    uint32_t label_samples;
    uint32_t label_class_count;

    *feature_file = fopen(feature_path, "rb");
    *label_file = fopen(label_path, "rb");
    if (*feature_file == NULL || *label_file == NULL) {
        osdzu3_set_error(error, error_size, "could not open dense_bin files");
        return false;
    }

    if (fread(&feature_magic, sizeof(uint32_t), 1, *feature_file) != 1 ||
        fread(&feature_samples, sizeof(uint32_t), 1, *feature_file) != 1 ||
        fread(&feature_count, sizeof(uint32_t), 1, *feature_file) != 1 ||
        fread(&label_magic, sizeof(uint32_t), 1, *label_file) != 1 ||
        fread(&label_samples, sizeof(uint32_t), 1, *label_file) != 1 ||
        fread(&label_class_count, sizeof(uint32_t), 1, *label_file) != 1) {
        osdzu3_set_error(error, error_size, "failed reading dense_bin headers");
        return false;
    }

    if (feature_magic != OSDZU3_DENSE_FEATURE_MAGIC || label_magic != OSDZU3_DENSE_LABEL_MAGIC) {
        osdzu3_set_error(error, error_size, "dense_bin headers are invalid");
        return false;
    }
    if (feature_count != expected_feature_count) {
        osdzu3_set_error(error, error_size, "dense_bin feature count %u does not match expected input size %u",
                         feature_count, expected_feature_count);
        return false;
    }
    if (label_class_count != expected_class_count) {
        osdzu3_set_error(error, error_size, "dense_bin class count %u does not match expected class count %u",
                         label_class_count, expected_class_count);
        return false;
    }
    if (feature_samples != label_samples) {
        osdzu3_set_error(error, error_size, "dense_bin feature and label sample counts do not match");
        return false;
    }

    *feature_data_offset = ftell(*feature_file);
    *label_data_offset = ftell(*label_file);
    *sample_count = feature_samples;
    return true;
}

bool osdzu3_dataset_open(osdzu3_dataset_t *dataset,
                         const osdzu3_dataset_config_t *config,
                         uint32_t expected_feature_count,
                         uint32_t expected_class_count,
                         char *error,
                         size_t error_size) {
    memset(dataset, 0, sizeof(*dataset));
    dataset->format = config->format;
    dataset->feature_count = expected_feature_count;
    dataset->class_count = expected_class_count;

    if (config->format == OSDZU3_DATASET_SYNTHETIC_XOR) {
        dataset->train_samples = 4;
        dataset->test_samples = 4;
        if (expected_feature_count != 2 || expected_class_count != 2) {
            osdzu3_set_error(error, error_size, "synthetic_xor requires input_size=2 and class_count=2");
            return false;
        }
        return true;
    }

    if (config->format == OSDZU3_DATASET_DENSE_BIN) {
        if (!osdzu3_open_dense_file_pair(config->train_features,
                                         config->train_labels,
                                         &dataset->train_features,
                                         &dataset->train_labels,
                                         &dataset->train_feature_data_offset,
                                         &dataset->train_label_data_offset,
                                         expected_feature_count,
                                         expected_class_count,
                                         &dataset->train_samples,
                                         error,
                                         error_size)) {
            osdzu3_dataset_close(dataset);
            return false;
        }
        if (!osdzu3_open_dense_file_pair(config->test_features,
                                         config->test_labels,
                                         &dataset->test_features,
                                         &dataset->test_labels,
                                         &dataset->test_feature_data_offset,
                                         &dataset->test_label_data_offset,
                                         expected_feature_count,
                                         expected_class_count,
                                         &dataset->test_samples,
                                         error,
                                         error_size)) {
            osdzu3_dataset_close(dataset);
            return false;
        }
        return true;
    }

    if (!osdzu3_open_mnist_file_pair(config->train_features,
                                     config->train_labels,
                                     &dataset->train_features,
                                     &dataset->train_labels,
                                     expected_feature_count,
                                     &dataset->train_samples,
                                     error,
                                     error_size)) {
        osdzu3_dataset_close(dataset);
        return false;
    }
    if (!osdzu3_open_mnist_file_pair(config->test_features,
                                     config->test_labels,
                                     &dataset->test_features,
                                     &dataset->test_labels,
                                     expected_feature_count,
                                     &dataset->test_samples,
                                     error,
                                     error_size)) {
        osdzu3_dataset_close(dataset);
        return false;
    }
    return true;
}

void osdzu3_dataset_close(osdzu3_dataset_t *dataset) {
    if (dataset->train_features != NULL) {
        fclose(dataset->train_features);
        dataset->train_features = NULL;
    }
    if (dataset->train_labels != NULL) {
        fclose(dataset->train_labels);
        dataset->train_labels = NULL;
    }
    if (dataset->test_features != NULL) {
        fclose(dataset->test_features);
        dataset->test_features = NULL;
    }
    if (dataset->test_labels != NULL) {
        fclose(dataset->test_labels);
        dataset->test_labels = NULL;
    }
}

uint32_t osdzu3_dataset_sample_count(const osdzu3_dataset_t *dataset, osdzu3_dataset_split_t split) {
    return (split == OSDZU3_SPLIT_TRAIN) ? dataset->train_samples : dataset->test_samples;
}

bool osdzu3_dataset_read_sample(const osdzu3_dataset_t *dataset,
                                osdzu3_dataset_split_t split,
                                uint32_t index,
                                float *features_out,
                                uint32_t *label_out,
                                char *error,
                                size_t error_size) {
    if (dataset->format == OSDZU3_DATASET_SYNTHETIC_XOR) {
        static const float xor_features[4][2] = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}
        };
        static const uint32_t xor_labels[4] = {0U, 1U, 1U, 0U};
        OSDZU3_UNUSED(split);
        if (index >= 4U) {
            osdzu3_set_error(error, error_size, "synthetic_xor index out of range");
            return false;
        }
        memcpy(features_out, xor_features[index], sizeof(xor_features[index]));
        *label_out = xor_labels[index];
        return true;
    }

    {
        FILE *feature_file = (split == OSDZU3_SPLIT_TRAIN) ? dataset->train_features : dataset->test_features;
        FILE *label_file = (split == OSDZU3_SPLIT_TRAIN) ? dataset->train_labels : dataset->test_labels;
        long feature_offset;
        long label_offset;
        uint32_t i;
        uint32_t label_u32 = 0U;
        int label;

        if (index >= osdzu3_dataset_sample_count(dataset, split)) {
            osdzu3_set_error(error, error_size, "sample index out of range");
            return false;
        }
        if (dataset->format == OSDZU3_DATASET_DENSE_BIN) {
            feature_offset = ((split == OSDZU3_SPLIT_TRAIN) ? dataset->train_feature_data_offset : dataset->test_feature_data_offset) +
                             ((long) index * (long) dataset->feature_count * (long) sizeof(float));
            label_offset = ((split == OSDZU3_SPLIT_TRAIN) ? dataset->train_label_data_offset : dataset->test_label_data_offset) +
                           ((long) index * (long) sizeof(uint32_t));
            if (fseek(feature_file, feature_offset, SEEK_SET) != 0 ||
                fseek(label_file, label_offset, SEEK_SET) != 0) {
                osdzu3_set_error(error, error_size, "failed to seek dense_bin files");
                return false;
            }
            if (fread(features_out, sizeof(float), dataset->feature_count, feature_file) != dataset->feature_count ||
                fread(&label_u32, sizeof(uint32_t), 1, label_file) != 1) {
                osdzu3_set_error(error, error_size, "unexpected EOF while reading dense_bin sample");
                return false;
            }
            if (label_u32 >= dataset->class_count) {
                osdzu3_set_error(error, error_size, "label %u exceeds configured class count %u",
                                 label_u32, dataset->class_count);
                return false;
            }
            *label_out = label_u32;
            return true;
        }

        feature_offset = 16L + ((long) index * (long) dataset->feature_count);
        label_offset = 8L + (long) index;
        if (fseek(feature_file, feature_offset, SEEK_SET) != 0 ||
            fseek(label_file, label_offset, SEEK_SET) != 0) {
            osdzu3_set_error(error, error_size, "failed to seek dataset files");
            return false;
        }
        for (i = 0; i < dataset->feature_count; i++) {
            int value = fgetc(feature_file);
            if (value == EOF) {
                osdzu3_set_error(error, error_size, "unexpected EOF while reading features");
                return false;
            }
            features_out[i] = (float) value / 255.0f;
        }
        label = fgetc(label_file);
        if (label == EOF) {
            osdzu3_set_error(error, error_size, "unexpected EOF while reading labels");
            return false;
        }
        if ((uint32_t) label >= dataset->class_count) {
            osdzu3_set_error(error, error_size, "label %d exceeds configured class count %u", label, dataset->class_count);
            return false;
        }
        *label_out = (uint32_t) label;
        return true;
    }
}
