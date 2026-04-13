#include "osdzu3_network.h"
#include "osdzu3_board.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define OSDZU3_CHECKPOINT_MAGIC 0x4F53445AU

static void osdzu3_set_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list args;
    if (error == NULL || error_size == 0) {
        return;
    }
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static float osdzu3_rand_uniform(float scale) {
    int bucket = (rand() % 21) - 10;
    float value = ((float) bucket * scale) / 10.0f;
    return (value == 0.0f) ? scale / 10.0f : value;
}

static float osdzu3_activate(osdzu3_activation_t activation, float x) {
    switch (activation) {
        case OSDZU3_ACT_LINEAR: return x;
        case OSDZU3_ACT_RELU: return (x > 0.0f) ? x : 0.01f * x;
        case OSDZU3_ACT_LEAKY_RELU: return (x > 0.0f) ? x : 0.01f * x;
        case OSDZU3_ACT_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case OSDZU3_ACT_SOFTMAX: return x;
        default: return x;
    }
}

static float osdzu3_activation_derivative(osdzu3_activation_t activation, float preactivation, float activation_value) {
    switch (activation) {
        case OSDZU3_ACT_LINEAR: return 1.0f;
        case OSDZU3_ACT_RELU: return preactivation > 0.0f ? 1.0f : 0.01f;
        case OSDZU3_ACT_LEAKY_RELU: return preactivation > 0.0f ? 1.0f : 0.01f;
        case OSDZU3_ACT_SIGMOID: return activation_value * (1.0f - activation_value);
        case OSDZU3_ACT_SOFTMAX: return 1.0f;
        default: return 1.0f;
    }
}

static float osdzu3_clip(float value, float limit) {
    if (value > limit) {
        return limit;
    }
    if (value < -limit) {
        return -limit;
    }
    return value;
}

static size_t osdzu3_weight_count(const osdzu3_network_t *network, uint32_t layer) {
    return (size_t) network->sizes[layer] * (size_t) network->sizes[layer + 1U];
}

static float *osdzu3_cache_activation_ptr(const osdzu3_network_t *network, uint32_t label, uint32_t layer) {
    return network->cached_activations[layer] + ((size_t) label * network->sizes[layer]);
}

static float *osdzu3_cache_preactivation_ptr(const osdzu3_network_t *network, uint32_t label, uint32_t layer) {
    return network->cached_preactivations[layer] + ((size_t) label * network->sizes[layer + 1U]);
}

static void osdzu3_zero_gradients(osdzu3_network_t *network) {
    uint32_t layer;
    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        memset(network->grad_weights[layer], 0, osdzu3_weight_count(network, layer) * sizeof(float));
        memset(network->grad_biases[layer], 0, (size_t) network->sizes[layer + 1U] * sizeof(float));
        memset(network->spec_grad_weights[layer], 0, osdzu3_weight_count(network, layer) * sizeof(float));
        memset(network->spec_grad_biases[layer], 0, (size_t) network->sizes[layer + 1U] * sizeof(float));
    }
}

static void osdzu3_zero_spec_gradients(osdzu3_network_t *network) {
    uint32_t layer;
    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        memset(network->spec_grad_weights[layer], 0, osdzu3_weight_count(network, layer) * sizeof(float));
        memset(network->spec_grad_biases[layer], 0, (size_t) network->sizes[layer + 1U] * sizeof(float));
    }
}

static void osdzu3_apply_gradients(osdzu3_network_t *network) {
    uint32_t layer;
    float rate = network->config.training.learning_rate;
    float clip = network->config.training.gradient_clip;

    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        uint32_t src = network->sizes[layer];
        uint32_t dst = network->sizes[layer + 1U];
        uint32_t i;
        uint32_t j;
        for (i = 0; i < src; i++) {
            for (j = 0; j < dst; j++) {
                size_t idx = (size_t) i * dst + j;
                network->weights[layer][idx] -= rate * osdzu3_clip(network->grad_weights[layer][idx], clip);
            }
        }
        for (j = 0; j < dst; j++) {
            network->biases[layer][j] -= rate * osdzu3_clip(network->grad_biases[layer][j], clip);
        }
    }
    osdzu3_zero_gradients(network);
}

static void osdzu3_apply_spec_gradients(osdzu3_network_t *network) {
    uint32_t layer;
    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        size_t count = osdzu3_weight_count(network, layer);
        uint32_t j;
        size_t i;
        for (i = 0; i < count; i++) {
            network->grad_weights[layer][i] += network->spec_grad_weights[layer][i];
        }
        for (j = 0; j < network->sizes[layer + 1U]; j++) {
            network->grad_biases[layer][j] += network->spec_grad_biases[layer][j];
        }
    }
}

static void osdzu3_softmax(float *values, uint32_t count) {
    uint32_t i;
    float max_value = values[0];
    float denom = 0.0f;

    for (i = 1; i < count; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }
    for (i = 0; i < count; i++) {
        values[i] = expf(values[i] - max_value);
        denom += values[i];
    }
    if (denom == 0.0f) {
        return;
    }
    for (i = 0; i < count; i++) {
        values[i] /= denom;
    }
}

static void osdzu3_forward(osdzu3_network_t *network, const float *features) {
    uint32_t layer;

    memcpy(network->activations[0], features, (size_t) network->sizes[0] * sizeof(float));

    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        uint32_t src = network->sizes[layer];
        uint32_t dst = network->sizes[layer + 1U];
        uint32_t i;
        uint32_t j;

        for (j = 0; j < dst; j++) {
            float sum = network->biases[layer][j];
            for (i = 0; i < src; i++) {
                sum += network->weights[layer][((size_t) i * dst) + j] * network->activations[layer][i];
            }
            network->preactivations[layer][j] = sum;
            network->activations[layer + 1U][j] =
                osdzu3_activate(network->config.network.layers[layer].activation, sum);
        }

        if (network->config.network.layers[layer].activation == OSDZU3_ACT_SOFTMAX) {
            osdzu3_softmax(network->activations[layer + 1U], dst);
        }
    }
}

static void osdzu3_cache_current_state(osdzu3_network_t *network, uint32_t label) {
    uint32_t layer;
    network->cache_valid[label] = true;
    for (layer = 0; layer <= network->config.network.layer_count; layer++) {
        memcpy(osdzu3_cache_activation_ptr(network, label, layer),
               network->activations[layer],
               (size_t) network->sizes[layer] * sizeof(float));
    }
    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        memcpy(osdzu3_cache_preactivation_ptr(network, label, layer),
               network->preactivations[layer],
               (size_t) network->sizes[layer + 1U] * sizeof(float));
    }
}

static float osdzu3_average_output_delta(const osdzu3_network_t *network, uint32_t label) {
    float total = 0.0f;
    uint32_t i;
    uint32_t count = network->sizes[network->config.network.layer_count];
    const float *cached = osdzu3_cache_activation_ptr(network, label, network->config.network.layer_count);

    for (i = 0; i < count; i++) {
        total += fabsf(network->activations[network->config.network.layer_count][i] - cached[i]);
    }
    return total / (float) count;
}

static float osdzu3_compute_loss(const osdzu3_network_t *network, uint32_t label) {
    float prob = network->activations[network->config.network.layer_count][label];
    if (prob < 1.0e-6f) {
        prob = 1.0e-6f;
    }
    return -logf(prob);
}

static uint32_t osdzu3_argmax(const float *values, uint32_t count, float *confidence_out) {
    uint32_t i;
    uint32_t best_index = 0;
    float best_value = values[0];

    for (i = 1; i < count; i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }
    if (confidence_out != NULL) {
        *confidence_out = best_value;
    }
    return best_index;
}

static void osdzu3_backward_from_state(osdzu3_network_t *network,
                                       uint32_t label,
                                       bool use_cached_state,
                                       bool write_speculative_buffers) {
    int layer;
    float *output_activations;
    uint32_t output_count = network->sizes[network->config.network.layer_count];
    uint32_t i;
    float **target_weights = write_speculative_buffers ? network->spec_grad_weights : network->grad_weights;
    float **target_biases = write_speculative_buffers ? network->spec_grad_biases : network->grad_biases;

    if (use_cached_state) {
        output_activations = osdzu3_cache_activation_ptr(network, label, network->config.network.layer_count);
    } else {
        output_activations = network->activations[network->config.network.layer_count];
    }

    for (i = 0; i < output_count; i++) {
        float target = (i == label) ? 1.0f : 0.0f;
        network->deltas[network->config.network.layer_count - 1U][i] = output_activations[i] - target;
    }

    for (layer = (int) network->config.network.layer_count - 1; layer >= 0; layer--) {
        uint32_t src = network->sizes[layer];
        uint32_t dst = network->sizes[layer + 1U];
        const float *src_activations =
            use_cached_state ? osdzu3_cache_activation_ptr(network, label, (uint32_t) layer) : network->activations[layer];
        uint32_t src_index;
        uint32_t dst_index;

        for (src_index = 0; src_index < src; src_index++) {
            for (dst_index = 0; dst_index < dst; dst_index++) {
                size_t idx = (size_t) src_index * dst + dst_index;
                target_weights[layer][idx] += src_activations[src_index] * network->deltas[layer][dst_index];
            }
        }
        for (dst_index = 0; dst_index < dst; dst_index++) {
            target_biases[layer][dst_index] += network->deltas[layer][dst_index];
        }

        if (layer > 0) {
            const float *prev_activations =
                use_cached_state ? osdzu3_cache_activation_ptr(network, label, (uint32_t) layer) : network->activations[layer];
            const float *prev_preactivations =
                use_cached_state ? osdzu3_cache_preactivation_ptr(network, label, (uint32_t) layer - 1U) : network->preactivations[layer - 1];
            for (src_index = 0; src_index < src; src_index++) {
                float sum = 0.0f;
                for (dst_index = 0; dst_index < dst; dst_index++) {
                    size_t idx = (size_t) src_index * dst + dst_index;
                    sum += network->weights[layer][idx] * network->deltas[layer][dst_index];
                }
                network->deltas[layer - 1][src_index] =
                    sum * osdzu3_activation_derivative(network->config.network.layers[layer - 1].activation,
                                                       prev_preactivations[src_index],
                                                       prev_activations[src_index]);
            }
        }
    }
}

static bool osdzu3_evaluate(osdzu3_network_t *network,
                            const osdzu3_dataset_t *dataset,
                            uint32_t limit,
                            float *accuracy_out,
                            char *error,
                            size_t error_size) {
    uint32_t sample_count = osdzu3_dataset_sample_count(dataset, OSDZU3_SPLIT_TEST);
    uint32_t features_count = network->sizes[0];
    float *features = (float *) malloc((size_t) features_count * sizeof(float));
    uint32_t correct = 0;
    uint32_t i;

    if (features == NULL) {
        osdzu3_set_error(error, error_size, "failed to allocate evaluation buffer");
        return false;
    }
    if (limit > 0 && limit < sample_count) {
        sample_count = limit;
    }

    for (i = 0; i < sample_count; i++) {
        uint32_t label;
        uint32_t prediction;
        if (!osdzu3_dataset_read_sample(dataset, OSDZU3_SPLIT_TEST, i, features, &label, error, error_size)) {
            free(features);
            return false;
        }
        osdzu3_forward(network, features);
        prediction = osdzu3_argmax(network->activations[network->config.network.layer_count],
                                   network->sizes[network->config.network.layer_count],
                                   NULL);
        if (prediction == label) {
            correct++;
        }
    }

    free(features);
    *accuracy_out = sample_count ? (float) correct / (float) sample_count : 0.0f;
    return true;
}

bool osdzu3_network_init(osdzu3_network_t *network,
                         const osdzu3_app_config_t *config,
                         char *error,
                         size_t error_size) {
    uint32_t layer;
    memset(network, 0, sizeof(*network));
    network->config = *config;
    network->sizes[0] = config->network.input_size;
    for (layer = 0; layer < config->network.layer_count; layer++) {
        size_t src;
        size_t dst;
        size_t count;
        uint32_t class_count = config->network.class_count;

        network->sizes[layer + 1U] = config->network.layers[layer].units;
        src = network->sizes[layer];
        dst = network->sizes[layer + 1U];
        count = src * dst;

        network->weights[layer] = (float *) malloc(count * sizeof(float));
        network->biases[layer] = (float *) calloc(dst, sizeof(float));
        network->grad_weights[layer] = (float *) calloc(count, sizeof(float));
        network->grad_biases[layer] = (float *) calloc(dst, sizeof(float));
        network->spec_grad_weights[layer] = (float *) calloc(count, sizeof(float));
        network->spec_grad_biases[layer] = (float *) calloc(dst, sizeof(float));
        network->preactivations[layer] = (float *) calloc(dst, sizeof(float));
        network->deltas[layer] = (float *) calloc(dst, sizeof(float));
        network->cached_preactivations[layer] = (float *) calloc(class_count * dst, sizeof(float));
        network->activations[layer] = (float *) calloc(src, sizeof(float));
        network->cached_activations[layer] = (float *) calloc(class_count * src, sizeof(float));

        if (network->weights[layer] == NULL || network->biases[layer] == NULL ||
            network->grad_weights[layer] == NULL || network->grad_biases[layer] == NULL ||
            network->spec_grad_weights[layer] == NULL || network->spec_grad_biases[layer] == NULL ||
            network->preactivations[layer] == NULL || network->deltas[layer] == NULL ||
            network->cached_preactivations[layer] == NULL || network->activations[layer] == NULL ||
            network->cached_activations[layer] == NULL) {
            osdzu3_set_error(error, error_size, "memory allocation failed during network initialization");
            osdzu3_network_free(network);
            return false;
        }
    }

    network->activations[config->network.layer_count] =
        (float *) calloc(network->sizes[config->network.layer_count], sizeof(float));
    network->cached_activations[config->network.layer_count] =
        (float *) calloc((size_t) config->network.class_count * network->sizes[config->network.layer_count], sizeof(float));
    if (network->activations[config->network.layer_count] == NULL ||
        network->cached_activations[config->network.layer_count] == NULL) {
        osdzu3_set_error(error, error_size, "memory allocation failed during output buffer initialization");
        osdzu3_network_free(network);
        return false;
    }

    srand(config->training.seed);
    for (layer = 0; layer < config->network.layer_count; layer++) {
        size_t count = osdzu3_weight_count(network, layer);
        size_t i;
        for (i = 0; i < count; i++) {
            network->weights[layer][i] = osdzu3_rand_uniform(config->network.weight_init_scale);
        }
    }
    osdzu3_zero_gradients(network);
    memset(network->cache_valid, 0, sizeof(network->cache_valid));
    return true;
}

void osdzu3_network_free(osdzu3_network_t *network) {
    uint32_t layer;
    for (layer = 0; layer < OSDZU3_MAX_LAYERS; layer++) {
        free(network->weights[layer]);
        free(network->biases[layer]);
        free(network->grad_weights[layer]);
        free(network->grad_biases[layer]);
        free(network->spec_grad_weights[layer]);
        free(network->spec_grad_biases[layer]);
        free(network->activations[layer]);
        free(network->preactivations[layer]);
        free(network->deltas[layer]);
        free(network->cached_activations[layer]);
        free(network->cached_preactivations[layer]);
        network->weights[layer] = NULL;
        network->biases[layer] = NULL;
        network->grad_weights[layer] = NULL;
        network->grad_biases[layer] = NULL;
        network->spec_grad_weights[layer] = NULL;
        network->spec_grad_biases[layer] = NULL;
        network->activations[layer] = NULL;
        network->preactivations[layer] = NULL;
        network->deltas[layer] = NULL;
        network->cached_activations[layer] = NULL;
        network->cached_preactivations[layer] = NULL;
    }
    if (network->config.network.layer_count == OSDZU3_MAX_LAYERS) {
        free(network->activations[OSDZU3_MAX_LAYERS]);
        free(network->cached_activations[OSDZU3_MAX_LAYERS]);
        network->activations[OSDZU3_MAX_LAYERS] = NULL;
        network->cached_activations[OSDZU3_MAX_LAYERS] = NULL;
    }
}

void osdzu3_network_describe(FILE *stream, const osdzu3_network_t *network) {
    uint32_t layer;
    fprintf(stream, "OSDZU3 network topology:\n");
    fprintf(stream, "  Input: %u\n", network->sizes[0]);
    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        fprintf(stream, "  Layer %u: %u -> %u (%s)\n",
                layer,
                network->sizes[layer],
                network->sizes[layer + 1U],
                osdzu3_activation_name(network->config.network.layers[layer].activation));
    }
}

bool osdzu3_network_train(osdzu3_network_t *network,
                          const osdzu3_dataset_t *dataset,
                          const osdzu3_train_options_t *options,
                          char *error,
                          size_t error_size) {
    uint32_t epoch_count = (options != NULL && options->epoch_override) ? options->epoch_override : network->config.training.epochs;
    uint32_t sample_count = osdzu3_dataset_sample_count(dataset, OSDZU3_SPLIT_TRAIN);
    uint32_t batch_size = network->config.training.batch_size;
    uint32_t features_count = network->sizes[0];
    float *features = (float *) malloc((size_t) features_count * sizeof(float));
    osdzu3_metrics_logger_t logger;
    uint32_t epoch;
    const char *metrics_path =
        (options != NULL && options->metrics_override_path != NULL) ? options->metrics_override_path : network->config.logging.metrics_csv;
    const char *checkpoint_path =
        (options != NULL && options->checkpoint_override_path != NULL) ? options->checkpoint_override_path : network->config.logging.checkpoint_bin;

    if (features == NULL) {
        osdzu3_set_error(error, error_size, "failed to allocate training feature buffer");
        return false;
    }
    if (network->config.training.train_limit > 0 && network->config.training.train_limit < sample_count) {
        sample_count = network->config.training.train_limit;
    }
    if (!osdzu3_metrics_open(&logger, metrics_path)) {
        free(features);
        osdzu3_set_error(error, error_size, "failed to open metrics CSV: %s", metrics_path);
        return false;
    }

    for (epoch = 0; epoch < epoch_count; epoch++) {
        uint32_t i;
        uint32_t correct = 0;
        uint32_t speculative_updates = 0;
        uint32_t fallback_updates = 0;
        float loss_total = 0.0f;
        uint64_t epoch_start = osdzu3_board_time_us();
        osdzu3_epoch_metrics_t metrics;

        osdzu3_zero_gradients(network);
        for (i = 0; i < sample_count; i++) {
            uint32_t label;
            uint32_t prediction;
            if (!osdzu3_dataset_read_sample(dataset, OSDZU3_SPLIT_TRAIN, i, features, &label, error, error_size)) {
                osdzu3_metrics_close(&logger);
                free(features);
                return false;
            }

            osdzu3_forward(network, features);
            prediction = osdzu3_argmax(network->activations[network->config.network.layer_count],
                                       network->sizes[network->config.network.layer_count],
                                       NULL);
            if (prediction == label) {
                correct++;
            }
            loss_total += osdzu3_compute_loss(network, label);

            if (!network->config.network.speculative_enabled || !network->cache_valid[label]) {
                osdzu3_backward_from_state(network, label, false, false);
                fallback_updates++;
                network->cache_valid[label] = true;
            } else {
                osdzu3_zero_spec_gradients(network);
                osdzu3_backward_from_state(network, label, true, true);
                if (osdzu3_average_output_delta(network, label) <= network->config.network.speculative_threshold) {
                    osdzu3_apply_spec_gradients(network);
                    speculative_updates++;
                } else {
                    osdzu3_backward_from_state(network, label, false, false);
                    fallback_updates++;
                }
            }

            if (network->config.network.speculative_enabled) {
                osdzu3_cache_current_state(network, label);
            }

            if ((i % batch_size) == 0U) {
                osdzu3_apply_gradients(network);
            }
        }
        osdzu3_apply_gradients(network);

        memset(&metrics, 0, sizeof(metrics));
        metrics.epoch = epoch + 1U;
        metrics.samples = sample_count;
        metrics.batches = (sample_count + batch_size - 1U) / batch_size;
        metrics.speculative_updates = speculative_updates;
        metrics.fallback_updates = fallback_updates;
        metrics.average_loss = sample_count ? (loss_total / (float) sample_count) : 0.0f;
        metrics.train_accuracy = sample_count ? ((float) correct / (float) sample_count) : 0.0f;
        metrics.threshold = network->config.network.speculative_threshold;
        strncpy(metrics.framework_variant,
                (options != NULL && options->framework_variant != NULL) ? options->framework_variant : OSDZU3_FRAMEWORK_VARIANT,
                sizeof(metrics.framework_variant) - 1U);
        strncpy(metrics.config_path,
                (options != NULL && options->config_path != NULL) ? options->config_path : "",
                sizeof(metrics.config_path) - 1U);
        strncpy(metrics.run_id,
                (options != NULL && options->run_id != NULL) ? options->run_id : "",
                sizeof(metrics.run_id) - 1U);
        strncpy(metrics.benchmark_group,
                (options != NULL && options->benchmark_group != NULL) ? options->benchmark_group : "",
                sizeof(metrics.benchmark_group) - 1U);
        if (!osdzu3_evaluate(network, dataset, network->config.training.test_limit, &metrics.eval_accuracy, error, error_size)) {
            osdzu3_metrics_close(&logger);
            free(features);
            return false;
        }
        metrics.epoch_ms = (double) (osdzu3_board_time_us() - epoch_start) / 1000.0;
        metrics.average_sample_us = sample_count ? ((metrics.epoch_ms * 1000.0) / (double) sample_count) : 0.0;

        osdzu3_metrics_write(&logger, &metrics);
        osdzu3_board_printf("epoch %u/%u loss=%.5f train_acc=%.4f eval_acc=%.4f epoch_ms=%.3f spec=%u fallback=%u\n",
                            metrics.epoch,
                            epoch_count,
                            metrics.average_loss,
                            metrics.train_accuracy,
                            metrics.eval_accuracy,
                            metrics.epoch_ms,
                            metrics.speculative_updates,
                            metrics.fallback_updates);
    }

    osdzu3_metrics_close(&logger);
    if (checkpoint_path[0] != '\0' &&
        !osdzu3_network_save_checkpoint(network,
                                        checkpoint_path,
                                        error,
                                        error_size)) {
        free(features);
        return false;
    }
    free(features);
    return true;
}

bool osdzu3_network_infer(const osdzu3_network_t *network,
                          const osdzu3_dataset_t *dataset,
                          osdzu3_dataset_split_t split,
                          uint32_t index,
                          uint32_t *prediction_out,
                          float *confidence_out,
                          char *error,
                          size_t error_size) {
    osdzu3_network_t mutable_network = *network;
    float *features = (float *) malloc((size_t) network->sizes[0] * sizeof(float));
    uint32_t label;
    if (features == NULL) {
        osdzu3_set_error(error, error_size, "failed to allocate inference buffer");
        return false;
    }
    if (!osdzu3_dataset_read_sample(dataset, split, index, features, &label, error, error_size)) {
        free(features);
        return false;
    }
    osdzu3_forward(&mutable_network, features);
    *prediction_out = osdzu3_argmax(mutable_network.activations[mutable_network.config.network.layer_count],
                                    mutable_network.sizes[mutable_network.config.network.layer_count],
                                    confidence_out);
    free(features);
    return true;
}

bool osdzu3_network_save_checkpoint(const osdzu3_network_t *network,
                                    const char *path,
                                    char *error,
                                    size_t error_size) {
    FILE *stream;
    uint32_t magic = OSDZU3_CHECKPOINT_MAGIC;
    uint32_t layer;

    stream = fopen(path, "wb");
    if (stream == NULL) {
        osdzu3_set_error(error, error_size, "failed to open checkpoint for writing: %s", path);
        return false;
    }

    if (fwrite(&magic, sizeof(magic), 1, stream) != 1 ||
        fwrite(&network->config.network.layer_count, sizeof(uint32_t), 1, stream) != 1 ||
        fwrite(&network->sizes[0], sizeof(uint32_t), 1, stream) != 1 ||
        fwrite(&network->config.network.class_count, sizeof(uint32_t), 1, stream) != 1) {
        fclose(stream);
        osdzu3_set_error(error, error_size, "failed to write checkpoint header");
        return false;
    }

    for (layer = 0; layer < network->config.network.layer_count; layer++) {
        size_t weight_count = osdzu3_weight_count(network, layer);
        uint32_t units = network->sizes[layer + 1U];
        uint32_t activation = (uint32_t) network->config.network.layers[layer].activation;
        if (fwrite(&units, sizeof(uint32_t), 1, stream) != 1 ||
            fwrite(&activation, sizeof(uint32_t), 1, stream) != 1 ||
            fwrite(network->weights[layer], sizeof(float), weight_count, stream) != weight_count ||
            fwrite(network->biases[layer], sizeof(float), units, stream) != units) {
            fclose(stream);
            osdzu3_set_error(error, error_size, "failed to write checkpoint payload");
            return false;
        }
    }

    fclose(stream);
    return true;
}

bool osdzu3_network_load_checkpoint(osdzu3_network_t *network,
                                    const char *path,
                                    char *error,
                                    size_t error_size) {
    FILE *stream;
    uint32_t magic;
    uint32_t layer_count;
    uint32_t input_size;
    uint32_t class_count;
    uint32_t layer;

    stream = fopen(path, "rb");
    if (stream == NULL) {
        return true;
    }

    if (fread(&magic, sizeof(uint32_t), 1, stream) != 1 ||
        fread(&layer_count, sizeof(uint32_t), 1, stream) != 1 ||
        fread(&input_size, sizeof(uint32_t), 1, stream) != 1 ||
        fread(&class_count, sizeof(uint32_t), 1, stream) != 1) {
        fclose(stream);
        osdzu3_set_error(error, error_size, "failed to read checkpoint header");
        return false;
    }

    if (magic != OSDZU3_CHECKPOINT_MAGIC ||
        layer_count != network->config.network.layer_count ||
        input_size != network->sizes[0] ||
        class_count != network->config.network.class_count) {
        fclose(stream);
        osdzu3_set_error(error, error_size, "checkpoint topology does not match current configuration");
        return false;
    }

    for (layer = 0; layer < layer_count; layer++) {
        uint32_t units;
        uint32_t activation;
        size_t weight_count = osdzu3_weight_count(network, layer);
        if (fread(&units, sizeof(uint32_t), 1, stream) != 1 ||
            fread(&activation, sizeof(uint32_t), 1, stream) != 1) {
            fclose(stream);
            osdzu3_set_error(error, error_size, "failed to read checkpoint layer header");
            return false;
        }
        if (units != network->sizes[layer + 1U] ||
            activation != (uint32_t) network->config.network.layers[layer].activation) {
            fclose(stream);
            osdzu3_set_error(error, error_size, "checkpoint layer %u does not match configuration", layer);
            return false;
        }
        if (fread(network->weights[layer], sizeof(float), weight_count, stream) != weight_count ||
            fread(network->biases[layer], sizeof(float), units, stream) != units) {
            fclose(stream);
            osdzu3_set_error(error, error_size, "failed to read checkpoint weights");
            return false;
        }
    }

    fclose(stream);
    return true;
}
