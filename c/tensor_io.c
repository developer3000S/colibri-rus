#include "tensor_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size) {
    if (!output || !index || !prefix)
        return set_error(error, error_size, "invalid FP8 tensor arguments");
    memset(output, 0, sizeof(*output));
    size_t length = strlen(prefix) + sizeof(".weight");
    char *name = malloc(length);
    if (!name) return set_error(error, error_size, "out of memory building tensor name");
    snprintf(name, length, "%s.weight", prefix);
    const ColiSafetensorsTensor *weight = coli_st_find(index, name);
    snprintf(name, length, "%s.scale", prefix);
    const ColiSafetensorsTensor *scale = coli_st_find(index, name);
    free(name);
    if (!weight || !scale || weight->dtype != COLI_ST_F8_E4M3 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        scale->shape[0] != (weight->shape[0] + 127) / 128 ||
        scale->shape[1] != (weight->shape[1] + 127) / 128)
        return set_error(error, error_size, "invalid native FP8 tensor: %s", prefix);
    output->data_allocation = malloc((size_t)weight->nbytes);
    output->scale_allocation = malloc((size_t)scale->nbytes);
    if (!output->data_allocation || !output->scale_allocation) {
        coli_owned_tensor_free(output);
        return set_error(error, error_size, "out of memory loading FP8 tensor: %s", prefix);
    }
    if (coli_st_read_tensor(index, weight, output->data_allocation) != 0 ||
        coli_st_read_tensor(index, scale, output->scale_allocation) != 0) {
        coli_owned_tensor_free(output);
        return set_error(error, error_size, "cannot read FP8 tensor: %s", prefix);
    }
    output->view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0,
        output->data_allocation, output->scale_allocation,
        (size_t)weight->nbytes, (size_t)scale->nbytes,
        weight->shape[0], weight->shape[1], 128, 128
    };
    return 0;
}

void coli_owned_tensor_free(ColiOwnedTensor *tensor) {
    if (!tensor) return;
    free(tensor->data_allocation);
    free(tensor->scale_allocation);
    memset(tensor, 0, sizeof(*tensor));
}

int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size) {
    if (!output || !index || !name)
        return set_error(error, error_size, "invalid float tensor arguments");
    memset(output, 0, sizeof(*output));
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor || (tensor->dtype != COLI_ST_F32 && tensor->dtype != COLI_ST_BF16))
        return set_error(error, error_size, "missing BF16/F32 tensor: %s", name);
    void *raw = malloc((size_t)tensor->nbytes);
    output->data = malloc((size_t)tensor->numel * sizeof(*output->data));
    if (!raw || !output->data) {
        free(raw);
        coli_float_tensor_free(output);
        return set_error(error, error_size, "out of memory loading tensor: %s", name);
    }
    if (coli_st_read_tensor(index, tensor, raw) != 0) {
        free(raw);
        coli_float_tensor_free(output);
        return set_error(error, error_size, "cannot read tensor: %s", name);
    }
    if (tensor->dtype == COLI_ST_F32)
        memcpy(output->data, raw, (size_t)tensor->nbytes);
    else {
        const uint16_t *values = raw;
        for (uint64_t index_value = 0; index_value < tensor->numel; index_value++)
            output->data[index_value] = coli_bf16_decode(values[index_value]);
    }
    free(raw);
    output->count = tensor->numel;
    output->rank = tensor->rank;
    memcpy(output->shape, tensor->shape, sizeof(output->shape));
    return 0;
}

void coli_float_tensor_free(ColiFloatTensor *tensor) {
    if (!tensor) return;
    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}
