#ifndef COLIBRI_TENSOR_IO_H
#define COLIBRI_TENSOR_IO_H

#include <stddef.h>
#include <stdint.h>

#include "safetensors_index.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ColiTensorView view;
    void *data_allocation;
    void *scale_allocation;
} ColiOwnedTensor;

typedef struct {
    float *data;
    uint64_t count;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiFloatTensor;

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size);
void coli_owned_tensor_free(ColiOwnedTensor *tensor);

int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size);
void coli_float_tensor_free(ColiFloatTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
