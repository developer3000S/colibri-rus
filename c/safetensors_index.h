#ifndef COLIBRI_SAFETENSORS_INDEX_H
#define COLIBRI_SAFETENSORS_INDEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_ST_MAX_RANK 8

typedef enum {
    COLI_ST_BF16 = 0,
    COLI_ST_F16,
    COLI_ST_F32,
    COLI_ST_U8,
    COLI_ST_I8,
    COLI_ST_I64,
    COLI_ST_F8_E4M3,
    COLI_ST_F8_E8M0
} ColiSafetensorsDType;

typedef struct {
    const char *name;
    ColiSafetensorsDType dtype;
    int shard;
    uint64_t offset;
    uint64_t nbytes;
    uint64_t numel;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiSafetensorsTensor;

typedef struct ColiSafetensorsIndex ColiSafetensorsIndex;

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size);
void coli_st_index_close(ColiSafetensorsIndex *index);
size_t coli_st_tensor_count(const ColiSafetensorsIndex *index);
size_t coli_st_shard_count(const ColiSafetensorsIndex *index);
const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard);
const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name);
int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination);
int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination);
int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length);
const char *coli_st_dtype_name(ColiSafetensorsDType dtype);

#ifdef __cplusplus
}
#endif

#endif
