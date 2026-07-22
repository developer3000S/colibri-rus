#define _GNU_SOURCE
#include "safetensors_index.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "compat.h"
#include "json.h"

#define COLI_ST_MAX_SHARDS 512
#define COLI_ST_MAX_HEADER (512ULL << 20)

typedef struct {
    char *path;
    int fd;
    uint64_t size;
} ColiSafetensorsShard;

struct ColiSafetensorsIndex {
    ColiSafetensorsTensor *tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    int *hash;
    size_t hash_capacity;
    ColiSafetensorsShard shards[COLI_ST_MAX_SHARDS];
    size_t shard_count;
};

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static uint64_t hash_name(const char *text) {
    uint64_t hash = 1469598103934665603ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static int has_safetensors_suffix(const char *name) {
    const char *suffix = strrchr(name, '.');
    return suffix && !strcmp(suffix, ".safetensors");
}

static int dtype_from_name(const char *name, ColiSafetensorsDType *dtype) {
    static const struct {
        const char *name;
        ColiSafetensorsDType dtype;
    } formats[] = {
        {"BF16", COLI_ST_BF16}, {"F16", COLI_ST_F16},
        {"F32", COLI_ST_F32}, {"U8", COLI_ST_U8},
        {"I8", COLI_ST_I8}, {"I64", COLI_ST_I64},
        {"F8_E4M3", COLI_ST_F8_E4M3},
        {"F8_E8M0", COLI_ST_F8_E8M0},
    };
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        if (!strcmp(name, formats[i].name)) {
            *dtype = formats[i].dtype;
            return 0;
        }
    }
    return -1;
}

static int dtype_width(ColiSafetensorsDType dtype, uint64_t *width) {
    switch (dtype) {
        case COLI_ST_U8:
        case COLI_ST_I8:
        case COLI_ST_F8_E4M3:
        case COLI_ST_F8_E8M0:
            *width = 1;
            return 0;
        case COLI_ST_BF16:
        case COLI_ST_F16:
            *width = 2;
            return 0;
        case COLI_ST_F32:
            *width = 4;
            return 0;
        case COLI_ST_I64:
            *width = 8;
            return 0;
    }
    return -1;
}

const char *coli_st_dtype_name(ColiSafetensorsDType dtype) {
    switch (dtype) {
        case COLI_ST_BF16: return "BF16";
        case COLI_ST_F16: return "F16";
        case COLI_ST_F32: return "F32";
        case COLI_ST_U8: return "U8";
        case COLI_ST_I8: return "I8";
        case COLI_ST_I64: return "I64";
        case COLI_ST_F8_E4M3: return "F8_E4M3";
        case COLI_ST_F8_E8M0: return "F8_E8M0";
    }
    return "UNKNOWN";
}

static int append_tensor(ColiSafetensorsIndex *index,
                         const ColiSafetensorsTensor *tensor) {
    if (index->tensor_count == index->tensor_capacity) {
        size_t capacity = index->tensor_capacity ? index->tensor_capacity * 2 : 4096;
        void *allocation = realloc(index->tensors, capacity * sizeof(*index->tensors));
        if (!allocation) return -1;
        index->tensors = (ColiSafetensorsTensor *)allocation;
        index->tensor_capacity = capacity;
    }
    index->tensors[index->tensor_count++] = *tensor;
    return 0;
}

static int json_nonnegative_integer(const jval *value, double upper_exclusive,
                                    uint64_t *result) {
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        value->num < 0.0 || value->num >= upper_exclusive ||
        floor(value->num) != value->num)
        return -1;
    *result = (uint64_t)value->num;
    return 0;
}

static int index_shard(ColiSafetensorsIndex *index, const char *path,
                       char *error, size_t error_size) {
    struct stat status;
    int result = -1;
    int fd = -1;
    char *header = NULL;
    char *arena = NULL;
    jval *root = NULL;

    if (index->shard_count >= COLI_ST_MAX_SHARDS)
        return set_error(error, error_size, "too many safetensors shards");

    fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) {
        result = set_error(error, error_size, "cannot open %s: %s",
                           path, strerror(errno));
        goto cleanup;
    }
    if (fstat(fd, &status) != 0) {
        result = set_error(error, error_size, "cannot stat %s: %s",
                           path, strerror(errno));
        goto cleanup;
    }
    if (status.st_size < 8) {
        result = set_error(error, error_size, "invalid safetensors header: %s", path);
        goto cleanup;
    }

    uint64_t shard_size = (uint64_t)status.st_size;
    uint64_t header_length = 0;
    if (pread(fd, &header_length, 8, 0) != 8 || header_length < 2 ||
        header_length > shard_size - 8 || header_length > COLI_ST_MAX_HEADER) {
        result = set_error(error, error_size, "invalid safetensors header: %s", path);
        goto cleanup;
    }

    header = (char *)malloc((size_t)header_length + 1);
    if (!header) {
        result = set_error(error, error_size, "out of memory reading %s", path);
        goto cleanup;
    }
    if (pread(fd, header, (size_t)header_length, 8) != (ssize_t)header_length) {
        result = set_error(error, error_size, "short safetensors header: %s", path);
        goto cleanup;
    }
    header[header_length] = 0;

    root = json_parse(header, &arena);
    if (!root || root->t != J_OBJ) {
        result = set_error(error, error_size,
                           "safetensors header is not an object: %s", path);
        goto cleanup;
    }

    int shard = (int)index->shard_count;
    uint64_t data_start = 8 + header_length;
    for (int i = 0; i < root->len; i++) {
        const char *name = root->keys[i];
        if (!name) {
            result = set_error(error, error_size, "invalid tensor name in %s", path);
            goto cleanup;
        }
        if (!strcmp(name, "__metadata__")) continue;
        jval *metadata = root->kids[i];
        jval *dtype_value = json_get(metadata, "dtype");
        jval *offsets = json_get(metadata, "data_offsets");
        jval *shape = json_get(metadata, "shape");
        ColiSafetensorsTensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        if (!metadata || metadata->t != J_OBJ ||
            !dtype_value || dtype_value->t != J_STR || !dtype_value->str ||
            !offsets || offsets->t != J_ARR || offsets->len != 2 ||
            !shape || shape->t != J_ARR ||
            shape->len > COLI_ST_MAX_RANK ||
            dtype_from_name(dtype_value->str, &tensor.dtype) != 0) {
            result = set_error(error, error_size,
                               "unsupported tensor metadata: %s in %s", name, path);
            goto cleanup;
        }

        uint64_t start = 0, end = 0;
        if (json_nonnegative_integer(offsets->kids[0], ldexp(1.0, 64), &start) != 0 ||
            json_nonnegative_integer(offsets->kids[1], ldexp(1.0, 64), &end) != 0 ||
            start > end || end > shard_size - data_start) {
            result = set_error(error, error_size,
                               "invalid offsets for %s in %s", name, path);
            goto cleanup;
        }

        tensor.rank = shape->len;
        tensor.numel = 1;
        for (int dimension = 0; dimension < shape->len; dimension++) {
            uint64_t extent = 0;
            if (json_nonnegative_integer(shape->kids[dimension],
                                         ldexp(1.0, 63), &extent) != 0 ||
                (extent && tensor.numel > UINT64_MAX / extent)) {
                result = set_error(error, error_size,
                                   "invalid shape for %s in %s", name, path);
                goto cleanup;
            }
            tensor.shape[dimension] = (int64_t)extent;
            tensor.numel *= extent;
        }

        uint64_t width = 0;
        if (dtype_width(tensor.dtype, &width) != 0 ||
            tensor.numel > UINT64_MAX / width ||
            end - start != tensor.numel * width) {
            result = set_error(error, error_size,
                               "tensor byte size does not match dtype and shape: %s in %s",
                               name, path);
            goto cleanup;
        }

        tensor.name = strdup(name);
        if (!tensor.name) {
            result = set_error(error, error_size, "out of memory indexing %s", path);
            goto cleanup;
        }
        tensor.shard = shard;
        tensor.offset = data_start + start;
        tensor.nbytes = end - start;
        if (append_tensor(index, &tensor) != 0) {
            free((char *)tensor.name);
            result = set_error(error, error_size, "out of memory indexing %s", path);
            goto cleanup;
        }
    }

    char *stored_path = strdup(path);
    if (!stored_path) {
        result = set_error(error, error_size, "out of memory storing %s", path);
        goto cleanup;
    }
    index->shards[shard].path = stored_path;
    index->shards[shard].fd = fd;
    index->shards[shard].size = shard_size;
    index->shard_count++;
    fd = -1;
    result = 0;

cleanup:
    json_free(root);
    free(arena);
    free(header);
    if (fd >= 0) close(fd);
    return result;
}

static int build_hash(ColiSafetensorsIndex *index, char *error, size_t error_size) {
    size_t capacity = 1;
    while (capacity < index->tensor_count * 2) capacity <<= 1;
    index->hash = (int *)malloc(capacity * sizeof(*index->hash));
    if (!index->hash)
        return set_error(error, error_size, "out of memory building tensor index");
    index->hash_capacity = capacity;
    for (size_t i = 0; i < capacity; i++) index->hash[i] = -1;
    for (size_t i = 0; i < index->tensor_count; i++) {
        size_t slot = (size_t)hash_name(index->tensors[i].name) & (capacity - 1);
        while (index->hash[slot] >= 0) {
            if (!strcmp(index->tensors[index->hash[slot]].name,
                        index->tensors[i].name))
                return set_error(error, error_size, "duplicate tensor: %s",
                                 index->tensors[i].name);
            slot = (slot + 1) & (capacity - 1);
        }
        index->hash[slot] = (int)i;
    }
    return 0;
}

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size) {
    if (!out || !directory)
        return set_error(error, error_size, "invalid safetensors index arguments");
    *out = NULL;
    DIR *stream = opendir(directory);
    if (!stream)
        return set_error(error, error_size, "cannot open %s: %s", directory, strerror(errno));
    char **paths = NULL;
    size_t path_count = 0, path_capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream))) {
        if (!has_safetensors_suffix(entry->d_name)) continue;
        if (path_count == path_capacity) {
            size_t capacity = path_capacity ? path_capacity * 2 : 16;
            void *allocation = realloc(paths, capacity * sizeof(*paths));
            if (!allocation) {
                closedir(stream);
                free(paths);
                return set_error(error, error_size, "out of memory listing %s", directory);
            }
            paths = (char **)allocation;
            path_capacity = capacity;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[path_count] = (char *)malloc(length);
        if (!paths[path_count]) {
            closedir(stream);
            for (size_t i = 0; i < path_count; i++) free(paths[i]);
            free(paths);
            return set_error(error, error_size, "out of memory listing %s", directory);
        }
        snprintf(paths[path_count++], length, "%s/%s", directory, entry->d_name);
    }
    closedir(stream);
    if (!path_count) {
        free(paths);
        return set_error(error, error_size, "no safetensors shards in %s", directory);
    }
    qsort(paths, path_count, sizeof(*paths), compare_paths);
    ColiSafetensorsIndex *index = (ColiSafetensorsIndex *)calloc(1, sizeof(*index));
    if (!index) {
        for (size_t i = 0; i < path_count; i++) free(paths[i]);
        free(paths);
        return set_error(error, error_size, "out of memory opening %s", directory);
    }
    int result = 0;
    for (size_t i = 0; i < path_count; i++) {
        if (index_shard(index, paths[i], error, error_size) != 0) {
            result = -1;
            break;
        }
    }
    for (size_t i = 0; i < path_count; i++) free(paths[i]);
    free(paths);
    if (!result) result = build_hash(index, error, error_size);
    if (result) {
        coli_st_index_close(index);
        return -1;
    }
    *out = index;
    return 0;
}

void coli_st_index_close(ColiSafetensorsIndex *index) {
    if (!index) return;
    for (size_t i = 0; i < index->tensor_count; i++)
        free((char *)index->tensors[i].name);
    for (size_t i = 0; i < index->shard_count; i++) {
        if (index->shards[i].fd >= 0) close(index->shards[i].fd);
        free(index->shards[i].path);
    }
    free(index->hash);
    free(index->tensors);
    free(index);
}

size_t coli_st_tensor_count(const ColiSafetensorsIndex *index) {
    return index ? index->tensor_count : 0;
}

size_t coli_st_shard_count(const ColiSafetensorsIndex *index) {
    return index ? index->shard_count : 0;
}

const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard) {
    return index && shard >= 0 && (size_t)shard < index->shard_count
        ? index->shards[shard].path : NULL;
}

const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name) {
    if (!index || !name || !index->hash_capacity) return NULL;
    size_t slot = (size_t)hash_name(name) & (index->hash_capacity - 1);
    while (index->hash[slot] >= 0) {
        const ColiSafetensorsTensor *tensor = &index->tensors[index->hash[slot]];
        if (!strcmp(tensor->name, name)) return tensor;
        slot = (slot + 1) & (index->hash_capacity - 1);
    }
    return NULL;
}

int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination) {
    if (!index || !destination || shard < 0 || (size_t)shard >= index->shard_count ||
        offset > index->shards[shard].size ||
        length > index->shards[shard].size - offset)
        return -1;
    unsigned char *output = (unsigned char *)destination;
    size_t done = 0;
    while (done < length) {
        ssize_t count = pread(index->shards[shard].fd, output + done,
                              length - done, (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        done += (size_t)count;
    }
    return 0;
}

int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination) {
    return tensor ? coli_st_read_at(index, tensor->shard, tensor->offset,
                                    (size_t)tensor->nbytes, destination) : -1;
}

int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length) {
    if (!index || shard < 0 || (size_t)shard >= index->shard_count ||
        offset > index->shards[shard].size ||
        length > index->shards[shard].size - offset)
        return -1;
    return posix_fadvise(index->shards[shard].fd, (off_t)offset,
                         (off_t)length, POSIX_FADV_WILLNEED);
}
