#define _GNU_SOURCE
#include "../safetensors_index.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TEST_ST_MAX_HEADER (512ULL << 20)

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int write_fixture(const char *path, const char *header,
                         const void *payload, size_t payload_size) {
    uint64_t header_length = (uint64_t)strlen(header);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 (payload_size && write_all(fd, payload, payload_size));
    close(fd);
    return result ? -1 : 0;
}

static int write_declared_length(const char *path, uint64_t header_length,
                                 uint64_t file_size) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length));
    if (!result && ftruncate(fd, (off_t)file_size) != 0) result = -1;
    close(fd);
    return result ? -1 : 0;
}

static int expect_rejected(const char *directory, const char *label) {
    ColiSafetensorsIndex *index = NULL;
    char error[256] = {0};
    if (coli_st_index_open(&index, directory, error, sizeof(error)) == 0) {
        fprintf(stderr, "%s: malformed fixture was accepted\n", label);
        coli_st_index_close(index);
        return -1;
    }
    if (index) {
        fprintf(stderr, "%s: failed open returned a live index\n", label);
        coli_st_index_close(index);
        return -1;
    }
    if (!error[0]) {
        fprintf(stderr, "%s: failed open did not report an error\n", label);
        return -1;
    }
    return 0;
}

static int check_valid_fixture(const char *directory) {
    ColiSafetensorsIndex *index = NULL;
    char error[256] = {0};
    if (coli_st_index_open(&index, directory, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return -1;
    }
    int result = 0;
    if (coli_st_shard_count(index) != 1 || coli_st_tensor_count(index) != 3)
        result = -1;
    const ColiSafetensorsTensor *weight = coli_st_find(index, "expert.fp4");
    const ColiSafetensorsTensor *scale = coli_st_find(index, "expert.scale");
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0)
        result = -1;
    if (weight && (weight->rank != 2 || weight->shape[0] != 2 ||
                   weight->shape[1] != 1 || weight->nbytes != 2 ||
                   weight->numel != 2))
        result = -1;
    unsigned char bytes[4] = {0};
    if (weight && (coli_st_read_tensor(index, weight, bytes) != 0 ||
                   bytes[0] != 5 || bytes[1] != 6))
        result = -1;
    if (weight && scale &&
        (coli_st_read_at(index, scale->shard, weight->offset, 4, bytes) != 0 ||
         memcmp(bytes, "\5\6\7\10", 4)))
        result = -1;
    if (strcmp(coli_st_dtype_name(COLI_ST_F8_E4M3), "F8_E4M3"))
        result = -1;
    coli_st_index_close(index);
    return result;
}

int main(void) {
    static const char valid_header[] =
        "{\"dense.fp8\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,2],"
        "\"data_offsets\":[0,4]},"
        "\"expert.fp4\":{\"dtype\":\"I8\",\"shape\":[2,1],"
        "\"data_offsets\":[4,6]},"
        "\"expert.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2,1],"
        "\"data_offsets\":[6,8]}}";
    static const unsigned char valid_payload[] = {1, 2, 3, 4, 5, 6, 7, 8};
    static const unsigned char one_byte[] = {0};
    static const unsigned char mismatch_payload[4096] = {0};
    static const struct {
        const char *label;
        const char *header;
    } malformed[] = {
        {"root-not-object", "[]"},
        {"metadata-not-object", "{\"w\":[]}"},
        {"dtype-not-string", "{\"w\":{\"dtype\":12,\"shape\":[1],\"data_offsets\":[0,1]}}"},
        {"offsets-not-array", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":{}}}"},
        {"offsets-wrong-length", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0]}}"},
        {"offsets-not-numbers", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[\"a\",1]}}"},
        {"negative-offset", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[-1,1]}}"},
        {"fractional-offset", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0.5,1]}}"},
        {"nan-offset", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,NaN]}}"},
        {"infinite-offset", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1e309]}}"},
        {"offset-past-file", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,2]}}"},
        {"shape-not-array", "{\"w\":{\"dtype\":\"U8\",\"shape\":{},\"data_offsets\":[0,1]}}"},
        {"negative-shape", "{\"w\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,1]}}"},
        {"fractional-shape", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1.5],\"data_offsets\":[0,1]}}"},
        {"shape-out-of-range", "{\"w\":{\"dtype\":\"U8\",\"shape\":[9223372036854775808],\"data_offsets\":[0,1]}}"},
        {"shape-product-overflow", "{\"w\":{\"dtype\":\"U8\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,1]}}"},
        {"rank-too-large", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,1]}}"},
    };

    /* Native MinGW binaries do not resolve the MSYS /tmp mount. */
    char directory[] = "colibri-st-XXXXXX";
    char path[256];
    int failures = 0;
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", directory);

    if (write_fixture(path, valid_header, valid_payload, sizeof(valid_payload)) != 0) {
        failures++;
    } else {
        /* Repetition makes this test useful under ASan/LSan. */
        for (int i = 0; i < 64; i++) {
            if (check_valid_fixture(directory) != 0) {
                fprintf(stderr, "valid fixture failed on iteration %d\n", i);
                failures++;
                break;
            }
        }
    }

    if (write_declared_length(path, 64, 8) != 0 ||
        expect_rejected(directory, "header-past-file") != 0)
        failures++;
    if (write_declared_length(path, TEST_ST_MAX_HEADER + 1,
                              TEST_ST_MAX_HEADER + 9) != 0 ||
        expect_rejected(directory, "header-over-limit") != 0)
        failures++;

    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++) {
        if (write_fixture(path, malformed[i].header, one_byte, sizeof(one_byte)) != 0 ||
            expect_rejected(directory, malformed[i].label) != 0)
            failures++;
    }

    static const struct {
        const char *label;
        const char *header;
        size_t payload_size;
    } size_mismatches[] = {
        {"f32-byte-size-mismatch",
         "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4096]}}",
         4096},
        {"bf16-byte-size-mismatch",
         "{\"w\":{\"dtype\":\"BF16\",\"shape\":[4096],\"data_offsets\":[0,2]}}",
         2},
        {"i64-byte-size-mismatch",
         "{\"w\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,4]}}",
         4},
        {"fp8-byte-size-mismatch",
         "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[2],\"data_offsets\":[0,1]}}",
         1},
    };
    for (size_t i = 0;
         i < sizeof(size_mismatches) / sizeof(size_mismatches[0]); i++) {
        if (write_fixture(path, size_mismatches[i].header, mismatch_payload,
                          size_mismatches[i].payload_size) != 0 ||
            expect_rejected(directory, size_mismatches[i].label) != 0)
            failures++;
    }

    unlink(path);
    rmdir(directory);
    if (failures) {
        fprintf(stderr, "safetensors index tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("safetensors index tests: ok");
    return 0;
}
