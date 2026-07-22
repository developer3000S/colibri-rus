#define _GNU_SOURCE
#include "../tensor_io.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int write_fixture(const char *path) {
    static const char header[] =
        "{\"linear.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[128,128],\"data_offsets\":[0,16384]},"
        "\"linear.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,1],\"data_offsets\":[16384,16385]},"
        "\"norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[16385,16389]}}";
    uint64_t header_length = sizeof(header) - 1;
    unsigned char *payload = calloc(16389, 1);
    if (!payload) return -1;
    memset(payload, 0x38, 16384);
    payload[16384] = 0x7f;
    uint16_t norm[2] = {0x3f80, 0x4000};
    memcpy(payload + 16385, norm, sizeof(norm));
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) { free(payload); return -1; }
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, 16389);
    close(fd);
    free(payload);
    return result ? -1 : 0;
}

int main(int argc, char **argv) {
    /* Native MinGW binaries do not resolve the MSYS /tmp mount. */
    char directory[] = "colibri-tensor-io-XXXXXX";
    char path[256], error[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture(path) != 0) return 1;
    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, directory, error, sizeof(error)) != 0) return 1;
    ColiOwnedTensor linear;
    ColiFloatTensor norm;
    if (coli_tensor_load_fp8(&linear, index, "linear", error, sizeof(error)) != 0 ||
        coli_tensor_load_f32(&norm, index, "norm.weight", error, sizeof(error)) != 0)
        return 1;
    if (linear.view.rows != 128 || linear.view.columns != 128 ||
        ((const uint8_t *)linear.view.data)[0] != 0x38 ||
        norm.count != 2 || norm.data[0] != 1.0f || norm.data[1] != 2.0f)
        return 1;
    coli_float_tensor_free(&norm);
    coli_owned_tensor_free(&linear);
    coli_st_index_close(index);
    unlink(path);
    rmdir(directory);

    if (argc > 1) {
        if (coli_st_index_open(&index, argv[1], error, sizeof(error)) != 0) return 1;
        if (coli_tensor_load_fp8(&linear, index, "layers.0.attn.wq_a",
                                 error, sizeof(error)) != 0 ||
            coli_tensor_load_f32(&norm, index, "layers.0.attn_norm.weight",
                                 error, sizeof(error)) != 0) return 1;
        if (linear.view.rows != 1024 || linear.view.columns != 4096 ||
            norm.count != 4096) return 1;
        coli_float_tensor_free(&norm);
        coli_owned_tensor_free(&linear);
        coli_st_index_close(index);
    }
    puts("tensor I/O tests: ok");
    return 0;
}
