#include <immintrin.h>

#define coli_fp8_matmul_batch_ref coli_fp8_matmul_batch_baseline_v6
/* ---- begin inlined native_quant_batch_interleaved_v4.c ---- */
/* Keep the established FP8 path and replace only FP4 batch execution. */
#define coli_fp4_matmul_batch_ref coli_fp4_matmul_batch_baseline_v4
#include "native_quant_batch.c"
#undef coli_fp4_matmul_batch_ref

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    enum { BATCH_MAX = 64, ROW_TILE = 4, SIMD_WIDTH = 8 };
    if (!outputs || !weight || !inputs || batch < 1 || batch > BATCH_MAX ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride) return -1;
    int stride = (batch + SIMD_WIDTH - 1) & -SIMD_WIDTH;

    float *item_activation = malloc(columns * sizeof(*item_activation));
    float *interleaved = calloc(columns * (size_t)stride, sizeof(*interleaved));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!item_activation || !interleaved || !activation_scales) {
        free(activation_scales); free(interleaved); free(item_activation);
        return -1;
    }
    for (int item = 0; item < batch; item++) {
        if (coli_fp8_activation_qdq_ref(
                item_activation, activation_scales,
                inputs + (size_t)item * columns, columns, 128)) {
            free(activation_scales); free(interleaved); free(item_activation);
            return -1;
        }
        for (size_t column = 0; column < columns; column++)
            interleaved[column * (size_t)stride + (size_t)item] =
                item_activation[column];
    }
    free(activation_scales); free(item_activation);

    float fp4[16], e8[256];
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) e8[i] = coli_e8m0_decode((uint8_t)i);
    const uint8_t *packed = weight->data, *scales = weight->scales;
    int64_t tiles = (weight->rows + ROW_TILE - 1) / ROW_TILE;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < tiles; tile++) {
        int lanes = (int)(weight->rows - tile * ROW_TILE);
        if (lanes > ROW_TILE) lanes = ROW_TILE;
        size_t row_data[ROW_TILE], row_scale[ROW_TILE];
        float sums[ROW_TILE][BATCH_MAX] = {{0}};
        for (int lane = 0; lane < lanes; lane++) {
            size_t row = (size_t)tile * ROW_TILE + (size_t)lane;
            row_data[lane] = row * packed_stride;
            row_scale[lane] = row * scale_stride;
        }
        for (size_t base = 0; base < columns; base += 32) {
            float block_scales[ROW_TILE];
            for (int lane = 0; lane < lanes; lane++)
                block_scales[lane] =
                    e8[scales[row_scale[lane] + base / 32]];
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                const float *activation =
                    interleaved + column * (size_t)stride;
                for (int lane = 0; lane < lanes; lane++) {
                    uint8_t byte = packed[row_data[lane] + column / 2];
                    float value = fp4[column & 1 ? byte >> 4 : byte & 15];
                    float scale = block_scales[lane];
                    #pragma omp simd
                    for (int item = 0; item < stride; item++)
                        sums[lane][item] += activation[item] * value * scale;
                }
            }
        }
        for (int lane = 0; lane < lanes; lane++) {
            size_t row = (size_t)tile * ROW_TILE + (size_t)lane;
            for (int item = 0; item < batch; item++)
                outputs[(size_t)item * rows + row] = sums[lane][item];
        }
    }
    free(interleaved);
    return 0;
}
/* ---- end inlined native_quant_batch_interleaved_v4.c ---- */

#undef coli_fp8_matmul_batch_ref

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
#ifndef __AVX512F__
    return coli_fp8_matmul_batch_baseline_v6(outputs, weight, inputs, batch);
#else
    if (!outputs || !weight || !inputs || batch < 1 || batch > 64 ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 128 || weight->block_columns != 128) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128;
    size_t scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns) return -1;

    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * scale_columns);
    if (!activations || !activation_scales) {
        free(activation_scales); free(activations); return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * scale_columns,
                inputs + (size_t)item * columns, columns, 128)) {
            free(activation_scales); free(activations); return -1;
        }

    float fp8[256], e8[256];
    for (int i = 0; i < 256; i++) {
        fp8[i] = coli_e4m3fn_decode((uint8_t)i);
        e8[i] = coli_e8m0_decode((uint8_t)i);
    }
    const uint8_t *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < weight->rows; row++) {
        float sums[64] = {0};
        size_t row_data = (size_t)row * columns;
        size_t scale_row = (size_t)row / 128;
        for (size_t base = 0; base < columns; base += 128) {
            __m512 scale = _mm512_set1_ps(
                e8[scales[scale_row * scale_columns + base / 128]]);
            for (size_t chunk = 0; chunk < 128; chunk += 16) {
                __m128i packed_codes = _mm_loadu_si128(
                    (const __m128i *)(data + row_data + base + chunk));
                __m512i codes = _mm512_cvtepu8_epi32(packed_codes);
                __m512 values = _mm512_i32gather_ps(codes, fp8, 4);
                for (int item = 0; item < batch; item++) {
                    __m512 activation = _mm512_loadu_ps(
                        activations + (size_t)item * columns + base + chunk);
                    __m512 product = _mm512_mul_ps(
                        _mm512_mul_ps(activation, values), scale);
                    float products[16];
                    _mm512_storeu_ps(products, product);
                    for (int lane = 0; lane < 16; lane++)
                        sums[item] += products[lane];
                }
            }
        }
        for (int item = 0; item < batch; item++)
            outputs[(size_t)item * rows + (size_t)row] = sums[item];
    }
    free(activation_scales); free(activations);
    return 0;
#endif
}
