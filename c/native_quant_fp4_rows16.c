#include "native_quant_fp4_rows16.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

static int source_valid(const ColiTensorView *weight) {
    if (!weight || weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 || !weight->data ||
        !weight->scales || weight->rows < 1 || weight->rows % 16 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return 0;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    return weight->data_bytes == rows * columns / 2 &&
           weight->scale_bytes == rows * columns / 32;
}

static int packed_valid(const ColiTensorView *weight) {
    if (!weight || weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 || !weight->data ||
        !weight->scales || weight->rows < 1 || weight->rows % 16 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 16 || weight->block_columns != 32) return 0;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    return weight->data_bytes == rows * columns / 2 &&
           weight->scale_bytes == rows * columns / 32;
}

int coli_fp4_pack_rows16_v10(unsigned char *packed_data,
                             unsigned char *packed_scales,
                             const ColiTensorView *source) {
    if (!packed_data || !packed_scales || !source_valid(source)) return -1;
    const unsigned char *data = source->data;
    const unsigned char *scales = source->scales;
    size_t rows = (size_t)source->rows, columns = (size_t)source->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    for (size_t row = 0; row < rows; row++) {
        size_t tile = row / 16, lane = row % 16;
        for (size_t column = 0; column < data_stride; column++)
            packed_data[(tile * data_stride + column) * 16 + lane] =
                data[row * data_stride + column];
        for (size_t column = 0; column < scale_stride; column++)
            packed_scales[(tile * scale_stride + column) * 16 + lane] =
                scales[row * scale_stride + column];
    }
    return 0;
}

#ifdef __AVX512F__
static void decode_tables(__m512 *fp4_table, float e8[256]) {
    float fp4[16];
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) e8[i] = coli_e8m0_decode((uint8_t)i);
    *fp4_table = _mm512_loadu_ps(fp4);
}

static __m512 decode_fp4_rows16(const unsigned char *data, int high,
                                __m512 fp4_table) {
    __m512i codes = _mm512_cvtepu8_epi32(
        _mm_loadu_si128((const __m128i *)data));
    codes = high ? _mm512_srli_epi32(codes, 4)
                 : _mm512_and_si512(codes, _mm512_set1_epi32(15));
    return _mm512_permutexvar_ps(codes, fp4_table);
}

static __m512 decode_scales_rows16(const unsigned char *scales,
                                   const float e8[256]) {
    __m512i codes = _mm512_cvtepu8_epi32(
        _mm_loadu_si128((const __m128i *)scales));
    return _mm512_i32gather_ps(codes, e8, 4);
}
#endif

int coli_fp4_matvec_rows16_v10(float *output,
                               const ColiTensorView *weight,
                               const float *input) {
#ifndef __AVX512F__
    (void)output; (void)weight; (void)input; return -1;
#else
    if (!output || !input || !packed_valid(weight)) return -1;
    size_t columns = (size_t)weight->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    __m512 fp4_table; float e8[256]; decode_tables(&fp4_table, e8);
    const unsigned char *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < weight->rows / 16; tile++) {
        __m512 sum = _mm512_setzero_ps();
        for (size_t base = 0; base < columns; base += 32) {
            __m512 scale = decode_scales_rows16(
                scales + ((size_t)tile * scale_stride + base / 32) * 16, e8);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                const unsigned char *codes = data +
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m512 values = decode_fp4_rows16(codes, column & 1, fp4_table);
                __m512 product = _mm512_mul_ps(
                    _mm512_mul_ps(_mm512_set1_ps(activation[column]), values),
                    scale);
                sum = _mm512_add_ps(sum, product);
            }
        }
        _mm512_storeu_ps(output + (size_t)tile * 16, sum);
    }
    free(activation_scales); free(activation); return 0;
#endif
}

int coli_fp4_dual_matvec_rows16_v10(float *output_a, float *output_b,
                                    const ColiTensorView *a,
                                    const ColiTensorView *b,
                                    const float *input) {
#ifndef __AVX512F__
    (void)output_a; (void)output_b; (void)a; (void)b; (void)input; return -1;
#else
    if (!output_a || !output_b || !input || !packed_valid(a) ||
        !packed_valid(b) || a->rows != b->rows ||
        a->columns != b->columns) return -1;
    size_t columns = (size_t)a->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    __m512 fp4_table; float e8[256]; decode_tables(&fp4_table, e8);
    const unsigned char *data_a = a->data, *data_b = b->data;
    const unsigned char *scales_a = a->scales, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < a->rows / 16; tile++) {
        __m512 sum_a = _mm512_setzero_ps(), sum_b = _mm512_setzero_ps();
        for (size_t base = 0; base < columns; base += 32) {
            const unsigned char *scale_offset_a = scales_a +
                ((size_t)tile * scale_stride + base / 32) * 16;
            const unsigned char *scale_offset_b = scales_b +
                ((size_t)tile * scale_stride + base / 32) * 16;
            __m512 scale_a = decode_scales_rows16(scale_offset_a, e8);
            __m512 scale_b = decode_scales_rows16(scale_offset_b, e8);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                size_t packed_offset =
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m512 values_a = decode_fp4_rows16(
                    data_a + packed_offset, column & 1, fp4_table);
                __m512 values_b = decode_fp4_rows16(
                    data_b + packed_offset, column & 1, fp4_table);
                __m512 x = _mm512_set1_ps(activation[column]);
                sum_a = _mm512_add_ps(sum_a, _mm512_mul_ps(
                    _mm512_mul_ps(x, values_a), scale_a));
                sum_b = _mm512_add_ps(sum_b, _mm512_mul_ps(
                    _mm512_mul_ps(x, values_b), scale_b));
            }
        }
        _mm512_storeu_ps(output_a + (size_t)tile * 16, sum_a);
        _mm512_storeu_ps(output_b + (size_t)tile * 16, sum_b);
    }
    free(activation_scales); free(activation); return 0;
#endif
}
