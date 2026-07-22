#include "native_quant.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

float coli_e8m0_decode(uint8_t value) {
    if (value == 0xff) return NAN;
    return ldexpf(1.0f, (int)value - 127);
}

float coli_e2m1_decode(uint8_t nibble) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 15];
}

float coli_e4m3fn_decode(uint8_t value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number;
    if (!exponent)
        number = ldexpf((float)mantissa, -9);
    else
        number = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

uint8_t coli_e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uint8_t)((negative ? 0x80 : 0) | 0x7e);

    uint8_t best = 0;
#if FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128
    /* Exact binary32 round-to-nearest-even conversion in constant time. */
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1)))
            rounded++;
        best = rounded;
    } else {
        uint32_t bits;
        memcpy(&bits, &magnitude, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu);
        uint32_t rounded = significand >> 20;
        uint32_t remainder = significand & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (rounded & 1u)))
            rounded++;
        if (rounded == 16u) {
            rounded = 8u;
            exponent++;
        }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
#else
    float best_distance = FLT_MAX;
    for (uint8_t code = 0; code <= 0x7e; code++) {
        float candidate = coli_e4m3fn_decode(code);
        float distance = fabsf(candidate - magnitude);
        if (distance < best_distance ||
            (distance == best_distance && !(code & 1) && (best & 1))) {
            best = code;
            best_distance = distance;
        }
    }
#endif
    return (uint8_t)(best | (negative ? 0x80 : 0));
}

float coli_bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float coli_bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

void coli_bf16_round_array(float *values, size_t count) {
    if (!values) return;
    for (size_t index = 0; index < count; index++)
        values[index] = coli_bf16_round(values[index]);
}

static int ceil_log2_positive(float value) {
    int exponent;
    float fraction = frexpf(value, &exponent);
    return fraction == 0.5f ? exponent - 1 : exponent;
}

int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size) {
    if (!output || !scales || !input || !length || !block_size)
        return -1;
    for (size_t base = 0; base < length; base += block_size) {
        size_t count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        maximum = fmaxf(maximum, 1e-4f);
        int scale_exponent = ceil_log2_positive(maximum / 448.0f);
        if (scale_exponent < -127) scale_exponent = -127;
        if (scale_exponent > 127) scale_exponent = 127;
        uint8_t encoded_scale = (uint8_t)(scale_exponent + 127);
        float scale = coli_e8m0_decode(encoded_scale);
        scales[base / block_size] = encoded_scale;
        for (size_t i = 0; i < count; i++) {
            float normalized = fmaxf(-448.0f,
                                     fminf(448.0f, input[base + i] / scale));
            output[base + i] = coli_e4m3fn_decode(
                coli_e4m3fn_encode(normalized)) * scale;
        }
    }
    return 0;
}

int coli_fp4_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size) {
    if (!output || !scales || !input || !length || !block_size)
        return -1;
    for (size_t base = 0; base < length; base += block_size) {
        size_t count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        maximum = fmaxf(maximum, 6.0f * ldexpf(1.0f, -126));
        int exponent = ceil_log2_positive(maximum / 6.0f);
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
        scales[base / block_size] = (uint8_t)(exponent + 127);
        float scale = coli_e8m0_decode(scales[base / block_size]);
        for (size_t i = 0; i < count; i++) {
            float value = fmaxf(-6.0f, fminf(6.0f, input[base + i] / scale));
            int best = 0;
            float distance = fabsf(value - coli_e2m1_decode(0));
            for (int code = 1; code < 16; code++) {
                float candidate = fabsf(value - coli_e2m1_decode((uint8_t)code));
                if (candidate < distance) {
                    distance = candidate;
                    best = code;
                }
            }
            output[base + i] = coli_e2m1_decode((uint8_t)best) * scale;
        }
    }
    return 0;
}

int coli_hadamard_bf16_ref(float *values, size_t length) {
    if (!values || !length || (length & (length - 1))) return -1;
    for (size_t width = 1; width < length; width *= 2)
        for (size_t base = 0; base < length; base += 2 * width)
            for (size_t i = 0; i < width; i++) {
                float left = values[base + i];
                float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
    float scale = 1.0f / sqrtf((float)length);
    for (size_t i = 0; i < length; i++)
        values[i] = coli_bf16_round(values[i] * scale);
    return 0;
}

int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32)
        return -1;
    size_t rows = (size_t)weight->rows;
    size_t columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2;
    size_t scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride)
        return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    const uint8_t *packed = weight->data;
    const uint8_t *scales = weight->scales;
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (size_t column = 0; column < columns; column++) {
            uint8_t byte = packed[row * packed_stride + column / 2];
            uint8_t code = column & 1 ? byte >> 4 : byte & 15;
            float scale = coli_e8m0_decode(
                scales[row * scale_stride + column / 32]);
            sum += activation[column] * coli_e2m1_decode(code) * scale;
        }
        output[row] = sum;
    }
    free(activation_scales);
    free(activation);
    return 0;
}

int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 128 || weight->block_columns != 128)
        return -1;
    size_t rows = (size_t)weight->rows;
    size_t columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128;
    size_t scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns)
        return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(scale_columns);
    if (!activation || !activation_scales) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    const uint8_t *data = weight->data;
    const uint8_t *scales = weight->scales;
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        size_t scale_row = row / 128;
        for (size_t column = 0; column < columns; column++) {
            float value = coli_e4m3fn_decode(data[row * columns + column]);
            float scale = coli_e8m0_decode(
                scales[scale_row * scale_columns + column / 128]);
            sum += activation[column] * value * scale;
        }
        output[row] = sum;
    }
    free(activation_scales);
    free(activation);
    return 0;
}
