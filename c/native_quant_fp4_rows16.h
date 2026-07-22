#ifndef COLIBRI_NATIVE_QUANT_FP4_ROWS16_H
#define COLIBRI_NATIVE_QUANT_FP4_ROWS16_H

#include "native_quant.h"

int coli_fp4_pack_rows16_v10(unsigned char *packed_data,
                             unsigned char *packed_scales,
                             const ColiTensorView *source);

int coli_fp4_matvec_rows16_v10(float *output,
                               const ColiTensorView *weight,
                               const float *input);

int coli_fp4_dual_matvec_rows16_v10(float *output_a, float *output_b,
                                    const ColiTensorView *a,
                                    const ColiTensorView *b,
                                    const float *input);

#endif
