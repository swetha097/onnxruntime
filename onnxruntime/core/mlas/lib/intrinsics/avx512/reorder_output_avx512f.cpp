/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    reorder_output_avx512f.cpp

Abstract:

    This module implements a 16x4 float transpose kernel using AVX-512F
    intrinsics, used to accelerate NCHWc -> NCHW output reordering.

--*/

#include "mlasi.h"

void
MLASCALL
MlasTransposeFloat32x16x4KernelAvx512F(
    const float* S,
    float* D,
    size_t GatherStride,
    size_t ScatterStride
    )
/*++

Routine Description:

    Transposes a 4-row x 16-column float block to a 16-row x 4-column block.

    Reads 4 rows of 16 floats from S (rows separated by GatherStride elements).
    Writes 16 rows of 4 floats to D (rows separated by ScatterStride elements).

    Used by MlasReorderOutputNchwThreaded when BlockSize==16 (AVX-512F path):
        GatherStride = BlockSize = 16  (NCHWc: 16 channels per block)
        ScatterStride = OutputSize     (NCHW: spatial size)

    Replaces 4 consecutive SSE2 4x4 calls.

--*/
{
    __m512 zmm0 = _mm512_loadu_ps(S);
    __m512 zmm1 = _mm512_loadu_ps(S + GatherStride);
    __m512 zmm2 = _mm512_loadu_ps(S + GatherStride * 2);
    __m512 zmm3 = _mm512_loadu_ps(S + GatherStride * 3);

    // Interleave pairs across the two row-pair groups.
    __m512 zmm4 = _mm512_unpacklo_ps(zmm0, zmm1);
    __m512 zmm5 = _mm512_unpackhi_ps(zmm0, zmm1);
    __m512 zmm6 = _mm512_unpacklo_ps(zmm2, zmm3);
    __m512 zmm7 = _mm512_unpackhi_ps(zmm2, zmm3);

    // Combine pairs to form complete transposed rows.
    __m512 zmm8  = _mm512_shuffle_ps(zmm4, zmm6, 0x44);
    __m512 zmm9  = _mm512_shuffle_ps(zmm4, zmm6, 0xEE);
    __m512 zmm10 = _mm512_shuffle_ps(zmm5, zmm7, 0x44);
    __m512 zmm11 = _mm512_shuffle_ps(zmm5, zmm7, 0xEE);

    // Store 16 output rows.  Extract one 128-bit lane per store; each lane
    // carries the 4 spatial values for one channel.
    _mm_storeu_ps(D + ScatterStride *  0, _mm512_extractf32x4_ps(zmm8,  0));
    _mm_storeu_ps(D + ScatterStride *  1, _mm512_extractf32x4_ps(zmm9,  0));
    _mm_storeu_ps(D + ScatterStride *  2, _mm512_extractf32x4_ps(zmm10, 0));
    _mm_storeu_ps(D + ScatterStride *  3, _mm512_extractf32x4_ps(zmm11, 0));
    _mm_storeu_ps(D + ScatterStride *  4, _mm512_extractf32x4_ps(zmm8,  1));
    _mm_storeu_ps(D + ScatterStride *  5, _mm512_extractf32x4_ps(zmm9,  1));
    _mm_storeu_ps(D + ScatterStride *  6, _mm512_extractf32x4_ps(zmm10, 1));
    _mm_storeu_ps(D + ScatterStride *  7, _mm512_extractf32x4_ps(zmm11, 1));
    _mm_storeu_ps(D + ScatterStride *  8, _mm512_extractf32x4_ps(zmm8,  2));
    _mm_storeu_ps(D + ScatterStride *  9, _mm512_extractf32x4_ps(zmm9,  2));
    _mm_storeu_ps(D + ScatterStride * 10, _mm512_extractf32x4_ps(zmm10, 2));
    _mm_storeu_ps(D + ScatterStride * 11, _mm512_extractf32x4_ps(zmm11, 2));
    _mm_storeu_ps(D + ScatterStride * 12, _mm512_extractf32x4_ps(zmm8,  3));
    _mm_storeu_ps(D + ScatterStride * 13, _mm512_extractf32x4_ps(zmm9,  3));
    _mm_storeu_ps(D + ScatterStride * 14, _mm512_extractf32x4_ps(zmm10, 3));
    _mm_storeu_ps(D + ScatterStride * 15, _mm512_extractf32x4_ps(zmm11, 3));
}
