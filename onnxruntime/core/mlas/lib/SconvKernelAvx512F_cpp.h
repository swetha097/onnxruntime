// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SconvKernelAvx512F_cpp.h
//
// C++ AVX-512 intrinsics equivalent of:
//   MlasConvNchwcFloatKernelAvx512F   (hotspot #1 in VTune)
//   MlasConvPointwiseFloatKernelAvx512F  (hotspot #2 in VTune)
//
// Translated from the assembly macro framework in:
//   onnxruntime/core/mlas/lib/amd64/SconvKernelAvx512F.asm
//
// PURPOSE
//   The assembly implementation is opaque to instruction-level profilers —
//   VTune cannot attribute time to individual load vs. FMA instructions
//   inside a multi-page macro-generated function.  This C++ version uses
//   _mm512_* intrinsics so that the compiler emits equivalent instructions
//   with debuggable line-number attribution.  Profiling this version lets
//   you determine whether the hotspot is:
//     - Memory-bound  (samples on _mm512_loadu_ps / _mm512_set1_ps lines)
//     - Compute-bound (samples on _mm512_fmadd_ps lines)
//     - Output-store  (samples on _mm512_storeu_ps lines)
//
// CALLING THIS CODE
//   In snchwc.cpp, where the platform kernel pointer is selected, add:
//
//     #if defined(PROFILING_BUILD)
//       #include "SconvKernelAvx512F_cpp.h"
//       Kernel = MlasConvNchwcFloatKernelAvx512F_Cpp;
//     #endif
//
// DATA LAYOUTS  (NCHWc16 / OIhw16i16o)
//
//   Input   [N, C/16, H, W, 16f]
//     Channel blocks are contiguous; within a block the 16 channels are the
//     innermost dimension.  Accessing channel i of spatial position (h,w):
//       input_ptr + (c_blk*H*W + h*W + w) * 16 * sizeof(float) + i*sizeof(float)
//
//   Filter  [OC/16, IC/16, KH, KW, 16ic, 16oc]   ("OIhw16i16o" blocked)
//     For one output filter row, one kernel position, one input channel
//     block:  a 16×16 float tile = 1024 bytes.
//     Layout within that tile: [ic=0..15][oc=0..15]
//       filter_ptr + f*FilterStride + (kr*KW+kc)*16*16*4 + ic*16*4 + oc*4
//
//   Output  [N, OC/16, H_out, W_out, 16f]
//     Mirror of input layout.  Each output position is a 16-float block.
//
// KEY STRIDE PARAMETERS (from snchwc.cpp)
//   StrideWidthBytes   = BlockSize * spatial_stride * sizeof(float)
//   DilationWidthBytes = BlockSize * dilation_w   * sizeof(float)
//   FilterStrideBytes  = BlockSize * InputChannels * KernelSize * sizeof(float)
//   OutputStrideBytes  = BlockSize * OutputH * OutputW * sizeof(float)
//   InputStrideBytes   = DilatedInputWidthBytes - KW * DilationWidthBytes
//   (Pointwise InputStride = BlockSize * InputH * InputW * sizeof(float))
//
// ACCUMULATOR REGISTER LAYOUT  (mirrors assembly zmm assignment)
//   The assembly assigns zmm0..23 as a FilterCount × OutputCount 2-D tile:
//     filter_row 0 : zmm0  zmm4  zmm8  zmm12 zmm16 zmm20  (output cols 0-5)
//     filter_row 1 : zmm1  zmm5  zmm9  zmm13 zmm17 zmm21
//     filter_row 2 : zmm2  zmm6  zmm10 zmm14 zmm18 zmm22
//     filter_row 3 : zmm3  zmm7  zmm11 zmm15 zmm19 zmm23
//   In C++ we represent this as acc[FilterCount][OutputCount].
//   Scratch registers zmm24..31 are used for filter vectors and input broadcasts.
//
#pragma once

#include <immintrin.h>
#include <cstddef>

#if defined(_MSC_VER)
#  define SCONV_FORCEINLINE __forceinline
#else
#  define SCONV_FORCEINLINE __attribute__((always_inline)) inline
#endif



// Kernel flags (must match SconvKernelCommon.inc / mlasi.h)
#ifndef MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT 0x01u
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION     0x02u
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION   0x04u
#endif

// AVX-512 NCHWc block size: 16 floats = one ZMM register
static constexpr int AVX512_BS = 16;

// ─────────────────────────────────────────────────────────────────────────────
// PostProcess
//
// Mirrors MlasConvPostProcessFloatAvx512FFilterFC_OutputOC assembly functions.
//
// Optionally accumulates existing output, adds bias, applies ReLU, then stores
// the FC × OC tile of ZMM accumulators to the output buffer.
//
// acc[f][o]   — accumulator for filter row f, output column o
// out_ptr     — base of output tile  (filter_row=0, output_col=0)
// out_stride  — byte stride between consecutive filter-row output rows
//               (= OutputStrideBytes passed to the kernel)
// bias        — FC blocks of 16 floats (one ZMM per filter row)
// flags       — MLAS_CONV_KERNEL_FLAG_* bitmask
// ─────────────────────────────────────────────────────────────────────────────
template<int FC, int OC>
static void Avx512PostProcess(
    __m512 (&acc)[FC][OC],
    float* out_ptr,
    size_t out_stride,
    const float* bias,
    unsigned flags)
{
    // OPT-2: merge accumulate+bias+relu+store into one for-f for-o pass.
    // OPT-5: when bias+accum both active, fuse via fmadd(1, existing_out, acc+bias)
    //        emitting vfmadd231ps instead of two vaddps.
    const bool do_accum = (flags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool do_bias  = (flags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION)     != 0;
    const bool do_relu  = (flags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)   != 0;
    const __m512 zero   = _mm512_setzero_ps();
    const __m512 one    = _mm512_set1_ps(1.0f);
    for (int f = 0; f < FC; f++) {
        float* row = reinterpret_cast<float*>(
            reinterpret_cast<char*>(out_ptr) + f * out_stride);
        const __m512 bvec = do_bias ? _mm512_loadu_ps(bias + f * AVX512_BS)  // [LOAD-BIAS]
                                    : _mm512_setzero_ps();
        for (int o = 0; o < OC; o++) {
            __m512 v = do_bias ? _mm512_add_ps(acc[f][o], bvec) : acc[f][o];
            if (do_accum) v = _mm512_fmadd_ps(one, _mm512_loadu_ps(row + o * AVX512_BS), v);  // [LOAD-OUTPUT + FMA]
            if (do_relu)  v = _mm512_max_ps(zero, v);  // [RELU]
            _mm512_storeu_ps(row + o * AVX512_BS, v);  // [STORE-OUTPUT]
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// NCHWc CONVOLUTION KERNEL
// Equivalent of MlasConvNchwcFloatKernelAvx512F
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// ComputeNchwcBlock
//
// Mirrors the ComputeBlock macro (KernelType=Nchwc) in SconvKernelAvx512F.asm.
//
// For each of the 16 input channel indices within the current 16i×16o filter
// tile (one kernel spatial position, one input channel block):
//   1. [LOAD-INPUT]  Broadcast input scalar for each output column
//                    Assembly: vbroadcastss zmm26..31, [rcx+o*StrideWidth+i*4]
//                    For FC=1, assembly uses embedded DWORD BCST form in FMA.
//   2. [LOAD-FILTER] Load 16 output-channel weights for each filter row
//                    Assembly: vmovups zmm24, [rdx+f*FilterStride+i*64]
//   3. [FMA]         acc[f][o] += filter_vec * input_broadcast
//                    Assembly: vfmadd231ps zmm_acc, zmm_in_bc, zmm24
//
// input_block       — ptr to the 16-float input block for output_col=0
// stride_width      — bytes between consecutive output columns (= StrideWidth)
// filter_base       — ptr to start of 16i×16o tile for filter_row=0
// filter_stride     — bytes from filter row 0 to filter row 1 (= FilterStride)
// ─────────────────────────────────────────────────────────────────────────────
template<int FC, int OC>
static SCONV_FORCEINLINE void ComputeNchwcBlock(
    __m512 (&acc)[FC][OC],
    const float* input_block,
    size_t stride_width,
    const float* filter_base,
    size_t filter_stride)
{
    // OPT-1: f-outer loop — filter loaded once per (f,i), reused across all OC.
    //        in_bc[] eliminated; broadcast inlined → vfmadd231ps zmm, zmm, [mem]{1to16}.
    //        ZMM pressure: FC=4,OC=6 was 31 (24 acc+6 in_bc+1 filt), now 25 (24 acc+1 filt).
    // OPT-3: _mm512_load_ps (vmovaps) — 16×16 filter tiles are 64-byte aligned.
    for (int i = 0; i < AVX512_BS; i++) {
        for (int f = 0; f < FC; f++) {
            const float* fp = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(filter_base) + f * filter_stride)
                + i * AVX512_BS;
            const __m512 fvec = _mm512_load_ps(fp);   // [LOAD-FILTER] vmovaps 64 bytes (aligned)
            for (int o = 0; o < OC; o++) {
                const float* p = reinterpret_cast<const float*>(
                    reinterpret_cast<const char*>(input_block) + o * stride_width) + i;
                acc[f][o] = _mm512_fmadd_ps(fvec, _mm512_set1_ps(*p), acc[f][o]);  // [FMA+BCST]
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessNchwcOutputs
//
// Mirrors ProcessOutputCountN (Avx512F, Nchwc, BlockSize=16, FilterCount, OC).
//
// Iterates over all kernel rows and columns to accumulate into a
// FC × OC output tile, then post-processes and stores.
//
// This handles only interior (non-padded) output positions.
// For padded positions use ProcessNchwcSingleOutput (below).
//
// input_pos    — input pointer for output_col=0, already accounting for the
//               current kernel row start (outer code adjusts for top padding)
// filter_base  — filter for current kernel-height-start, filter_row=0
// out_ptr      — in/out: advanced by OC blocks after storing
// ─────────────────────────────────────────────────────────────────────────────
template<int FC, int OC>
static void ProcessNchwcOutputs(
    const float* input_pos,
    const float* filter_base,
    float*& out_ptr,
    size_t stride_width,
    size_t dilation_width,
    size_t filter_stride,
    size_t out_stride,
    size_t input_stride,
    size_t kernel_height,
    size_t kernel_width,
    const float* bias,
    unsigned flags)
{
    // printf("Processing Nchwc Outputs 512\n");
    // fflush(stdout);
    // Clear accumulators — vpxord zmm0..zmm23
    __m512 acc[FC][OC];
    for (int f = 0; f < FC; f++)
        for (int o = 0; o < OC; o++)
            acc[f][o] = _mm512_setzero_ps();

    const float* row_input = input_pos;
    const float* row_filter = filter_base;

    for (size_t kr = 0; kr < kernel_height; kr++) {
        const float* col_input  = row_input;
        const float* col_filter = row_filter;

        for (size_t kc = 0; kc < kernel_width; kc++) {
            // OPT-4: prefetch next filter tile into L1 (T0); next input block into L2 (T1)
            _mm_prefetch(reinterpret_cast<const char*>(col_filter) + AVX512_BS * AVX512_BS * sizeof(float), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(col_input)  + dilation_width, _MM_HINT_T1);

            // Inner unrolled loop over 16 input channel indices
            // (mirrors "IRP Index, <0,1,...,15>" in assembly)
            ComputeNchwcBlock<FC, OC>(acc, col_input, stride_width, col_filter, filter_stride);

            // Advance input by DilationWidth bytes (next kernel column)
            col_input = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(col_input) + dilation_width);
            // Advance filter by one 16i×16o tile = 1024 bytes
            col_filter += AVX512_BS * AVX512_BS;  // = 256 floats = 1024 bytes
        }

        // Advance input to next kernel row (InputStride bytes)
        // row_input = reinterpret_cast<const float*>(
        //     reinterpret_cast<const char*>(row_input) + input_stride);
        row_input = reinterpret_cast<const float*>(    reinterpret_cast<const char*>(col_input) + input_stride);  // ← col_input is correct
        // Filter stays anchored at row_filter — each kernel column step above
        // already advanced col_filter, and we restart col_filter each row.
        // But note: assembly does NOT reset rdx per kernel row — it keeps
        // advancing.  To match: row_filter is NOT reset, it continues.
        row_filter = col_filter;  // already advanced past all KW columns
    }

    // Post-process and store the FC × OC tile
    Avx512PostProcess<FC, OC>(acc, out_ptr, out_stride, bias, flags);
    // Advance output pointer by OC * 16 floats (= OC * 64 bytes)
    out_ptr += OC * AVX512_BS;  // = out_ptr + OC*16*sizeof(float) bytes
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessNchwcSingleOutput
//
// Handles padded output positions (OutputCountLeftPad / OutputCountRightPad),
// where each output column must be bounds-checked against the valid input
// range before contributing to the accumulator.
//
// Mirrors MlasConvNchwcFloatSingleAvx512FFilterFC in the assembly, which loops
// one output at a time and uses:
//   lea rbx, [rcx+r13]      ; Input - InputBase
//   cmp rbx, r14            ; >= InputWidth (bytes) ?
//   jae SkipOverPadding     ; yes → treat as zero (padding)
//
// input_base  — valid input start pointer (no left-pad bias)
// input_width — valid input width in bytes (= BlockSize * InputWidth * 4)
// ─────────────────────────────────────────────────────────────────────────────
template<int FC>
static void ProcessNchwcSingleOutput(
    const float* input_pos,     // biased (may be in left-pad region)
    const float* input_base,    // unbiased valid input start
    size_t input_width_bytes,   // valid input width in bytes
    size_t dilated_input_width, // bytes to advance input_base per kernel row
    const float* filter_base,
    float*& out_ptr,
    size_t stride_width,
    size_t dilation_width,
    size_t filter_stride,
    size_t out_stride,
    size_t input_stride,
    size_t kernel_height,
    size_t kernel_width,
    const float* bias,
    unsigned flags)
{
    __m512 acc[FC][1];
    for (int f = 0; f < FC; f++)
        acc[f][0] = _mm512_setzero_ps();

    const float* row_input      = input_pos;
    const float* row_input_base = input_base;
    const float* row_filter     = filter_base;

    for (size_t kr = 0; kr < kernel_height; kr++) {
        const float* col_input  = row_input;
        const float* col_filter = row_filter;

        for (size_t kc = 0; kc < kernel_width; kc++) {
            // Padding bounds check:
            //   (col_input - input_base) >= input_width_bytes  → padding, skip
            size_t offset = reinterpret_cast<size_t>(col_input)
                          - reinterpret_cast<size_t>(row_input_base);
            if (offset < input_width_bytes) {
                ComputeNchwcBlock<FC, 1>(acc, col_input, stride_width, col_filter, filter_stride);
            }

            col_input = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(col_input) + dilation_width);
            col_filter += AVX512_BS * AVX512_BS;
        }

        row_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(col_input) + input_stride);
        row_input_base = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(row_input_base) + dilated_input_width);
        row_filter = col_filter;
    }

    Avx512PostProcess<FC, 1>(acc, out_ptr, out_stride, bias, flags);
    out_ptr += AVX512_BS;  // advance output by one block
}

// ─────────────────────────────────────────────────────────────────────────────
// RunNchwcFilterCount
//
// Outer spatial loop for a fixed FilterCount.
// Mirrors ProcessFilterCountN macro in SconvKernelAvx512F.asm:
//   - Calls MlasConvNchwcFloatSingleAvx512FFilterN for padded outputs (OC=1)
//   - Processes 6 / 3 / 2 output blocks at a time for the main region
//   - Then calls Single again for right-pad + remaining
// ─────────────────────────────────────────────────────────────────────────────
template<int FC>
static void RunNchwcFilterCount(
    const float* input_pos,
    const float* input_base,
    size_t input_width_bytes,
    size_t dilated_input_width,
    const float* filter_start,
    float*& out_ptr,
    size_t stride_width,
    size_t dilation_width,
    size_t filter_stride,
    size_t out_stride,
    size_t input_stride,
    size_t kernel_height,
    size_t kernel_width,
    size_t out_count_left,
    size_t out_count_main,
    size_t out_count_right,
    const float* bias,
    unsigned flags)
{
    const float* cur_input = input_pos;

    // ── Left-padded outputs (each processed one at a time with bounds check) ──
    for (size_t n = 0; n < out_count_left; n++) {
        ProcessNchwcSingleOutput<FC>(
            cur_input, input_base, input_width_bytes, dilated_input_width,
            filter_start, out_ptr,
            stride_width, dilation_width, filter_stride, out_stride,
            input_stride, kernel_height, kernel_width, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + stride_width);
    }

    // ── Main outputs: process in batches of 6, then 3, then 2 ──
    size_t remaining = out_count_main;

    // Batch of 6 (AVX-512 processes 6 output columns × 4 filter rows)
    while (remaining >= 6) {
        ProcessNchwcOutputs<FC, 6>(
            cur_input, filter_start, out_ptr,
            stride_width, dilation_width, filter_stride, out_stride,
            input_stride, kernel_height, kernel_width, bias, flags);
        // Advance input by 6 × StrideWidth
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + 6 * stride_width);
        remaining -= 6;
    }

    // Batch of 3
    if (remaining >= 3) {
        ProcessNchwcOutputs<FC, 3>(
            cur_input, filter_start, out_ptr,
            stride_width, dilation_width, filter_stride, out_stride,
            input_stride, kernel_height, kernel_width, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + 3 * stride_width);
        remaining -= 3;
    }

    // Batch of 2
    if (remaining >= 2) {
        ProcessNchwcOutputs<FC, 2>(
            cur_input, filter_start, out_ptr,
            stride_width, dilation_width, filter_stride, out_stride,
            input_stride, kernel_height, kernel_width, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + 2 * stride_width);
        remaining -= 2;
    }
    // If remaining == 1, it falls into the right-pad handler below.

    // ── Right-padded outputs + any 1 remaining from main batch ──
    size_t right_total = out_count_right + remaining;
    for (size_t n = 0; n < right_total; n++) {
        ProcessNchwcSingleOutput<FC>(
            cur_input, input_base, input_width_bytes, dilated_input_width,
            filter_start, out_ptr,
            stride_width, dilation_width, filter_stride, out_stride,
            input_stride, kernel_height, kernel_width, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + stride_width);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MlasConvNchwcFloatKernelAvx512F_Cpp
//
// Drop-in C++ replacement for MlasConvNchwcFloatKernelAvx512F.
// Signature matches MLAS_CONV_FLOAT_KERNEL typedef (see mlasi.h).
//
// Called from snchwc.cpp for each input channel block (ic loop):
//   Kernel(input + BS*(ih*W - PadLeftX), filter + BS*ic*KernelSize,
//          output, StrideWidthBytes, DilationWidthBytes, FilterCount,
//          InputStrideBytes, FilterStrideBytes, OutputStrideBytes,
//          EffKernelHeight, KernelWidth,
//          input + BS*(ih*W), InputWidthBytes, DilatedInputWidthBytes,
//          OutputCountLeftPad, OutputCount, OutputCountRightPad,
//          Bias, Flags)
// ─────────────────────────────────────────────────────────────────────────────
inline void MlasConvNchwcFloatKernelAvx512F_Cpp(
    const float* Input,           // biased input (may point into left-pad region)
    const float* Filter,          // OIhw16i16o filter block for this channel group
    float* Output,
    size_t StrideWidth,           // bytes between output columns
    size_t DilationWidth,         // bytes between kernel columns in input
    size_t FilterCount,           // 1-4: output filter rows to process
    size_t InputStride,           // bytes to advance input per kernel row
    size_t FilterStride,          // bytes between filter output rows
    size_t OutputStride,          // bytes between output rows
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,       // unbiased valid input start
    size_t InputWidth,            // valid input width in bytes
    size_t DilatedInputWidth,     // bytes to advance InputBase per kernel row
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned Flags)
{
    float* out_ptr = Output;

    // Dispatch on FilterCount (mirrors cmp r11,3 / jb / je chains in assembly)
    switch (FilterCount) {
    case 4:
        RunNchwcFilterCount<4>(
            Input, InputBase, InputWidth, DilatedInputWidth, Filter, out_ptr,
            StrideWidth, DilationWidth, FilterStride, OutputStride, InputStride,
            KernelHeight, KernelWidth,
            OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, Flags);
        break;
    case 3:
        RunNchwcFilterCount<3>(
            Input, InputBase, InputWidth, DilatedInputWidth, Filter, out_ptr,
            StrideWidth, DilationWidth, FilterStride, OutputStride, InputStride,
            KernelHeight, KernelWidth,
            OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, Flags);
        break;
    case 2:
        RunNchwcFilterCount<2>(
            Input, InputBase, InputWidth, DilatedInputWidth, Filter, out_ptr,
            StrideWidth, DilationWidth, FilterStride, OutputStride, InputStride,
            KernelHeight, KernelWidth,
            OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, Flags);
        break;
    default:  // FilterCount == 1
        RunNchwcFilterCount<1>(
            Input, InputBase, InputWidth, DilatedInputWidth, Filter, out_ptr,
            StrideWidth, DilationWidth, FilterStride, OutputStride, InputStride,
            KernelHeight, KernelWidth,
            OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, Flags);
        break;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// POINTWISE CONVOLUTION KERNEL
// Equivalent of MlasConvPointwiseFloatKernelAvx512F
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// ComputePointwiseBlock
//
// Mirrors the inner "ProcessNextInputBlock" body in ProcessPointwiseOutputCountN
// (SconvKernelCommon.inc / SconvKernelAvx512F.asm).
//
// For each of the 16 input channel indices within the current channel block:
//   1. [LOAD-INPUT]  Broadcast input scalar for each output column
//   2. [LOAD-FILTER] Load 16 output-channel weights for each filter row
//   3. [FMA]         acc[f][o] += filter_vec * input_broadcast
//
// Identical pattern to ComputeNchwcBlock; pointwise just calls this once per
// channel block without the spatial kernel row/col loops.
//
// input_block    — ptr to the 16-float input block for channel_blk, output_col=0
// stride_width   — bytes between output columns (= StrideWidth)
// filter_block   — ptr to 16i×16o filter tile for filter_row=0, channel_blk
// filter_stride  — bytes between filter output rows (= FilterStride)
// ─────────────────────────────────────────────────────────────────────────────
template<int FC, int OC>
static SCONV_FORCEINLINE void ComputePointwiseBlock(
    __m512 (&acc)[FC][OC],
    const float* input_block,
    size_t stride_width,
    const float* filter_block,
    size_t filter_stride)
{
    // OPT-1: f-outer loop; in_bc[] eliminated; broadcast inlined into fmadd.
    // OPT-3: _mm512_load_ps — filter tiles are 64-byte aligned.
    for (int i = 0; i < AVX512_BS; i++) {
        for (int f = 0; f < FC; f++) {
            const float* fp = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(filter_block) + f * filter_stride)
                + i * AVX512_BS;
            const __m512 fvec = _mm512_load_ps(fp);   // [LOAD-FILTER] vmovaps 64 bytes (aligned)
            for (int o = 0; o < OC; o++) {
                const float* p = reinterpret_cast<const float*>(
                    reinterpret_cast<const char*>(input_block) + o * stride_width) + i;
                acc[f][o] = _mm512_fmadd_ps(fvec, _mm512_set1_ps(*p), acc[f][o]);  // [FMA+BCST]
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProcessPointwiseOutputs
//
// Mirrors ProcessPointwiseOutputCountN (Avx512F, BlockSize=16, FC, OC).
//
// Iterates over all InputChannels channel blocks (the pointwise "kernel loop"),
// accumulating into a FC × OC tile, then post-processes and stores.
//
// input_pos      — input pointer for output_col=0, current output position
// filter_base    — filter pointer at start of this kernel call's channel range
// out_ptr        — in/out: advanced by OC blocks after storing
// input_stride   — bytes to advance input to the next channel block
//                  (= BlockSize * InputH * InputW * sizeof(float))
// filter_stride  — bytes between filter output rows
//                  (= BlockSize * TotalInputChannels * sizeof(float))
// input_channels — number of 16-channel blocks to process (InputChannelBatch/BS)
// ─────────────────────────────────────────────────────────────────────────────
template<int FC, int OC>
static void ProcessPointwiseOutputs(
    const float* input_pos,
    const float* filter_base,
    float*& out_ptr,
    size_t stride_width,
    size_t input_stride,
    size_t filter_stride,
    size_t out_stride,
    size_t input_channels,
    const float* bias,
    unsigned flags)
{
    // printf("ProcessPointwiseOutputs: FC=%d, OC=%d\n", FC, OC);
    // fflush(stdout);
    __m512 acc[FC][OC];
    for (int f = 0; f < FC; f++)
        for (int o = 0; o < OC; o++)
            acc[f][o] = _mm512_setzero_ps();

    const float* cur_input  = input_pos;
    const float* cur_filter = filter_base;

    // Outer loop: iterate over input channel blocks
    // Each iteration processes one 16i×16o filter tile
    for (size_t ch = 0; ch < input_channels; ch++) {
        // OPT-4: prefetch next filter tile into L1 (T0); next input block into L2 (T1)
        _mm_prefetch(reinterpret_cast<const char*>(cur_filter) + AVX512_BS * AVX512_BS * sizeof(float), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(cur_input)  + input_stride, _MM_HINT_T1);

        ComputePointwiseBlock<FC, OC>(acc, cur_input, stride_width, cur_filter, filter_stride);

        // Advance input to next channel block (InputStride bytes)
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + input_stride);
        // Advance filter by one 16i×16o tile = 1024 bytes (= BS*BS*sizeof(float))
        cur_filter += AVX512_BS * AVX512_BS;
    }

    Avx512PostProcess<FC, OC>(acc, out_ptr, out_stride, bias, flags);
    out_ptr += OC * AVX512_BS;
}

// ─────────────────────────────────────────────────────────────────────────────
// RunPointwiseFilterCount
//
// Outer spatial loop for a fixed FilterCount.
// Mirrors ProcessPointwiseFilterCountN macro in SconvKernelAvx512F.asm:
//   - Process 6 output blocks at a time (main loop)
//   - Then 3, then 2, then 1
// ─────────────────────────────────────────────────────────────────────────────
template<int FC>
static void RunPointwiseFilterCount(
    const float* input_pos,
    const float* filter_base,
    float*& out_ptr,
    size_t stride_width,
    size_t input_stride,
    size_t filter_stride,
    size_t out_stride,
    size_t input_channels,
    size_t out_count,
    const float* bias,
    unsigned flags)
{
    const float* cur_input = input_pos;
    size_t remaining = out_count;

    // Batch of 6
    while (remaining >= 6) {
        ProcessPointwiseOutputs<FC, 6>(
            cur_input, filter_base, out_ptr,
            stride_width, input_stride, filter_stride, out_stride,
            input_channels, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + 6 * stride_width);
        remaining -= 6;
    }

    // Batch of 3
    if (remaining >= 3) {
        ProcessPointwiseOutputs<FC, 3>(
            cur_input, filter_base, out_ptr,
            stride_width, input_stride, filter_stride, out_stride,
            input_channels, bias, flags);
        cur_input = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(cur_input) + 3 * stride_width);
        remaining -= 3;
    }

    // Batch of 2
    if (remaining >= 2) {
        ProcessPointwiseOutputs<FC, 2>(
            cur_input, filter_base, out_ptr,
            stride_width, input_stride, filter_stride, out_stride,
            input_channels, bias, flags);
        remaining -= 2;
        // (cur_input advance not needed — no more batches after this)
    }

    // Single remaining output
    if (remaining == 1) {
        ProcessPointwiseOutputs<FC, 1>(
            cur_input, filter_base, out_ptr,
            stride_width, input_stride, filter_stride, out_stride,
            input_channels, bias, flags);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MlasConvPointwiseFloatKernelAvx512F_Cpp
//
// Drop-in C++ replacement for MlasConvPointwiseFloatKernelAvx512F.
// Signature matches MLAS_CONV_POINTWISE_FLOAT_KERNEL typedef (see mlasi.h).
//
// Called from snchwc.cpp inner channel-batch loop:
//   Kernel(input, filter, output,
//          StrideWidthBytes,
//          InputChannelBatch / BlockSize,   ← "InputChannels" param
//          FilterCount,
//          InputStrideBytes,                ← BS * InputSize * sizeof(float)
//          FilterStrideBytes,               ← BS * TotalInputChannels * sizeof(float)
//          OutputStrideBytes,
//          OutputThisIteration,
//          Bias, Flags)
//
// Note: InputChannels here is InputChannelBatch/BlockSize (number of 16-channel
// blocks in this call), NOT the total model input channels.
// ─────────────────────────────────────────────────────────────────────────────
inline void MlasConvPointwiseFloatKernelAvx512F_Cpp(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,      // bytes between output columns
    size_t InputChannels,    // number of 16-channel blocks to process
    size_t FilterCount,      // 1-4: output filter rows to process
    size_t InputStride,      // bytes to next input channel block
    size_t FilterStride,     // bytes between filter output rows
    size_t OutputStride,     // bytes between output rows
    size_t OutputCount,      // number of output spatial positions
    const float* Bias,
    unsigned Flags)
{
    float* out_ptr = Output;

    switch (FilterCount) {
    case 4:
        RunPointwiseFilterCount<4>(
            Input, Filter, out_ptr,
            StrideWidth, InputStride, FilterStride, OutputStride,
            InputChannels, OutputCount, Bias, Flags);
        break;
    case 3:
        RunPointwiseFilterCount<3>(
            Input, Filter, out_ptr,
            StrideWidth, InputStride, FilterStride, OutputStride,
            InputChannels, OutputCount, Bias, Flags);
        break;
    case 2:
        RunPointwiseFilterCount<2>(
            Input, Filter, out_ptr,
            StrideWidth, InputStride, FilterStride, OutputStride,
            InputChannels, OutputCount, Bias, Flags);
        break;
    default:  // FilterCount == 1
        RunPointwiseFilterCount<1>(
            Input, Filter, out_ptr,
            StrideWidth, InputStride, FilterStride, OutputStride,
            InputChannels, OutputCount, Bias, Flags);
        break;
    }
}
