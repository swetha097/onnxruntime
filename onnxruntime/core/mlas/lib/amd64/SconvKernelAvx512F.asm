;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision convolution
;   operation.
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SconvKernelCommon.inc
        .list

;
; Macro Description:
;
;   This macro generates code to clear the block accumulators.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   zmm0-zmm23 - Supplies the block accumulators.
;

ClearBlock MACRO FilterCount, OutputCount

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vpxord zmm0,zmm0,zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vpxord zmm4,zmm4,zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vpxord zmm8,zmm8,zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vpxord zmm12,zmm12,zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vpxord zmm16,zmm16,zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vpxord zmm20,zmm20,zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vpxord zmm1,zmm1,zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vpxord zmm5,zmm5,zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vpxord zmm9,zmm9,zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vpxord zmm13,zmm13,zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vpxord zmm17,zmm17,zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vpxord zmm21,zmm21,zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vpxord zmm2,zmm2,zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vpxord zmm6,zmm6,zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vpxord zmm10,zmm10,zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vpxord zmm14,zmm14,zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vpxord zmm18,zmm18,zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vpxord zmm22,zmm22,zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vpxord zmm3,zmm3,zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vpxord zmm7,zmm7,zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vpxord zmm11,zmm11,zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vpxord zmm15,zmm15,zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vpxord zmm19,zmm19,zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vpxord zmm23,zmm23,zmm23>

        ENDM

;
; Macro Description:
;
;   This macro multiplies and accumulates for FilterCount by OutputCount block
;   of the output buffer.
;
; Arguments:
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
;   VectorOffset - Supplies the byte offset from the filter buffer to fetch
;       elements.
;
;   BroadcastOffset - Supplies the byte offset from the input buffer to fetch
;       elements.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the input buffer.
;
;   rdx - Supplies the address of the filter buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rbx - Supplies the address of the filter buffer plus 2 * FilterStride.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r14 - Supplies the address of the input buffer plus 3 * StrideWidth.
;
;   zmm0-zmm23 - Supplies the block accumulators.
;

ComputeBlock MACRO KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

IFIDNI <KernelType>, <Depthwise>
        vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE OutputCount, 1, <vfmadd231ps zmm0,zmm24,ZMMWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vfmadd231ps zmm4,zmm24,ZMMWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vfmadd231ps zmm8,zmm24,ZMMWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vfmadd231ps zmm12,zmm24,ZMMWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vfmadd231ps zmm16,zmm24,ZMMWORD PTR [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vfmadd231ps zmm20,zmm24,ZMMWORD PTR [r14+r9*2+BroadcastOffset]>
ELSE
IF FilterCount EQ 1
        vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE OutputCount, 1, <vfmadd231ps zmm0,zmm24,DWORD BCST [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vfmadd231ps zmm4,zmm24,DWORD BCST [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vfmadd231ps zmm8,zmm24,DWORD BCST [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vfmadd231ps zmm12,zmm24,DWORD BCST [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vfmadd231ps zmm16,zmm24,DWORD BCST [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vfmadd231ps zmm20,zmm24,DWORD BCST [r14+r9*2+BroadcastOffset]>
ELSE
        EmitIfCountGE OutputCount, 1, <vbroadcastss zmm26,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vbroadcastss zmm27,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vbroadcastss zmm28,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vbroadcastss zmm29,DWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vbroadcastss zmm30,DWORD PTR [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vbroadcastss zmm31,DWORD PTR [r14+r9*2+BroadcastOffset]>
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vfmadd231ps zmm0,zmm26,ZMMWORD PTR [rdx+VectorOffset]>
        EmitIfCountGE FilterCount, 2, <vfmadd231ps zmm1,zmm26,ZMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 3, <vfmadd231ps zmm2,zmm26,ZMMWORD PTR [rbx+VectorOffset]>
        EmitIfCountGE FilterCount, 4, <vfmadd231ps zmm3,zmm26,ZMMWORD PTR [rbx+rsi+VectorOffset]>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vfmadd231ps zmm0,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vfmadd231ps zmm4,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vfmadd231ps zmm8,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vfmadd231ps zmm12,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vfmadd231ps zmm16,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vfmadd231ps zmm20,zmm31,zmm24>
        EmitIfCountGE FilterCount, 2, <vmovups zmm24,ZMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vfmadd231ps zmm1,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vfmadd231ps zmm5,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vfmadd231ps zmm9,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vfmadd231ps zmm13,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vfmadd231ps zmm17,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vfmadd231ps zmm21,zmm31,zmm24>
        EmitIfCountGE FilterCount, 3, <vmovups zmm24,ZMMWORD PTR [rbx+VectorOffset]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vfmadd231ps zmm2,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vfmadd231ps zmm6,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vfmadd231ps zmm10,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vfmadd231ps zmm14,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vfmadd231ps zmm18,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vfmadd231ps zmm22,zmm31,zmm24>
        EmitIfCountGE FilterCount, 4, <vmovups zmm24,ZMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vfmadd231ps zmm3,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vfmadd231ps zmm7,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vfmadd231ps zmm11,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vfmadd231ps zmm15,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vfmadd231ps zmm19,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vfmadd231ps zmm23,zmm31,zmm24>
ENDIF
ENDIF
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute the convolution for a specified number
;   of filter rows.
;
; Arguments:
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description) when
;       KernelType!=Depthwise. Supplies the address of the filter buffer when
;       KernelType=Depthwise.
;
;   rbp - Supplies the DilationWidth parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r15 - Supplies the InputStride parameter (see function description).
;

ProcessFilterCountN MACRO KernelFrame, KernelType, FilterCount

        LOCAL   ProcessOutputCount
        LOCAL   ProcessNextOutputCountBy6
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCountLessThan3
        LOCAL   ProcessRemainingOutputCount1
        LOCAL   ProcessOutputCountRightPadAndRemaining

;
; Process the output blocks that include left padding.
;

        mov     r10,KernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      ProcessOutputCount
        call    MlasConv&KernelType&FloatSingleAvx512FFilter&FilterCount

;
; Process the output blocks that do not include any padding.
;

ProcessOutputCount:
        mov     r10,KernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy6:
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             ; advance input by 6 elements
        sub     r10,6
        jae     ProcessNextOutputCountBy6

ProcessRemainingOutputCount:
        add     r10,6                       ; correct for over-subtract above
        jz      ProcessOutputCountRightPadAndRemaining
        cmp     r10,3
        jb      ProcessRemainingOutputCountLessThan3
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jz      ProcessOutputCountRightPadAndRemaining

ProcessRemainingOutputCountLessThan3:
        cmp     r10,1
        je      ProcessOutputCountRightPadAndRemaining
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 2
        lea     rdi,[rdi+r9*2]              ; advance input by 2 elements
        sub     r10,2

;
; Process the output blocks that include right padding plus any remaining output
; blocks from above.
;

ProcessOutputCountRightPadAndRemaining:
        add     r10,KernelFrame.OutputCountRightPad[rsp]
        jz      ExitKernel
        call    MlasConv&KernelType&FloatSingleAvx512FFilter&FilterCount

        ENDM

;
; Macro Description:
;
;   OPT-1: For each filter row f, load the filter ZMM once (vmovups) then FMA
;   all output columns using embedded DWORD BCST memory-form, eliminating the
;   vbroadcastss scratch registers zmm26-31 used in the base ComputeBlock path.
;
; Arguments:
;
;   FC  - FilterCount (1-4)
;   OC  - OutputCount (1-6)
;   VO  - byte offset into filter tile (Index*16*4)
;   BO  - byte offset into input (Index*4)
;
; Implicit Arguments:
;
;   rcx - input row base
;   rdx - filter row 0 base
;   rsi - FilterStride
;   rbx - filter row 2 base (rdx + 2*rsi), valid when FC > 2
;   r9  - StrideWidth
;   r14 - input col 3 base (rcx + 3*r9), valid when OC > 3
;

KernelStep MACRO FC, OC, VO, BO
        EmitIfCountGE FC, 1, <vmovups zmm24,ZMMWORD PTR [rdx+VO]>
        EmitIfCount2GE FC, 1, OC, 1, <vfmadd231ps zmm0,zmm24,DWORD BCST [rcx+BO]>
        EmitIfCount2GE FC, 1, OC, 2, <vfmadd231ps zmm4,zmm24,DWORD BCST [rcx+r9+BO]>
        EmitIfCount2GE FC, 1, OC, 3, <vfmadd231ps zmm8,zmm24,DWORD BCST [rcx+r9*2+BO]>
        EmitIfCount2GE FC, 1, OC, 4, <vfmadd231ps zmm12,zmm24,DWORD BCST [r14+BO]>
        EmitIfCount2GE FC, 1, OC, 5, <vfmadd231ps zmm16,zmm24,DWORD BCST [r14+r9+BO]>
        EmitIfCount2GE FC, 1, OC, 6, <vfmadd231ps zmm20,zmm24,DWORD BCST [r14+r9*2+BO]>
        EmitIfCountGE FC, 2, <vmovups zmm24,ZMMWORD PTR [rdx+rsi+VO]>
        EmitIfCount2GE FC, 2, OC, 1, <vfmadd231ps zmm1,zmm24,DWORD BCST [rcx+BO]>
        EmitIfCount2GE FC, 2, OC, 2, <vfmadd231ps zmm5,zmm24,DWORD BCST [rcx+r9+BO]>
        EmitIfCount2GE FC, 2, OC, 3, <vfmadd231ps zmm9,zmm24,DWORD BCST [rcx+r9*2+BO]>
        EmitIfCount2GE FC, 2, OC, 4, <vfmadd231ps zmm13,zmm24,DWORD BCST [r14+BO]>
        EmitIfCount2GE FC, 2, OC, 5, <vfmadd231ps zmm17,zmm24,DWORD BCST [r14+r9+BO]>
        EmitIfCount2GE FC, 2, OC, 6, <vfmadd231ps zmm21,zmm24,DWORD BCST [r14+r9*2+BO]>
        EmitIfCountGE FC, 3, <vmovups zmm24,ZMMWORD PTR [rbx+VO]>
        EmitIfCount2GE FC, 3, OC, 1, <vfmadd231ps zmm2,zmm24,DWORD BCST [rcx+BO]>
        EmitIfCount2GE FC, 3, OC, 2, <vfmadd231ps zmm6,zmm24,DWORD BCST [rcx+r9+BO]>
        EmitIfCount2GE FC, 3, OC, 3, <vfmadd231ps zmm10,zmm24,DWORD BCST [rcx+r9*2+BO]>
        EmitIfCount2GE FC, 3, OC, 4, <vfmadd231ps zmm14,zmm24,DWORD BCST [r14+BO]>
        EmitIfCount2GE FC, 3, OC, 5, <vfmadd231ps zmm18,zmm24,DWORD BCST [r14+r9+BO]>
        EmitIfCount2GE FC, 3, OC, 6, <vfmadd231ps zmm22,zmm24,DWORD BCST [r14+r9*2+BO]>
        EmitIfCountGE FC, 4, <vmovups zmm24,ZMMWORD PTR [rbx+rsi+VO]>
        EmitIfCount2GE FC, 4, OC, 1, <vfmadd231ps zmm3,zmm24,DWORD BCST [rcx+BO]>
        EmitIfCount2GE FC, 4, OC, 2, <vfmadd231ps zmm7,zmm24,DWORD BCST [rcx+r9+BO]>
        EmitIfCount2GE FC, 4, OC, 3, <vfmadd231ps zmm11,zmm24,DWORD BCST [rcx+r9*2+BO]>
        EmitIfCount2GE FC, 4, OC, 4, <vfmadd231ps zmm15,zmm24,DWORD BCST [r14+BO]>
        EmitIfCount2GE FC, 4, OC, 5, <vfmadd231ps zmm19,zmm24,DWORD BCST [r14+r9+BO]>
        EmitIfCount2GE FC, 4, OC, 6, <vfmadd231ps zmm23,zmm24,DWORD BCST [r14+r9*2+BO]>
        ENDM

;
; Macro Description:
;
;   Unrolled 16-element compute block: for each of 16 input channels within the
;   current filter tile, call KernelStep with the appropriate offsets.
;
; Arguments:
;
;   FC - FilterCount (1-4)
;   OC - OutputCount (1-6)
;

KernelComputeBlock MACRO FC, OC
        IRP     Index, <0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15>
            KernelStep FC, OC, %(Index*16*4), %(Index*4)
        ENDM
        ENDM

;
; Macro Description:
;
;   Generate the inner compute body for one NCHWc output block. Loops over
;   KernelHeight x KernelWidth, performing KernelComputeBlock at each position,
;   with OPT-4 software prefetch. OC=1 adds the padded-column bounds check.
;
; Arguments:
;
;   FC    - FilterCount (1-4)
;   OC    - OutputCount (1-6)
;   Frame - frame struct name (SconvKernelFrame or SconvKernelSingleFrame.KernelFrame)
;   Sfx   - unique label suffix
;

NchwcOutputs MACRO FC, OC, Frame, Sfx

        mov     rcx,rdi
        mov     rdx,Frame.PreviousP2Home[rsp]
        mov     r11,Frame.KernelHeight[rsp]
        mov     r12,Frame.KernelWidth[rsp]
IF OC EQ 1
        mov     r13,Frame.InputBase[rsp]
        mov     rax,Frame.InputWidth[rsp]
        neg     r13
ENDIF
        ClearBlock FC, OC
        test    r11,r11
        jz      @NchwcPost&Sfx

@NchwcRow&Sfx:
        mov     rax,r12
@NchwcCol&Sfx:
IF OC EQ 1
        lea     rbx,[rcx+r13]
        cmp     rbx,Frame.InputWidth[rsp]
        jae     @NchwcSkip&Sfx
ENDIF
IF OC GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx
ENDIF
IF FC GT 2
        lea     rbx,[rdx+rsi*2]
ENDIF
        prefetcht0 [rdx+16*16*4]
        prefetcht1 [rcx+rbp]
        KernelComputeBlock FC, OC
@NchwcSkip&Sfx:
        add     rcx,rbp
        add     rdx,16*16*4
        dec     rax
        jnz     @NchwcCol&Sfx
        add     rcx,r15
IF OC EQ 1
        sub     r13,Frame.DilatedInputWidth[rsp]
ENDIF
        dec     r11
        jnz     @NchwcRow&Sfx

@NchwcPost&Sfx:
        mov     edx,DWORD PTR Frame.Flags[rsp]
IF FC GT 1
        mov     rax,Frame.OutputStride[rsp]
ENDIF
        mov     rcx,Frame.Bias[rsp]
        call    MlasConvPostProcessFloatAvx512FFilter&FC&Output&OC

        ENDM

;
; Macro Description:
;
;   Generate the inner compute body for one pointwise output block. Loops over
;   InputChannels, performing KernelComputeBlock at each channel block, with
;   OPT-4 software prefetch.
;
; Arguments:
;
;   FC  - FilterCount (1-4)
;   OC  - OutputCount (1-6)
;   Sfx - unique label suffix
;

PwOutputs MACRO FC, OC, Sfx

        mov     rcx,rdi
        mov     rdx,r12
        mov     r11,SconvKernelPointwiseFrame.InputChannels[rsp]
        ClearBlock FC, OC
IF OC GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx
ENDIF
IF FC GT 2
        lea     rbx,[rdx+rsi*2]
ENDIF
@PwCh&Sfx:
        prefetcht0 [rdx+16*16*4]
        prefetcht1 [rcx+rbp]
IF OC GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx
ENDIF
IF FC GT 2
        lea     rbx,[rdx+rsi*2]
ENDIF
        KernelComputeBlock FC, OC
        add     rcx,rbp
        add     rdx,16*16*4
        dec     r11
        jnz     @PwCh&Sfx
        mov     edx,DWORD PTR SconvKernelPointwiseFrame.Flags[rsp]
IF FC GT 1
        mov     rax,SconvKernelPointwiseFrame.OutputStride[rsp]
ENDIF
        mov     rcx,SconvKernelPointwiseFrame.Bias[rsp]
        call    MlasConvPostProcessFloatAvx512FFilter&FC&Output&OC

        ENDM

;
; Generate the NCHW convolution kernel (uses macro framework, unchanged).
;

SconvKernelFunction Nchw, 16, Avx512F

;
; Generate the NCHWc convolution kernel (explicit assembly, OPT-1/4).
;

        NESTED_ENTRY MlasConvNchwcFloatKernelAvx512F, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r15
        push_reg r14
        push_reg r13
        push_reg r12
        alloc_stack (SconvKernelFrame.SavedR12 - SconvKernelFrame.SavedXmm6)
        save_xmm128 xmm6,  SconvKernelFrame.SavedXmm6
        save_xmm128 xmm7,  SconvKernelFrame.SavedXmm7
        save_xmm128 xmm8,  SconvKernelFrame.SavedXmm8
        save_xmm128 xmm9,  SconvKernelFrame.SavedXmm9
        save_xmm128 xmm10, SconvKernelFrame.SavedXmm10
        save_xmm128 xmm11, SconvKernelFrame.SavedXmm11
        save_xmm128 xmm12, SconvKernelFrame.SavedXmm12
        save_xmm128 xmm13, SconvKernelFrame.SavedXmm13
        save_xmm128 xmm14, SconvKernelFrame.SavedXmm14
        save_xmm128 xmm15, SconvKernelFrame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx                                     ; rdi = Input
        mov     SconvKernelFrame.PreviousP2Home[rsp],rdx    ; save Filter
        mov     rsi,SconvKernelFrame.FilterStride[rsp]      ; rsi = FilterStride
        mov     rbp,SconvKernelFrame.DilationWidth[rsp]     ; rbp = DilationWidth
        mov     r11,SconvKernelFrame.FilterCount[rsp]       ; r11 = FilterCount
        mov     r15,SconvKernelFrame.InputStride[rsp]       ; r15 = InputStride
        ; r8 (Output) and r9 (StrideWidth) remain in their argument registers

        cmp     r11,3
        ja      NchwcKernel_FC4
        je      NchwcKernel_FC3
        cmp     r11,1
        je      NchwcKernel_FC1

;
; FilterCount == 2
;

NchwcKernel_FC2:
        mov     r10,SconvKernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      NchwcMain2
        call    MlasConvNchwcFloatSingleAvx512FFilter2
NchwcMain2:
        mov     r10,SconvKernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      NchwcRem2
NchwcBy6_2:
        NchwcOutputs 2, 6, SconvKernelFrame, N2O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     NchwcBy6_2
NchwcRem2:
        add     r10,6
        jz      NchwcRPad2
        cmp     r10,3
        jb      NchwcLt3_2
        NchwcOutputs 2, 3, SconvKernelFrame, N2O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      NchwcRPad2
NchwcLt3_2:
        cmp     r10,2
        jb      NchwcRPad2
        NchwcOutputs 2, 2, SconvKernelFrame, N2O2
        lea     rdi,[rdi+r9*2]
        sub     r10,2
NchwcRPad2:
        add     r10,SconvKernelFrame.OutputCountRightPad[rsp]
        jz      NchwcKernelExit
        call    MlasConvNchwcFloatSingleAvx512FFilter2
        jmp     NchwcKernelExit

;
; FilterCount == 3
;

NchwcKernel_FC3:
        mov     r10,SconvKernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      NchwcMain3
        call    MlasConvNchwcFloatSingleAvx512FFilter3
NchwcMain3:
        mov     r10,SconvKernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      NchwcRem3
NchwcBy6_3:
        NchwcOutputs 3, 6, SconvKernelFrame, N3O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     NchwcBy6_3
NchwcRem3:
        add     r10,6
        jz      NchwcRPad3
        cmp     r10,3
        jb      NchwcLt3_3
        NchwcOutputs 3, 3, SconvKernelFrame, N3O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      NchwcRPad3
NchwcLt3_3:
        cmp     r10,2
        jb      NchwcRPad3
        NchwcOutputs 3, 2, SconvKernelFrame, N3O2
        lea     rdi,[rdi+r9*2]
        sub     r10,2
NchwcRPad3:
        add     r10,SconvKernelFrame.OutputCountRightPad[rsp]
        jz      NchwcKernelExit
        call    MlasConvNchwcFloatSingleAvx512FFilter3
        jmp     NchwcKernelExit

;
; FilterCount == 4
;

NchwcKernel_FC4:
        mov     r10,SconvKernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      NchwcMain4
        call    MlasConvNchwcFloatSingleAvx512FFilter4
NchwcMain4:
        mov     r10,SconvKernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      NchwcRem4
NchwcBy6_4:
        NchwcOutputs 4, 6, SconvKernelFrame, N4O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     NchwcBy6_4
NchwcRem4:
        add     r10,6
        jz      NchwcRPad4
        cmp     r10,3
        jb      NchwcLt3_4
        NchwcOutputs 4, 3, SconvKernelFrame, N4O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      NchwcRPad4
NchwcLt3_4:
        cmp     r10,2
        jb      NchwcRPad4
        NchwcOutputs 4, 2, SconvKernelFrame, N4O2
        lea     rdi,[rdi+r9*2]
        sub     r10,2
NchwcRPad4:
        add     r10,SconvKernelFrame.OutputCountRightPad[rsp]
        jz      NchwcKernelExit
        call    MlasConvNchwcFloatSingleAvx512FFilter4
        jmp     NchwcKernelExit

;
; FilterCount == 1
;

NchwcKernel_FC1:
        mov     r10,SconvKernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      NchwcMain1
        call    MlasConvNchwcFloatSingleAvx512FFilter1
NchwcMain1:
        mov     r10,SconvKernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      NchwcRem1
NchwcBy6_1:
        NchwcOutputs 1, 6, SconvKernelFrame, N1O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     NchwcBy6_1
NchwcRem1:
        add     r10,6
        jz      NchwcRPad1
        cmp     r10,3
        jb      NchwcLt3_1
        NchwcOutputs 1, 3, SconvKernelFrame, N1O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      NchwcRPad1
NchwcLt3_1:
        cmp     r10,2
        jb      NchwcRPad1
        NchwcOutputs 1, 2, SconvKernelFrame, N1O2
        lea     rdi,[rdi+r9*2]
        sub     r10,2
NchwcRPad1:
        add     r10,SconvKernelFrame.OutputCountRightPad[rsp]
        jz      NchwcKernelExit
        call    MlasConvNchwcFloatSingleAvx512FFilter1

NchwcKernelExit:
        vzeroupper
        movaps  xmm15,SconvKernelFrame.SavedXmm15[rsp]
        movaps  xmm14,SconvKernelFrame.SavedXmm14[rsp]
        movaps  xmm13,SconvKernelFrame.SavedXmm13[rsp]
        movaps  xmm12,SconvKernelFrame.SavedXmm12[rsp]
        movaps  xmm11,SconvKernelFrame.SavedXmm11[rsp]
        movaps  xmm10,SconvKernelFrame.SavedXmm10[rsp]
        movaps  xmm9,SconvKernelFrame.SavedXmm9[rsp]
        movaps  xmm8,SconvKernelFrame.SavedXmm8[rsp]
        movaps  xmm7,SconvKernelFrame.SavedXmm7[rsp]
        movaps  xmm6,SconvKernelFrame.SavedXmm6[rsp]
        add     rsp,(SconvKernelFrame.SavedR12 - SconvKernelFrame.SavedXmm6)
        BEGIN_EPILOGUE
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConvNchwcFloatKernelAvx512F, _TEXT

;
; Single-output helpers for NCHWc (padded columns). For each FilterCount,
; loop over r10 outputs, calling NchwcOutputs OC=1 via SconvKernelSingleFrame.
;

        IRP     FilterCount, <1, 2, 3, 4>

        LEAF_ENTRY MlasConvNchwcFloatSingleAvx512FFilter&FilterCount, _TEXT

@NchwcSingle&FilterCount&Loop:
        NchwcOutputs FilterCount, 1, SconvKernelSingleFrame.KernelFrame, S&FilterCount
        add     rdi,r9
        dec     r10
        jnz     @NchwcSingle&FilterCount&Loop
        ret

        LEAF_END MlasConvNchwcFloatSingleAvx512FFilter&FilterCount, _TEXT

        ENDM

;
; Generate the depthwise convolution kernel (uses macro framework, unchanged).
;

SconvKernelDepthwiseFunction 16, Avx512F

;
; Generate the pointwise convolution kernel (explicit assembly, OPT-1/4).
;

        NESTED_ENTRY MlasConvPointwiseFloatKernelAvx512F, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r14
        push_reg r12
        alloc_stack (SconvKernelPointwiseFrame.SavedR12 - SconvKernelPointwiseFrame.SavedXmm6)
        save_xmm128 xmm6,  SconvKernelPointwiseFrame.SavedXmm6
        save_xmm128 xmm7,  SconvKernelPointwiseFrame.SavedXmm7
        save_xmm128 xmm8,  SconvKernelPointwiseFrame.SavedXmm8
        save_xmm128 xmm9,  SconvKernelPointwiseFrame.SavedXmm9
        save_xmm128 xmm10, SconvKernelPointwiseFrame.SavedXmm10
        save_xmm128 xmm11, SconvKernelPointwiseFrame.SavedXmm11
        save_xmm128 xmm12, SconvKernelPointwiseFrame.SavedXmm12
        save_xmm128 xmm13, SconvKernelPointwiseFrame.SavedXmm13
        save_xmm128 xmm14, SconvKernelPointwiseFrame.SavedXmm14
        save_xmm128 xmm15, SconvKernelPointwiseFrame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx                                             ; rdi = Input
        mov     r12,rdx                                             ; r12 = Filter base
        mov     r10,SconvKernelPointwiseFrame.OutputCount[rsp]      ; r10 = OutputCount
        mov     r11,SconvKernelPointwiseFrame.FilterCount[rsp]      ; r11 = FilterCount
        mov     rsi,SconvKernelPointwiseFrame.FilterStride[rsp]     ; rsi = FilterStride
        mov     rbp,SconvKernelPointwiseFrame.InputStride[rsp]      ; rbp = InputStride
        ; r8 (Output) and r9 (StrideWidth) remain in their argument registers

        cmp     r11,3
        ja      PwKernel_FC4
        je      PwKernel_FC3
        cmp     r11,1
        je      PwKernel_FC1

;
; FilterCount == 2
;

PwKernel_FC2:
        sub     r10,6
        jb      PwRem2
PwBy6_2:
        PwOutputs 2, 6, P2O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     PwBy6_2
PwRem2:
        add     r10,6
        jz      PwKernelExit
        cmp     r10,3
        jb      PwLt3_2
        PwOutputs 2, 3, P2O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      PwKernelExit
PwLt3_2:
        cmp     r10,2
        jb      PwSingle2
        PwOutputs 2, 2, P2O2
        jmp     PwKernelExit
PwSingle2:
        PwOutputs 2, 1, P2O1
        jmp     PwKernelExit

;
; FilterCount == 3
;

PwKernel_FC3:
        sub     r10,6
        jb      PwRem3
PwBy6_3:
        PwOutputs 3, 6, P3O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     PwBy6_3
PwRem3:
        add     r10,6
        jz      PwKernelExit
        cmp     r10,3
        jb      PwLt3_3
        PwOutputs 3, 3, P3O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      PwKernelExit
PwLt3_3:
        cmp     r10,2
        jb      PwSingle3
        PwOutputs 3, 2, P3O2
        jmp     PwKernelExit
PwSingle3:
        PwOutputs 3, 1, P3O1
        jmp     PwKernelExit

;
; FilterCount == 4
;

PwKernel_FC4:
        sub     r10,6
        jb      PwRem4
PwBy6_4:
        PwOutputs 4, 6, P4O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     PwBy6_4
PwRem4:
        add     r10,6
        jz      PwKernelExit
        cmp     r10,3
        jb      PwLt3_4
        PwOutputs 4, 3, P4O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      PwKernelExit
PwLt3_4:
        cmp     r10,2
        jb      PwSingle4
        PwOutputs 4, 2, P4O2
        jmp     PwKernelExit
PwSingle4:
        PwOutputs 4, 1, P4O1
        jmp     PwKernelExit

;
; FilterCount == 1
;

PwKernel_FC1:
        sub     r10,6
        jb      PwRem1
PwBy6_1:
        PwOutputs 1, 6, P1O6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     PwBy6_1
PwRem1:
        add     r10,6
        jz      PwKernelExit
        cmp     r10,3
        jb      PwLt3_1
        PwOutputs 1, 3, P1O3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      PwKernelExit
PwLt3_1:
        cmp     r10,2
        jb      PwSingle1
        PwOutputs 1, 2, P1O2
        jmp     PwKernelExit
PwSingle1:
        PwOutputs 1, 1, P1O1

PwKernelExit:
        vzeroupper
        movaps  xmm15,SconvKernelPointwiseFrame.SavedXmm15[rsp]
        movaps  xmm14,SconvKernelPointwiseFrame.SavedXmm14[rsp]
        movaps  xmm13,SconvKernelPointwiseFrame.SavedXmm13[rsp]
        movaps  xmm12,SconvKernelPointwiseFrame.SavedXmm12[rsp]
        movaps  xmm11,SconvKernelPointwiseFrame.SavedXmm11[rsp]
        movaps  xmm10,SconvKernelPointwiseFrame.SavedXmm10[rsp]
        movaps  xmm9,SconvKernelPointwiseFrame.SavedXmm9[rsp]
        movaps  xmm8,SconvKernelPointwiseFrame.SavedXmm8[rsp]
        movaps  xmm7,SconvKernelPointwiseFrame.SavedXmm7[rsp]
        movaps  xmm6,SconvKernelPointwiseFrame.SavedXmm6[rsp]
        add     rsp,(SconvKernelPointwiseFrame.SavedR12 - SconvKernelPointwiseFrame.SavedXmm6)
        BEGIN_EPILOGUE
        pop     r12
        pop     r14
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConvPointwiseFloatKernelAvx512F, _TEXT

;
; Macro Description:
;
;   This macro generates code to process an output block after the inner
;   convolution kernel has executed and then stores the output block to the
;   output buffer.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;

        IRP     FilterCount, <1, 2, 3, 4>
        IRP     OutputCount, <1, 2, 3, 6>

        LEAF_ENTRY MlasConvPostProcessFloatAvx512FFilter&FilterCount&Output&OutputCount, _TEXT

IF FilterCount GT 2
        lea     rbx,[r8+rax*2]              ; compute output plus 2 rows
ENDIF

;
; Test if the existing contents of the output buffer should be accumulated
; with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        jz      SkipAccumulateOutput
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vaddps zmm0,zmm0,ZMMWORD PTR [r8]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vaddps zmm4,zmm4,ZMMWORD PTR [r8+16*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vaddps zmm8,zmm8,ZMMWORD PTR [r8+32*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vaddps zmm12,zmm12,ZMMWORD PTR [r8+48*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vaddps zmm16,zmm16,ZMMWORD PTR [r8+64*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vaddps zmm20,zmm20,ZMMWORD PTR [r8+80*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vaddps zmm1,zmm1,ZMMWORD PTR [r8+rax]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vaddps zmm5,zmm5,ZMMWORD PTR [r8+rax+16*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vaddps zmm9,zmm9,ZMMWORD PTR [r8+rax+32*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vaddps zmm13,zmm13,ZMMWORD PTR [r8+rax+48*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vaddps zmm17,zmm17,ZMMWORD PTR [r8+rax+64*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vaddps zmm21,zmm21,ZMMWORD PTR [r8+rax+80*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vaddps zmm2,zmm2,ZMMWORD PTR [rbx]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vaddps zmm6,zmm6,ZMMWORD PTR [rbx+16*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vaddps zmm10,zmm10,ZMMWORD PTR [rbx+32*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vaddps zmm14,zmm14,ZMMWORD PTR [rbx+48*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vaddps zmm18,zmm18,ZMMWORD PTR [rbx+64*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vaddps zmm22,zmm22,ZMMWORD PTR [rbx+80*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vaddps zmm3,zmm3,ZMMWORD PTR [rbx+rax]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vaddps zmm7,zmm7,ZMMWORD PTR [rbx+rax+16*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vaddps zmm11,zmm11,ZMMWORD PTR [rbx+rax+32*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vaddps zmm15,zmm15,ZMMWORD PTR [rbx+rax+48*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vaddps zmm19,zmm19,ZMMWORD PTR [rbx+rax+64*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vaddps zmm23,zmm23,ZMMWORD PTR [rbx+rax+80*4]>

SkipAccumulateOutput:

;
; Test if the bias buffer should be accumulated with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        jz      SkipBiasAddition
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vaddps zmm0,zmm0,ZMMWORD PTR [rcx]>
        EmitIfCountGE FilterCount, 2, <vaddps zmm1,zmm1,ZMMWORD PTR [rcx+16*4]>
        EmitIfCountGE FilterCount, 3, <vaddps zmm2,zmm2,ZMMWORD PTR [rcx+32*4]>
        EmitIfCountGE FilterCount, 4, <vaddps zmm3,zmm3,ZMMWORD PTR [rcx+48*4]>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups zmm28,ZMMWORD PTR [rcx]>
        EmitIfCountGE FilterCount, 2, <vmovups zmm29,ZMMWORD PTR [rcx+16*4]>
        EmitIfCountGE FilterCount, 3, <vmovups zmm30,ZMMWORD PTR [rcx+32*4]>
        EmitIfCountGE FilterCount, 4, <vmovups zmm31,ZMMWORD PTR [rcx+48*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vaddps zmm0,zmm0,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vaddps zmm4,zmm4,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vaddps zmm8,zmm8,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vaddps zmm12,zmm12,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vaddps zmm16,zmm16,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vaddps zmm20,zmm20,zmm28>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vaddps zmm1,zmm1,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vaddps zmm5,zmm5,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vaddps zmm9,zmm9,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vaddps zmm13,zmm13,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vaddps zmm17,zmm17,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vaddps zmm21,zmm21,zmm29>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vaddps zmm2,zmm2,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vaddps zmm6,zmm6,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vaddps zmm10,zmm10,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vaddps zmm14,zmm14,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vaddps zmm18,zmm18,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vaddps zmm22,zmm22,zmm30>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vaddps zmm3,zmm3,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vaddps zmm7,zmm7,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vaddps zmm11,zmm11,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vaddps zmm15,zmm15,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vaddps zmm19,zmm19,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vaddps zmm23,zmm23,zmm31>
ENDIF

SkipBiasAddition:

;
; Test for fused ReLU activation.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        jz      SkipReluActivation
        vpxord  zmm24,zmm24,zmm24
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vmaxps zmm0,zmm24,zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vmaxps zmm4,zmm24,zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vmaxps zmm8,zmm24,zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vmaxps zmm12,zmm24,zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vmaxps zmm16,zmm24,zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vmaxps zmm20,zmm24,zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vmaxps zmm1,zmm24,zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vmaxps zmm5,zmm24,zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vmaxps zmm9,zmm24,zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmaxps zmm13,zmm24,zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmaxps zmm17,zmm24,zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmaxps zmm21,zmm24,zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vmaxps zmm2,zmm24,zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vmaxps zmm6,zmm24,zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vmaxps zmm10,zmm24,zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vmaxps zmm14,zmm24,zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vmaxps zmm18,zmm24,zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vmaxps zmm22,zmm24,zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vmaxps zmm3,zmm24,zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vmaxps zmm7,zmm24,zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vmaxps zmm11,zmm24,zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vmaxps zmm15,zmm24,zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vmaxps zmm19,zmm24,zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vmaxps zmm23,zmm24,zmm23>

SkipReluActivation:

;
; Store the output block in the output buffer.
;

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vmovups ZMMWORD PTR [r8],zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vmovups ZMMWORD PTR [r8+16*4],zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vmovups ZMMWORD PTR [r8+32*4],zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vmovups ZMMWORD PTR [r8+48*4],zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vmovups ZMMWORD PTR [r8+64*4],zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vmovups ZMMWORD PTR [r8+80*4],zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vmovups ZMMWORD PTR [r8+rax],zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vmovups ZMMWORD PTR [r8+rax+16*4],zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vmovups ZMMWORD PTR [r8+rax+32*4],zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmovups ZMMWORD PTR [r8+rax+48*4],zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmovups ZMMWORD PTR [r8+rax+64*4],zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmovups ZMMWORD PTR [r8+rax+80*4],zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vmovups ZMMWORD PTR [rbx],zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vmovups ZMMWORD PTR [rbx+16*4],zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vmovups ZMMWORD PTR [rbx+32*4],zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vmovups ZMMWORD PTR [rbx+48*4],zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vmovups ZMMWORD PTR [rbx+64*4],zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vmovups ZMMWORD PTR [rbx+80*4],zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vmovups ZMMWORD PTR [rbx+rax],zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vmovups ZMMWORD PTR [rbx+rax+16*4],zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vmovups ZMMWORD PTR [rbx+rax+32*4],zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vmovups ZMMWORD PTR [rbx+rax+48*4],zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vmovups ZMMWORD PTR [rbx+rax+64*4],zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vmovups ZMMWORD PTR [rbx+rax+80*4],zmm23>
        add_immed r8,OutputCount*16*4       ; advance output by N nchw16c blocks
        ret

        LEAF_END MlasConvPostProcessFloatAvx512FFilter&FilterCount&Output&OutputCount, _TEXT

        ENDM
        ENDM

        END
