# YOLOX FP32 MLAS Optimization Journal

**Goal:** ≥5–10% latency improvement for FP32 YOLOX-S, YOLOX-Nano, YOLOX-Tiny on ONNX Runtime MLAS/CPU EP, without mAP regression vs. base branch. Close the gap with Intel OpenVINO.

- **Models:** YOLOX-S (640×640), YOLOX-Nano (416×416), YOLOX-Tiny (416×416)
- **Accuracy metric:** mAP@IoU=0.5 on 10-image COCO subset; conf=0.45, NMS=0.45
- **Base branch:** `main` @ `57f6079d06` (Jun 15 2026 snapshot; same base as MobileNetV3 work)
- **Working branch:** `perf/mobilenetv3-opt` (shared with MobileNetV3 work; same MLAS changes apply)
- **Benchmark harness:** `yolox_inference_kit/run_vtune_cpu.bat` (run from CMD), 10 threads, 10wu/50trials
- **Build:** VS2022 build tree at `check_jun17_reldeb_onnx/RelWithDebInfo/RelWithDebInfo`
- **Statistical gates:** CoV <10% preferred, >20% = discard/re-run; z-score outlier filter (|z|>3)
- **Machine:** AMD 7840U, 16 logical cores, Windows 11, AVX-512 capable

**Key difference from MobileNetV3 goal:** YOLOX uses **SiLU** (Swish: `x * sigmoid(x)`), not HardSwish. The model task is object detection so the accuracy metric is mAP (not top-1 classification). All three YOLOX variants must pass the gating criteria simultaneously.

---

## Relevant inherited optimizations from MobileNetV3 branch

The `perf/mobilenetv3-opt` branch already contains:
- **OPT4** (`26396e06e1`): bias zmm28 reuse in `SconvKernelAvx512F.asm` — applies to all NCHWc pointwise convolutions, including those in YOLOX.
- **HardSwish conv-activation fusion** (`0073b63840`): `TryFuseNchwcHardSwish` in `nchwc_transformer.cc` — YOLOX does NOT use HardSwish, so this does not help YOLOX directly, but the diamond-pattern fusion mechanism it establishes is the template for the SiLU diamond fusion.

---

## YOLOX architecture and op mix (research)

YOLOX is a detection model: backbone (CSPDarknet or MobileNet-derived), FPN neck (PAFPN), and a decoupled head.

**Key ops in all three YOLOX variants:**
- **Conv + SiLU (the dominant pattern):** Every "BaseConv" block = `Conv → BatchNorm → SiLU`. In ONNX, BatchNorm is folded into Conv (as bias). SiLU (ONNX opset 17 introduces it as a primitive; older models decompose it to `Mul(x, Sigmoid(x))`). YOLOX-S/Nano/Tiny ONNX models use decomposed SiLU = `x * sigmoid(x)` (confirmed by MLAS graph analysis research below).
- **Concat:** Feature pyramid requires multiple tensor concatenations along the channel axis — already handled by NCHWc transformer.
- **Upsample/Resize:** FPN upsampling — already handled by NCHWc transformer.
- **Add (shortcut):** ResNet-style shortcut additions — already handled as NCHWc binary.
- **No HardSwish, no depthwise:** YOLOX-S uses standard Conv. YOLOX-Nano uses depth-multiplier-based Conv but not separable depthwise groups. No HardSwish anywhere.

**SiLU decomposition in ONNX:**  
ONNX opset 17 added SiLU as a standalone op, but most exported YOLOX models use older opsets. The runtime sees `Conv → Sigmoid → Mul(conv_out, sigmoid_out)` — the same diamond pattern as `Mul(x, HardSigmoid(x))` for HardSwish, just with plain Sigmoid instead of HardSigmoid.

**MLAS SiLU status before this work:**
- `MlasComputeSilu` exists in `mlas/lib/silu.cpp` — standalone SiLU kernel used for layer-norm / attention ops.
- `MlasSiluKernelAvx512F` exists in `mlas/lib/intrinsics/avx512/silu_avx512f.cpp` — fused AVX-512F kernel computing `x * logistic_approx(x)` in a single pass without a temporary buffer.
- **No `MlasSiLUActivation` enum** in `MLAS_ACTIVATION_KIND` — Sigmoid is `MlasLogisticActivation` and runs via `MlasComputeLogistic` separately.
- **NCHWc transformer does not have a SiLU diamond fusion** — `TryFuseNchwcHardSwish` handles HardSwish; no analogous `TryFuseNchwcSiLU` exists.

**Why SiLU diamond fusion is the top lever for YOLOX:**  
Every "BaseConv" block in YOLOX = Conv + BN (folded) + SiLU. With ≥30 such blocks per YOLOX-S inference, the unfused path runs:
1. Conv epilogue: bias + identity (NCHW → NCHWc already handled)
2. Sigmoid (full tensor read/write via `MlasComputeLogistic` + separately)
3. Mul (full tensor read/write via elementwise kernel)

Fusing Conv + SiLU eliminates steps 2 and 3 as separate full-tensor passes, computing SiLU in-register in the conv epilogue. For MobileNetV3 HardSwish, this was the difference between +21% (large) and +35% (small). YOLOX has a similar density of activation passes, so the expected gain is in the same range.

---

## Experiment log

### 2026-07-02 — R0: Research + baseline infrastructure setup

- **Actions:**
  - Created `journal_yolox.md` (this file).
  - Fixed `yolox_inference_kit/run_vtune_cpu.bat`: updated MODEL_DIR, ONNX_INCLUDE, IMAGE_DIR, OV_BIN, TBB_BIN to correct `sloganat` machine paths. Changed default NUM_THREADS from 1 → 10 (goal requirement).
  - Surveyed MLAS activation infrastructure: confirmed `MlasComputeSilu` + `MlasSiluKernelAvx512F` exist but are not wired into the NCHWc conv activation path.
  - Surveyed NCHWc transformer: `TryFuseNchwcHardSwish` (added for MobileNetV3) is the exact template for `TryFuseNchwcSiLU`. NCHWc transformer already dispatches Sigmoid nodes through `TransformActivation`; the Sigmoid-single-use case fuses `Conv → Sigmoid` into the conv (activation=Sigmoid). The Sigmoid-as-SiLU-gate diamond `Conv → {Sigmoid, Mul(conv_out, sig_out)}` is NOT yet fused.
- **Plan for E1:** Implement `MlasSiLUActivation` + `TryFuseNchwcSiLU` using the HardSwish machinery as the template.
- **Status:** ✅ research complete, code changes in progress.

---

*[Subsequent entries will be added after each experiment.]*
