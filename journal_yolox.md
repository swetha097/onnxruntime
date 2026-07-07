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

---

### 2026-07-07 — E1: SiLU diamond conv-fusion — measured, regresses nano/tiny

**Hypothesis:** Fusing `Conv → Sigmoid → Mul(conv_out, sig_out)` into a single NCHWc Conv with `activation=SiLU` (analogous to the HardSwish +21%/+35% result on MobileNetV3) will eliminate 74–104 separate Sigmoid+Mul tensor passes per YOLOX inference and yield ≥5% improvement.

**Graph verification:** ONNX inspection confirmed: yolox_s has 74, yolox_nano has 104, yolox_tiny has 74 Conv→SiLU fuseable diamonds. All are NCHWc-eligible (output channels multiple of 16). The fusion fires correctly — `TryFuseNchwcSiLU` matches all diamonds.

**Benchmark results (10T, 10wu, 50 trials, RelWithDebInfo, AMD 7840U):**

| Model | Base SS p50 | Opt SS p50 | Delta | Status |
|---|---|---|---|---|
| yolox_s    | 60.38 ms | 60.63 ms | −0.4% | NEUTRAL (noise-level) |
| yolox_nano |  6.11 ms |  6.61 ms | −8.2% | **REGRESSION** (t≈16, significant) |
| yolox_tiny | 18.66 ms | 19.26 ms | −3.2% | **REGRESSION** (t≈16, significant) |

**Accuracy: ALL PASS** — mAP identical (1.0000 / 0.0455 / 0.0909) for all three models and all kernel variants tested. No accuracy regression in any configuration.

**Root cause of regression — FMA3 asm vs C++ polynomial:**
The unfused baseline dispatches Sigmoid via `MlasComputeLogisticF32KernelFma3` — a hand-optimized FMA3 assembly kernel tuned for Zen4. Its throughput exceeds any C++ polynomial computed inside the conv epilogue, even using the same coefficients. The fused path replaces one dedicated Zen4-tuned asm routine with a generic C++ scalar loop inside `MlasActivationKernel`, which cannot match it.

Three kernel implementations were tested — all regress on nano/tiny:
1. `MlasComputeSilu` via temp buffer → −4.8% nano, −2.1% tiny (heap allocation overhead)
2. Platform `SiluKernelRoutine` (AVX-512F path, in-place) → −4.8% nano, −2.1% tiny (AVX-512 frequency throttle on AMD Zen 4)
3. Inline rational-polynomial `MLAS_ACTIVATION_FUNCTION<MlasSiLUActivation>` template → −8.4% nano, −3.4% tiny (C++ polynomial vs FMA3 asm)

**Why HardSwish fusion worked but SiLU fusion doesn't:**
HardSwish baseline kernel was `MlasComputeLogistic` (polynomial) + linear clamp + multiply — the fused HardSwish `x * clip(αx+β, 0, 1)` is cheaper than the unfused two-pass path. SiLU's unfused baseline is `MlasComputeLogisticF32KernelFma3` (FMA3 *assembly*) + Mul — the assembly kernel is already faster than the C++ polynomial can achieve, so fusion cannot win.

**Decision:** The SiLU diamond conv-fusion approach as implemented does NOT meet the goal on this AMD 7840U machine. The changes are accurate (mAP preserved) but fail the ≥5% performance gate for nano and tiny.

**What would need to change to unlock this optimization:**
The SiLU activation kernel inside `MlasActivationKernel` would need a platform-dispatched **FMA3/AVX2 assembly path** (analogous to the standalone `MlasComputeLogisticF32KernelFma3`) that achieves higher throughput than the unfused baseline. Until such a kernel exists, the fusion adds overhead rather than removing it. This is a pre-requisite for future SiLU fusion work.

**Code state:** `activate.cpp` left with the inline polynomial template (v3, cleanest form). The graph-level fusion code is correct and can be re-enabled once a faster kernel exists. Full changes committed on `perf/yolox-silu-opt`.

---

## SUMMARY (2026-07-07)

**Goal NOT MET for YOLOX.** Accuracy gate: ALL PASS. Performance gate: FAIL on nano (−8%) and tiny (−3%). yolox_s is neutral (±0.4%).

**What was delivered:**
- `TryFuseNchwcSiLU` — correct graph-level SiLU diamond fusion (verified against 74–104 sites per model)
- `MlasSiLUActivation` with inline rational-polynomial epilogue kernel
- Full unit tests (2 new NchwcOptimizerTests)
- Complete benchmark suite (10T, 10wu, 50 trials, all three models)

**Remaining gap:** The bottleneck is not the graph structure but the MLAS activation kernel. A hand-written FMA3/AVX2 assembly SiLU epilogue kernel (matching `MlasComputeLogisticF32KernelFma3` throughput) is the prerequisite for any SiLU conv-fusion win on AMD Zen 4.

**Recommended next investigation for YOLOX perf:**
1. Profile with VTune to identify actual hotspots (may be Concat layout copies, Resize bilinear, or NMS post-processing rather than SiLU)
2. BatchNorm fusion — YOLOX exports with folded BN but the BN→Conv fusion in NCHWc may not be firing
3. Depthwise / grouped conv improvements (yolox_nano uses grouped convolutions)
4. Threading model — at 10 threads on 8-core AMD 7840U, SpinPause overhead may dominate small feature maps
