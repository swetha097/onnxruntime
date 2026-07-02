# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this kit is

A self-contained Windows benchmark/profiling harness for YOLOX object-detection inference via ONNX Runtime CPU (and optionally OpenVINO/oneDNN/ZenDNN EPs). It measures steady-state latency, E2E latency, throughput, and mAP against COCO annotations, and optionally captures Intel VTune hotspot profiles.

Key files:
- `yolox_unified_inference.cpp` — C++ inference harness (compile + run directly)
- `preprocess_yolox_images.py` — converts raw JPEG images to `.tensor` files consumed by the harness
- `run_vtune_cpu.bat` — driver: compiles the harness, stages DLLs, runs benchmarks, optionally collects VTune profiles, and appends a consolidated summary table
- `real_annotations.txt` — COCO-format ground truth for mAP computation
- `models/yolox/` — ONNX model files (`yolox_nano.onnx`, `yolox_s.onnx`, `yolox_tiny.onnx`)
- `real_test_images/` — source JPEG images

## Build and compile

Requires VS 2022 x64 Developer Command Prompt (or `vcvars64.bat x64`) and a built ORT (`onnxruntime.lib` + `onnxruntime.dll`).

```cmd
cl yolox_unified_inference.cpp ^
  /MD /EHsc /std:c++17 /O2 /Zi ^
  /Fe:yolox_inference_standard.exe ^
  /D_CRT_SECURE_NO_WARNINGS ^
  /I"<ORT_ROOT>\include\onnxruntime\core\session" ^
  /link /DEBUG:FULL /INCREMENTAL:NO /OPT:NOREF /OPT:NOICF ^
  /LIBPATH:"<ORT_BUILD>\Release\Release" ^
  onnxruntime.lib
```

Copy `onnxruntime.dll` and `onnxruntime_providers_shared.dll` from the build output to the working directory before running.

## Preprocess images

The harness does not decode JPEGs. It reads `.tensor` files — raw HWC float32 normalized to [0,1] — from a sibling directory named `<image_dir>_tensors_<size>`.

```cmd
pip install Pillow numpy
python preprocess_yolox_images.py --input real_test_images --size 640
# for nano/tiny models use --size 416
```

Output lands in `real_test_images_tensors_640/` (or `_416/`).

## Run inference

```cmd
# Batch mode with mAP (CPU, default)
yolox_inference_standard.exe ^
    --model models\yolox\yolox_s.onnx ^
    --image-dir real_test_images ^
    --ep CPU ^
    --ground_truth real_annotations.txt ^
    --conf_threshold 0.45 --nms_threshold 0.45 ^
    --warmup 10 --trials 3 --threads 1

# Other EPs: --ep OPENVINO | DNNL | ZENDNN
```

Key CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--model` | required | Path to `.onnx` |
| `--image-dir` | required (or `--image`) | Directory of images |
| `--ep` | CPU | CPU \| OPENVINO \| DNNL \| ZENDNN |
| `--ground_truth` | — | Annotations file for mAP |
| `--save-gt` | — | Append predicted boxes to a file |
| `--conf_threshold` | 0.45 | |
| `--nms_threshold` | 0.45 | |
| `--warmup` | 10 | Max warmup iterations; stops early on steady-state |
| `--trials` | 3 | Measurement trials |
| `--threads` | 1 | ORT intra-op thread count |

## Run full benchmark with VTune

Update config variables at the top of `run_vtune_cpu.bat` (MODEL_DIR, ONNX_INCLUDE, IMAGE_DIR) then:

```cmd
# Timing only (no VTune)
run_vtune_cpu.bat <threads> <path\to\ORT\Release\Release> 0

# With VTune hotspot collection
run_vtune_cpu.bat <threads> <path\to\ORT\Release\Release> 1
```

Output lands in `check_newscript_base_<N>t\` (per-model timing logs + `summary_steady_state_only.txt`) and `check_newscript_base_<N>t_vtune\` (CSV hotspot/top-down/summary reports).

## Ground truth file format

`real_annotations.txt` — one annotated box per line, absolute pixel coordinates, COCO 80-class IDs:

```
<filename> <class_id> <x1> <y1> <x2> <y2>
```

Common COCO IDs: 0=person, 2=car, 7=truck, 15=cat, 56=chair, 62=tv.

If the file does not exist, `run_vtune_cpu.bat` auto-generates it from `yolox_s` + CPU EP with `--save-gt`.

## Summary table format

```
Model           EP         Mode  Latency_ms  Throughput  FPS         p50_ms      p90_ms      mAP        trials wu trial-std  ci95_ms   cv_pct
-----------------------------------------------------------------------------------------------------------------------------------------------
yolox_s         CPU        SS    ...         ...         ...         ...         ...         ...        50     10 ...        ...       ...
yolox_s         CPU        E2E   ...         ...         ...         ...         ...         ...        50     10 ...        ...       ...
```

CV stability gates: `<10%` = OK, `10–20%` = questionable, `>20%` = unreliable.

## Architecture: inference pipeline

`SessionContext` (`yolox_unified_inference.cpp:569`) creates the ORT session and reads the input shape from the model. `run_yolox_inference()` (line 641) drives: load tensor → `session.Run()` → decode raw output → NMS → validate boxes → compute mAP. `main()` adds warmup loop (steady-state detection via rolling std-dev < 3% of mean over a 5-iteration window) + measurement trials + Z-score outlier filtering on latencies before computing final `LatencyMetrics`.

Model output is either pre-decoded (cx, cy, w, h already absolute) or needs grid decoding — `needs_decode` (line 716) is `true` when the number of output rows equals the total number of grid cells across strides 8/16/32. In that case the harness decodes the raw YOLOX head outputs using the standard `(pred + grid) * stride` formula.
