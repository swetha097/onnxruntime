# journal_mt_30_jul.md — MLAS multi-threading optimization round

**Branch:** `perf/all_mt-opt` (worktree `C:\Users\sloganat\Documents\onnxruntime-mt-all`)
**Base:** `57f6079d06` (`swe_fork/perf/base`, June 17) — fresh worktree, no prior-round commits
**Objective:** ≥5–10% SS throughput vs base on FP32 yolox_{s,tiny,nano}, MobileNetV3-{large,small},
mobileclip_s0, accuracy-neutral, no 1T/2T regression, VTune SPIN%/QUEUE% down.

---

## E0 — Environment and harness

**Hardware caveat (important, affects how the reference profiles may be read).**
The supplied baseline profile is `Strix365-Base-MLAS-Profiling-Jun17.xlsx`, and its header records
`AMD Ryzen AI 9 365 w/ Radeon` (Strix Point: hybrid Zen5 + Zen5c, two CCXs). **This machine is not
that part.** `PROCESSOR_IDENTIFIER` reports `AMD64 Family 25 Model 116 Stepping 1` = Zen4 Phoenix
(Ryzen 7 PRO 7840U), 8 cores / 16 logical, single CCX, non-hybrid. Consequences:

* Absolute latencies in the reference workbook are not comparable to anything measured here, so
  **all baselines below were regenerated locally**; the workbook is used only for hotspot structure.
* `ThreadPool::DegreeOfParallelism` returns `(NumThreads()+1) * 4` on hybrid parts and
  `(NumThreads()+1)` otherwise (`onnxruntime/core/common/threadpool.cc:644-656`). On Strix the MLAS
  index space is therefore 4× oversubscribed and self-balancing; **on this Zen4 part it is not**.
  That difference motivated E1.

**Build.** `_build.bat` / `_build_inc.bat` in the worktree root. FetchContent is pointed at the
same-commit cached sources under `Documents/onnxruntime/check_jun17_onnx/Release/_deps/*-src` via
`FETCHCONTENT_SOURCE_DIR_*`, which makes the build independent of the flaky GitHub egress.
Base DLL archived at `_dlls/onnxruntime.base.dll`.

**Harness.** The three mandated scripts are used, copied into `bench/{mnv3,yolox,clip}` and
re-pointed at local paths, driven from CMD through clean-PATH wrappers `bench/_run_*.bat`
(the MSYS `find.exe` hazard is avoided by resetting PATH to Windows-only). Two harness bugs had to
be fixed to make them run on this machine; neither changes measurement semantics:

1. `run_vtune_cpu_fixed.bat` wrapped the exe in `start ... cmd /c ".\yolox_inference_standard.exe ^ ..."`.
   Caret continuations are not processed inside that quoted argument, so cmd saw `--model` as a
   command and every YOLOX run produced an empty result file. Rewritten to the direct
   `start /wait /high /b "" .\exe ^ ...` form already used by the MobileNetV3 harness.
2. `mobileclip_hotspots_run_all.bat` called Intel `setvars.bat`, which on this machine switches to
   VS 2025 Community and drops the UCRT paths. Replaced with the same manual Windows-SDK
   include/lib injection the other two harnesses use.

`MODEL_DIR` for YOLOX also had a stale trailing `\yolox` component.

---

## E1 — Local baselines (base DLL, mandated harnesses)

SS mean latency, 30 trials (50 for YOLOX), 10 images (20 for YOLOX):

| model | 1T ms | 2T ms | 4T ms | acc/mAP | 1T→4T |
|---|---|---|---|---|---|
| yolox_s | 216.31 | 171.49 | 106.72 | 1.0000 | 2.03× |
| yolox_tiny | 55.43 | 48.66 | 30.30 | 0.0909 | 1.83× |
| yolox_nano | 12.97 | 10.53 | 7.08 | 0.0455 | 1.83× |
| mobilenetv3_large | 6.58 | 4.06 | 3.02 | 0.9000 | 2.18× |
| mobilenetv3_small | 2.19 | 1.66 | 1.34 | 0.9000 | 1.63× |
| mobileclip_s0 | 59.68 | 47.01 | 28.70 | n/a¹ | 2.08× |

¹ The MobileClip harness has no `real_labels.txt` anywhere on this machine, so its Top-1 column is
blank. Accuracy for that model is therefore gated on **bit-identical predictions** (class + confidence)
between arms, which is a stronger check than a 10-image Top-1 rate.

**Reading:** 4 threads buy only 1.63–2.18×. Between 45% and 59% of the added capacity is lost to
parallelization tax — on *every* model, including yolox_s. (A previous round recorded yolox_s
1T→4T as "near-linear"; at 2.03× on 4 threads that is 51% efficiency, so there is headroom in
principle even there.)

**Stability.** 1T is clean (CV 1.1–4.6%). Some 2T/4T configurations breached the 10% CV gate
(mobileclip 2T 20.0% / 4T 11.4%, yolox_s 2T 12.9%, yolox_nano 2T 17.7% / 4T 14.8%). Because
cross-session drift on this machine turned out to exceed the effect sizes under test (demonstrated
in E2), **lever selection is done with interleaved A/B + Welch t, not with single harness runs.**

---

## E2 — Lever: NCHWc work decomposition (clamp + oversubscription)

**Hypothesis.** `MlasNchwcConv` sets `WorkBlock.tids = MlasGetMaximumThreadCount(ThreadPool)`
unconditionally (`snchwc.cpp:1403`) and never clamps it to the amount of work that exists — unlike
`convolve.cpp:1103-1105` and `reorder.cpp:613-615`, which do clamp. Two defects follow:

1. When `TotalWork < d_of_p`, surplus indices receive an empty range from `MlasPartitionWork`
   (`mlasi.h:1759-1779`) yet still cost a `LoopCounter` claim and a full algorithm-object construction.
2. `MlasPartitionWork` balances the *count* of indices, but indices are not equal cost:
   `FilterCount = min(4, OutputChannels/BlockSize - FilterSet*4)` (`snchwc.cpp:551`) makes a trailing
   filter set as cheap as ¼ of a full one, and padding rows evaluate a shorter kernel. With exactly
   one index per thread there is no opportunity to rebalance.

**Change.** `MlasNchwcWorkCount()` in `snchwc.cpp`: `tids = clamp(G * d_of_p, 1, TotalWork)`, with
`G` selected by `MLAS_NCHWC_WORK_GRAN` so both arms run from the *same DLL* (G=0 reproduces base
scheduling exactly).

**Result** — interleaved A/B, 8 cycles × 30 trials, MobileNetV3 @ 4T:

| G | mnv3_large | mnv3_small | note |
|---|---|---|---|
| 1 (clamp only) | **−3.26%** (t=−1.06) | +0.95% (t=0.32) | A-then-B ordering |
| 4 (clamp + 4× oversubscribe) | **+1.83%** (t=2.01) | +0.82% (t=0.88) | ABBA ordering |

Predictions bit-identical in every arm.

**Outcome: REJECTED as a headline lever** (best case +1.8%, far below the gate).

**Methodological note worth recording.** A naive non-interleaved sweep of the same knob reported
G=1 at **−7%** (i.e. a 7% *win*): `off` measured 3.28 ms in that sweep but 3.02 ms in the baseline
sweep and 3.14–3.18 ms in the interleaved runs. Cross-session drift on this machine is ±8%, which is
larger than every effect measured in this round. Any single-shot before/after comparison here is
uninterpretable.

---

## E3 — Root cause: where the 4T tax actually goes

Instrument: `onnxruntime_perf_test -p` per-node profiling, MobileNetV3-large, 300 iterations,
1T vs 4T. (Profiling inflates absolute per-node times by roughly 55%; the **1T→4T delta** is the
meaningful quantity and profiler overhead largely cancels in it.)

Per-op-type mean cost per call:

| op | 1T µs | 4T µs | |
|---|---|---|---|
| Conv | 77.96 | 42.22 | scales ✓ |
| Gemm | 233.53 | 175.52 | scales ✓ |
| Mul | 20.60 | **25.93** | +26% |
| HardSigmoid | 19.17 | **25.06** | +31% |
| ReorderInput | 7.73 | **14.96** | +93% |
| ReorderOutput | 7.93 | **14.47** | +82% |
| ReduceMean | 13.27 | **16.13** | +22% |

Everything that is not Conv/Gemm gets *slower* when threads are added, and those ops are 38.8% of
4T node time.

The decisive split is by the cost model's own decision. Reconstructing Eigen's
`TensorCostModel::numThreads` for each node from the shapes recorded in the profile
(`ThreadPool::ParallelFor`, `threadpool.cc:619-633`; per-element cost from
`element_wise_ranged_transform.h:101`):

| op | cost-model verdict | #nodes | 1T µs | 4T µs | delta |
|---|---|---|---|---|---|
| HardSigmoid | serial | 16 | 216.4 | 395.6 | **+82.8%** |
| HardSigmoid | 2 threads | 3 | 92.4 | 72.0 | −22.1% |
| HardSigmoid | 3 threads | 2 | 93.7 | 58.5 | −37.5% |
| Mul | serial | 18 | 262.0 | 479.0 | **+82.8%** |
| Mul | 2 threads | 5 | 113.1 | 107.2 | −5.2% |
| Mul | 3 threads | 4 | 124.6 | 104.6 | −16.1% |
| Mul | 4 threads | 2 | 97.8 | 61.2 | −37.4% |
| ReduceMean | serial | 5 | 68.4 | 82.4 | +20.5% |

**This overturns the working assumption of the previous three rounds.** The ops ORT chooses to
parallelize really do speed up — the cost model's per-op decisions are *locally correct*. The loss is
concentrated in ops that ORT already runs **serially**, which become **83% slower** merely because
the pool has four threads instead of one. A serial op pays no fork-join barrier, so this is not
dispatch overhead. `Shape` and `Reshape` — pure metadata kernels that do no arithmetic and never
parallelize — regress too, which rules out anything intrinsic to the kernels themselves.

The tax on the serial critical path is therefore environmental: the neighbouring parallel ops scatter
their outputs across four private L1/L2s so the next serial op reads lines owned by other cores, and
three spinning workers hold the package at all-core clocks on a 15–28 W mobile part.

Two corollaries, both confirmed below: making ORT parallelize *less* cannot recover this (the
parallel decisions are already right), and turning spinning *off* cannot either (E4).

---

## E4 — Session-level probes (no code change, via `perf_test -C/-T/-D/-Z`)

MobileNetV3-large @ 4T, P50 of 400 iterations.

**Spin policy**

| config | P50 ms | |
|---|---|---|
| default | 3.185 | |
| `-D` (allow_spinning=0) | 3.946 | **+24% worse** |
| `-Z` (force_spinning_stop) | 3.239 | ~neutral |

Parking the workers is decisively worse, reproducing the earlier rounds' finding that the spin is
load-bearing across ~159 nodes of sub-µs dispatches. Note that `-D` cannot help the *main*-thread
side of the barrier in any case: `EndParallelSectionInternal` spins on five unbounded
`SpinPause` loops (`EigenNonBlockingThreadPool.h:1016-1049`) that consult neither `spin_count_` nor
`allow_spinning`.

**Thread affinity** (tests the SMT-sibling hypothesis; the previous round rejected affinity on CCX
grounds, which is a different question)

| config | P50 ms | |
|---|---|---|
| default | 3.172 | |
| `-T 3;5;7` (one logical each, distinct cores) | 3.324 | worse |
| `-T 3,4;5,6;7,8` (one full core each) | 3.139 | −1.0%, within noise |
| `-T 1,2;1,2;1,2` (all workers onto one core) | 4.148 | **+31% worse** |

The deliberate-collision control confirms SMT collision is genuinely expensive, and therefore that
the Windows scheduler is *already* spreading these threads well. Explicit affinity has ≤1% to offer.
**Affinity rejected**, now on measured rather than architectural grounds.

**Spin intensity** — `spin_backoff_max` (upstream PR #28096, present in the base commit; distinct
from the `allow_spinning`/`spin_duration_us` knobs the earlier round rejected, because it keeps
workers unparked and merely reduces how often they re-poll `RunQueue::PopFront`)

| config | P50 ms | |
|---|---|---|
| default (backoff=1) | 3.224 | |
| `--spin_backoff_max 2` | **3.092** | −4.1% |
| `--spin_backoff_max 4` | 3.109 | −3.6% |
| `--spin_backoff_max 16` | 3.107 | −3.6% |
| `--spin_backoff_max 64` | 3.159 | −2.0% |
| `--spin_duration_us 20 / 100 / 1000` | 3.234 / 3.194 / 3.337 | no gain |

Backoff is the only knob that helps. It is the one lever that acts directly on the E3 root cause:
it lowers the duty cycle of the spinning workers, which is what taxes the serial critical path,
without parking them. Full interleaved A/B across all six models and 1T/2T/4T is in progress.

---

---

## E5 — Lever: lower the parallelization threshold (`ORT_PARALLEL_COST_SCALE`)

**Hypothesis, straight out of E3.** ORT's per-op decisions are locally correct but the *serial* side
is being over-charged. Eigen's `TensorCostModel` requires `kStartupCycles = 100000` cycles (~30 µs)
of estimated work before splitting a loop, which is far above this pool's actual fork-join cost.
Meanwhile E3 shows that staying serial is *not* free — it costs ~83%. So the threshold is in the
wrong place: ORT is **under**-parallelizing, not over-parallelizing. (This is the opposite of the
"size gate" reading, and only the per-node data distinguishes the two.)

**Change.** `ThreadPool::ParallelFor` (`onnxruntime/core/common/threadpool.cc:619`) — the single
choke point through which every cost-model-driven loop passes (`TryParallelFor` delegates to it at
`threadpool.cc:703-710`). A scale is applied to the cost used *only* for the go/no-go decision;
`CalculateParallelForBlock` keeps the unscaled cost so block granularity is unchanged. Env-overridable
via `ORT_PARALLEL_COST_SCALE` so both arms run from one DLL.

Sequential sweep, MobileNetV3-large @ 4T, P50 of 400 iterations: scale 1 → 3.631 ms, 2 → 3.347,
4 → 3.313, 8 → **3.080**, 16 → 3.092. Saturates at 8.

**Interleaved A/B (6 cycles, ABBA, per-model rep counts), scale = 8:**

| model | 1T | 2T | 4T |
|---|---|---|---|
| mobilenetv3_large | +1.79 (t=1.4) | **+4.61** (t=3.1) | **+4.44** (t=2.6) |
| mobilenetv3_small | −0.97 (t=−0.5) | **+6.86** (t=3.9) | +3.50 (t=1.2) |
| mobileclip_s0 | +1.37 | +0.25 | +2.72 |
| yolox_s | −0.31 | +3.85 | −4.82 (t=−0.8, cv 5.7%) |
| yolox_tiny | −0.23 | −0.40 | +0.23 |
| yolox_nano | −0.32 | +1.20 (t=2.9) | **+3.53** (t=3.0) |

Note again the drift lesson from E2: the sequential sweep implied −15%, the interleaved
measurement says +4.4%. **The interleaved number is the real one.**

---

## E6 — Combined: cost scale 8 + spin backoff 2

Both levers act on the same root cause from different sides — backoff lowers the spinners' duty
cycle, the cost scale moves work off the taxed serial path — so they were expected to partly
overlap rather than add.

**Interleaved A/B, 6 cycles:**

| model | 1T | 2T | 4T |
|---|---|---|---|
| mobilenetv3_large | −0.15 | +3.07 (t=4.1) | **+5.39 (t=5.4)** |
| mobilenetv3_small | −2.70 (cv 10.1%) | +3.09 (t=1.4) | **+4.58** (t=2.1) |
| mobileclip_s0 | −1.09 | −0.78 | +2.36 (t=1.4) |
| yolox_s | +0.15 | −8.47 (t=−1.1, cv 9.7%) | −1.87 (t=−0.7) |
| yolox_tiny | +0.43 | +0.45 | +1.70 (t=1.5) |
| yolox_nano | +0.22 | +2.46 (t=2.2) | +3.72 (t=1.7) |

**On the 1T column.** Every 1T entry is noise by construction, not by luck: with
`intra_op_num_threads = 1` the pool has zero worker threads, so `DegreeOfParallelism` returns 1,
`CostModel::numThreads` can only return 1, and no worker exists to spin. Neither lever can execute
at 1T. The −2.70% on mobilenetv3_small carries CV 10.1% and t = −0.49; it is measurement noise.

**On yolox_s 2T (−8.47%).** CV 9.7% and t = −1.08 — not significant. yolox_s at 2T is the single
noisiest cell in the matrix across every run in this round.

**Shipped as defaults** (both are threading/session-level, in scope):
* `OrtThreadPoolParams::spin_backoff_max` 1 → 2 (`onnxruntime/core/util/thread_utils.h:46`)
* `ORT_DEFAULT_PARALLEL_COST_SCALE` 8.0 (`onnxruntime/core/common/threadpool.cc`)

Both remain overridable — the first through the existing
`session.intra_op.spin_backoff_max` config key, the second through `ORT_PARALLEL_COST_SCALE`.

---

---

## E7 — Validation through the mandated harnesses

Base and candidate DLLs run back-to-back per (harness, thread count) via `bench/_sweep_ab.bat`, so
each pair is adjacent in time. SS mean latency, ms:

| model | 1T base→cand | 2T base→cand | 4T base→cand |
|---|---|---|---|
| yolox_s | 214.39 → 211.90 (+1.2%) | 165.46 → 168.31 (−1.7%) ⚠ | 106.47 → 106.61 (−0.1%) |
| yolox_tiny | 55.02 → 55.27 (−0.5%) | 45.67 → 47.91 (−4.9%) ⚠ | 30.66 → 29.61 (+3.4%) |
| yolox_nano | 12.34 → 12.50 (−1.3%) | 9.77 → 10.34 (−5.8%) ⚠ | 7.15 → 7.18 (−0.4%) |
| mobilenetv3_large | 6.11 → 5.92 (+3.1%) | 4.05 → 3.96 (+2.2%) | 3.01 → 2.98 (+1.0%) |
| mobilenetv3_small | 2.23 → 2.11 (+5.4%) | 1.67 → 1.57 (+6.0%) | 1.53 → 1.41 (**+7.8%**) |
| mobileclip_s0 | 60.72 → 61.61 (−1.5%) | 46.53 → 44.97 (+3.4%) ⚠ | 26.89 → 28.37 (−5.5%) ⚠ |

⚠ = one or both arms breached the 10% CV gate and the cell must be treated as unreliable. Every 2T
YOLOX cell and both flagged MobileClip cells are in that category (CV 12–23%). The 1T column is
inert by construction (see E6) and its ±5% spread calibrates this harness's single-run noise floor.

**Accuracy — no regression anywhere.**
* MobileNetV3-large / small: Top-1 **0.9000** in both arms at 1T, 2T and 4T.
* YOLOX s / tiny / nano: mAP **1.0000 / 0.0909 / 0.0455** in both arms at all thread counts.
* MobileClip-S0 (no ground-truth labels exist on this machine): Top-1 class **and** score strings
  are **bit-identical** between arms at 1T, 2T and 4T. Both levers are pure scheduling changes that
  alter neither the arithmetic nor the accumulation order, so bit-identity is the expected result.

---

## E8 — VTune SPIN% / QUEUE% (criterion 6)

Release builds emit no PDB, and VTune's own `Spin Time` column only classifies OS synchronization
APIs, not ORT's user-space spin loop — an unsymbolized Release profile reports SPIN as 0.00%. A
RelWithDebInfo tree (`build_sym/`) was therefore built for attribution. Both arms are driven from the
**same** symbolized DLL with the levers toggled by `ORT_PARALLEL_COST_SCALE` / `--spin_backoff_max`,
so there is no build confound. MobileNetV3-large, 4T, 400 iterations, hotspots/sw sampling:

| metric | base | candidate | change |
|---|---|---|---|
| total CPU time | 6.307 s | 6.133 s | −2.8% |
| `concurrency::SpinPause` | 1.892 s | 1.442 s | **−23.8% absolute** |
| **SPIN%** | **29.99%** | **23.52%** | **−6.5 points** |
| `concurrency::RunQueue<...>` | 0.615 s | 0.659 s | +7.2% absolute |
| **QUEUE%** | **9.76%** | **10.75%** | **+1.0 point** |

**SPIN% is down decisively; QUEUE% is slightly up — criterion 6 is only half met, and the reason is
structural rather than incidental.** The two levers pull the two counters in opposite directions:
`spin_backoff_max` halves the spinners' polling rate, which is what removes 0.45 s of `SpinPause`;
but the cost-scale lever works precisely *by creating more parallel dispatches*, and every extra
dispatch is extra `RunQueue` traffic. A QUEUE% reduction is not reachable while the cost-scale lever
is enabled. Reverting to backoff alone would take QUEUE% down with SPIN%, at the price of roughly
half the latency gain (E4 vs E6).

---

## E9 — OpenVINO comparison

ORT built with `--use_openvino CPU` (`openvino_jun17_base_perf`, same June-17 base) against the
MLAS candidate, `onnxruntime_perf_test` P50, ms:

| model | MLAS 1T | OV 1T | MLAS 4T | OV 4T | 4T verdict |
|---|---|---|---|---|---|
| mobilenetv3_large | 5.76 | 6.82 | 2.67 | 2.71 | MLAS +1.4% |
| mobilenetv3_small | 1.96 | 2.46 | 1.17 | 1.16 | parity |
| mobileclip_s0 | 58.88 | 46.22 | 18.26 | 14.54 | **OV +20.4%** |
| yolox_s | 208.11 | 225.97 | 68.39 | 72.28 | MLAS +5.4% |
| yolox_tiny | 54.25 | 55.77 | 19.35 | 18.26 | OV +5.6% |
| yolox_nano | 12.09 | 14.73 | 5.32 | 5.23 | OV +1.8% |

1T→4T scaling: OV 2.12–3.18×, MLAS 1.68–3.22×.

**Explaining the gap.** MLAS wins 1T on five of six models — its kernels are not the problem. OV's
advantage is concentrated in two places:

1. **MobileClip-S0, where OV is ahead at *every* thread count (1T 46.2 vs 58.9 ms).** This is not a
   threading gap at all; it is graph shape. The optimized MobileClip graph still executes its GELU
   as an unfused `Div→Erf→Add→Mul→Mul` chain plus a large number of NCHW⇄NCHWc reorders, while OV's
   whole-graph JIT fuses the transformer blocks per shape. No threading lever can close this;
   it needs op fusion, which is out of scope for this round.
2. **Scaling on the small models.** OV reaches 2.8–3.2× on the YOLOX family versus MLAS's
   2.3–3.0×, because TBB gives it fewer, larger parallel regions over a fused graph, whereas MLAS
   pays a fork-join per node across 150+ nodes.

A measurement caveat worth recording: the mandated harnesses report MLAS 1T→4T scaling as only
1.5–2.3× (E1/E7), but `perf_test` on the identical DLLs reports 2.3–3.2×. The harness figure
includes per-image file I/O and post-processing that does not scale with intra-op threads, so it
understates the compute scaling. Both are reported; the harness number is the one the gate uses.

---

# FINAL SUMMARY

## Best configuration

Two threading defaults, both scheduling-only and both accuracy-neutral by construction:

| change | file | from | to |
|---|---|---|---|
| thread-pool spin duty cycle | `onnxruntime/core/util/thread_utils.h:46` | `spin_backoff_max = 1` | `2` |
| parallelization threshold | `onnxruntime/core/common/threadpool.cc` | cost scale 1.0 | `ORT_DEFAULT_PARALLEL_COST_SCALE 8.0` |

Both remain overridable at runtime (`session.intra_op.spin_backoff_max`, `ORT_PARALLEL_COST_SCALE`).
A third change, NCHWc work-count clamping/oversubscription in `snchwc.cpp`, is present but **left
defaulted OFF** (`MLAS_NCHWC_DEFAULT_WORK_GRANULARITY 0`, i.e. bit-identical to base scheduling)
because it did not earn its place — see E2.

## Does it meet the gate?

**No — not across all six models. Reported honestly:**

| criterion | verdict |
|---|---|
| 1. ≥5–10% improvement | **Partially.** MobileNetV3-small +7.8% (4T harness) / +4.6% (A/B); MobileNetV3-large +5.4% (A/B 4T, t=5.4). MobileClip +2.4%, yolox_nano +3.7%, yolox_tiny +1.7%, yolox_s ~0%. Two of six models reach ≥5% at 4T; none reach 10%. |
| 2. Accuracy | **Met.** Top-1 0.9000 and mAP 1.0000/0.0909/0.0455 unchanged; MobileClip predictions bit-identical. |
| 3. Statistical validity | **Met for the accepted cells** (Welch t=5.4 and 2.1 at 4T on MobileNetV3 over 6 interleaved cycles). **Not met** for most 2T YOLOX/MobileClip cells, which breached the CV gate and are reported as unreliable rather than claimed. |
| 4. OpenVINO comparison | **Met** (E9), with the gap attributed to graph fusion on MobileClip and region granularity on the small models. |
| 5. No 1T/2T regression | **Met at 1T** — structurally inert (`d_of_p`=1, no workers exist to spin), the ±5% spread is noise. **At 2T the picture is mixed**: MobileNetV3 +2.2/+6.0%, but the YOLOX 2T cells show −1.7 to −5.8% at CV 12–20%, i.e. not distinguishable from noise but not demonstrably safe either. |
| 6. VTune SPIN% and QUEUE% both down | **Half met.** SPIN% 29.99→23.52 (−6.5 pts, −23.8% absolute). QUEUE% 9.76→10.75 (+1.0 pt) — unavoidable, since the cost-scale lever's mechanism *is* more dispatches. |

## Overhead breakdown, before and after

MobileNetV3-large @4T, symbolized VTune: SpinPause 29.99% → 23.52%, RunQueue 9.76% → 10.75%,
total CPU −2.8%. Per-node profiling attributes the residual 4T tax to ops that remain on the calling
thread, which run ~83% slower than at 1T (E3).

## What was learned that changes the picture

The three previous rounds concluded that MLAS over-parallelizes tiny ops and that only
dispatch-count reduction (fusion) can help. **Per-node profiling contradicts the first half of
that.** Every op ORT chooses to parallelize genuinely speeds up (−5% to −37%); the entire 4T loss is
concentrated in ops it runs **serially**, which are ~83% slower merely because the pool has four
threads. `Shape` and `Reshape` — which do no arithmetic and never parallelize — regress too, so this
is not a kernel or barrier effect: it is the cost of neighbouring parallel ops scattering their
outputs across four private caches, plus spinners holding a 15–28 W part at all-core clocks. The
correct response is therefore to parallelize *more*, not less, which is the opposite of the "size
gate" that was next on the lever list.

## Recommended further work, in expected-value order

1. **MobileClip GELU fusion and NCHWc reorder elimination.** E9 shows OV beating MLAS by 21% at
   *1T* on this model, so the gap is graph shape, not threading. This is the single largest
   remaining item and it is a fusion lever, deliberately out of scope here.
2. **Cache-affine partitioning across adjacent ops.** E3 identifies the real 4T tax as cross-core
   scatter. If consecutive ops shared a partitioning so each thread re-reads what it wrote, the
   83% serial-op penalty would shrink at its source. `LoopCounter` already has home-shard affinity
   (`threadpool.cc:271-333`) to build on.
3. **Recalibrate Eigen's cost model properly** rather than with a blanket 8× scale: measure this
   pool's real fork-join cost and set `kStartupCycles` from it, per-machine at session init. The
   flat 8× helps MobileNetV3 but does nothing for YOLOX and is unlikely to be portable.
4. **Fix the main-thread barrier spin.** `EndParallelSectionInternal` spins on five *unbounded*
   `SpinPause` loops (`EigenNonBlockingThreadPool.h:1016-1049`) that honour neither `spin_count_`
   nor `allow_spinning`. No session knob can reach them today; bounding them would make the spin
   knobs behave predictably.
5. **Do not re-attempt** on this hardware: `allow_spinning=0` / `spin_duration_us` (E4, and two
   prior rounds), thread affinity (E4 — now rejected on measured grounds, with a collision control),
   NCHWc tids gating/proportional tids (E2 and two prior rounds).

## Reproduction

```
# build            : _build.bat (full)  /  _build_inc.bat (incremental)  /  _build_sym.bat (VTune)
# harness sweep    : bench\_sweep_ab.bat            -> vbase_*t / vcand_*t
# interleaved A/B  : bench\_pt_ab.py --cand-env ORT_PARALLEL_COST_SCALE=8 --cand "--spin_backoff_max 2"
# VTune SPIN/QUEUE : bench\_vtune_ab.bat
# OpenVINO         : bench\_ov.bat
```

