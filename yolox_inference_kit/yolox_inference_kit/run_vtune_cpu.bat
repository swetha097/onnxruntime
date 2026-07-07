@echo off

REM ============================================================================
REM YoloX Multi-EP VTune Profiling
REM FIXES applied:
REM   - start /wait /high /b for inference launches (reduces OS scheduling jitter)
REM   - VTune launch left unchanged (VTune controls priority itself)
REM ============================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM === Configuration ===
set MODEL_DIR=C:\Users\sloganat\Documents\amd-onnxruntime\onnxruntime\yolox\models
set ONNX_INCLUDE=C:\Users\sloganat\Documents\amd-onnxruntime\onnxruntime\include\onnxruntime\core\session
REM  Usage:  run_vtune_cpu_fixed.bat <threads> <onnx_lib_dir> [run_vtune]
REM    %~1  = thread count (default 1)
REM    %~2  = path to folder containing onnxruntime.dll + onnxruntime.lib
REM    %~3  = 0 (default) timing-only benchmark  |  1 full VTune hotspot collection
set ONNX_LIB=
if not "%~2"=="" set ONNX_LIB=%~2
if "%ONNX_LIB%"=="" (
    echo.
    echo ERROR: ONNX_LIB not set. Pass the build output folder as the 2nd argument.
    echo   Usage: run_vtune_cpu_fixed.bat ^<threads^> ^<path\to\Release\Release^> [0^|1]
    echo   Example - benchmark only : run_vtune_cpu_fixed.bat 1 C:\...\Release\Release 0
    echo   Example - VTune profiling: run_vtune_cpu_fixed.bat 1 C:\...\Release\Release 1
    echo.
    exit /b 1
)
set IMAGE_DIR=C:\Users\sloganat\Documents\amd-onnxruntime\onnxruntime\yolox\real_test_images
set CONF=0.45
set NMS=0.45
set VTUNE_BIN=C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64

REM  OpenVINO install paths — only used when EPS contains "OpenVINO"
set OV_BIN=C:\Users\sloganat\Downloads\yolox\openvino\openvino_install_Release\runtime\bin\intel64\Release
set TBB_BIN=C:\Users\sloganat\Downloads\yolox\openvino\openvino_install_Release\runtime\3rdparty\tbb\bin

set EPS=CPU
@REM OpenVINO
set MODELS=yolox_s yolox_nano yolox_tiny

REM  0 = skip VTune, fast timing only (default).  1 = full VTune hotspots.
set RUN_VTUNE=0
if not "%~3"=="" set RUN_VTUNE=%~3

set CV_WARN_THRESH=10
set CV_FAIL_THRESH=20
set WARMUP_OpenVINO=10
set WARMUP_CPU=10
set WARMUP_DNNL=10
set WARMUP_ZenDNN=10
set WARMUP_DEFAULT=10
set TRIALS=50
set LOCAL_SYMBOL_CACHE=C:\Symbols

REM === Thread count ===
set NUM_THREADS=10
if not "%~1"=="" set NUM_THREADS=%~1

set RESULT_DIR=check_newscript_base_%NUM_THREADS%t
set VTUNE_DIR=%RESULT_DIR%_vtune

REM === Output folders ===
if not exist %RESULT_DIR%      mkdir %RESULT_DIR%
if not exist %VTUNE_DIR%       mkdir %VTUNE_DIR%
if not exist "%LOCAL_SYMBOL_CACHE%" mkdir "%LOCAL_SYMBOL_CACHE%"

echo.
echo ============================================================================
echo  YoloX Steady-State + End-to-End Profiling
echo ============================================================================
echo  Models  : %MODELS%
echo  EPs     : %EPS%
echo  VTune   : %VTUNE_BIN%\vtune.exe
echo  Images  : %IMAGE_DIR%
echo ============================================================================
echo.

REM === STEP 1: Check images ===
echo [STEP 1] Checking images...
if not exist "%IMAGE_DIR%" (
    echo ERROR: IMAGE_DIR not found: %IMAGE_DIR%
    exit /b 1
)
echo OK: Using real images from %IMAGE_DIR%
echo.

REM === STEP 2: VS x64 + Compile ===
echo [STEP 2] Setting up Visual Studio x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed.
    exit /b 1
)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

cl /? 2>&1 | findstr /i "x64\|AMD64\|for x64" >nul
if errorlevel 1 (
    echo ERROR: cl.exe is NOT targeting x64.
    exit /b 1
)
echo OK: cl.exe x64 confirmed
echo.

echo [STEP 2] Compiling with full debug symbols...
cl yolox_unified_inference.cpp ^
  /MD /EHsc /std:c++17 /O2 /Zi ^
  /Fe:yolox_inference_standard.exe ^
  /D_CRT_SECURE_NO_WARNINGS ^
  /I"%ONNX_INCLUDE%" ^
  /link /DEBUG:FULL /INCREMENTAL:NO /OPT:NOREF /OPT:NOICF ^
  /LIBPATH:"%ONNX_LIB%" ^
  onnxruntime.lib
if errorlevel 1 (
    echo ERROR: Compilation failed
    exit /b 1
)
echo OK: Compiled -^> yolox_inference_standard.exe
echo.

REM === STEP 2.5: Copy DLLs ===
echo [STEP 2.5] Copying runtime DLLs for EPs: %EPS%
echo   Source (ORT): %ONNX_LIB%
echo.

copy /Y "%ONNX_LIB%\onnxruntime.dll"                  . >nul 2>&1
copy /Y "%ONNX_LIB%\onnxruntime_providers_shared.dll" . >nul 2>&1
if not exist onnxruntime.dll (
    echo ERROR: onnxruntime.dll not found in %ONNX_LIB%
    echo        Check that ONNX_LIB points to the correct build output folder.
    exit /b 1
)
echo   OK [MLAS/CPU] onnxruntime.dll
echo   OK [MLAS/CPU] onnxruntime_providers_shared.dll

echo %EPS% | findstr /I "OpenVINO" >nul
if not errorlevel 1 (
    echo.
    echo   [OpenVINO] Source OV bin : %OV_BIN%
    echo   [OpenVINO] Source TBB bin: %TBB_BIN%

    copy /Y "%ONNX_LIB%\onnxruntime_providers_openvino.dll" . >nul 2>&1
    if not exist onnxruntime_providers_openvino.dll (
        echo.
        echo   ERROR: onnxruntime_providers_openvino.dll not found in %ONNX_LIB%
        exit /b 1
    )
    echo   OK [OpenVINO] onnxruntime_providers_openvino.dll

    for %%F in (
        openvino.dll
        openvino_auto_batch_plugin.dll
        openvino_auto_plugin.dll
        openvino_c.dll
        openvino_hetero_plugin.dll
        openvino_intel_cpu_plugin.dll
        openvino_ir_frontend.dll
        openvino_onnx_frontend.dll
    ) do (
        copy /Y "%OV_BIN%\%%F" . >nul 2>&1
        if exist "%%F" (
            echo   OK [OpenVINO] %%F
        ) else (
            echo   WARNING [OpenVINO] %%F not found in %OV_BIN%
        )
    )

    for %%F in (tbb12.dll tbbbind_2_5.dll tbbmalloc.dll) do (
        copy /Y "%TBB_BIN%\%%F" . >nul 2>&1
        if exist "%%F" (
            echo   OK [TBB] %%F
        ) else (
            echo   WARNING [TBB] %%F not found in %TBB_BIN%
        )
    )
)

echo.
echo OK: All DLLs staged for EPs: %EPS%
echo.

REM === STEP 3: Ground truth ===
if not exist real_annotations.txt (
    echo [STEP 3] Generating ground truth with yolox_s + CPU EP...
    REM FIX: high-priority launch to reduce OS scheduling jitter
    start /wait /high /b "" .\yolox_inference_standard.exe ^
      --model "%MODEL_DIR%\yolox_s.onnx" ^
      --image-dir "%IMAGE_DIR%" ^
      --ep CPU ^
      --save-gt real_annotations.txt ^
      --conf_threshold %CONF% ^
      --nms_threshold %NMS% ^
      --warmup 5 > nul 2>&1
    echo OK: Ground truth saved to real_annotations.txt
) else (
    echo [STEP 3] Using existing real_annotations.txt
)
echo.

REM === STEP 4: Symbols + env ===
set VS_REDIST=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Redist\MSVC
set MSVC_DEBUG_CRT=
for /f "delims=" %%V in ('dir /b /ad "%VS_REDIST%" 2^>nul ^| sort /r') do (
    set MSVC_DEBUG_CRT=%VS_REDIST%\%%V\debug_nonredist\x64\Microsoft.VC143.DebugCRT
    goto :symbol_path_set
)
:symbol_path_set

set SCRIPT_DIR_CLEAN=%SCRIPT_DIR:~0,-1%
set _NT_SYMBOL_PATH=srv*%LOCAL_SYMBOL_CACHE%*https://msdl.microsoft.com/download/symbols;%MSVC_DEBUG_CRT%;%SCRIPT_DIR_CLEAN%
echo [STEP 4] Symbol path:
echo   %_NT_SYMBOL_PATH%
echo.

set "PATH=%VTUNE_BIN%;%PATH%"
set OV_LOG_LEVEL=0
set OV_LOG_SINK=none

REM === STEP 5: Profiling loop ===
echo [STEP 5] Running profiling...
echo.

set SUMMARY_FILE=%RESULT_DIR%\summary_steady_state_only.txt
if exist "%SUMMARY_FILE%" del "%SUMMARY_FILE%"
if exist "%RESULT_DIR%\cv_warnings.txt" del "%RESULT_DIR%\cv_warnings.txt"

echo YoloX Steady-State + End-to-End Profiling Summary> "%SUMMARY_FILE%"
echo EPs benchmarked: %EPS%   Warmup: %WARMUP_CPU%   Trials: %TRIALS%>> "%SUMMARY_FILE%"
echo Logging: OV_LOG_LEVEL=0  ORT=WARNING>> "%SUMMARY_FILE%"
echo.>> "%SUMMARY_FILE%"
echo NOTE on metrics:>> "%SUMMARY_FILE%"
echo   SS  = Steady State -- measurement trials only, after warmup>> "%SUMMARY_FILE%"
echo   E2E = End-to-End   -- warmup + measurement trials pooled>> "%SUMMARY_FILE%"
echo   Latency    = mean per-image latency in ms>> "%SUMMARY_FILE%"
echo   Throughput = N_samples / sum_latencies in FPS -- real throughput>> "%SUMMARY_FILE%"
echo   FPS        = 1000 / p50 latency -- 4-stream peak>> "%SUMMARY_FILE%"
echo.>> "%SUMMARY_FILE%"
echo Model           EP         Mode  Latency_ms  Throughput  FPS         p50_ms      p90_ms      mAP        trials wu trial-std  ci95_ms   cv_pct>> "%SUMMARY_FILE%"
echo --------------------------------------------------------------------------------------------------------------------------------------------------->> "%SUMMARY_FILE%"

for %%M in (%MODELS%) do (
    for %%E in (%EPS%) do (
        set WU=!WARMUP_DEFAULT!
        if defined WARMUP_%%E set WU=!WARMUP_%%E!
        call :PROFILE_MODEL %%M %%E !WU!
    )
)

goto :PRINT_SUMMARY

REM ============================================================================
REM Subroutine: PROFILE_MODEL
REM ============================================================================
:PROFILE_MODEL
setlocal
set M=%1
set E=%2
set WU=%3
set RESULT=%RESULT_DIR%\%M%_%E%_steady.txt
set VTUNE_RESULT_DIR=%VTUNE_DIR%\%M%_%E%

set ONNX_LIB_SEARCH=%ONNX_LIB%
if "%ONNX_LIB_SEARCH:~-1%"=="\" set ONNX_LIB_SEARCH=%ONNX_LIB_SEARCH:~0,-1%

echo ============================================
echo  Model : %M%   EP : %E%   Warmup: %WU% runs
echo ============================================

REM --- [a] Inference --- FIX: high-priority launch reduces OS scheduling jitter
echo  [a] Standalone inference -- measuring SS + E2E...
start /wait /high /b "" .\yolox_inference_standard.exe ^
    --model "%MODEL_DIR%\%M%.onnx" ^
    --image-dir "%IMAGE_DIR%" ^
    --ep %E% ^
    --ground_truth real_annotations.txt ^
    --conf_threshold %CONF% ^
    --nms_threshold %NMS% ^
    --warmup %WU% ^
    --trials %TRIALS% ^
    --threads %NUM_THREADS% > "%RESULT%" 2>&1
if errorlevel 1 (
    echo   WARNING: Inference returned error. Check %RESULT%
) else (
    echo   OK: Metrics saved to %RESULT%
)

echo.
echo  --- Performance Metrics ---
findstr /C:"SS mean" /C:"SS p50" /C:"SS p90" /C:"SS Sustained" /C:"SS Inv p50" /C:"E2E mean" /C:"E2E p50" /C:"E2E p90" /C:"E2E Sustained" /C:"E2E Inv p50" /C:"Overall mAP" /C:"Warmup runs used" /C:"Measurement trials" /C:"Trial std-dev" "%RESULT%" 2>nul
echo.

REM --- [b/c/d] VTune --- NOTE: no start /high here — VTune controls priority itself
if /I not "%RUN_VTUNE%"=="1" goto :SKIP_VTUNE

echo  [b] Collecting VTune hotspots...
if exist "%VTUNE_RESULT_DIR%" rmdir /s /q "%VTUNE_RESULT_DIR%"

vtune -collect hotspots -knob sampling-mode=sw ^
    -result-dir "%VTUNE_RESULT_DIR%" ^
    -search-dir "." ^
    -search-dir "%ONNX_LIB_SEARCH%" ^
    -- .\yolox_inference_standard.exe ^
    --model "%MODEL_DIR%\%M%.onnx" ^
    --image-dir "%IMAGE_DIR%" ^
    --ep %E% ^
    --ground_truth real_annotations.txt ^
    --conf_threshold %CONF% ^
    --nms_threshold %NMS% ^
    --warmup %WU% ^
    --trials %TRIALS% ^
    --threads %NUM_THREADS% > "%VTUNE_DIR%\%M%_%E%_vtune_run.txt" 2>&1
if errorlevel 1 (
    echo   WARNING: VTune collection returned error.
) else (
    echo   OK: VTune hotspot data collected
)

echo  [c] Exporting hotspots CSV...
vtune -report hotspots ^
    -r "%VTUNE_RESULT_DIR%" ^
    -format csv ^
    -csv-delimiter comma ^
    -report-output "%VTUNE_DIR%\%M%_%E%_hotspots.csv" ^
    -search-dir "." ^
    -search-dir "%ONNX_LIB_SEARCH%" > nul 2>&1
if errorlevel 1 (
    echo   WARNING: VTune hotspots CSV export failed
) else (
    echo   OK: Hotspots CSV saved to %VTUNE_DIR%\%M%_%E%_hotspots.csv
)

echo  [d] Exporting top-down CSV...
vtune -report top-down ^
    -r "%VTUNE_RESULT_DIR%" ^
    -format csv ^
    -csv-delimiter comma ^
    -report-output "%VTUNE_DIR%\%M%_%E%_topdown.csv" ^
    -search-dir "." ^
    -search-dir "%ONNX_LIB_SEARCH%" > nul 2>&1
if errorlevel 1 (
    echo   WARNING: VTune top-down CSV export failed
) else (
    echo   OK: Top-down CSV saved to %VTUNE_DIR%\%M%_%E%_topdown.csv
)

echo  [e] Exporting summary CSV...
vtune -report summary ^
    -r "%VTUNE_RESULT_DIR%" ^
    -format csv ^
    -csv-delimiter comma ^
    -report-output "%VTUNE_DIR%\%M%_%E%_summary.csv" ^
    -search-dir "." ^
    -search-dir "%ONNX_LIB_SEARCH%" > nul 2>&1
if errorlevel 1 (
    echo   WARNING: VTune summary CSV export failed
) else (
    echo   OK: Summary CSV saved to %VTUNE_DIR%\%M%_%E%_summary.csv
)

echo.
echo  --- VTune Top Hotspots ---
findstr /C:"MlasCon" /C:"SpinPause" /C:"WorkerLoop" /C:"Elapsed Time:" /C:"CPU Time:" /C:"Thread Count:" "%VTUNE_DIR%\%M%_%E%_vtune_run.txt" 2>nul
echo.
echo  VTune reports saved to: %VTUNE_DIR%\
echo.
goto :AFTER_VTUNE

:SKIP_VTUNE
echo  [b/c/d/e] VTune profiling SKIPPED. Set RUN_VTUNE=1 to enable.
echo.

:AFTER_VTUNE

set CV_WARN_THRESH=10
set CV_FAIL_THRESH=20

REM --- Append SS summary row ---
powershell -NoProfile -Command "try { $r=[System.IO.File]::ReadAllText('%RESULT%'); $p50=[regex]::Match($r,'SS p50\s*:\s*([\d.]+)').Groups[1].Value; $p90=[regex]::Match($r,'SS p90\s*:\s*([\d.]+)').Groups[1].Value; $mean=[regex]::Match($r,'SS mean\s*:\s*([\d.]+)').Groups[1].Value; $sus=[regex]::Match($r,'SS Sustained\s*:\s*([\d.]+)').Groups[1].Value; $invp50=[regex]::Match($r,'SS Inv p50\s*:\s*([\d.]+)').Groups[1].Value; $map=[regex]::Match($r,'Overall mAP[^:]*:\s*([\d.]+)').Groups[1].Value; $wu=[regex]::Match($r,'Warmup runs used\s*:\s*(\d+)').Groups[1].Value; $tr=[regex]::Match($r,'Measurement trials\s*:\s*(\d+)').Groups[1].Value; $tsd=[regex]::Match($r,'Trial std-dev\s*:\s*([\d.]+)').Groups[1].Value; $n=0.0; if($tr){$n=[double]$tr}; $ci95=''; if($n -gt 0 -and $tsd){try{$ci95=[math]::Round(1.96*[double]$tsd/[math]::Sqrt($n),3).ToString('F3')}catch{$ci95=''}}; $cv=''; if($mean -and $tsd){try{$cv=[math]::Round([double]$tsd/[double]$mean*100,1).ToString('F1')+[char]37}catch{$cv=''}}; if($p50){$row=('%M%').PadRight(16)+('%E%').PadRight(11)+'SS'.PadRight(6)+$mean.PadRight(12)+$sus.PadRight(12)+$invp50.PadRight(12)+$p50.PadRight(12)+$p90.PadRight(12)+$map.PadRight(11)+$tr.PadRight(7)+$wu.PadRight(3)+$tsd.PadRight(11)+$ci95.PadRight(10)+$cv; Add-Content '%RESULT_DIR%\summary_steady_state_only.txt' $row}else{Write-Host ('  [SUMROW-WARN] SS %M%/%E%: p50 not found in result file') -ForegroundColor Yellow} } catch { Write-Host ('  [SUMROW-ERR] SS %M%/%E%: '+$_.Exception.Message) -ForegroundColor Red }"

REM --- Append E2E summary row ---
powershell -NoProfile -Command "try { $r=[System.IO.File]::ReadAllText('%RESULT%'); $p50=[regex]::Match($r,'E2E p50\s*:\s*([\d.]+)').Groups[1].Value; $p90=[regex]::Match($r,'E2E p90\s*:\s*([\d.]+)').Groups[1].Value; $mean=[regex]::Match($r,'E2E mean\s*:\s*([\d.]+)').Groups[1].Value; $sus=[regex]::Match($r,'E2E Sustained\s*:\s*([\d.]+)').Groups[1].Value; $invp50=[regex]::Match($r,'E2E Inv p50\s*:\s*([\d.]+)').Groups[1].Value; $map=[regex]::Match($r,'Overall mAP[^:]*:\s*([\d.]+)').Groups[1].Value; $wu=[regex]::Match($r,'Warmup runs used\s*:\s*(\d+)').Groups[1].Value; $tr=[regex]::Match($r,'Measurement trials\s*:\s*(\d+)').Groups[1].Value; $tsd=[regex]::Match($r,'Trial std-dev\s*:\s*([\d.]+)').Groups[1].Value; $n=0.0; if($tr){$n=[double]$tr}; $ci95=''; if($n -gt 0 -and $tsd){try{$ci95=[math]::Round(1.96*[double]$tsd/[math]::Sqrt($n),3).ToString('F3')}catch{$ci95=''}}; $cv=''; if($mean -and $tsd){try{$cv=[math]::Round([double]$tsd/[double]$mean*100,1).ToString('F1')+[char]37}catch{$cv=''}}; if($p50){$row=('%M%').PadRight(16)+('%E%').PadRight(11)+'E2E'.PadRight(6)+$mean.PadRight(12)+$sus.PadRight(12)+$invp50.PadRight(12)+$p50.PadRight(12)+$p90.PadRight(12)+$map.PadRight(11)+$tr.PadRight(7)+$wu.PadRight(3)+$tsd.PadRight(11)+$ci95.PadRight(10)+$cv; Add-Content '%RESULT_DIR%\summary_steady_state_only.txt' $row}else{Write-Host ('  [SUMROW-WARN] E2E %M%/%E%: p50 not found in result file') -ForegroundColor Yellow} } catch { Write-Host ('  [SUMROW-ERR] E2E %M%/%E%: '+$_.Exception.Message) -ForegroundColor Red }"

REM --- CV Gate ---
powershell -NoProfile -Command "$r=(Get-Content '%RESULT%') -join ' '; $tsd=[regex]::Match($r,'Trial std-dev\s*:\s*([\d.]+)').Groups[1].Value; $mean=[regex]::Match($r,'SS mean\s*:\s*([\d.]+)').Groups[1].Value; $tr=[regex]::Match($r,'Measurement trials\s*:\s*(\d+)').Groups[1].Value; if($mean -and [double]$mean -gt 0 -and $tsd -ne ''){$cv=[math]::Round([double]$tsd/[double]$mean*100,1); $tag='%M%/%E% SS (n='+$tr+')  mean='+$mean+'ms  std='+$tsd+'ms  CV='+$cv+'pct'; if($cv -gt %CV_FAIL_THRESH%){$msg='  [CV-FAIL] '+$tag+'  >%CV_FAIL_THRESH%pct -- UNRELIABLE: results cannot be trusted'; Write-Host $msg -ForegroundColor Red; Add-Content '%RESULT_DIR%\cv_warnings.txt' $msg}elseif($cv -gt %CV_WARN_THRESH%){$msg='  [CV-WARN] '+$tag+'  >%CV_WARN_THRESH%pct -- questionable: increase TRIALS or isolate machine'; Write-Host $msg -ForegroundColor Yellow; Add-Content '%RESULT_DIR%\cv_warnings.txt' $msg}else{Write-Host ('  [CV-OK]   '+$tag) -ForegroundColor Green}}" 2>nul

echo  Result file  : %RESULT%
if /I "%RUN_VTUNE%"=="1" echo  VTune reports : %VTUNE_DIR%\
echo.
endlocal
goto :eof

REM ============================================================================
REM STEP 6: Final summary
REM ============================================================================
:PRINT_SUMMARY
echo.
echo ============================================================================
echo  FINAL COMPARISON SUMMARY  (SS = Steady State,  E2E = End-to-End)
echo ============================================================================
type "%RESULT_DIR%\summary_steady_state_only.txt"
echo.

if exist "%RESULT_DIR%\cv_warnings.txt" (
    echo ============================================================================
    echo  CV STABILITY WARNINGS  -- flagged runs may not be trustworthy
    echo ============================================================================
    type "%RESULT_DIR%\cv_warnings.txt"
    echo.
    echo  Thresholds  : CV ^<%CV_WARN_THRESH%pct=stable  %CV_WARN_THRESH%--%CV_FAIL_THRESH%pct=questionable  ^>%CV_FAIL_THRESH%pct=unreliable
    echo  Fix actions : increase TRIALS (current=%TRIALS%^), close background apps,
    echo                check thermal throttling, re-run flagged model/EP pairs
    echo ============================================================================
    echo.
) else (
    echo  CV check: all runs within acceptable stability limits.
    echo.
)

echo Column legend:
echo   Mode       = SS (measurement only) or E2E (warmup + measurement)
echo   Latency_ms = mean per-image latency
echo   Throughput = real images-per-second, N / sum-latency
echo   FPS        = 1000 / p50_ms, 4-stream peak
echo   p50/p90    = latency percentiles in ms
echo   mAP        = detection accuracy at IoU 0.5
echo   trial-std  = per-trial std-dev (ms)
echo   ci95_ms    = 95%% confidence interval half-width: 1.96 * std / sqrt(n)
echo   cv_pct     = coefficient of variation: std / mean * 100
echo               ^< 5%% = very stable,  5-15%% = acceptable,  ^>15%% = noisy / suspect
echo.
echo Detailed results : %RESULT_DIR%\
if /I "%RUN_VTUNE%"=="1" (
    echo VTune CSV reports : %VTUNE_DIR%\*_hotspots.csv
    echo VTune CSV reports : %VTUNE_DIR%\*_topdown.csv
    echo VTune CSV reports : %VTUNE_DIR%\*_summary.csv
    echo VTune run logs   : %VTUNE_DIR%\*_vtune_run.txt
)
echo ============================================================================

endlocal
exit /b 0
