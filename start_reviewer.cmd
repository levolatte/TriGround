@echo off
cd /d "%~dp0"
if not exist ".conda-env\python.exe" (
  echo Missing .conda-env\python.exe
  pause
  exit /b 1
)
".conda-env\python.exe" tools\review_grounding.py ^
  --manifest "..\city_detection_prepared\train\grounding_final_val.json" ^
  --data-root "..\city_detection_prepared\train" ^
  --predictions "..\qwen_rgb_error_analysis\results.jsonl" ^
  --reviews "..\qwen_rgb_error_analysis\grounding_final_val_reviews.json" ^
  --host 127.0.0.1 ^
  --port 8765 ^
  --open
