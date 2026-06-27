"""
run_pipeline.py — Pitwall Pipeline Orchestrator
─────────────────────────────────────────────────
Runs locally via `python run_pipeline.py` or in GitHub Actions.
Calls each stage notebook in sequence, reports status, and stops on failure.

USAGE
─────
Set the PIPELINE env var, then run:
  PIPELINE=weekend_rf python run_pipeline.py

Pipelines
─────────
  weekend_rf   Today's race prediction (RF/GBT baseline — use until MAE is trained)
               01_ingest → 02_clean → 03_features → 05_train → 06_predict

  weekend_mae  Race prediction using trained MAE model
               07_tel_ingest → 08_tel_preprocess → 11_mae_predict
               (11_mae_predict will exit gracefully if model not yet trained)

  tel_backfill Historical telemetry ingestion + preprocessing (one-time)
               07_tel_ingest → 08_tel_preprocess

  mae_pretrain MAE self-supervised pre-training (run repeatedly; resumes each time)
               09_mae_pretrain
               Checkpoints every epoch. Re-run until 200 epochs are complete.

  mae_finetune Fine-tune the pre-trained encoder for position prediction
               10_mae_finetune
               Run once after mae_pretrain reaches 200 epochs.

  full_mae     Full MAE pipeline (one-time historical setup, then training)
               07_tel_ingest → 08_tel_preprocess → 09_mae_pretrain → 10_mae_finetune
"""

import os
import pathlib
import sys
import time
from datetime import datetime, timezone

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── PIPELINE SELECTOR ────────────────────────────────────────────────────────

PIPELINE = os.environ.get("PIPELINE", "weekend_rf")

# ── PIPELINE DEFINITIONS ──────────────────────────────────────────────────────

NOTEBOOKS_ROOT = str(PROJECT_ROOT / "notebooks")

# Each step: (notebook_path_relative_to_project, description)
PIPELINE_STEPS = {
    "weekend_rf": [
        ("notebooks/01_ingest",       "Ingest lap data from FastF1"),
        ("notebooks/02_clean",        "Clean Bronze → Silver Parquet"),
        ("notebooks/03_features",     "Engineer features → Gold Parquet"),
        ("notebooks/05_train",        "Train RF/GBT baseline model"),
        ("notebooks/06_predict",      "Generate predictions + dashboard JSON"),
    ],
    "weekend_mae": [
        ("notebooks/07_tel_ingest",    "Ingest raw telemetry (current season)"),
        ("notebooks/08_tel_preprocess","Preprocess → Silver tensors"),
        ("notebooks/11_mae_predict",   "MAE inference + dashboard JSON"),
    ],
    "tel_backfill": [
        ("notebooks/07_tel_ingest",    "Ingest raw telemetry (historical backfill)"),
        ("notebooks/08_tel_preprocess","Preprocess → Silver tensors"),
    ],
    "mae_pretrain": [
        ("notebooks/09_mae_pretrain",  "MAE self-supervised pre-training (resumes from checkpoint)"),
    ],
    "mae_finetune": [
        ("notebooks/10_mae_finetune",  "Fine-tune encoder for position prediction"),
    ],
    "retrain_predict": [
        ("notebooks/05_train",        "Train RF/GBT baseline model"),
        ("notebooks/06_predict",      "Generate predictions + dashboard JSON"),
    ],
    "full_mae": [
        ("notebooks/07_tel_ingest",    "Ingest raw telemetry (historical backfill)"),
        ("notebooks/08_tel_preprocess","Preprocess → Silver tensors"),
        ("notebooks/09_mae_pretrain",  "MAE self-supervised pre-training"),
        ("notebooks/10_mae_finetune",  "Fine-tune encoder for position prediction"),
    ],
    "dev_eda": [
        ("notebooks/01_ingest",       "Ingest lap data from FastF1"),
        ("notebooks/02_clean",        "Clean Bronze → Silver Parquet"),
        ("notebooks/03_features",     "Engineer features → Gold Parquet"),
        ("notebooks/04_eda",          "EDA + correlation analysis (dev only)"),
    ],
}

if PIPELINE not in PIPELINE_STEPS:
    raise ValueError(
        f"Unknown pipeline: '{PIPELINE}'. "
        f"Valid options: {list(PIPELINE_STEPS.keys())}"
    )

steps = PIPELINE_STEPS[PIPELINE]

# ── DISPLAY PLAN ──────────────────────────────────────────────────────────────

print(f"{'='*60}")
print(f"  PITWALL PIPELINE: {PIPELINE.upper()}")
print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Steps   : {len(steps)}")
print(f"{'='*60}")
for i, (nb, desc) in enumerate(steps, 1):
    print(f"  {i}. {desc}")
    print(f"     |-- {nb}")
print()

# ── RUN STEPS ─────────────────────────────────────────────────────────────────

import runpy

results   = []
t_overall = time.time()

for step_idx, (notebook_rel, description) in enumerate(steps, 1):
    py_file = str(PROJECT_ROOT / f"{notebook_rel}.py")
    t_start = time.time()

    print(f"[{step_idx}/{len(steps)}] START  ── {description}")
    print(f"          File     : {py_file}")

    try:
        runpy.run_path(py_file, run_name="__main__")
        result = "OK"

        elapsed = time.time() - t_start
        print(f"[{step_idx}/{len(steps)}] DONE   ── {description}  ({elapsed:.0f}s)")
        results.append((step_idx, description, "OK", elapsed, None))

    except Exception as e:
        elapsed  = time.time() - t_start
        err_msg  = str(e)[:300]
        print(f"\n{'!'*60}")
        print(f"[{step_idx}/{len(steps)}] FAILED ── {description}  ({elapsed:.0f}s)")
        print(f"  Error: {err_msg}")
        print(f"{'!'*60}\n")
        results.append((step_idx, description, "FAILED", elapsed, err_msg))
        # Stop pipeline on failure — don't run downstream steps on bad data
        break

    print()

# ── SUMMARY ───────────────────────────────────────────────────────────────────

total_elapsed = time.time() - t_overall
n_ok     = sum(1 for r in results if r[2] == "OK")
n_failed = sum(1 for r in results if r[2] == "FAILED")
n_skip   = len(steps) - len(results)   # steps not reached due to earlier failure

print(f"\n{'='*60}")
print(f"  PIPELINE SUMMARY: {PIPELINE.upper()}")
print(f"  Finished : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Total    : {total_elapsed:.0f}s  ({total_elapsed/60:.1f} min)")
print(f"  Passed   : {n_ok} / {len(steps)}")
if n_failed:
    print(f"  Failed   : {n_failed}")
if n_skip:
    print(f"  Skipped  : {n_skip}  (not reached)")
print(f"{'='*60}")

for step_idx, desc, status, elapsed, err in results:
    icon = "✅" if status == "OK" else "❌"
    print(f"  {icon} [{step_idx}] {desc:45s}  {elapsed:>6.0f}s")

for i in range(len(results) + 1, len(steps) + 1):
    print(f"  ⏭  [{i}] {steps[i-1][1]:45s}  (skipped)")

if n_failed:
    print(f"\n  Pipeline stopped at step {len(results)}.")
    print(f"  Fix the error above, then re-run from step {len(results)}.")
else:
    print(f"\n  All steps completed successfully.")
    if PIPELINE == "weekend_rf":
        print(f"  ► Predictions written to Parquet + Supabase.")
        print(f"  ► predictions.json updated locally for dashboard.")
    elif PIPELINE == "weekend_mae":
        print(f"  ► MAE predictions written to Parquet + Supabase.")
    elif PIPELINE in ("mae_pretrain",):
        print(f"  ► Checkpoint saved. Re-run mae_pretrain to continue training.")
        print(f"     When epoch 199 is reached, switch to mae_finetune.")
