"""
run_pipeline.py — Pitwall Pipeline Orchestrator
─────────────────────────────────────────────────
Run this notebook on a Databricks CE cluster. It calls each stage notebook
in sequence using dbutils.notebook.run(), reports status, and stops cleanly
on failure.

USAGE
─────
Set the PIPELINE widget at the top of the notebook, then Run All.

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
               CE-safe: checkpoints every epoch. Re-run this pipeline after every
               2-hr CE session until 200 epochs are complete.

  mae_finetune Fine-tune the pre-trained encoder for position prediction
               10_mae_finetune
               Run once after mae_pretrain reaches 200 epochs.

  full_mae     Full MAE pipeline (one-time historical setup, then training)
               07_tel_ingest → 08_tel_preprocess → 09_mae_pretrain → 10_mae_finetune
               ⚠ Only use if you have a long CE session or are willing to restart.
"""

import os
import pathlib
import sys
import time
from datetime import datetime, timezone

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── WIDGET / PIPELINE SELECTOR ────────────────────────────────────────────────
# On Databricks this creates a dropdown widget at the top of the notebook.
# Locally it reads from the environment variable PIPELINE (default: weekend_rf).

try:
    dbutils.widgets.dropdown(
        "PIPELINE",
        "weekend_rf",
        [
            "weekend_rf",
            "weekend_mae",
            "tel_backfill",
            "mae_pretrain",
            "mae_finetune",
            "full_mae",
        ],
        "Select Pipeline",
    )
    PIPELINE = dbutils.widgets.get("PIPELINE")
    ON_DATABRICKS = True
except NameError:
    PIPELINE      = os.environ.get("PIPELINE", "weekend_rf")
    ON_DATABRICKS = False

# ── PIPELINE DEFINITIONS ──────────────────────────────────────────────────────

NOTEBOOKS_ROOT = str(PROJECT_ROOT / "notebooks")

# Each step: (notebook_path_relative_to_workspace, timeout_seconds, description)
# timeout_seconds for dbutils.notebook.run():
#   CE 2-hr hard limit = 7200s, but we set lower so failures surface faster.
#   Long-running notebooks (09, 10) are set to 7100s (just under CE limit).

PIPELINE_STEPS = {
    "weekend_rf": [
        ("notebooks/01_ingest",       1800, "Ingest lap data from FastF1"),
        ("notebooks/02_clean",         600, "Clean Bronze → Silver Delta"),
        ("notebooks/03_features",      600, "Engineer features → Gold Delta"),
        ("notebooks/05_train",         900, "Train RF/GBT baseline model"),
        ("notebooks/06_predict",       600, "Generate predictions + dashboard JSON"),
    ],
    "weekend_mae": [
        ("notebooks/07_tel_ingest",   3600, "Ingest raw telemetry (current season)"),
        ("notebooks/08_tel_preprocess",1800, "Preprocess → Silver tensors"),
        ("notebooks/11_mae_predict",   600, "MAE inference + dashboard JSON"),
    ],
    "tel_backfill": [
        ("notebooks/07_tel_ingest",   7100, "Ingest raw telemetry (historical backfill)"),
        ("notebooks/08_tel_preprocess",3600, "Preprocess → Silver tensors"),
    ],
    "mae_pretrain": [
        ("notebooks/09_mae_pretrain", 7100, "MAE self-supervised pre-training (resumes from checkpoint)"),
    ],
    "mae_finetune": [
        ("notebooks/10_mae_finetune", 7100, "Fine-tune encoder for position prediction"),
    ],
    "full_mae": [
        ("notebooks/07_tel_ingest",   7100, "Ingest raw telemetry (historical backfill)"),
        ("notebooks/08_tel_preprocess",3600, "Preprocess → Silver tensors"),
        ("notebooks/09_mae_pretrain", 7100, "MAE self-supervised pre-training"),
        ("notebooks/10_mae_finetune", 7100, "Fine-tune encoder for position prediction"),
    ],
}

if PIPELINE not in PIPELINE_STEPS:
    raise ValueError(
        f"Unknown pipeline: '{PIPELINE}'. "
        f"Valid options: {list(PIPELINE_STEPS.keys())}"
    )

steps = PIPELINE_STEPS[PIPELINE]

# ── RESOLVE ABSOLUTE NOTEBOOK PATHS ──────────────────────────────────────────
# dbutils.notebook.run() needs absolute Workspace paths.
# When running via Repos the root is /Workspace/Repos/<repo-name>.

WORKSPACE_ROOT = "/Workspace/Repos/pitwall"

# ── DISPLAY PLAN ──────────────────────────────────────────────────────────────

print(f"{'='*60}")
print(f"  PITWALL PIPELINE: {PIPELINE.upper()}")
print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"  Steps   : {len(steps)}")
print(f"{'='*60}")
for i, (nb, timeout, desc) in enumerate(steps, 1):
    print(f"  {i}. {desc}")
    print(f"     └─ {nb}  (timeout: {timeout//60} min)")
print()

# ── RUN STEPS ─────────────────────────────────────────────────────────────────

results   = []
t_overall = time.time()

for step_idx, (notebook_rel, timeout_s, description) in enumerate(steps, 1):
    nb_abs  = f"{WORKSPACE_ROOT}/{notebook_rel}"
    t_start = time.time()

    print(f"[{step_idx}/{len(steps)}] START  ── {description}")
    print(f"          Notebook : {nb_abs}")
    print(f"          Timeout  : {timeout_s // 60} min")

    try:
        if ON_DATABRICKS:
            result = dbutils.notebook.run(nb_abs, timeout_s, arguments={})
        else:
            # Local fallback — execute the notebook file directly as a script
            # (useful for testing outside Databricks)
            import runpy
            nb_local = str(PROJECT_ROOT / notebook_rel) + ".py"
            runpy.run_path(nb_local, run_name="__main__")
            result = "OK (local)"

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
    _, _, desc = steps[i - 1]
    print(f"  ⏭  [{i}] {steps[i-1][2]:45s}  (skipped)")

if n_failed:
    print(f"\n  Pipeline stopped at step {len(results)}.")
    print(f"  Fix the error above, then re-run from step {len(results)}.")
else:
    print(f"\n  All steps completed successfully.")
    if PIPELINE == "weekend_rf":
        print(f"  ► predictions.json is live. Push to GitHub to trigger Vercel redeploy.")
    elif PIPELINE == "weekend_mae":
        print(f"  ► MAE predictions written. predictions.json updated.")
    elif PIPELINE in ("mae_pretrain",):
        print(f"  ► Checkpoint saved. Re-run mae_pretrain to continue training.")
        print(f"     When epoch 199 is reached, switch to mae_finetune.")
