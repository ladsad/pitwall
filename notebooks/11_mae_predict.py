"""
11_mae_predict.py
──────────────────
PRIMARY PREDICTION NOTEBOOK — supersedes 06_predict.py.

Runs the fine-tuned F1MAE encoder on the current race weekend's telemetry,
writes predictions to the shared PREDICTIONS_PATH Delta table (model_version='mae'),
and exports predictions.json for the dashboard.

Prerequisites (run in order each race weekend):
  01_ingest.py → 02_clean.py → 03_features.py   (Gold Delta — needed for labels)
  07_tel_ingest.py → 08_tel_preprocess.py        (Silver telemetry)
  [09_mae_pretrain.py + 10_mae_finetune.py done once historically]

06_predict.py remains available as an RF/GBT comparison baseline.
"""

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import torch
import numpy as np
from pyspark.sql import functions as F

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.spark_session import get_spark_session
from utils.mae_model import F1MAE, F1PositionHead
from utils.tel_dataset import F1TelemetryDataset
from utils.tel_schema import TEL_SILVER_SCHEMA
from config import (
    SEASON, EVENT, ROUND_NUMBER,
    TELEMETRY_CLEAN_PATH, BASE_PATH, PREDICTIONS_PATH,
    ENCODER_HPARAMS  # added to config.py alongside other ML settings
)

spark = get_spark_session("pitwall-mae-predict")

# ── PATHS ─────────────────────────────────────────────────────────────────────

SILVER_PATH     = f"{TELEMETRY_CLEAN_PATH}/silver"
FINETUNED_MODEL = f"{BASE_PATH}/models/mae_finetuned.pt"
MAE_PRED_PATH   = f"{PREDICTIONS_PATH}/season={SEASON}/event={EVENT}/model=mae"

# ── DEVICE ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Predicting: Season {SEASON} | Event: {EVENT} | Round: {ROUND_NUMBER}")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

if not os.path.exists(FINETUNED_MODEL):
    print(
        "\n" + "=" * 65 + "\n"
        "  MAE MODEL NOT YET TRAINED\n"
        "=" * 65 + "\n"
        f"  Fine-tuned model not found at:\n    {FINETUNED_MODEL}\n\n"
        "  The MAE pipeline requires ~20–45 hours of compute on CE\n"
        "  before it can generate predictions.\n\n"
        "  ► FOR TODAY'S RACE: run 06_predict.py (RF/GBT baseline)\n"
        "    It uses the existing Gold features and produces the same\n"
        "    predictions.json output for the dashboard.\n\n"
        "  ► TO ENABLE MAE FOR FUTURE RACES, run in order on CE:\n"
        "    1. 07_tel_ingest.py     (~6–10 hrs, restartable)\n"
        "    2. 08_tel_preprocess.py (~1–2 hrs)\n"
        "    3. 09_mae_pretrain.py   (~15–38 hrs, checkpointed per epoch)\n"
        "    4. 10_mae_finetune.py   (~2–4 hrs)\n"
        "    Then re-run this notebook for the next race weekend.\n"
        + "=" * 65
    )
    # Exit cleanly — do not crash the Databricks job
    import sys
    sys.exit(0)


# Reconstruct model architecture then load weights
# (ENCODER_HPARAMS must match those used in 09/10 — stored in config.py)
mae_backbone = F1MAE(**ENCODER_HPARAMS)

model = F1PositionHead(
    encoder   = mae_backbone.encoder,
    d_model   = ENCODER_HPARAMS["d_model"],
    n_classes = 20,
).to(device)

model.load_state_dict(torch.load(FINETUNED_MODEL, map_location=device))
model.eval()

print(f"Model loaded from: {FINETUNED_MODEL}")

# ── LOAD SILVER FOR CURRENT RACE WEEKEND ──────────────────────────────────────
# We reuse 08_tel_preprocess.py's output. If it hasn't run yet for the current
# season, the user should run it first (same as the existing 01→06 cadence).

print(f"\nLoading Silver telemetry for {EVENT} {SEASON}...")

try:
    silver_sdf = (
        spark.read.schema(TEL_SILVER_SCHEMA).parquet(SILVER_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("event") == EVENT)
             )
    )
    row_count = silver_sdf.count()
    print(f"Silver rows found: {row_count:,}")
    silver_sdf.groupBy("session_type").count().orderBy("session_type").show()
except Exception as e:
    raise RuntimeError(
        f"No Silver telemetry found for {EVENT} {SEASON}.\n"
        f"Run 07_tel_ingest.py → 08_tel_preprocess.py first.\n"
        f"Error: {e}"
    )

if row_count == 0:
    raise ValueError(f"Silver telemetry is empty for {EVENT} {SEASON}.")

# ── COLLECT SILVER TO DRIVER TENSORS ─────────────────────────────────────────
# We aggregate to one prediction per driver (same as RF/LR pipeline convention).
# Strategy: take the lap with the lowest lap_time_delta proxy — the driver's
# representative best-pace lap — and predict position from that lap's telemetry.
# This mirrors how the existing Gold feature store picks best-lap metrics.

silver_pdf = silver_sdf.toPandas()
silver_pdf["channels"] = silver_pdf["channels"].apply(
    lambda ch: np.array(ch, dtype=np.float32)  # (6, 1024)
)

# Group by driver, pick best lap (min lap_number as proxy for quicklap ordering)
driver_groups = silver_pdf.groupby("driver")

driver_tensors = {}   # {driver: (6, 1024) tensor}
for driver, group in driver_groups:
    # Sort by lap_number ascending — pick the lap FastF1 returned first
    # (pick_quicklaps already sorted by pace quality in 07_tel_ingest.py)
    best_row = group.sort_values("lap_number").iloc[0]
    arr = best_row["channels"]
    if arr.shape == (6, 1024):
        driver_tensors[driver] = torch.from_numpy(arr)

print(f"\nDrivers with valid telemetry: {len(driver_tensors)}")

# ── RUN INFERENCE ─────────────────────────────────────────────────────────────

results = []

with torch.no_grad():
    for driver, tensor in driver_tensors.items():
        x = tensor.unsqueeze(0).to(device)            # (1, 6, 1024)
        logits = model(x)                             # (1, 20)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()   # (20,)

        # Predicted position: argmax of softmax → 0-indexed class → 1-indexed position
        pred_class    = int(np.argmax(probs))
        pred_position = pred_class + 1

        # Win probability: prob of class 0 (position 1)
        win_prob = float(probs[0])

        # Confidence: entropy-based — low entropy = model is more certain
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        # Normalise entropy to [0, 1] where 0 = perfectly certain
        max_entropy = np.log(20)
        uncertainty = float(entropy / max_entropy)

        results.append({
            "driver":             driver,
            "predicted_position": pred_position,
            "win_probability":    win_prob,
            "uncertainty":        uncertainty,
            "raw_probs":          probs.tolist(),
        })

# Sort by win probability descending — rerank positions accordingly
results.sort(key=lambda r: r["win_probability"], reverse=True)
for rank, r in enumerate(results, start=1):
    r["predicted_position"] = rank   # rerank to avoid ties

print("\nMAE Predictions:")
print(f"  {'Driver':<6} {'Pred Pos':>8} {'Win Prob':>10} {'Uncertainty':>13}")
print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*13}")
for r in results:
    print(
        f"  {r['driver']:<6} "
        f"{r['predicted_position']:>8} "
        f"{r['win_probability']:>10.4f} "
        f"{r['uncertainty']:>13.4f}"
    )

# ── WRITE PREDICTIONS PARQUET ─────────────────────────────────────────────────

output_schema = """
    driver string, team string, event string, round int, season int,
    model_version string, predicted_position int,
    win_probability float, uncertainty float
"""

pred_sdf = spark.createDataFrame(
    [
        (
            r["driver"],
            "MAE",       # no team info at telemetry level — placeholder
            EVENT,
            ROUND_NUMBER,
            SEASON,
            "mae",
            r["predicted_position"],
            r["win_probability"],
            r["uncertainty"],
        )
        for r in results
    ],
    schema=output_schema.strip(),
)

(
    pred_sdf.write
    .format("delta")
    .mode("overwrite")
    .option(
        "replaceWhere",
        f"season = {SEASON} AND event = '{EVENT}' AND model_version = 'mae'"
    )
    .save(PREDICTIONS_PATH)
)

print(f"\nMAE predictions written to: {PREDICTIONS_PATH} (model_version='mae')")

# ── COMPARISON WITH EXISTING RF/LR PIPELINE ───────────────────────────────────
# Load the RF predictions for the same event and compare top-3 accuracy if
# race results are available (base_r{N} model exists post-race).

print(f"\n── Comparison: MAE vs RF/LR (if post-race results available) ──────────")

try:
    rf_pred_path = f"{PREDICTIONS_PATH}"
    rf_preds_sdf = (
        spark.read.format("delta").load(PREDICTIONS_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("event") == EVENT)
                 & (F.col("model_version").startswith("base_r"))
             )
             .select("driver", "predicted_position", "win_probability")
             .withColumnRenamed("predicted_position", "rf_pred_pos")
             .withColumnRenamed("win_probability",    "rf_win_prob")
    )

    if rf_preds_sdf.count() == 0:
        print("  No post-race RF predictions found (race may not have completed yet).")
    else:
        # Load actual results from Gold
        from config import FEATURES_PATH
        actuals_sdf = (
            spark.read.format("delta").load(FEATURES_PATH)
                 .filter(
                     (F.col("season") == SEASON)
                     & (F.col("event") == EVENT)
                     & (F.col("session_type") == "R")
                 )
                 .select("driver", "race_position")
                 .distinct()
        )

        if actuals_sdf.count() == 0:
            print("  Actual race results not available yet — run pipeline after the race.")
        else:
            # Build comparison table
            mae_spark = spark.createDataFrame(
                [(r["driver"], r["predicted_position"]) for r in results],
                ["driver", "mae_pred_pos"]
            )

            comparison = (
                actuals_sdf
                .join(mae_spark,   on="driver", how="left")
                .join(rf_preds_sdf, on="driver", how="left")
            )

            total = comparison.count()

            mae_exact = comparison.filter(F.col("mae_pred_pos") == F.col("race_position")).count()
            rf_exact  = comparison.filter(F.col("rf_pred_pos")  == F.col("race_position")).count()

            mae_top3  = comparison.filter(
                F.abs(F.col("mae_pred_pos") - F.col("race_position")) <= 2
            ).count()
            rf_top3   = comparison.filter(
                F.abs(F.col("rf_pred_pos") - F.col("race_position")) <= 2
            ).count()

            print(f"\n  {'Metric':<20} {'MAE Model':>12} {'RF Model':>12}")
            print(f"  {'─'*20} {'─'*12} {'─'*12}")
            print(f"  {'Exact accuracy':<20} {mae_exact/total:>12.3f} {rf_exact/total:>12.3f}")
            print(f"  {'Top-3 accuracy':<20} {mae_top3/total:>12.3f} {rf_top3/total:>12.3f}")
            print(f"  {'(drivers evaluated)':<20} {total:>12}")

            print("\n  Per-driver comparison:")
            comparison.select(
                "driver", "race_position", "mae_pred_pos", "rf_pred_pos"
            ).orderBy("race_position").show(25)

except Exception as e:
    print(f"  Comparison skipped — RF predictions not available: {e}")

# ── HISTORICAL ACCURACY (MAE, season so far) ──────────────────────────────────
# Reads from the shared PREDICTIONS_PATH Delta table filtering model_version='mae'
# and post-race Gold results. Same logic as 06_predict.py.

print(f"\n── Historical MAE accuracy (season so far) ──────────────────────────────")

try:
    from config import FEATURES_PATH

    history_df = (
        spark.read.format("delta").load(PREDICTIONS_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("model_version") == "mae")
                 & (F.col("predicted_position") == 1)
             )
             .select("event", "driver", "round")
             .withColumnRenamed("driver", "predicted")
    )

    actuals_df = (
        spark.read.format("delta").load(FEATURES_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("session_type") == "R")
                 & (F.col("race_position") == 1)
             )
             .select("event", "driver")
             .distinct()
             .withColumnRenamed("driver", "actual")
    )

    history_rows = (
        history_df
        .join(actuals_df, on="event", how="inner")
        .withColumn("top3_hit", F.col("predicted") == F.col("actual"))
        .orderBy("round")
        .select("event", "predicted", "actual", "top3_hit")
        .collect()
    )

    history = [
        {
            "event":     row["event"],
            "predicted": row["predicted"],
            "actual":    row["actual"],
            "top3_hit":  bool(row["top3_hit"]),
        }
        for row in history_rows
    ]

    season_hits     = sum(1 for h in history if h["top3_hit"])
    season_accuracy = round(season_hits / len(history), 4) if history else 0.0
    print(f"  MAE winner accuracy this season: {season_hits}/{len(history)} ({season_accuracy:.1%})")

except Exception as e:
    print(f"  Historical accuracy skipped: {e}")
    history         = []
    season_accuracy = 0.0

# ── EXPORT predictions.json FOR DASHBOARD ─────────────────────────────────────
# Schema-compatible with 06_predict.py so the dashboard works unchanged.
# This is now the primary predictions.json — 06_predict.py is a comparison baseline.

payload = {
    "model_version":   "mae",
    "generated_at":    datetime.now(timezone.utc).isoformat(),
    "event":           EVENT,
    "round":           ROUND_NUMBER,
    "season":          SEASON,
    "sessions_used":   list(
        {row["session_type"] for row in silver_pdf["session_type"].items()}
        if "session_type" in silver_pdf.columns
        else ["R"]
    ),
    "season_accuracy": {
        "top3_pct": season_accuracy,
        "races":    len(history),
    },
    "recency_lambda": None,   # not applicable for MAE
    "predictions": [
        {
            "driver":             r["driver"],
            "team":               None,     # not available from telemetry alone
            "predicted_position": r["predicted_position"],
            "win_probability":    round(r["win_probability"], 4),
            "uncertainty":        round(r["uncertainty"], 4),
            "trend":              {"label": "flat", "value": None},  # pace trend N/A for MAE
            "sessions":           {},
        }
        for r in results
    ],
    "feature_importance": [],  # not applicable for MAE
    "history":            history,
}

dashboard_json_path = BASE_PATH + "/dashboard/public/predictions.json"
try:
    dbutils.fs.put(dashboard_json_path, json.dumps(payload, indent=2), overwrite=True)
    print(f"\npredictions.json written to: {dashboard_json_path}")
    print(f"Next step: git add dashboard/public/predictions.json && git push")
    print(f"Vercel will redeploy automatically within ~30 seconds.")
except Exception as e:
    print(f"\nCould not write predictions.json (dbutils not available outside CE): {e}")

print(f"\nDone. MAE predictions for {EVENT} {SEASON} | Drivers: {len(results)}")

