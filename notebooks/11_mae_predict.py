"""
11_mae_predict.py — PRIMARY PREDICTION NOTEBOOK
─────────────────────────────────────────────────
Runs the fine-tuned F1MAE encoder on the current race weekend's telemetry,
writes predictions to Parquet + Supabase, and exports predictions.json.

Prerequisites (per race weekend):
  01_ingest.py → 02_clean.py → 03_features.py   (Gold — for labels)
  07_tel_ingest.py → 08_tel_preprocess.py        (Silver telemetry)
  [09_mae_pretrain.py + 10_mae_finetune.py — one-time historical training]

06_predict.py remains available as RF/GBT comparison baseline.
"""

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
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.spark_session import get_spark_session
from utils.mae_model import F1MAE, F1PositionHead
from utils.tel_schema import TEL_SILVER_SCHEMA
from utils.predict_utils import (
    season_history,
    build_payload,
    write_dashboard_json,
)
from utils.db import upsert_predictions
from config import (
    SEASON, EVENT, ROUND_NUMBER,
    BASE_PATH, TELEMETRY_CLEAN_PATH, PREDICTIONS_PATH, FEATURES_PATH,
    DASHBOARD_JSON_PATH,
    ENCODER_HPARAMS,
)

spark = get_spark_session("pitwall-mae-predict")

SILVER_PATH     = str(TELEMETRY_CLEAN_PATH / "silver")
FINETUNED_MODEL = str(BASE_PATH / "models" / "mae_finetuned.pt")

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
        "  The MAE pipeline requires ~20–45 hours of compute\n"
        "  before it can generate predictions.\n\n"
        "  ► FOR TODAY'S RACE: run 06_predict.py (RF/GBT baseline)\n\n"
        "  ► TO ENABLE MAE FOR FUTURE RACES, run in order:\n"
        "    1. 07_tel_ingest.py     (~6–10 hrs, restartable)\n"
        "    2. 08_tel_preprocess.py (~1–2 hrs)\n"
        "    3. 09_mae_pretrain.py   (~15–38 hrs, checkpointed per epoch)\n"
        "    4. 10_mae_finetune.py   (~2–4 hrs)\n"
        "    Then re-run this notebook for the next race weekend.\n"
        + "=" * 65
    )
    sys.exit(0)

mae_backbone = F1MAE(**ENCODER_HPARAMS)
model = F1PositionHead(
    encoder   = mae_backbone.encoder,
    d_model   = ENCODER_HPARAMS["d_model"],
    n_classes = 30,
).to(device)

model.load_state_dict(torch.load(FINETUNED_MODEL, map_location=device))
model.eval()
print(f"Model loaded from: {FINETUNED_MODEL}")

# ── LOAD SILVER TELEMETRY ─────────────────────────────────────────────────────

print(f"\nLoading Silver telemetry for {EVENT} {SEASON}...")

try:
    silver_sdf = (
        spark.read.schema(TEL_SILVER_SCHEMA).parquet(SILVER_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("event") == EVENT)
                 & (F.col("session_type") != "R")
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

sessions_used = silver_sdf.select("session_type").distinct().rdd.flatMap(lambda r: r).collect()

# ── COLLECT DRIVER TENSORS ────────────────────────────────────────────────────

silver_pdf = silver_sdf.toPandas()
silver_pdf["channels"] = silver_pdf["channels"].apply(
    lambda ch: np.array(ch, dtype=np.float32)  # (6, 1024)
)

driver_tensors = {}
for driver, group in silver_pdf.groupby("driver"):
    # Enforce: Driver MUST have participated in Qualifying (or Sprint Qualifying)
    valid_sessions = group["session_type"].values
    if "Q" not in valid_sessions and "SQ" not in valid_sessions:
        continue
        
    # Use their qualifying lap for the prediction tensor
    q_laps = group[group["session_type"].isin(["Q", "SQ"])]
    best_row = q_laps.sort_values("lap_number").iloc[0]
    
    arr = best_row["channels"]
    if arr.shape == (6, 1024):
        driver_tensors[driver] = torch.from_numpy(arr)

print(f"\nDrivers with valid telemetry: {len(driver_tensors)}")

# Fetch Team mappings from Gold Features to ensure correct dashboard colors
driver_teams = {}
try:
    teams_sdf = spark.read.parquet(str(FEATURES_PATH)).filter(
        (F.col("season") == SEASON) & (F.col("event") == EVENT)
    ).select("driver", "team").distinct()
    for row in teams_sdf.collect():
        if row.driver and row.team:
            driver_teams[row.driver] = row.team
except Exception as e:
    print(f"Warning: Could not fetch team mappings: {e}")

# ── INFERENCE ─────────────────────────────────────────────────────────────────

channel_names = ["Speed", "Throttle", "Brake", "RPM", "Gear", "DRS"]
channel_impacts = {ch: [] for ch in channel_names}
results = []

with torch.no_grad():
    for driver, tensor in driver_tensors.items():
        x      = tensor.unsqueeze(0).to(device)                          # (1, 6, 1024)
        logits = model(x)                                                  # (1, 30)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()    # (30,)

        pred_position = int(np.argmax(probs)) + 1
        win_prob      = float(probs[0])
        entropy       = float(-np.sum(probs * np.log(probs + 1e-9)))
        uncertainty   = float(entropy / np.log(30))

        # Occlusion Sensitivity (Interpretability)
        for c_idx, c_name in enumerate(channel_names):
            x_occ = x.clone()
            x_occ[0, c_idx, :] = 0.0  # Zero out (since data is z-score normalized, 0 = mean)
            logits_occ = model(x_occ)
            probs_occ = torch.softmax(logits_occ, dim=1).squeeze(0).cpu().numpy()
            impact = abs(win_prob - float(probs_occ[0]))
            channel_impacts[c_name].append(impact)

        results.append({
            "driver":             driver,
            "team":               driver_teams.get(driver, "Unknown"),
            "predicted_position": pred_position,
            "win_probability":    win_prob,
            "uncertainty":        uncertainty,
        })

# Average and normalize channel impacts for the dashboard
feature_importance = []
total_impact = 0.0
avg_impacts = {}
for ch in channel_names:
    avg = float(np.mean(channel_impacts[ch])) if channel_impacts[ch] else 0.0
    avg_impacts[ch] = avg
    total_impact += avg

if total_impact > 0:
    for ch, avg in avg_impacts.items():
        feature_importance.append({
            "feature": f"{ch} Telemetry",
            "importance": float(avg / total_impact)
        })
feature_importance.sort(key=lambda x: x["importance"], reverse=True)

results.sort(key=lambda r: r["win_probability"], reverse=True)
for rank, r in enumerate(results, start=1):
    r["predicted_position"] = rank

print(f"\n  {'Driver':<6} {'Pred Pos':>8} {'Win Prob':>10} {'Uncertainty':>13}")
print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*13}")
for r in results:
    print(f"  {r['driver']:<6} {r['predicted_position']:>8} {r['win_probability']:>10.4f} {r['uncertainty']:>13.4f}")

# ── WRITE PREDICTIONS PARQUET ─────────────────────────────────────────────────

pred_sdf = spark.createDataFrame(
    [
        (
            r["driver"], r["team"], EVENT, ROUND_NUMBER, SEASON,
            "mae", r["predicted_position"],
            r["win_probability"], r["uncertainty"],
        )
        for r in results
    ],
    schema=(
        "driver string, team string, event string, round int, season int, "
        "model_version string, predicted_position int, "
        "win_probability float, uncertainty float"
    ),
)

pred_output = str(PREDICTIONS_PATH / f"season={SEASON}" / f"event={EVENT}" / "model=mae")
pred_sdf.write.mode("overwrite").parquet(pred_output)
print(f"\nMAE predictions written to: {pred_output}")

# ── WRITE TO SUPABASE ─────────────────────────────────────────────────────────

try:
    rows_for_db = [row.asDict() for row in pred_sdf.collect()]
    upsert_predictions(rows_for_db)
    print("Predictions upserted to Supabase.")
except Exception as e:
    print(f"Supabase write skipped (set SUPABASE_URL/KEY to enable): {e}")

# ── COMPARISON WITH RF/GBT ────────────────────────────────────────────────────

print("\n── Comparison: MAE vs RF/GBT (if post-race results available) ──────────")

try:
    rf_preds_sdf = (
        spark.read.parquet(str(PREDICTIONS_PATH))
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
        actuals_sdf = (
            spark.read.parquet(str(FEATURES_PATH))
                 .filter(
                     (F.col("season") == SEASON)
                     & (F.col("event") == EVENT)
                     & (F.col("session_type") == "R")
                 )
                 .select("driver", "race_position")
                 .distinct()
        )

        if actuals_sdf.count() == 0:
            print("  Actual race results not available yet.")
        else:
            mae_spark  = spark.createDataFrame(
                [(r["driver"], r["predicted_position"]) for r in results],
                ["driver", "mae_pred_pos"],
            )
            comparison = actuals_sdf.join(mae_spark, on="driver", how="left").join(rf_preds_sdf, on="driver", how="left")
            total      = comparison.count()

            mae_exact = comparison.filter(F.col("mae_pred_pos") == F.col("race_position")).count()
            rf_exact  = comparison.filter(F.col("rf_pred_pos")  == F.col("race_position")).count()
            mae_top3  = comparison.filter(F.abs(F.col("mae_pred_pos") - F.col("race_position")) <= 2).count()
            rf_top3   = comparison.filter(F.abs(F.col("rf_pred_pos")  - F.col("race_position")) <= 2).count()

            print(f"\n  {'Metric':<20} {'MAE Model':>12} {'RF Model':>12}")
            print(f"  {'─'*20} {'─'*12} {'─'*12}")
            print(f"  {'Exact accuracy':<20} {mae_exact/total:>12.3f} {rf_exact/total:>12.3f}")
            print(f"  {'Top-3 accuracy':<20} {mae_top3/total:>12.3f} {rf_top3/total:>12.3f}")
            print(f"  {'(drivers evaluated)':<20} {total:>12}")
            comparison.select("driver", "race_position", "mae_pred_pos", "rf_pred_pos").orderBy("race_position").show(25)

except Exception as e:
    print(f"  Comparison skipped: {e}")

# ── HISTORICAL ACCURACY ───────────────────────────────────────────────────────

try:
    history, season_acc = season_history(
        spark, PREDICTIONS_PATH, FEATURES_PATH,
        season=SEASON,
        model_version_filter="mae",
    )
    print(f"\nMAE winner accuracy this season: {sum(1 for h in history if h['top3_hit'])}/{len(history)} ({season_acc:.1%})")
except Exception as e:
    print(f"  Historical accuracy skipped: {e}")
    history, season_acc = [], 0.0

# ── EXPORT predictions.json ───────────────────────────────────────────────────

payload = build_payload(
    model_version="mae",
    event=EVENT,
    round_number=ROUND_NUMBER,
    season=SEASON,
    sessions_used=sessions_used,
    season_accuracy=(season_acc, len(history)),
    recency_lambda=None,
    predictions=[
        {
            "driver":             r["driver"],
            "team":               r["team"],
            "predicted_position": r["predicted_position"],
            "win_probability":    round(r["win_probability"], 4),
            "uncertainty":        round(r["uncertainty"], 4),
            "trend":              {"label": "flat", "value": None},
            "sessions":           {},
        }
        for r in results
    ],
    feature_importance=feature_importance,
    history=history,
)

write_dashboard_json(payload, DASHBOARD_JSON_PATH)
print(f"\nDone. MAE predictions for {EVENT} {SEASON} | Drivers: {len(results)}")
