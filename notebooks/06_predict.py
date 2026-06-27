"""
06_predict.py — RF/GBT Prediction (Baseline)
──────────────────────────────────────────────
Generates race predictions using the trained RF/GBT ensemble on Gold tabular
features. Writes predictions to Parquet + Supabase + dashboard JSON.

  - When MAE model (mae_finetuned.pt) is trained: use 11_mae_predict.py instead.
  - This notebook remains as the RF/GBT comparison baseline.

Run order:
  01_ingest.py → 02_clean.py → 03_features.py → 05_train.py → 06_predict.py
"""

import os
import pathlib
import sys
from datetime import datetime, timezone

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

from utils.spark_session import get_spark_session
from utils.predict_utils import (
    session_scores,
    trend_map,
    bootstrap_uncertainty,
    season_history,
    build_payload,
    write_dashboard_json,
)
from utils.db import upsert_predictions
from config import (
    SEASON, EVENT, ROUND_NUMBER,
    BASE_PATH, FEATURES_PATH, MODELS_PATH, PREDICTIONS_PATH,
    DASHBOARD_JSON_PATH,
    SESSION_TYPES, SESSION_WEIGHTS,
    RECENCY_LAMBDA,
)

spark = get_spark_session("pitwall-predict")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

qualifying_path = str(MODELS_PATH / f"qualifying_r{ROUND_NUMBER:02d}")
base_path       = str(MODELS_PATH / f"base_r{ROUND_NUMBER:02d}")

if os.path.exists(qualifying_path):
    model_path    = qualifying_path
    model_version = f"qualifying_r{ROUND_NUMBER:02d}"
elif os.path.exists(base_path):
    model_path    = base_path
    model_version = f"base_r{ROUND_NUMBER:02d}"
else:
    raise FileNotFoundError(
        f"No model found for round {ROUND_NUMBER}. "
        f"Run 05_train.py first.\nLooked at:\n  {qualifying_path}\n  {base_path}"
    )

print(f"Predicting: Season {SEASON} | Event: {EVENT} | Round: {ROUND_NUMBER}")
print(f"Model     : {model_version} ({model_path})")

model = PipelineModel.load(model_path)

# ── LOAD FEATURES FOR CURRENT WEEKEND ────────────────────────────────────────

predict_df = spark.read.parquet(
    str(FEATURES_PATH / f"season={SEASON}" / f"event={EVENT}")
)

# Prevent data leakage: exclude Race sessions. The model must predict using only pre-race data.
predict_df = predict_df.filter(F.col("session_type") != "R")

row_count = predict_df.count()
print(f"\nFeature rows for {EVENT}: {row_count:,}")
predict_df.groupBy("session_type").count().orderBy("session_type").show()

if row_count == 0:
    raise ValueError(f"No Gold features found for {EVENT}. Run 03_features.py first.")

# ── NaN / NULL HANDLING (matches training) ────────────────────────────────────

NUMERIC_COLS = [
    "lap_time_delta",
    "consistency_score",
    "best_sector_combo",
    "tyre_deg_rate",
    "pace_vs_teammate",
    "pace_trend",
]

for _col in NUMERIC_COLS:
    predict_df = predict_df.withColumn(
        _col,
        F.when(F.isnan(F.col(_col)), F.lit(0.0)).otherwise(F.col(_col)),
    )
predict_df = predict_df.fillna(0.0, subset=NUMERIC_COLS)
predict_df = predict_df.fillna(-1, subset=["race_position"])

# ── RUN MODEL ─────────────────────────────────────────────────────────────────

predictions_raw = model.transform(predict_df)

# ── MAP LABEL INDEX → FINISHING POSITION ─────────────────────────────────────

label_indexer_model = model.stages[3]
labels = label_indexer_model.labels

index_to_position = {float(i): int(labels[i]) for i in range(len(labels))}

win_index = next(
    (int(idx) for idx, pos in index_to_position.items() if pos == 1),
    None,
)
if win_index is None:
    print("WARNING: Position 1 not found in label index. Using index 0 as win proxy.")
    win_index = 0

print(f"\nLabel index mapping (first 5): { {k: v for k, v in list(index_to_position.items())[:5]} }")
print(f"Win probability vector slot  : index {win_index}")

# ── EXTRACT WIN PROBABILITY PER DRIVER ───────────────────────────────────────

extract_win_prob = F.udf(lambda v: float(v[win_index]))
extract_pred_pos = F.udf(lambda idx: index_to_position.get(float(idx), -1))

predictions_raw = (
    predictions_raw
    .withColumn("win_prob_lap", extract_win_prob(F.col("probability")).cast("float"))
    .withColumn("pred_pos_lap", extract_pred_pos(F.col("prediction")).cast("integer"))
)

from pyspark.sql.types import FloatType
def get_sess_weight(st):
    return float(SESSION_WEIGHTS.get(st, 0.1))
sess_weight_udf = F.udf(get_sess_weight, FloatType())

predictions_raw = predictions_raw.withColumn("sess_weight", sess_weight_udf(F.col("session_type")))

driver_preds = (
    predictions_raw
    .groupBy("driver")
    .agg(
        F.max("team").alias("team"),
        (F.sum(F.col("win_prob_lap") * F.col("sess_weight")) / F.sum("sess_weight")).alias("win_probability"),
        F.first("pred_pos_lap").alias("predicted_position"),
        F.count("*").alias("lap_count"),
    )
    .orderBy(F.desc("win_probability"))
)

print("\nDriver win probabilities (pre-normalisation):")
driver_preds.select("driver", "team", "win_probability", "predicted_position").show(25)

# ── NORMALISE WIN PROBABILITIES ───────────────────────────────────────────────

total_prob = driver_preds.agg(F.sum("win_probability")).collect()[0][0]
driver_preds = driver_preds.withColumn(
    "win_probability",
    (F.col("win_probability") / F.lit(total_prob)).cast("float"),
)

# ── BOOTSTRAP UNCERTAINTY ─────────────────────────────────────────────────────

print(f"\nBootstrap uncertainty estimation (N=20 resamples)...")
driver_list    = [row["driver"] for row in driver_preds.select("driver").collect()]
uncertainty_map = bootstrap_uncertainty(predictions_raw, driver_list)
print("Bootstrap complete.")

# ── ASSEMBLE FINAL PREDICTIONS ────────────────────────────────────────────────

from pyspark.sql.window import Window

uncertainty_rows = [(d, u) for d, u in uncertainty_map.items()]
uncertainty_df   = spark.createDataFrame(uncertainty_rows, ["driver", "uncertainty"])

final_preds = (
    driver_preds
    .join(uncertainty_df, on="driver", how="left")
    .fillna(0.0, subset=["uncertainty"])
    .orderBy(F.desc("win_probability"))
    .withColumn(
        "predicted_position",
        F.row_number().over(Window.orderBy(F.desc("win_probability"))),
    )
)

print("\nFinal predictions:")
final_preds.select("driver", "team", "win_probability", "uncertainty", "predicted_position").show(25)

# ── SESSION SCORES & TREND ────────────────────────────────────────────────────

_non_race_sessions = [s for s in SESSION_TYPES if s != "R"]
sess_scores = session_scores(predictions_raw, _non_race_sessions, SESSION_WEIGHTS)
trend       = trend_map(predictions_raw)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "lap_time_delta",
    "consistency_score",
    "best_sector_combo",
    "tyre_deg_rate",
    "pace_vs_teammate",
    "pace_trend",
    "compound_idx",
    "team_idx",
    "driver_idx",
]

gbt_model   = model.stages[-1]
importances = gbt_model.featureImportances.toArray().tolist()

feature_importance = [
    {"feature": name, "importance": round(float(imp), 4)}
    for name, imp in sorted(
        zip(FEATURE_COLS, importances[:len(FEATURE_COLS)]),
        key=lambda x: x[1],
        reverse=True,
    )
]

# Compute driver-specific feature impacts
global_imp = {item["feature"]: item["importance"] for item in feature_importance}
driver_features = predictions_raw.groupBy("driver").agg(
    *[F.avg(c).alias(c + "_avg") for c in FEATURE_COLS]
).collect()

driver_feat_dict = {r["driver"]: {c: r[c + "_avg"] for c in FEATURE_COLS} for r in driver_features}
feature_stats = {}
for c in FEATURE_COLS:
    vals = [r[c + "_avg"] for r in driver_features if r[c + "_avg"] is not None]
    mean_val = sum(vals)/len(vals) if vals else 0
    std_val = (sum((v - mean_val)**2 for v in vals) / len(vals))**0.5 if len(vals) > 1 else 1.0
    feature_stats[c] = (mean_val, std_val)

driver_local_importance = {}
for d, feats in driver_feat_dict.items():
    local_imp = []
    for c in FEATURE_COLS:
        g_imp = global_imp[c]
        val = feats[c] if feats[c] is not None else feature_stats[c][0]
        mean_val, std_val = feature_stats[c]
        z_score = abs(val - mean_val) / (std_val + 1e-6)
        score = g_imp * z_score
        local_imp.append({"feature": c, "importance": score})
    
    # Normalize to 1.0 or keep raw impacts? Let's normalize so it sums to 1 like global
    total = sum(item["importance"] for item in local_imp) + 1e-6
    local_imp = [{"feature": item["feature"], "importance": round(item["importance"]/total, 4)} for item in local_imp]
    local_imp.sort(key=lambda x: x["importance"], reverse=True)
    driver_local_importance[d] = local_imp


# ── HISTORICAL ACCURACY ───────────────────────────────────────────────────────

history, season_acc = season_history(
    spark, PREDICTIONS_PATH, FEATURES_PATH,
    season=SEASON,
    model_version_filter="base_r",
)

# ── WRITE PREDICTIONS PARQUET ─────────────────────────────────────────────────

import json
_sessions_udf = F.udf(lambda d: json.dumps(sess_scores.get(d, {})), "string")
_trend_udf    = F.udf(lambda d: json.dumps(trend.get(d, {"label": "flat", "value": None})), "string")
_feature_importance_udf = F.udf(lambda d: json.dumps(driver_local_importance.get(d, [])), "string")

output_df = final_preds.select(
    F.col("driver").cast("string"),
    F.col("team").cast("string"),
    F.lit(EVENT).cast("string").alias("event"),
    F.lit(ROUND_NUMBER).cast("integer").alias("round"),
    F.lit(SEASON).cast("integer").alias("season"),
    F.lit(model_version).cast("string").alias("model_version"),
    F.col("predicted_position").cast("integer"),
    F.col("win_probability").cast("float"),
    F.col("uncertainty").cast("float"),
    _sessions_udf(F.col("driver")).alias("sessions"),
    _trend_udf(F.col("driver")).alias("trend"),
    _feature_importance_udf(F.col("driver")).alias("feature_importance"),
    F.lit(datetime.now(timezone.utc).isoformat()).cast("string").alias("generated_at"),
)

pred_output = str(PREDICTIONS_PATH / f"season={SEASON}" / f"event={EVENT}")
output_df.write.mode("overwrite").parquet(pred_output)
print(f"\nPredictions Parquet written to: {pred_output}")

# ── WRITE TO SUPABASE ─────────────────────────────────────────────────────────

try:
    rows_for_db = [row.asDict() for row in output_df.collect()]
    for r in rows_for_db:
        r["sessions"] = json.loads(r["sessions"])
        r["trend"]    = json.loads(r["trend"])
        r["feature_importance"] = json.loads(r["feature_importance"])
    upsert_predictions(rows_for_db)
    print("Predictions upserted to Supabase.")
except Exception as e:
    print(f"Supabase write skipped (set SUPABASE_URL/KEY to enable): {e}")

# ── EXPORT predictions.json ───────────────────────────────────────────────────

rows = final_preds.orderBy(F.desc("win_probability")).collect()
sessions_used = [
    row["session_type"]
    for row in predictions_raw.select("session_type").distinct().collect()
]

payload = build_payload(
    model_version=model_version,
    event=EVENT,
    round_number=ROUND_NUMBER,
    season=SEASON,
    sessions_used=sessions_used,
    season_accuracy=(season_acc, len(history)),
    recency_lambda=RECENCY_LAMBDA,
    predictions=[
        {
            "driver":             row["driver"],
            "team":               row["team"],
            "predicted_position": row["predicted_position"],
            "win_probability":    round(float(row["win_probability"]), 4),
            "uncertainty":        round(float(row["uncertainty"]), 4),
            "trend":              trend.get(row["driver"], {"label": "flat", "value": None}),
            "sessions":           sess_scores.get(row["driver"], {}),
            "feature_importance": driver_local_importance.get(row["driver"], []),
        }
        for row in rows
    ],
    feature_importance=feature_importance,
    history=history,
)

write_dashboard_json(payload, DASHBOARD_JSON_PATH)
print(f"\nDone. Model: {model_version} | Drivers predicted: {len(rows)}")