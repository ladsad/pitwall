import os
import pathlib
import sys
import json
from datetime import datetime, timezone

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

from utils.spark_session import get_spark_session
from config import (
    SEASON, EVENT, ROUND_NUMBER,
    FEATURES_PATH, MODELS_PATH, PREDICTIONS_PATH,
)

spark = get_spark_session("pitwall-predict")

#  LOAD MODEL 

qualifying_path = f"{MODELS_PATH}/qualifying_r{ROUND_NUMBER:02d}"
base_path       = f"{MODELS_PATH}/base_r{ROUND_NUMBER:02d}"

def path_exists(path: str) -> bool:
    try:
        return len(dbutils.fs.ls(path)) > 0
    except Exception:
        return os.path.exists(path)

if path_exists(qualifying_path):
    model_path    = qualifying_path
    model_version = f"qualifying_r{ROUND_NUMBER:02d}"
elif path_exists(base_path):
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

#  LOAD FEATURES FOR CURRENT WEEKEND 

predict_df = (
    spark.read.format("delta").load(FEATURES_PATH)
         .filter(
             (F.col("season") == SEASON)
             & (F.col("event") == EVENT)
         )
)

row_count = predict_df.count()
print(f"\nFeature rows for {EVENT}: {row_count:,}")
predict_df.groupBy("session_type").count().orderBy("session_type").show()

if row_count == 0:
    raise ValueError(f"No Gold features found for {EVENT}. Run 03_features.py first.")

#  APPLY SAME NaN / NULL HANDLING AS TRAINING 

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

# race_position may be null mid-weekend — StringIndexer handleInvalid="keep" covers this
# but we need a placeholder so the label stage doesn't error on a completely missing column.
predict_df = predict_df.fillna(-1, subset=["race_position"])

#  RUN MODEL 

predictions_raw = model.transform(predict_df)

#  MAP LABEL INDEX → FINISHING POSITION 

label_indexer_model = model.stages[1]
labels = label_indexer_model.labels   # e.g. ["1", "2", "3", ...]

# Build index → position mapping
index_to_position = {float(i): int(labels[i]) for i in range(len(labels))}

# Find which probability vector slot corresponds to position 1 (the win)
win_index = None
for idx, pos in index_to_position.items():
    if pos == 1:
        win_index = int(idx)
        break

if win_index is None:
    # Position 1 not seen in training data — use index 0 as fallback with a warning
    print("WARNING: Position 1 not found in label index. Using index 0 as win proxy.")
    win_index = 0

print(f"\nLabel index mapping (first 5): { {k: v for k, v in list(index_to_position.items())[:5]} }")
print(f"Win probability vector slot  : index {win_index}")

#  EXTRACT WIN PROBABILITY PER DRIVER 

extract_win_prob = F.udf(lambda v: float(v[win_index]))
extract_pred_pos = F.udf(lambda idx: index_to_position.get(float(idx), -1))

predictions_raw = (
    predictions_raw
    .withColumn("win_prob_lap",   extract_win_prob(F.col("probability")).cast("float"))
    .withColumn("pred_pos_lap",   extract_pred_pos(F.col("prediction")).cast("integer"))
)

# Aggregate to one row per driver — max win probability across all laps/sessions
driver_preds = (
    predictions_raw
    .groupBy("driver", "team")
    .agg(
        F.max("win_prob_lap").alias("win_probability"),
        F.first("pred_pos_lap").alias("predicted_position"),   # most common predicted pos
        F.count("*").alias("lap_count"),
    )
    .orderBy(F.desc("win_probability"))
)

print("\nDriver win probabilities (pre-normalisation):")
driver_preds.select("driver", "team", "win_probability", "predicted_position").show(25)

#  NORMALISE WIN PROBABILITIES 

total_prob = driver_preds.agg(F.sum("win_probability")).collect()[0][0]
driver_preds = driver_preds.withColumn(
    "win_probability",
    (F.col("win_probability") / F.lit(total_prob)).cast("float")
)

#  BOOTSTRAP UNCERTAINTY 

N_BOOTSTRAP = 20
RESAMPLE_FRACTION = 0.8

print(f"\nBootstrap uncertainty estimation (N={N_BOOTSTRAP} resamples)...")

driver_list = [row["driver"] for row in driver_preds.select("driver").collect()]
bootstrap_probs = {d: [] for d in driver_list}

for i in range(N_BOOTSTRAP):
    sample = predictions_raw.sample(fraction=RESAMPLE_FRACTION, seed=i)
    if sample.count() == 0:
        continue

    sample_agg = (
        sample
        .groupBy("driver")
        .agg(F.max("win_prob_lap").alias("win_prob"))
    )

    sample_total = sample_agg.agg(F.sum("win_prob")).collect()[0][0]
    if not sample_total or sample_total == 0:
        continue

    sample_norm = sample_agg.withColumn(
        "win_prob_norm", (F.col("win_prob") / F.lit(sample_total)).cast("float")
    )

    for row in sample_norm.collect():
        if row["driver"] in bootstrap_probs:
            bootstrap_probs[row["driver"]].append(row["win_prob_norm"])

# Compute std dev across bootstrap samples per driver
import statistics
uncertainty_map = {}
for driver, probs in bootstrap_probs.items():
    if len(probs) >= 2:
        uncertainty_map[driver] = round(statistics.stdev(probs), 4)
    else:
        uncertainty_map[driver] = 0.0

print("Bootstrap complete.")

#  ASSEMBLE FINAL PREDICTIONS 

uncertainty_rows = [(d, u) for d, u in uncertainty_map.items()]
uncertainty_df = spark.createDataFrame(uncertainty_rows, ["driver", "uncertainty"])

final_preds = (
    driver_preds
    .join(uncertainty_df, on="driver", how="left")
    .fillna(0.0, subset=["uncertainty"])
    .orderBy(F.desc("win_probability"))
    .withColumn("predicted_position", F.row_number().over(
        __import__("pyspark.sql", fromlist=["Window"]).Window.orderBy(F.desc("win_probability"))
    ))
)

print("\nFinal predictions:")
final_preds.select(
    "driver", "team", "win_probability", "uncertainty", "predicted_position"
).show(25)

#  WRITE PREDICTIONS DELTA 

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
    F.lit(datetime.now(timezone.utc).isoformat()).cast("string").alias("generated_at"),
)

(
    output_df.write
    .format("delta")
    .mode("overwrite")
    .option("replaceWhere", f"season = {SEASON} AND event = '{EVENT}'")
    .save(PREDICTIONS_PATH)
)

print(f"\nPredictions Delta written to: {PREDICTIONS_PATH}")

#  EXPORT predictions.json FOR DASHBOARD 

rows = output_df.orderBy(F.desc("win_probability")).collect()

payload = {
    "model_version": model_version,
    "generated_at":  datetime.now(timezone.utc).isoformat(),
    "event":         EVENT,
    "round":         ROUND_NUMBER,
    "season":        SEASON,
    "predictions": [
        {
            "driver":             row["driver"],
            "team":               row["team"],
            "predicted_position": row["predicted_position"],
            "win_probability":    round(float(row["win_probability"]), 4),
            "uncertainty":        round(float(row["uncertainty"]), 4),
        }
        for row in rows
    ],
}

# Write locally in the repo so git push triggers Vercel redeploy
dashboard_json_path = PROJECT_ROOT / "dashboard" / "public" / "predictions.json"
dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)

with open(dashboard_json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"predictions.json written to: {dashboard_json_path}")
print(f"\nNext step: git add dashboard/public/predictions.json && git push")
print(f"Vercel will redeploy automatically within ~30 seconds.")
print(f"\nDone. Model: {model_version} | Drivers predicted: {len(rows)}")