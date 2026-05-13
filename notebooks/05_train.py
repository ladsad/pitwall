import os
import pathlib
import sys

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    # Running as a Databricks notebook — __file__ is not defined
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from utils.spark_session import get_spark_session
from utils.weights import add_sample_weight
from config import SEASON, EVENT, ROUND_NUMBER, FEATURES_PATH, MODELS_PATH

spark = get_spark_session("pitwall-train")

#  MODEL VERSION 

gold_df = spark.read.format("delta").load(FEATURES_PATH).filter(
    F.col("season") == SEASON
)

current_round_has_result = (
    gold_df
    .filter((F.col("round_number") == ROUND_NUMBER) & (F.col("race_position").isNotNull()))
    .limit(1)
    .count() > 0
)

model_version = f"base_r{ROUND_NUMBER:02d}" if current_round_has_result else f"qualifying_r{ROUND_NUMBER:02d}"
model_path    = f"{MODELS_PATH}/{model_version}"

print(f"Training : Season {SEASON} | Round: {ROUND_NUMBER} | Version: {model_version}")
print(f"Model output: {model_path}")

#  TRAINING DATA 

train_df = gold_df.filter(F.col("race_position").isNotNull())

row_count = train_df.count()
print(f"\nTraining rows (labelled): {row_count:,}")
train_df.groupBy("session_type").count().orderBy("session_type").show()

if row_count < 100:
    print(f"WARNING: Only {row_count} training rows — model accuracy will be low. "
          f"More rounds needed for reliable predictions.")

#  APPLY DUAL WEIGHTS 

train_df = add_sample_weight(train_df)

print("\nWeight sanity check — avg sample_weight by session_type and round:")
train_df.groupBy("round_number", "session_type").agg(
    F.round(F.avg("recency_weight"),  3).alias("avg_recency"),
    F.round(F.avg("session_weight"),  3).alias("avg_session"),
    F.round(F.avg("sample_weight"),   3).alias("avg_sample"),
).orderBy("round_number", "session_type").show(30)

#  FEATURE COLUMNS 

CATEGORICAL_COLS = ["compound"]
NUMERIC_COLS     = [
    "lap_time_delta",
    "consistency_score",
    "best_sector_combo",
    "tyre_deg_rate",
    "pace_vs_teammate",
    "pace_trend",
]

# fillna() only fixes SQL null — it does NOT replace IEEE NaN.
# consistency_score is NaN (not null) when a driver has a single lap in a session
# (stddev of one value is undefined). Replace NaN → 0.0 across all numeric cols first,
# then fill any remaining SQL nulls (e.g. pace_trend on first round).
for _col in NUMERIC_COLS:
    train_df = train_df.withColumn(
        _col,
        F.when(F.isnan(F.col(_col)), F.lit(0.0)).otherwise(F.col(_col)),
    )
train_df = train_df.fillna(0.0, subset=NUMERIC_COLS)

#  BUILD MLLLIB PIPELINE 

compound_indexer = StringIndexer(
    inputCol="compound",
    outputCol="compound_idx",
    handleInvalid="keep",   # unseen compounds at prediction time → extra index bucket
)

# race_position indexer: maps position integers (1, 2, 3...) to 0-based index.
# GBTClassifier requires the label column to be 0-indexed doubles.
label_indexer = StringIndexer(
    inputCol="race_position",
    outputCol="label",
    handleInvalid="keep",
)

assembler = VectorAssembler(
    inputCols=NUMERIC_COLS + ["compound_idx"],
    outputCol="features",
    handleInvalid="keep",   # rows with remaining nulls → zero imputation
)

# GBTClassifier only supports binary labels — use RandomForestClassifier for multiclass (20 positions).
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="sample_weight",
    numTrees=100,        # 100 trees — stable estimates at this data size
    maxDepth=4,          # shallow trees reduce overfitting on ~1300 rows
    seed=42,
)

pipeline = Pipeline(stages=[
    compound_indexer,
    label_indexer,
    assembler,
    rf,
])

#  TRAIN / VALIDATION SPLIT 

max_round = train_df.agg(F.max("round_number")).collect()[0][0]
holdout_df = train_df.filter(F.col("round_number") == max_round)
fit_df     = train_df.filter(F.col("round_number") <  max_round)

fit_count     = fit_df.count()
holdout_count = holdout_df.count()

print(f"\nTrain/validation split:")
print(f"  Fit set     : {fit_count:,} rows (rounds 1–{max_round - 1})")
print(f"  Holdout set : {holdout_count:,} rows (round {max_round})")

if fit_count < 50:
    print("WARNING: Fit set is very small — training on full dataset instead.")
    fit_df = train_df

#  FIT 

print("\nFitting GBT pipeline...")
model = pipeline.fit(fit_df)
print("Done.")

#  EVALUATE ON HOLDOUT 

if holdout_count > 0:
    holdout_preds = model.transform(holdout_df)

    # Exact position accuracy
    exact_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    exact_acc = exact_evaluator.evaluate(holdout_preds)

    # Top-3 accuracy: predicted position within 3 of actual
    # prediction column holds the indexed label — map back to position
    label_model  = model.stages[1]          # the label StringIndexer model
    index_to_pos = {float(i): float(label_model.labels[i])
                    for i in range(len(label_model.labels))}
    index_to_pos_udf = F.udf(lambda idx: index_to_pos.get(idx, -1.0))

    holdout_preds = (
        holdout_preds
        .withColumn("pred_position",   index_to_pos_udf(F.col("prediction")).cast("float"))
        .withColumn("actual_position", F.col("race_position").cast("float"))
    )

    top3_acc = holdout_preds.filter(
        F.abs(F.col("pred_position") - F.col("actual_position")) <= 2
    ).count() / holdout_count

    print(f"\nHoldout evaluation (round {max_round}):")
    print(f"  Exact accuracy : {exact_acc:.3f}")
    print(f"  Top-3 accuracy : {top3_acc:.3f}  (predicted position within ±2 of actual)")

    print("\nSample predictions vs actual (Race rows):")
    holdout_preds.filter(F.col("session_type") == "R") \
                 .select("driver", "actual_position", "pred_position", "sample_weight") \
                 .orderBy("actual_position") \
                 .show(20)

#  REFIT ON FULL TRAINING SET 

print("\nRefitting on full training set for final model...")
final_model = pipeline.fit(train_df)
print("Done.")

#  FEATURE IMPORTANCE 

rf_model      = final_model.stages[-1]
feature_names = NUMERIC_COLS + ["compound_idx"]
importances   = rf_model.featureImportances.toArray()

print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    print(f"  {name:<25} {imp:.4f}  {bar}")

#  SAVE MODEL 

final_model.write().overwrite().save(model_path)
print(f"\nModel saved to: {model_path}")
print(f"Version: {model_version}")

# BASELINE: LOGISTIC REGRESSION 

print("\nTraining Logistic Regression baseline...")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="sample_weight",
    maxIter=100,
    regParam=0.01,
)

lr_pipeline = Pipeline(stages=[
    compound_indexer,
    label_indexer,
    assembler,
    lr,
])

lr_model = lr_pipeline.fit(train_df)

if holdout_count > 0:
    lr_preds   = lr_model.transform(holdout_df)
    lr_acc     = exact_evaluator.evaluate(lr_preds)
    print(f"LR baseline exact accuracy: {lr_acc:.3f}  (GBT: {exact_acc:.3f})")
    if exact_acc > lr_acc:
        print("GBT outperforms baseline ✓")
    else:
        print("WARNING: GBT does not outperform LR baseline — review features/weights.")

lr_model_path = f"{MODELS_PATH}/lr_baseline_r{ROUND_NUMBER:02d}"
lr_model.write().overwrite().save(lr_model_path)
print(f"LR baseline saved to: {lr_model_path}")