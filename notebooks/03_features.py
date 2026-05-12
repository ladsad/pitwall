import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.path.exists('/Workspace/Repos/pitwall') and '/Workspace/Repos/pitwall' not in sys.path:
    sys.path.insert(0, '/Workspace/Repos/pitwall')

from pyspark.sql import functions as F
from pyspark.sql import Window

from utils.spark_session import get_spark_session
from utils.schema import GOLD_SCHEMA, RESULTS_SCHEMA
from config import SEASON, EVENT, ROUND_NUMBER, CLEAN_PATH, FEATURES_PATH, RESULTS_PATH

spark = get_spark_session("pitwall-features")

print(f"Feature engineering: Season {SEASON} | Event: {EVENT} | Round: {ROUND_NUMBER}")
print(f"Input  (Silver): {CLEAN_PATH}")
print(f"Output (Gold)  : {FEATURES_PATH}")

#  READ SILVER 

silver_df = (
    spark.read.format("delta").load(CLEAN_PATH)
         .filter((F.col("season") == SEASON) & (F.col("event") == EVENT))
)

print(f"\nSilver rows loaded: {silver_df.count():,}")
silver_df.groupBy("session_type").count().orderBy("session_type").show()

#  2. RENAME & TAG 
# Rename Bronze column names (PascalCase from FastF1) to Gold snake_case.
# Tag round_number from config — not in Silver, needed for recency weighting.

base_df = (
    silver_df
    .withColumnRenamed("Driver",    "driver")
    .withColumnRenamed("Team",      "team")
    .withColumnRenamed("LapNumber", "lap_number")
    .withColumn("round_number", F.lit(ROUND_NUMBER))
)

#  LAP TIME DELTA 

driver_session_w = Window.partitionBy("season", "event", "session_type", "driver")

base_df = base_df.withColumn(
    "lap_time_delta",
    (F.col("LapTime") - F.min("LapTime").over(driver_session_w)).cast("float")
)

#  CONSISTENCY SCORE 

base_df = base_df.withColumn(
    "consistency_score",
    F.stddev("LapTime").over(driver_session_w).cast("float")
)

#  BEST SECTOR COMBO 

base_df = base_df.withColumn(
    "best_sector_combo",
    (
        F.min("Sector1Time").over(driver_session_w)
        + F.min("Sector2Time").over(driver_session_w)
        + F.min("Sector3Time").over(driver_session_w)
    ).cast("float")
)

#  TYRE DEG RATE 

compound_w = Window.partitionBy("season", "event", "session_type", "driver", "Compound")

base_df = base_df.withColumn(
    "tyre_deg_rate",
    F.when(
        (F.max("TyreLife").over(compound_w) - F.min("TyreLife").over(compound_w)) > 0,
        (F.max("LapTime").over(compound_w) - F.min("LapTime").over(compound_w))
        / (F.max("TyreLife").over(compound_w) - F.min("TyreLife").over(compound_w))
    ).otherwise(F.lit(0.0)).cast("float")
)

#  7. PACE VS TEAMMATE 

team_session_w = Window.partitionBy("season", "event", "session_type", "team")

base_df = base_df.withColumn(
    "pace_vs_teammate",
    (
        F.min("LapTime").over(driver_session_w)
        - F.min("LapTime").over(team_session_w)
    ).cast("float")
)

#  PACE TREND (CROSS-ROUND) 

try:
    prior_gold_df = (
        spark.read.format("delta").load(FEATURES_PATH)
             .filter(
                 (F.col("season") == SEASON)
                 & (F.col("round_number") < ROUND_NUMBER)
                 & (F.col("session_type") == "R")
             )
             .select("driver", "season", "round_number", "lap_time_delta")
    )

    prior_count = prior_gold_df.count()
    print(f"\nPrior Gold race rows loaded for pace_trend: {prior_count:,}")

    if prior_count > 0:
        # Avg lap_time_delta per driver per round (race pace proxy)
        round_avg = (
            prior_gold_df
            .groupBy("driver", "season", "round_number")
            .agg(F.avg("lap_time_delta").alias("avg_delta"))
        )

        # For each driver compute:
        #   recent_avg  = mean of avg_delta over last 2 rounds (rounds N-1, N-2)
        #   baseline_avg = mean of avg_delta over 4 rounds before that (N-3 to N-6)
        # Then pace_trend = recent_avg - baseline_avg
        # Negative = driver is closer to session best recently = improving

        recent_rounds   = ROUND_NUMBER - 1  # most recent completed round
        recent_cutoff   = ROUND_NUMBER - 2  # 2 rounds back
        baseline_cutoff = ROUND_NUMBER - 6  # 4 rounds before that

        recent_avg = (
            round_avg
            .filter(
                (F.col("round_number") <= recent_rounds)
                & (F.col("round_number") >= recent_cutoff)
            )
            .groupBy("driver", "season")
            .agg(F.avg("avg_delta").alias("recent_avg"))
        )

        baseline_avg = (
            round_avg
            .filter(
                (F.col("round_number") < recent_cutoff)
                & (F.col("round_number") >= baseline_cutoff)
            )
            .groupBy("driver", "season")
            .agg(F.avg("avg_delta").alias("baseline_avg"))
        )

        pace_trend_df = (
            recent_avg
            .join(baseline_avg, on=["driver", "season"], how="left")
            .withColumn(
                "pace_trend",
                (F.col("recent_avg") - F.col("baseline_avg")).cast("float")
            )
            .select("driver", "season", "pace_trend")
        )

        # Join pace_trend onto base_df
        base_df = base_df.join(pace_trend_df, on=["driver", "season"], how="left")
        print("pace_trend joined successfully.")

    else:
        print("No prior Gold data found — pace_trend will be null (expected on first run).")
        base_df = base_df.withColumn("pace_trend", F.lit(None).cast("float"))

except Exception as e:
    # Gold table doesn't exist yet on the very first run
    print(f"Gold table not found — pace_trend will be null. ({e})")
    base_df = base_df.withColumn("pace_trend", F.lit(None).cast("float"))

#  JOIN RACE RESULTS (LABEL) 

try:
    results_df = (
        spark.read
             .schema(RESULTS_SCHEMA)
             .parquet(f"{RESULTS_PATH}/season={SEASON}/event={EVENT}")
             .select(
                 F.col("Driver").alias("driver"),
                 F.col("Position").alias("race_position")
             )
    )

    base_df = base_df.join(results_df, on="driver", how="left")
    print(f"\nRace results joined — {results_df.count()} driver positions available.")

except Exception as e:
    print(f"Results not found — race_position will be null (expected mid-weekend). ({e})")
    base_df = base_df.withColumn("race_position", F.lit(None).cast("integer"))

#  SELECT & CAST TO GOLD SCHEMA 

gold_df = base_df.select(
    F.col("driver").cast("string"),
    F.col("team").cast("string"),
    F.col("round_number").cast("integer"),
    F.col("event").cast("string"),
    F.col("season").cast("integer"),
    F.col("session_type").cast("string"),
    F.col("lap_number").cast("integer"),
    F.col("Compound").alias("compound").cast("string"),
    F.col("lap_time_delta").cast("float"),
    F.col("consistency_score").cast("float"),
    F.col("best_sector_combo").cast("float"),
    F.col("tyre_deg_rate").cast("float"),
    F.col("pace_vs_teammate").cast("float"),
    F.col("pace_trend").cast("float"),
    F.col("race_position").cast("integer"),
    F.lit(None).cast("float").alias("recency_weight"),
    F.lit(None).cast("float").alias("session_weight"),
    F.lit(None).cast("float").alias("sample_weight"),
)

print(f"\nGold rows to write: {gold_df.count():,}")

#  WRITE GOLD DELTA

(
    gold_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .option("replaceWhere", f"season = {SEASON} AND event = '{EVENT}'")
    .partitionBy("season", "event", "session_type")
    .save(FEATURES_PATH)
)

print(f"Gold Delta written to: {FEATURES_PATH}")

#  VERIFICATION 

verify_df = (
    spark.read.format("delta").load(FEATURES_PATH)
         .filter((F.col("season") == SEASON) & (F.col("event") == EVENT))
)

print("\nRow counts by session_type (Gold — this event):")
verify_df.groupBy("session_type").count().orderBy("session_type").show()

print("Null rates for engineered features:")
feature_cols = [
    "lap_time_delta", "consistency_score", "best_sector_combo",
    "tyre_deg_rate", "pace_vs_teammate", "pace_trend", "race_position"
]
total = verify_df.count()
verify_df.select([
    F.round(F.count(F.when(F.col(c).isNull(), c)) / total * 100, 1).alias(c)
    for c in feature_cols
]).show()

print("Sample Gold rows (Race session):")
verify_df.filter(F.col("session_type") == "R") \
         .select("driver", "team", "lap_number", "lap_time_delta",
                 "consistency_score", "pace_vs_teammate", "race_position") \
         .orderBy("driver", "lap_number") \
         .show(20)