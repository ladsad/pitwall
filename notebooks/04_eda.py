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
from config import SEASON, EVENT, ROUND_NUMBER, FEATURES_PATH

spark = get_spark_session("pitwall-eda")

print(f"EDA: Season {SEASON} | Event: {EVENT} | Round: {ROUND_NUMBER}")
print(f"Reading Gold from: {FEATURES_PATH}")

#  LOAD GOLD 

gold_df = (
    spark.read.format("delta").load(FEATURES_PATH)
         .filter(F.col("season") == SEASON)
)

total_rows = gold_df.count()
print(f"\nTotal Gold rows (season {SEASON}): {total_rows:,}")

gold_df.groupBy("round_number", "event", "session_type") \
       .count() \
       .orderBy("round_number", "session_type") \
       .show(50)

# Completed rounds = rounds where race_position is populated
completed_df = gold_df.filter(F.col("race_position").isNotNull())
print(f"Rows from completed rounds (race_position not null): {completed_df.count():,}")

#  FEATURE DISTRIBUTIONS 

print("\n--- Feature distributions (all sessions, completed rounds) ---")
feature_cols = [
    "lap_time_delta", "consistency_score", "best_sector_combo",
    "tyre_deg_rate", "pace_vs_teammate", "pace_trend"
]
completed_df.select(feature_cols).summary("count", "mean", "stddev", "min", "25%", "75%", "max").show()

#  FEATURE DISTRIBUTIONS BY SESSION TYPE 

print("\n--- Avg feature values by session_type ---")
gold_df.groupBy("session_type").agg(
    F.round(F.avg("lap_time_delta"),    3).alias("avg_lap_delta"),
    F.round(F.avg("consistency_score"), 3).alias("avg_consistency"),
    F.round(F.avg("pace_vs_teammate"),  3).alias("avg_vs_teammate"),
    F.round(F.avg("tyre_deg_rate"),     3).alias("avg_tyre_deg"),
    F.count("*").alias("row_count"),
).orderBy("session_type").show()

#  CORRELATION: FEATURES vs RACE POSITION 

print("\n--- Pearson correlation with race_position (Race session rows only) ---")
race_df = completed_df.filter(F.col("session_type") == "R")

for col in feature_cols:
    corr = race_df.stat.corr(col, "race_position")
    direction = "↑ worse position" if corr > 0 else "↓ better position"
    print(f"  {col:<25} r = {corr:+.4f}   ({direction})")

#  CORRELATION ACROSS ALL SESSION TYPES 

print("\n--- Correlation with race_position by session_type ---")
print(f"  {'session_type':<8}", end="")
for col in feature_cols:
    print(f"  {col[:12]:<14}", end="")
print()

for session_type in ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]:
    session_rows = completed_df.filter(F.col("session_type") == session_type)
    count = session_rows.count()
    if count < 10:
        continue   # skip session types with too few rows to be meaningful
    print(f"  {session_type:<8}", end="")
    for col in feature_cols:
        try:
            corr = session_rows.stat.corr(col, "race_position")
            print(f"  {corr:+.3f}        ", end="")
        except Exception:
            print(f"  {'N/A':<14}", end="")
    print(f"  (n={count})")

#  TOP PERFORMERS BY FEATURE 

print(f"\n--- Current event feature leaders ({EVENT} {SEASON}) ---")
current_df = gold_df.filter(F.col("event") == EVENT)

# Best avg lap_time_delta per driver across all sessions (lower = better)
print("\nBest avg lap_time_delta by driver (lower = closer to session best):")
current_df.groupBy("driver", "team").agg(
    F.round(F.avg("lap_time_delta"), 3).alias("avg_lap_delta"),
    F.round(F.avg("consistency_score"), 3).alias("avg_consistency"),
    F.round(F.avg("pace_vs_teammate"), 3).alias("avg_vs_teammate"),
).orderBy("avg_lap_delta").show(20)

# Race session only: feature leaders on race day
print(f"\nRace day feature leaders ({EVENT}):")
current_race = current_df.filter(F.col("session_type") == "R")
if current_race.count() > 0:
    current_race.groupBy("driver", "team", "race_position").agg(
        F.round(F.avg("lap_time_delta"),   3).alias("avg_lap_delta"),
        F.round(F.avg("consistency_score"),3).alias("avg_consistency"),
        F.round(F.avg("tyre_deg_rate"),    3).alias("avg_tyre_deg"),
    ).orderBy("race_position").show(20)
else:
    print("  No Race session rows for this event yet.")

#  TEAM vs DRIVER PACE GAP 

print("\n--- pace_vs_teammate by driver (Race session, completed rounds) ---")
race_df.groupBy("driver", "team").agg(
    F.round(F.avg("pace_vs_teammate"), 3).alias("avg_vs_teammate"),
    F.round(F.min("pace_vs_teammate"), 3).alias("best_vs_teammate"),
    F.count("*").alias("laps"),
).orderBy("avg_vs_teammate").show(20)

#  TYRE COMPOUND EFFECT ON PACE 

print("\n--- avg lap_time_delta by compound (all sessions) ---")
gold_df.filter(F.col("compound") != "UNKNOWN").groupBy("compound").agg(
    F.round(F.avg("lap_time_delta"), 3).alias("avg_lap_delta"),
    F.round(F.avg("tyre_deg_rate"),  3).alias("avg_tyre_deg"),
    F.count("*").alias("laps"),
).orderBy("avg_lap_delta").show()

#  PACE TREND LEADERS 

print("\n--- pace_trend by driver (negative = improving) ---")
trend_df = gold_df.filter(
    (F.col("session_type") == "R") & F.col("pace_trend").isNotNull()
)
if trend_df.count() > 0:
    trend_df.groupBy("driver", "team").agg(
        F.round(F.avg("pace_trend"), 3).alias("avg_pace_trend"),
    ).orderBy("avg_pace_trend").show(20)
else:
    print("  pace_trend not yet available (need 3+ rounds of data).")

#  NULL AUDIT 

print("\n--- Null rates across full season Gold (%) ---")
all_cols = feature_cols + ["race_position", "pace_trend"]
total = gold_df.count()
gold_df.select([
    F.round(F.count(F.when(F.col(c).isNull(), c)) / total * 100, 1).alias(c)
    for c in all_cols
]).show()