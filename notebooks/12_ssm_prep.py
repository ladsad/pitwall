import os
import pathlib
import sys
import numpy as np

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import FloatType, IntegerType, ArrayType

from utils.spark_session import get_spark_session
from config import RAW_PATH, TELEMETRY_CLEAN_PATH
from utils.schema import BRONZE_SCHEMA
from utils.tel_schema import TEL_SILVER_SCHEMA

spark = get_spark_session("pitwall-ssm-prep")

print("SSM Data Prep: Processing all historical race sessions...")

# ── LOAD TELEMETRY (SILVER) ──────────────────────────────────────────────────
# Telemetry Silver has `channels` (6, 1024) and `lap_number`
silver_tel_path = str(TELEMETRY_CLEAN_PATH / "silver")
if not os.path.exists(silver_tel_path):
    print("Silver telemetry not found. Run 08_tel_preprocess.py first.")
    sys.exit(0)

silver_tel_df = (
    spark.read.schema(TEL_SILVER_SCHEMA).parquet(silver_tel_path)
         .filter(F.col("session_type") == "R")
)

print(f"Total race telemetry laps loaded: {silver_tel_df.count():,}")

# ── LOAD LAP TIMES (BRONZE) ──────────────────────────────────────────────────
# We use Bronze to get ALL laps, including pit laps and safety car laps, 
# which were dropped from Gold by the 107% rule.
if not os.path.exists(str(RAW_PATH)):
    print("Bronze laps not found.")
    sys.exit(0)

raw_laps_df = (
    spark.read.schema(BRONZE_SCHEMA)
         .option("basePath", str(RAW_PATH))
         .parquet(str(RAW_PATH))
         .filter(F.col("session_type") == "R")
)

# Clean critical nulls but KEEP outlier/pit laps
raw_laps_df = raw_laps_df.dropna(subset=["Driver", "Team", "LapNumber", "LapTime", "Compound"])

# Create Pit Stop Boolean and Safety Car proxies
session_w = Window.partitionBy("season", "event", "session_type")
raw_laps_df = raw_laps_df.withColumn(
    "_session_median", F.percentile_approx("LapTime", 0.5).over(session_w)
)

# Pit lap if PitInTime is not null. Safety Car if lap > 1.3 * median
raw_laps_df = raw_laps_df.withColumn(
    "pit_stop_bool", F.when(F.col("PitInTime").isNotNull(), 1).otherwise(0)
).withColumn(
    "track_status_sc", F.when(F.col("LapTime") > F.col("_session_median") * 1.3, 1).otherwise(0)
).withColumnRenamed("Driver", "driver").withColumnRenamed("LapNumber", "lap_number")

# Compute pace delta (target) compared to session best
raw_laps_df = raw_laps_df.withColumn(
    "_session_best", F.min("LapTime").over(session_w)
).withColumn(
    "pace_delta", (F.col("LapTime") - F.col("_session_best")).cast("float")
)

# ── JOIN METADATA WITH TELEMETRY ─────────────────────────────────────────────
# We now join the continuous bronze laps with the telemetry laps.
# Sometimes fastf1 fails to get telemetry for a lap, which might create a gap.
joined_df = raw_laps_df.join(
    silver_tel_df.select("season", "event", "driver", "lap_number", "channels"),
    on=["season", "event", "driver", "lap_number"],
    how="inner"
)

# ── EXTRACT CONTINUOUS STINTS ────────────────────────────────────────────────
# A continuous stint is unbroken if the difference between row_number and lap_number is constant.
driver_w = Window.partitionBy("season", "event", "driver").orderBy("lap_number")

joined_df = joined_df.withColumn(
    "row_num", F.row_number().over(driver_w)
).withColumn(
    "stint_group", F.col("lap_number") - F.col("row_num")
)

# Now, stint_id = driver + stint_group
joined_df = joined_df.withColumn(
    "stint_id", F.concat(F.col("driver"), F.lit("_"), F.col("stint_group").cast("string"))
)

# Aggregate into sequence-length stints
# PySpark collect_list is non-deterministic in order. We pack values into structs with lap_number,
# sort the array by lap_number, and then extract the values to guarantee chronological order.

def collect_sorted(col_name):
    return F.transform(
        F.array_sort(F.collect_list(F.struct("lap_number", col_name))),
        lambda x: x.getField(col_name)
    )

stints_df = (
    joined_df.groupBy("season", "event", "driver", "stint_id")
    .agg(
        F.count("lap_number").alias("stint_length"),
        F.array_sort(F.collect_list("lap_number")).alias("lap_numbers"),
        collect_sorted("Compound").alias("compounds"),
        collect_sorted("TyreLife").alias("tyre_ages"),
        collect_sorted("track_status_sc").alias("sc_flags"),
        collect_sorted("pit_stop_bool").alias("pit_flags"),
        collect_sorted("pace_delta").alias("pace_deltas"),
        collect_sorted("channels").alias("telemetry_sequence")
    )
    .filter(F.col("stint_length") >= 5)  # only keep stints with at least 5 continuous laps
)

print(f"\nContinuous racing stints found: {stints_df.count():,}")
stints_df.select("season", "event", "driver", "stint_id", "stint_length").orderBy(F.desc("stint_length")).show(20)

# Validate output columns
print("\nExtracted Columns for SSM:")
stints_df.printSchema()

# Save the prepped data to a new path for SSM training
SSM_PREP_PATH = str(PROJECT_ROOT / "data" / "ssm_prep")
stints_df.write.mode("overwrite").partitionBy("season", "event").parquet(SSM_PREP_PATH)
print(f"\nSSM Prep Data written to: {SSM_PREP_PATH}")
