import os
import pathlib
import sys

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.sql import Window
 
from utils.spark_session import get_spark_session
from utils.schema import BRONZE_SCHEMA
from utils.transforms import (
    drop_critical_nulls,
    cast_lap_number,
    normalise_compound,
    filter_outlier_laps,
)
from config import SEASON, EVENT, RAW_PATH, CLEAN_PATH

spark = get_spark_session("pitwall-cleanup")

#  READ BRONZE

bronze_path = str(RAW_PATH / f"season={SEASON}" / f"event={EVENT}")

if not os.path.exists(bronze_path):
    print(f"\n[SKIP] No data found at {bronze_path}. Exiting cleanly.")
    sys.exit(0)

raw_df = (
    spark.read
         .schema(BRONZE_SCHEMA)
         .parquet(bronze_path)
)

raw_count = raw_df.count()
print(f"\nBronze rows loaded: {raw_count:,}")
 
print("\nRow counts by session_type (Bronze):")
raw_df.groupBy("session_type").count().orderBy("session_type").show()

#  CLEAN-UP

after_nulls_df = drop_critical_nulls(raw_df)
 
dropped_nulls = raw_count - after_nulls_df.count()
print(f"Rows dropped (critical nulls): {dropped_nulls:,}")

cleaned_df = (
    after_nulls_df
    .transform(cast_lap_number)
    .transform(normalise_compound)
)

silver_df = filter_outlier_laps(cleaned_df, threshold=1.07)
 
dropped_outliers = after_nulls_df.count() - silver_df.count()
silver_count = silver_df.count()
 
print(f"Rows dropped (107% rule)  : {dropped_outliers:,}")
print(f"Rows remaining (Silver)   : {silver_count:,}")
print(f"Total rows removed        : {raw_count - silver_count:,} "
      f"({(raw_count - silver_count) / raw_count * 100:.1f}%)")

#  WRITE SILVER DATA — Parquet (replacing Delta on local)

silver_output = str(CLEAN_PATH / f"season={SEASON}" / f"event={EVENT}")

(
    silver_df
    .write
    .mode("overwrite")
    .partitionBy("session_type")
    .parquet(silver_output)
)
 
print(f"\nSilver Parquet written to: {silver_output}")

#  VERIFICATION

verify_df = spark.read.schema(BRONZE_SCHEMA).parquet(silver_output)
 
print("Row counts by session_type (Silver — this event):")
verify_df.groupBy("session_type").count().orderBy("session_type").show()
 
# Null check
critical_cols = ["Driver", "Team", "LapTime", "session_type", "Compound"]
print("Null counts in critical columns (should all be 0):")
verify_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in critical_cols
]).show()
 
# 107% sanity check
session_w = Window.partitionBy("season", "event", "session_type")
breach_count = (
    verify_df
    .withColumn("_session_best", F.min("LapTime").over(session_w))
    .filter(F.col("LapTime") > F.col("_session_best") * 1.07)
    .count()
)
print(f"Laps breaching 107% rule in Silver (should be 0): {breach_count}")
 
# Compound value check
print("Distinct Compound values (should be SOFT/MEDIUM/HARD/INTER/WET/UNKNOWN):")
verify_df.select("Compound").distinct().orderBy("Compound").show()