# Install FastF1 — run this cell first on Databricks
# %pip install fastf1

import fastf1
import pandas as pd

from pyspark.sql import SparkSession

from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType, TimestampType, BooleanType
)
import os
import pathlib
import sys

# ── Resolve project root (works locally and on Databricks) ──────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SEASON, EVENT, SESSION, RAW_PATH
from utils.spark_session import get_spark

spark = get_spark()

# ── 1. Pull session via FastF1 ──────────────────────────────────────────────
# Use /tmp on Databricks/Linux, local .cache dir on Windows
if os.name == "nt":
    CACHE_DIR = PROJECT_ROOT / ".cache" / "fastf1"
else:
    CACHE_DIR = pathlib.Path("/tmp/fastf1_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

session = fastf1.get_session(SEASON, EVENT, SESSION)
session.load()

# ── 2. Extract lap data into pandas first ───────────────────────────────────
laps_pd = session.laps.pick_quicklaps().reset_index(drop=True)

# Keep only the columns we care about
cols = [
    "Driver", "Team",
    "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife", "FreshTyre",
    "PitInTime", "PitOutTime", "IsPersonalBest"
]
laps_pd = laps_pd[cols].copy()

# Convert timedelta columns to float seconds — PySpark can't read timedeltas
time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
             "PitInTime", "PitOutTime"]
for col in time_cols:
    laps_pd[col] = laps_pd[col].dt.total_seconds()

# ── 3. Define schema explicitly — never let PySpark infer on raw data ───────
schema = StructType([
    StructField("Driver",          StringType(),  True),
    StructField("Team",            StringType(),  True), 
    StructField("LapNumber",       IntegerType(), True),
    StructField("LapTime",         FloatType(),   True),
    StructField("Sector1Time",     FloatType(),   True),
    StructField("Sector2Time",     FloatType(),   True),
    StructField("Sector3Time",     FloatType(),   True),
    StructField("SpeedI1",         FloatType(),   True),
    StructField("SpeedI2",         FloatType(),   True),
    StructField("SpeedFL",         FloatType(),   True),
    StructField("SpeedST",         FloatType(),   True),
    StructField("Compound",        StringType(),  True),
    StructField("TyreLife",        FloatType(),   True),
    StructField("FreshTyre",       BooleanType(), True),
    StructField("PitInTime",       FloatType(),   True),
    StructField("PitOutTime",      FloatType(),   True),
    StructField("IsPersonalBest",  BooleanType(), True),
])

# ── 4. Convert to PySpark DataFrame ─────────────────────────────────────────
laps_spark = spark.createDataFrame(laps_pd, schema=schema)

# ── 5. Add partition metadata columns ───────────────────────────────────────

from pyspark.sql.functions import lit

laps_spark = (
    laps_spark
    .withColumn("season",  lit(SEASON))
    .withColumn("event",   lit(EVENT))
    .withColumn("session", lit(SESSION))
)

# ── 6. Save as Parquet (partitioned — makes cleaning/querying faster) ────────
output_path = f"{RAW_PATH}/season={SEASON}/event={EVENT}/session={SESSION}"

laps_spark.write.mode("overwrite").parquet(output_path)

print(f"Saved {laps_spark.count()} laps to {output_path}")
laps_spark.show(5)