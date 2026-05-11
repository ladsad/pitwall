import fastf1
import pandas as pd 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F 
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, BooleanType
)

import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.path.exists('/Workspace/Repos/pitwall') and '/Workspace/Repos/pitwall' not in sys.path:
    sys.path.insert(0, '/Workspace/Repos/pitwall')

from config import SEASON, EVENT, RAW_PATH, SESSION_TYPES
from utils.spark_session import get_spark_session
from utils.schema import BRONZE_SCHEMA
from utils.transforms import timedeltas_to_seconds

if os.name == "nt":
    CACHE_DIR = PROJECT_ROOT / ".cache" / "fastf1"
else:
    CACHE_DIR = pathlib.Path("/tmp/fastf1_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

spark = get_spark_session("pitwall-ingest")

print(f"Ingesting: Season {SEASON} | Event: {EVENT}")
print(f"Sessions: {SESSION_TYPES}")
print(f"Output root: {RAW_PATH}")

# SESSION TYPE NORMALISER 
# FastF1 uses varied internal identifiers. We normalise to the short codes
# defined in config.py so session_type is consistent downstream.

SESSION_TYPE_MAP = {
    "Practice 1":           "FP1",
    "Practice 2":           "FP2",
    "Practice 3":           "FP3",
    "Qualifying":           "Q",
    "Sprint Qualifying":    "SQ",
    "Sprint":               "S",
    "Race":                 "R",
}

# COLUMN SELECTION HELPER
# FastF1 returns many columns we don't need (telemetry, weather snapshots, etc.).
# We select only the columns in BRONZE_SCHEMA — minus the partition + tag cols
# we add ourselves — to avoid schema drift if FastF1 adds new columns upstream.

FASTF1_COLS_TO_KEEP = [
    "Driver", "DriverNumber", "Team",
    "LapNumber",
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife", "FreshTyre",
    "PitInTime", "PitOutTime",
    "IsPersonalBest",
]

# MAIN INGESTION LOOP
# Iterates over all session types. For each:
#   a) Load via FastF1 (with pick_quicklaps to drop telemetry-broken laps)
#   b) Select and coerce columns
#   c) Convert timedeltas to float seconds
#   d) Tag session_type, season, event, session partition columns
#   e) Create Spark DataFrame with explicit schema
#   f) Write partitioned Parquet to Bronze
 
ingested_sessions = []
skipped_sessions  = []

for session_code in SESSION_TYPES:
    print(f"\n{'-'*60}")
    print(f"  Loading: {session_code}")

    try:
        # Load from FastF1
        session = fastf1.get_session(SEASON, EVENT, session_code)
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        laps_df = session.laps.pick_quicklaps()

        if laps_df.empty:
            print(f"  No quicklaps returned for {session_code} — skipping.")
            skipped_sessions.append(session_code)
            continue

        available_cols = [c for c in FASTF1_COLS_TO_KEEP if c in laps_df.columns]
        missing_cols = set(FASTF1_COLS_TO_KEEP) - set(available_cols)
        if missing_cols:
            print(f"  Missing columns in {session_code}: {missing_cols}")
            for col in missing_cols:
                laps_df[col] = None
        
        laps_df = laps_df[FASTF1_COLS_TO_KEEP].copy()

        # Convert timedeltas
        laps_df = timedeltas_to_seconds(laps_df)

        # Tag partition + metadata columns
        # session_type: normalise FastF1's full name to our short code
        raw_session_name = session.name
        session_type_tag = SESSION_TYPE_MAP.get(raw_session_name, session_code)

        laps_df["session_type"] = session_type_tag
        laps_df["season"]       = SEASON
        laps_df["event"]        = EVENT
        laps_df["session"]      = raw_session_name

        for bool_col in ("FreshTyre", "IsPersonalBest"):
            if bool_col in laps_df.columns:
                laps_df[bool_col] = laps_df[bool_col].astype(str)

        # Create Spark DataFrame with explicit schema
        sdf = spark.createDataFrame(laps_df, schema=BRONZE_SCHEMA)

        print(f"  {sdf.count()} laps loaded for {session_type_tag}")

        output_path = (
            f"{RAW_PATH}"
            f"/season={SEASON}"
            f"/event={EVENT}"
            f"/session={session_code}"
        )

        (
            sdf.write
               .mode("overwrite")
               .parquet(output_path)
        )

        print(f"  Written to: {output_path}")
        ingested_sessions.append(session_code)


    except Exception as e:
        print(f"  {session_code} failed — likely doesn't exist for this event.")
        print(f"  Error: {e}")
        skipped_sessions.append(session_code)

# SUMMARY

if ingested_sessions:
    last_code = SESSION_TYPES[[st for st in SESSION_TYPES if st in
                                [s.replace("FP1","FP1").replace("Qualifying","Q")
                                 for s in ingested_sessions]][-1]
                               if False else -1]
 
    verify_path = f"{RAW_PATH}/season={SEASON}/event={EVENT}"
    verify_df   = spark.read.schema(BRONZE_SCHEMA).parquet(verify_path)
 
    print(f"\nSchema (from explicit definition):")
    verify_df.printSchema()
 
    print(f"\nRow counts by session_type:")
    verify_df.groupBy("session_type").count().orderBy("session_type").show()
else:
    print("\nNo sessions ingested successfully.")