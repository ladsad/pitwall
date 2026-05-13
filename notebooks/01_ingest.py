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

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    # Running as a Databricks notebook — __file__ is not defined
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SEASON, EVENT, ROUND_NUMBER, RAW_PATH, SESSION_TYPES, RESULTS_PATH
from utils.spark_session import get_spark_session
from utils.schema import BRONZE_SCHEMA, RESULTS_SCHEMA
from utils.transforms import timedeltas_to_seconds

if os.name == "nt":
    CACHE_DIR = PROJECT_ROOT / ".cache" / "fastf1"
else:
    CACHE_DIR = pathlib.Path("/tmp/fastf1_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

spark = get_spark_session("pitwall-ingest")

print(f"Ingesting: Season {SEASON} | Event: {EVENT} | Round: {ROUND_NUMBER}")
print(f"Sessions: {SESSION_TYPES}")
print(f"Output root: {RAW_PATH}")

# SESSION TYPE NORMALISER 
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
ingested_sessions = []
skipped_sessions  = []
race_session_obj  = None   # held for results ingestion below

for session_code in SESSION_TYPES:
    print(f"\n{'-'*60}")
    print(f"  Loading: {session_code}")

    try:
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
        laps_df = timedeltas_to_seconds(laps_df)

        raw_session_name = session.name
        session_type_tag = SESSION_TYPE_MAP.get(raw_session_name, session_code)

        laps_df["session_type"] = session_type_tag
        laps_df["season"]       = SEASON
        laps_df["event"]        = EVENT
        laps_df["session"]      = raw_session_name

        for bool_col in ("FreshTyre", "IsPersonalBest"):
            if bool_col in laps_df.columns:
                laps_df[bool_col] = laps_df[bool_col].astype(str)

        sdf = spark.createDataFrame(laps_df, schema=BRONZE_SCHEMA)

        print(f"  {sdf.count()} laps loaded for {session_type_tag}")

        output_path = (
            f"{RAW_PATH}"
            f"/season={SEASON}"
            f"/event={EVENT}"
            f"/session={session_code}"
        )

        sdf.write.mode("overwrite").parquet(output_path)
        print(f"  Written to: {output_path}")
        ingested_sessions.append(session_code)

        # Hold on to the Race session object — used for results ingestion below.
        # We don't re-load it; session.results is already available after .load().
        if session_type_tag == "R":
            race_session_obj = session

    except Exception as e:
        print(f"  {session_code} failed — likely doesn't exist for this event.")
        print(f"  Error: {e}")
        skipped_sessions.append(session_code)

# SUMMARY — LAPS
print(f"\n{'='*60}")
print(f"  Lap ingestion complete")
print(f"  Sessions ingested : {ingested_sessions}")
print(f"  Sessions skipped  : {skipped_sessions}")

if ingested_sessions:
    verify_path = f"{RAW_PATH}/season={SEASON}/event={EVENT}"
    verify_df = spark.read.schema(BRONZE_SCHEMA).parquet(verify_path)
    print(f"\nRow counts by session_type:")
    verify_df.groupBy("session_type").count().orderBy("session_type").show()

# RACE RESULTS INGESTION

print(f"\n{'-'*60}")

if race_session_obj is None:
    print("  No Race session ingested — results write skipped.")
    print("  Re-run after the Race session is available to populate results.")
else:
    print("  Ingesting race results...")

    results_df = race_session_obj.results

    RESULTS_COLS_MAP = {
        "Abbreviation": "Driver",
        "TeamName":      "TeamName",
        "Position":      "Position",
        "GridPosition":  "GridPosition",
        "Status":        "Status",
        "Points":        "Points",
    }

    # Select and rename to our schema column names
    available = {k: v for k, v in RESULTS_COLS_MAP.items() if k in results_df.columns}
    missing   = [k for k in RESULTS_COLS_MAP if k not in results_df.columns]
    if missing:
        print(f"  Missing results columns: {missing} — will be null.")
        for col in missing:
            results_df[RESULTS_COLS_MAP[col]] = None

    results_pdf = results_df[list(available.keys())].rename(columns=available).copy()

    # Cast position columns to int — FastF1 returns them as float
    for pos_col in ("Position", "GridPosition"):
        if pos_col in results_pdf.columns:
            results_pdf[pos_col] = pd.to_numeric(results_pdf[pos_col], errors="coerce")
            results_pdf[pos_col] = results_pdf[pos_col].astype("Int64")  # nullable int

    results_pdf["Points"] = pd.to_numeric(results_pdf["Points"], errors="coerce").astype(float)

    # Tag partition columns
    results_pdf["season"]       = SEASON
    results_pdf["event"]        = EVENT
    results_pdf["round_number"] = ROUND_NUMBER

    # Nullable Int64 → standard int before Spark (Spark doesn't read pandas Int64)
    for pos_col in ("Position", "GridPosition"):
        results_pdf[pos_col] = results_pdf[pos_col].astype(object).where(
            results_pdf[pos_col].notna(), other=None
        )

    results_sdf = spark.createDataFrame(results_pdf, schema=RESULTS_SCHEMA)

    results_output_path = f"{RESULTS_PATH}/season={SEASON}/event={EVENT}"
    results_sdf.write.mode("overwrite").parquet(results_output_path)

    print(f"  {results_sdf.count()} driver results written to: {results_output_path}")
    print(f"\nResults:")
    results_sdf.select("Driver", "TeamName", "Position", "Status").orderBy("Position").show(25)