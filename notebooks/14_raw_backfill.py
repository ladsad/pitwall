import fastf1
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TEL_SEASONS, SESSION_TYPES, RAW_PATH
from utils.spark_session import get_spark_session
from utils.schema import BRONZE_SCHEMA
from utils.transforms import timedeltas_to_seconds

CACHE_DIR = PROJECT_ROOT / ".cache" / "fastf1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

spark = get_spark_session("pitwall-raw-backfill")

print("Raw Data Ingestion (Historical Backfill)")
print(f"Target Seasons: {TEL_SEASONS}")
print(f"Target Sessions: {SESSION_TYPES}")
print(f"Output root: {RAW_PATH}\n")

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

total_laps_written = 0

for year in TEL_SEASONS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    events = schedule["EventName"].tolist()
    
    print(f"\n── Season {year} — {len(events)} events ──────────────────────────────")
    for gp in events:
        print(f"\n  {gp}")
        for session_code in SESSION_TYPES:
            output_path = str(
                RAW_PATH
                / f"season={year}"
                / f"event={gp}"
                / f"session={session_code}"
            )
            
            # Simple skip if already exists
            if os.path.exists(output_path):
                print(f"    [skip] {session_code} already exists")
                continue

            try:
                session = fastf1.get_session(year, gp, session_code)
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps_df = session.laps.pick_quicklaps()
            except Exception as e:
                print(f"    [skip] {session_code} — data not available: {e}")
                continue

            if laps_df.empty:
                print(f"    [skip] {session_code} — No quicklaps returned.")
                continue

            available_cols = [c for c in FASTF1_COLS_TO_KEEP if c in laps_df.columns]
            missing_cols = set(FASTF1_COLS_TO_KEEP) - set(available_cols)
            for col in missing_cols:
                laps_df[col] = None
            
            laps_df = laps_df[FASTF1_COLS_TO_KEEP].copy()
            laps_df = timedeltas_to_seconds(laps_df)

            raw_session_name = session.name
            session_type_tag = SESSION_TYPE_MAP.get(raw_session_name, session_code)

            laps_df["session_type"] = session_type_tag
            laps_df["season"]       = year
            laps_df["event"]        = gp
            laps_df["session"]      = raw_session_name

            for bool_col in ("FreshTyre", "IsPersonalBest"):
                if bool_col in laps_df.columns:
                    laps_df[bool_col] = laps_df[bool_col].astype(str)

            # PySpark requires strict types, replace NaNs with Nones
            laps_df = laps_df.where(pd.notnull(laps_df), None)

            try:
                sdf = spark.createDataFrame(laps_df, schema=BRONZE_SCHEMA)
                sdf.write.mode("overwrite").parquet(output_path)
                lap_count = len(laps_df)
                total_laps_written += lap_count
                print(f"    [ok] {session_code} — {lap_count} laps written")
            except Exception as e:
                print(f"    [fail] {session_code} — Schema/write error: {e}")

print(f"\n{'='*60}")
print(f"Raw Lap Ingestion Complete")
print(f"Total historical laps written: {total_laps_written:,}")
