import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import fastf1
import pandas as pd

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path("/Workspace/Repos/pitwall")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, BooleanType, ArrayType,
)

from utils.spark_session import get_spark_session
from utils.tel_schema import TEL_BRONZE_SCHEMA
from config import TELEMETRY_RAW_PATH, TEL_SEASONS, SESSION_TYPES

spark = get_spark_session("pitwall-tel-ingest")

# ── CACHE ────────────────────────────────────────────────────────────────────

if os.name == "nt":
    CACHE_DIR = PROJECT_ROOT / ".cache" / "fastf1"
else:
    CACHE_DIR = pathlib.Path("/tmp/fastf1_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

TELEMETRY_CHANNELS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]

# Sessions to pull telemetry for — Race always included; FP/Quali optional
TEL_SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "R"]

SESSION_TYPE_MAP = {
    "Practice 1":        "FP1",
    "Practice 2":        "FP2",
    "Practice 3":        "FP3",
    "Qualifying":        "Q",
    "Sprint Qualifying": "SQ",
    "Sprint":            "S",
    "Race":              "R",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _success_flag(season: int, event: str, session_code: str) -> str:
    """Return DBFS path of the _SUCCESS marker for this session."""
    return f"{TELEMETRY_RAW_PATH}/season={season}/event={event}/session={session_code}/_SUCCESS"


def _session_already_done(season: int, event: str, session_code: str) -> bool:
    """Check whether this session's telemetry has already been written."""
    try:
        dbutils.fs.ls(_success_flag(season, event, session_code))
        return True
    except Exception:
        return False


def _write_success_flag(season: int, event: str, session_code: str) -> None:
    """Write an empty _SUCCESS marker so restarts skip completed sessions."""
    flag_path = _success_flag(season, event, session_code)
    dbutils.fs.put(flag_path, "", overwrite=True)


def _tel_to_row(lap, season: int, event: str, session_type_tag: str) -> dict | None:
    """
    Pull telemetry for a single lap and return a dict matching TEL_BRONZE_SCHEMA.
    Returns None if telemetry is unavailable or malformed.
    """
    try:
        tel = lap.get_telemetry()
    except Exception:
        return None

    if tel is None or tel.empty or len(tel) < 20:
        # Fewer than 20 samples → useless for resampling
        return None

    # Ensure Distance column exists (required for resampling in phase 2)
    if "Distance" not in tel.columns:
        return None

    driver      = str(lap["Driver"])
    lap_number  = int(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else -1
    lap_id      = f"{season}_{event}_{driver}_{lap_number}"

    def _to_float_list(series) -> list[float]:
        return [float(v) if pd.notna(v) else 0.0 for v in series]

    def _to_bool_list(series) -> list[bool]:
        return [bool(v) if pd.notna(v) else False for v in series]

    def _to_int_list(series) -> list[int]:
        return [int(v) if pd.notna(v) else 0 for v in series]

    return {
        "lap_id":      lap_id,
        "driver":      driver,
        "season":      season,
        "event":       event,
        "session_type": session_type_tag,
        "lap_number":  lap_number,
        "distance":    _to_float_list(tel["Distance"]),
        "speed":       _to_float_list(tel["Speed"])     if "Speed"    in tel.columns else [],
        "throttle":    _to_float_list(tel["Throttle"])  if "Throttle" in tel.columns else [],
        "brake":       _to_bool_list(tel["Brake"])      if "Brake"    in tel.columns else [],
        "rpm":         _to_float_list(tel["RPM"])        if "RPM"      in tel.columns else [],
        "gear":        _to_int_list(tel["nGear"])        if "nGear"    in tel.columns else [],
        "drs":         _to_int_list(tel["DRS"])          if "DRS"      in tel.columns else [],
    }


# ── MAIN INGESTION FUNCTION ───────────────────────────────────────────────────

def ingest_session_telemetry(season: int, event: str, session_code: str) -> int:
    """
    Fetch all quicklap telemetry for one session, write as Parquet to Bronze,
    mark with _SUCCESS flag. Returns the number of laps written.

    Idempotent: skips silently if _SUCCESS flag already exists.
    """
    if _session_already_done(season, event, session_code):
        print(f"    [skip] {season} | {event} | {session_code} — already done")
        return 0

    try:
        session = fastf1.get_session(season, event, session_code)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception as e:
        print(f"    [error] {season} | {event} | {session_code} — load failed: {e}")
        return 0

    laps = session.laps.pick_quicklaps()
    if laps.empty:
        print(f"    [skip] No quicklaps for {season} | {event} | {session_code}")
        return 0

    session_type_tag = SESSION_TYPE_MAP.get(session.name, session_code)

    # Fetch telemetry per lap (2 threads max to avoid OOM on CE)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_tel_to_row, lap, season, event, session_type_tag): i
                   for i, (_, lap) in enumerate(laps.iterrows())}
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)

    if not rows:
        print(f"    [warn] {season} | {event} | {session_code} — 0 valid telemetry laps")
        return 0

    sdf = spark.createDataFrame(rows, schema=TEL_BRONZE_SCHEMA)

    output_path = (
        f"{TELEMETRY_RAW_PATH}"
        f"/season={season}"
        f"/event={event}"
        f"/session={session_code}"
    )

    sdf.write.mode("overwrite").parquet(output_path)
    _write_success_flag(season, event, session_code)

    count = len(rows)
    print(f"    [ok] {season} | {event} | {session_code} — {count} laps written → {output_path}")
    return count


# ── LOOP OVER SEASONS / EVENTS / SESSIONS ────────────────────────────────────

print(f"Telemetry ingestion: seasons {TEL_SEASONS}")
print(f"Sessions targeted  : {TEL_SESSION_TYPES}")
print(f"Output root        : {TELEMETRY_RAW_PATH}\n")

total_laps      = 0
total_sessions  = 0
skipped_count   = 0

for year in TEL_SEASONS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    events   = schedule["EventName"].tolist()

    print(f"\n── Season {year} — {len(events)} events ──────────────────────────────")

    for gp in events:
        print(f"\n  {gp}")
        for session_code in TEL_SESSION_TYPES:
            if _session_already_done(year, gp, session_code):
                skipped_count += 1
                print(f"    [skip] {session_code} already done")
                continue

            n = ingest_session_telemetry(year, gp, session_code)
            if n > 0:
                total_laps     += n
                total_sessions += 1

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Telemetry ingestion complete")
print(f"  Sessions written  : {total_sessions}")
print(f"  Sessions skipped  : {skipped_count} (already done)")
print(f"  Total laps written: {total_laps:,}")

# Spot-check: count rows across Bronze
print("\nSpot-check — row counts by season (Bronze telemetry):")
try:
    all_bronze = (
        spark.read
             .schema(TEL_BRONZE_SCHEMA)
             .option("basePath", TELEMETRY_RAW_PATH)
             .parquet(TELEMETRY_RAW_PATH)
    )
    all_bronze.groupBy("season", "session_type").count().orderBy("season", "session_type").show()
except Exception as e:
    print(f"  Could not read Bronze for spot-check: {e}")
