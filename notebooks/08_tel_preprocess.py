import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

from utils.spark_session import get_spark_session
from utils.tel_schema import TEL_BRONZE_SCHEMA, TEL_SILVER_SCHEMA
from config import TELEMETRY_RAW_PATH, TELEMETRY_CLEAN_PATH, TEL_SEASONS, FEATURES_PATH

spark = get_spark_session("pitwall-tel-preprocess")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

N          = 1024       # fixed distance-axis length for all laps
CHANNELS   = ["speed", "throttle", "brake", "rpm", "gear", "drs"]
# Lap filters
MIN_SAMPLES           = 100    # fewer samples → bad telemetry
MAX_INTERP_FRACTION   = 0.20   # more than 20% interpolated points → reject lap
SC_LAP_MULTIPLIER     = 2.0    # laps > 2× session median are safety-car laps

SILVER_PATH = str(TELEMETRY_CLEAN_PATH / "silver")

print(f"Telemetry preprocessing: seasons {TEL_SEASONS}")
print(f"Resample length N      : {N}")
print(f"Input  (Bronze)        : {TELEMETRY_RAW_PATH}")
print(f"Output (Silver)        : {SILVER_PATH}")


# ── LOAD RACE RESULTS FOR LABELS ─────────────────────────────────────────────
# Labels (race_position) come from the existing Gold feature store.
# We join on (season, event, driver) — nullable for mid-weekend laps.

try:
    labels_df = (
        spark.read.parquet(str(FEATURES_PATH))
             .filter(F.col("session_type") == "R")
             .select("season", "event", "driver", "race_position")
             .distinct()
    )
    print(f"Labels loaded: {labels_df.count():,} driver-event rows from Gold")
    has_labels = True
except Exception as e:
    print(f"Could not load Gold labels (expected on first run): {e}")
    labels_df  = None
    has_labels = False

# ── RESAMPLE FUNCTION (runs in pandas UDF / local) ───────────────────────────

def resample_lap(
    distance: list[float],
    channel_arrays: dict[str, list],
    n: int = N,
) -> np.ndarray | None:
    """
    Resample a single lap's telemetry channels onto a uniform distance grid of
    length N using linear interpolation.

    Returns float32 array of shape (6, N), or None if the lap is invalid.
    """
    dist = np.array(distance, dtype=np.float64)
    if len(dist) < MIN_SAMPLES:
        return None

    # Distance must be monotonically increasing — occasionally FastF1 returns
    # duplicate distance values at the start of a lap.
    unique_mask = np.concatenate(([True], np.diff(dist) > 0))
    dist = dist[unique_mask]
    ch_arrays_filtered = {k: np.array(v, dtype=np.float64)[unique_mask]
                          for k, v in channel_arrays.items()}

    if len(dist) < MIN_SAMPLES:
        return None

    dist_uniform = np.linspace(dist[0], dist[-1], n)
    result_channels = []

    for ch in CHANNELS:
        raw = ch_arrays_filtered.get(ch, np.zeros(len(dist)))
        if len(raw) != len(dist):
            raw = np.zeros(len(dist))

        # Count how many target points fall outside the original range
        # (will be extrapolated — track as proxy for interpolation fraction)
        n_extrap = np.sum((dist_uniform < dist[0]) | (dist_uniform > dist[-1]))
        if n_extrap / n > MAX_INTERP_FRACTION:
            return None

        f = interp1d(dist, raw, kind="linear", fill_value="extrapolate")
        result_channels.append(f(dist_uniform).astype(np.float32))

    stacked = np.stack(result_channels, axis=0)
    if np.isnan(stacked).any():
        return None
    return stacked


def compute_session_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute per-channel mean and std over all valid resampled laps in a session.
    Used for z-score normalisation; stored alongside Silver data.
    Returns { channel_name: {mean: float, std: float} }.
    """
    stacked = {ch: [] for ch in CHANNELS}
    for row in rows:
        arr = np.array(row["channels"]).reshape(len(CHANNELS), N)
        for i, ch in enumerate(CHANNELS):
            stacked[ch].append(arr[i])

    stats = {}
    for ch in CHANNELS:
        data = np.concatenate(stacked[ch])  # all values for this channel, this session
        stats[ch] = {
            "mean": float(np.mean(data)),
            "std":  float(np.std(data)) if np.std(data) > 1e-6 else 1.0,
        }
    return stats


def normalise_row(channels_flat: list[float], stats: dict) -> list[float]:
    """Apply per-channel z-score using pre-computed stats."""
    arr = np.array(channels_flat, dtype=np.float32).reshape(len(CHANNELS), N)
    for i, ch in enumerate(CHANNELS):
        arr[i] = (arr[i] - stats[ch]["mean"]) / stats[ch]["std"]
    return arr.flatten().tolist()


# ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────

import shutil
if os.path.exists(SILVER_PATH):
    shutil.rmtree(SILVER_PATH)

total_laps_written = 0
sessions_ok       = 0
sessions_skipped  = 0
laps_rejected     = 0

for year in TEL_SEASONS:
    bronze_season_path = str(TELEMETRY_RAW_PATH / str(year))

    # ── Verify Bronze path exists before reading ─────────────────────────────
    if not os.path.exists(bronze_season_path):
        print(f"  [skip] Season {year} — Bronze path not found: {bronze_season_path}")
        print(f"         Run 07_tel_ingest.py first.")
        continue

    try:
        season_df = (
            spark.read
                 .schema(TEL_BRONZE_SCHEMA)
                 .option("recursiveFileLookup", "true")
                 .parquet(bronze_season_path)
        )
    except Exception as e:
        print(f"  [skip] Season {year} — could not read Bronze: {e}")
        continue

    events = [row["event"] for row in season_df.select("event").distinct().collect()]
    print(f"\n── Season {year} — {len(events)} events ──────────────────────────────")

    for event in events:
        event_rows = []
        session_types = [
            row["session_type"]
            for row in (
                season_df.filter(F.col("event") == event)
                         .select("session_type").distinct().collect()
            )
        ]

        for stype in session_types:
            session_pdf = (
                season_df
                .filter((F.col("event") == event) & (F.col("session_type") == stype))
                .toPandas()
            )

            if session_pdf.empty:
                continue

            # ── Safety-car lap filter (lap time proxy via distance range)
            # We don't have lap time here; use total distance range as proxy —
            # a safety-car lap covers a full lap distance but at much lower speed.
            # We flag this at Silver read time from the existing Gold lap times instead.
            # For now, filter only on sample count.

            # ── Resample all laps ──────────────────────────────────────────────
            raw_rows = []  # pre-normalisation

            for _, lap_row in session_pdf.iterrows():
                ch_arrays = {
                    "speed":    lap_row.get("speed",    []),
                    "throttle": lap_row.get("throttle", []),
                    "brake":    lap_row.get("brake",    []),
                    "rpm":      lap_row.get("rpm",      []),
                    "gear":     lap_row.get("gear",     []),
                    "drs":      lap_row.get("drs",      []),
                }
                arr = resample_lap(lap_row.get("distance", []), ch_arrays)
                if arr is None:
                    laps_rejected += 1
                    continue

                raw_rows.append({
                    "lap_id":       lap_row["lap_id"],
                    "driver":       lap_row["driver"],
                    "season":       int(lap_row["season"]),
                    "event":        lap_row["event"],
                    "session_type": lap_row["session_type"],
                    "lap_number":   int(lap_row["lap_number"]),
                    "channels":     arr.flatten().tolist(),   # (6*1024,) before Silver schema
                })

            if not raw_rows:
                sessions_skipped += 1
                continue

            # ── Per-session z-score normalisation ─────────────────────────────
            stats = compute_session_stats(raw_rows)

            for r in raw_rows:
                r["channels"] = normalise_row(r["channels"], stats)
                # Reshape channels to (6, 1024) list-of-lists for Silver schema
                arr_2d = np.array(r["channels"], dtype=np.float32).reshape(len(CHANNELS), N)
                r["channels"] = arr_2d.tolist()
                # label_position filled after join below
                r["label_position"] = None

            event_rows.extend(raw_rows)
            sessions_ok += 1

            print(f"  [ok] {year} | {event} | {stype} — {len(raw_rows)} laps valid")

        if event_rows:
            event_sdf = spark.createDataFrame(event_rows, schema=TEL_SILVER_SCHEMA)
            if has_labels:
                event_sdf = event_sdf.join(
                    labels_df.withColumnRenamed("race_position", "label_position_gold"),
                    on=["season", "event", "driver"],
                    how="left",
                ).drop("label_position").withColumnRenamed("label_position_gold", "label_position")
            
            (
                event_sdf
                .write
                .mode("append")
                .partitionBy("season")
                .parquet(SILVER_PATH)
            )
            total_laps_written += len(event_rows)

# ── VERIFICATION ─────────────────────────────────────────────────────────────

if total_laps_written == 0:
    print("\nNo valid rows to write — check Bronze data.")
else:

    print(f"\n{'='*60}")
    print(f"Silver written to: {SILVER_PATH}")
    print(f"  Sessions processed : {sessions_ok}")
    print(f"  Sessions skipped   : {sessions_skipped}")
    print(f"  Laps rejected      : {laps_rejected:,}")
    print(f"  Laps written       : {total_laps_written:,}")

    # ── Verification ──────────────────────────────────────────────────────────
    verify = spark.read.schema(TEL_SILVER_SCHEMA).parquet(SILVER_PATH)
    print("\nRow counts by season / session_type (Silver):")
    verify.groupBy("season", "session_type").count().orderBy("season", "session_type").show()

    print("Null check on channels (should be 0):")
    print(verify.filter(F.col("channels").isNull()).count(), "null channel rows")

    print("Label coverage (non-null label_position):")
    total = verify.count()
    labelled = verify.filter(F.col("label_position").isNotNull()).count()
    print(f"  {labelled:,} / {total:,} rows labelled ({100*labelled/total:.1f}%)")
