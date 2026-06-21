"""
utils/predict_utils.py — Shared prediction helpers
────────────────────────────────────────────────────
Used by both 06_predict.py (RF/GBT) and 11_mae_predict.py (MAE) to avoid
duplicating session scoring, trend extraction, bootstrap uncertainty, JSON
payload assembly, and dashboard write logic.
"""

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import functions as F


# ── SESSION SCORES ────────────────────────────────────────────────────────────

def session_scores(predictions_raw_df, session_types, weights):
    """
    Return {driver: {session_type: {"score": float, "weight": float} | None}}.

    Parameters
    ----------
    predictions_raw_df : pyspark DataFrame
        Must have columns: driver, session_type, lap_time_delta
    session_types : list[str]
        Ordered list of session keys for the payload (e.g. ["FP1", ..., "Q"])
    weights : dict[str, float]
        SESSION_WEIGHTS from config
    """
    raw = (
        predictions_raw_df
        .groupBy("driver", "session_type")
        .agg(F.avg("lap_time_delta").alias("avg_delta"))
        .collect()
    )

    driver_session_map = defaultdict(dict)
    for row in raw:
        driver_session_map[row["driver"]][row["session_type"]] = row["avg_delta"]

    def _sessions_for(driver):
        drv_map = driver_session_map[driver]
        return {
            s: (
                {
                    "score":  round(float(drv_map[s]), 2),
                    "weight": weights.get(s, 0.0),
                }
                if s in drv_map
                else None
            )
            for s in session_types
        }

    return {driver: _sessions_for(driver) for driver in driver_session_map}


# ── PACE TREND ────────────────────────────────────────────────────────────────

def trend_map(predictions_raw_df, threshold=0.15):
    """
    Return {driver: {"label": "up"|"flat"|"down", "value": float|None}}.

    Tries Race rows first; falls back to all sessions mid-weekend.
    """
    def _label(val):
        if val is None:
            return "flat"
        if val < -threshold:
            return "up"
        if val > threshold:
            return "down"
        return "flat"

    race_rows = (
        predictions_raw_df
        .filter(F.col("session_type") == "R")
        .groupBy("driver")
        .agg(F.avg("pace_trend").alias("avg_pace_trend"))
        .collect()
    )

    if not race_rows:
        race_rows = (
            predictions_raw_df
            .groupBy("driver")
            .agg(F.avg("pace_trend").alias("avg_pace_trend"))
            .collect()
        )

    return {
        row["driver"]: {
            "label": _label(row["avg_pace_trend"]),
            "value": round(float(row["avg_pace_trend"]), 4) if row["avg_pace_trend"] is not None else None,
        }
        for row in race_rows
    }


# ── BOOTSTRAP UNCERTAINTY ─────────────────────────────────────────────────────

def bootstrap_uncertainty(predictions_raw_df, driver_list, n=20, fraction=0.8):
    """
    Estimate per-driver win-probability std-dev via bootstrap resampling.

    Returns {driver: float (std dev across bootstrap samples)}.
    """
    bootstrap_probs = {d: [] for d in driver_list}

    for i in range(n):
        sample = predictions_raw_df.sample(fraction=fraction, seed=i)
        if sample.count() == 0:
            continue

        sample_agg = (
            sample
            .groupBy("driver")
            .agg(F.max("win_prob_lap").alias("win_prob"))
        )

        sample_total = sample_agg.agg(F.sum("win_prob")).collect()[0][0]
        if not sample_total or sample_total == 0:
            continue

        sample_norm = sample_agg.withColumn(
            "win_prob_norm",
            (F.col("win_prob") / F.lit(sample_total)).cast("float"),
        )

        for row in sample_norm.collect():
            if row["driver"] in bootstrap_probs:
                bootstrap_probs[row["driver"]].append(row["win_prob_norm"])

    return {
        driver: round(statistics.stdev(probs), 4) if len(probs) >= 2 else 0.0
        for driver, probs in bootstrap_probs.items()
    }


# ── SEASON HISTORY ────────────────────────────────────────────────────────────

def season_history(spark, predictions_path, features_path, season, model_version_filter):
    """
    Return (history list, season_accuracy float).

    Parameters
    ----------
    model_version_filter : str | callable
        String prefix (startswith) or a Column expression passed to .filter().
    """
    if isinstance(model_version_filter, str):
        mv_col = F.col("model_version").startswith(model_version_filter)
    else:
        mv_col = model_version_filter

    try:
        history_df = (
            spark.read.parquet(str(predictions_path))
                 .filter(
                     (F.col("season") == season)
                     & mv_col
                     & (F.col("predicted_position") == 1)
                 )
                 .select("event", "driver", "round")
                 .withColumnRenamed("driver", "predicted")
        )
    except Exception:
        return [], 0.0

    try:
        actuals_df = (
            spark.read.parquet(str(features_path))
                 .filter(
                     (F.col("season") == season)
                     & (F.col("session_type") == "R")
                     & (F.col("race_position") == 1)
                 )
                 .select("event", "driver")
                 .distinct()
                 .withColumnRenamed("driver", "actual")
        )
    except Exception:
        return [], 0.0

    rows = (
        history_df
        .join(actuals_df, on="event", how="inner")
        .withColumn("top3_hit", F.col("predicted") == F.col("actual"))
        .orderBy("round")
        .select("event", "predicted", "actual", "top3_hit")
        .collect()
    )

    history = [
        {
            "event":     row["event"],
            "predicted": row["predicted"],
            "actual":    row["actual"],
            "top3_hit":  bool(row["top3_hit"]),
        }
        for row in rows
    ]

    hits     = sum(1 for h in history if h["top3_hit"])
    accuracy = round(hits / len(history), 4) if history else 0.0

    return history, accuracy


# ── PAYLOAD BUILDER ───────────────────────────────────────────────────────────

def build_payload(
    *,
    model_version,
    event,
    round_number,
    season,
    sessions_used,
    season_accuracy,
    recency_lambda,
    predictions,        # list of dicts: driver, team, predicted_position, win_probability, uncertainty, trend, sessions
    feature_importance=None,
    history=None,
):
    """
    Return the canonical predictions.json dict understood by the dashboard.
    """
    return {
        "model_version":   model_version,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "event":           event,
        "round":           round_number,
        "season":          season,
        "sessions_used":   sessions_used,
        "season_accuracy": {
            "top3_pct": season_accuracy[0] if isinstance(season_accuracy, tuple) else season_accuracy,
            "races":    season_accuracy[1] if isinstance(season_accuracy, tuple) else None,
        },
        "recency_lambda":     recency_lambda,
        "predictions":        predictions,
        "feature_importance": feature_importance or [],
        "history":            history or [],
    }


# ── DASHBOARD JSON WRITER ─────────────────────────────────────────────────────

def write_dashboard_json(payload, path):
    """Write predictions.json to the local filesystem."""
    json_str = json.dumps(payload, indent=2)
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json_str, encoding="utf-8")
    print(f"predictions.json written to: {local_path}")
