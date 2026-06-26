"""
utils/db.py — Supabase client for writing and reading predictions.
Replaces the old spark_session.py + dbutils-based file I/O.
"""

from __future__ import annotations

import os
import math
from functools import lru_cache

from supabase import create_client, Client


def _clean_json(obj):
    """Recursively convert NaN and Infinity to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a singleton Supabase client. Reads URL/key from env vars."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set.\n"
            "Get them from: https://supabase.com → Project Settings → API."
        )
    return create_client(url, key)


def upsert_predictions(records: list[dict]) -> None:
    """
    Upsert prediction rows into the 'predictions' table.
    Conflict resolution on (season, event, driver, model_version).
    """
    sb = get_supabase()
    clean_records = _clean_json(records)
    sb.table("predictions").upsert(
        clean_records,
        on_conflict="season,event,driver,model_version",
    ).execute()


