"""
utils/db.py — Supabase client for writing predictions and reading history.
Replaces the old spark_session.py + dbutils-based file I/O.
"""

from __future__ import annotations

import os
from functools import lru_cache

from supabase import create_client, Client


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
    sb.table("predictions").upsert(
        records,
        on_conflict="season,event,driver,model_version",
    ).execute()


def upsert_history(records: list[dict]) -> None:
    """Upsert season history rows into the 'race_history' table."""
    sb = get_supabase()
    sb.table("race_history").upsert(
        records,
        on_conflict="season,event,model_version",
    ).execute()


def fetch_predictions(season: int, event: str, model_version: str | None = None) -> list[dict]:
    """Fetch predictions for a given season/event from Supabase."""
    sb = get_supabase()
    query = (
        sb.table("predictions")
        .select("*")
        .eq("season", season)
        .eq("event", event)
    )
    if model_version:
        query = query.eq("model_version", model_version)
    result = query.order("win_probability", desc=True).execute()
    return result.data


def fetch_history(season: int, model_version_prefix: str | None = None) -> list[dict]:
    """Fetch race history for a given season."""
    sb = get_supabase()
    query = sb.table("race_history").select("*").eq("season", season)
    if model_version_prefix:
        query = query.like("model_version", f"{model_version_prefix}%")
    result = query.order("round").execute()
    return result.data
