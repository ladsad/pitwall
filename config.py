import os
import pathlib
from dotenv import load_dotenv

try:
    _PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
except NameError:
    _PROJECT_ROOT = pathlib.Path.cwd()

load_dotenv(_PROJECT_ROOT / ".env.local")
load_dotenv(_PROJECT_ROOT / ".env")

SEASON = 2026
EVENT = "Barcelona Grand Prix"
ROUND_NUMBER = 7

SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Local paths replacing Unity Catalog Volumes.
# Set PITWALL_DATA env var to override, or default to ./data relative to project root.

BASE_PATH = pathlib.Path(os.environ.get("PITWALL_DATA", str(_PROJECT_ROOT / "data")))

RAW_PATH            = BASE_PATH / "raw"
RESULTS_PATH        = BASE_PATH / "raw" / "results"
CLEAN_PATH          = BASE_PATH / "clean"
FEATURES_PATH       = BASE_PATH / "features"
MODELS_PATH         = BASE_PATH / "models"
PREDICTIONS_PATH    = BASE_PATH / "predictions"
DASHBOARD_JSON_PATH = _PROJECT_ROOT / "dashboard" / "public" / "predictions.json"

# ML weighting
RECENCY_LAMBDA = 0.15

SESSION_WEIGHTS = {
    "R":   1.0,
    "Q":   0.7,
    "SQ":  0.6,
    "S":   0.5,
    "FP3": 0.35,
    "FP2": 0.25,
    "FP1": 0.15,
}

# Telemetry paths (Support independent overrides for Kaggle input vs working dirs)
TELEMETRY_RAW_PATH   = pathlib.Path(os.environ.get("TEL_RAW_PATH", str(BASE_PATH / "telemetry" / "raw")))
TELEMETRY_CLEAN_PATH = pathlib.Path(os.environ.get("TEL_CLEAN_PATH", str(BASE_PATH / "telemetry" / "clean")))

# Seasons to backfill during initial historical ingestion
TEL_SEASONS = [2024, 2025, 2026]

# MAE encoder hyperparameters — shared between 09, 10, and 11 so they stay in sync.
# If you change any of these, delete the checkpoint and retrain from scratch.
ENCODER_HPARAMS = dict(
    d_model         = 384,
    n_heads         = 6,
    encoder_layers  = 6,
    decoder_d_model = 192,
    decoder_n_heads = 4,
    decoder_layers  = 2,
    patch_stride    = 16,
    mask_ratio      = 0.75,
)

# ── SUPABASE ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")