SEASON = 2026
EVENT = "Canadian Grand Prix"
ROUND_NUMBER = 5

SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]

# Unity Catalog Volume paths
BASE_PATH = "/Volumes/workspace/default/pitwall"
RAW_PATH = f"{BASE_PATH}/raw"
RESULTS_PATH = f"{BASE_PATH}/raw/results"
CLEAN_PATH = f"{BASE_PATH}/clean"
FEATURES_PATH = f"{BASE_PATH}/features"
MODELS_PATH = f"{BASE_PATH}/models"
PREDICTIONS_PATH   = f"{BASE_PATH}/predictions"
DASHBOARD_JSON_PATH = f"{BASE_PATH}/dashboard/public/predictions.json"

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

# Telemetry paths
TELEMETRY_RAW_PATH   = f"{BASE_PATH}/telemetry/raw"
TELEMETRY_CLEAN_PATH = f"{BASE_PATH}/telemetry/clean"

# Seasons to backfill during initial historical ingestion
TEL_SEASONS = [2024, 2025]

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