SEASON = 2026
EVENT = "Melbourne Grand Prix"   
ROUND_NUMBER = 1               

SESSION_TYPES = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]

# DBFS paths
BASE_PATH = "dbfs:/pitwall"
RAW_PATH = f"{BASE_PATH}/raw"
CLEAN_PATH = f"{BASE_PATH}/clean"
FEATURES_PATH = f"{BASE_PATH}/features"
MODELS_PATH = f"{BASE_PATH}/models"
PREDICTIONS_PATH = f"{BASE_PATH}/predictions"

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