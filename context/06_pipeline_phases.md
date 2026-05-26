# Pipeline Phases

## RF/GBT Baseline (Notebooks 01–06)

### Phase 1 — Ingestion (`01_ingest.py`)
- **Input**: FastF1 API
- **Sessions**: FP1, FP2, FP3, Q, SQ, S, R — all types
- **Output**: `raw/season=…/event=…/session=…/` (Bronze Parquet)
- **Key steps**: `pick_quicklaps()`, timedelta→float, explicit schema, results ingestion for Race sessions

### Phase 2 — Cleaning (`02_clean.py`)
- **Input**: Bronze Parquet
- **Output**: `clean/` (Silver Delta)
- **Key steps**: drop critical nulls, 107% lap filter, cast + validate types

### Phase 3 — Feature Engineering (`03_features.py`)
- **Input**: Silver Delta
- **Output**: `features/` (Gold Delta) — lap-level granularity
- **Features**: `lap_time_delta`, `consistency_score`, `best_sector_combo`, `tyre_deg_rate`, `pace_vs_teammate`, `pace_trend`

### Phase 4 — EDA (`04_eda.py`)  *(dev only — not in production pipeline)*
- Spark SQL correlation analysis, circuit-specific patterns, team vs driver pace gap

### Phase 5 — Training (`05_train.py`)
- **Input**: Gold Delta + race results (labels)
- **Output**: `models/base_r{N}/` or `models/qualifying_r{N}/`
- MLlib Pipeline: VectorAssembler → StringIndexer → GBTClassifier (sample_weight = recency × session_type)

### Phase 6 — Prediction + Export (`06_predict.py`)
- **Input**: versioned MLlib model + Gold features for current weekend
- **Output**: `predictions/` Delta + `dashboard/public/predictions.json`
- Bootstrap uncertainty (N=20, 80%), session scores, pace trend, feature importance
- Uses `utils/predict_utils.py` for shared logic

---

## MAE Pipeline (Notebooks 07–11)

### Phase 7 — Telemetry Ingestion (`07_tel_ingest.py`)
- **Input**: FastF1 API (200 Hz lap telemetry)
- **Output**: `telemetry/raw/season=…/event=…/session=…/` (Bronze Parquet + `_SUCCESS` flags)
- `ThreadPoolExecutor(max_workers=2)` for OOM-safe parallel ingestion on CE

### Phase 8 — Telemetry Preprocessing (`08_tel_preprocess.py`)
- **Input**: Telemetry Bronze
- **Output**: `telemetry/clean/silver/season=…/` — shape (6, 1024) float32 per lap
- Distance-axis resampling (scipy interp), per-session z-score, label join from Gold

### Phase 9 — MAE Pre-training (`09_mae_pretrain.py`)
- **Input**: Telemetry Silver (all seasons, all sessions — unlabelled OK)
- **Output**: `models/mae_checkpoint.pt` (rolling, per-epoch), `models/mae_train_log.csv`
- 200 epochs total, cosine LR, CE-resumable — safe across Databricks 2-hr sessions

### Phase 10 — MAE Fine-tuning (`10_mae_finetune.py`)
- **Input**: `mae_checkpoint.pt` + labelled Telemetry Silver (race_position known)
- **Output**: `models/mae_finetuned.pt` (best top-3-acc), `models/mae_finetune_checkpoint.pt`
- Two-phase: 10 epochs linear probe (encoder frozen) → 40 epochs full fine-tune

### Phase 11 — MAE Prediction + Export (`11_mae_predict.py`)
- **Input**: `mae_finetuned.pt` + Telemetry Silver for current race weekend
- **Output**: `predictions/` Delta (`model_version='mae'`) + `dashboard/public/predictions.json`
- Entropy-based uncertainty; comparison block vs RF/GBT if post-race results available
- Uses `utils/predict_utils.py` for shared logic

---

## Execution Order (per race weekend)

```
RF/GBT (always):   01 → 02 → 03 → 05 → 06
MAE (once trained):07 → 08 → 11
```

## Shared Utilities

| Module | Purpose |
|---|---|
| `config.py` | All path constants, `ENCODER_HPARAMS`, `SESSION_WEIGHTS`, `DASHBOARD_JSON_PATH` |
| `utils/spark_session.py` | Singleton SparkSession |
| `utils/predict_utils.py` | Session scores, trend map, bootstrap, season history, payload builder, JSON writer |
| `utils/mae_model.py` | `F1MAE`, `F1PositionHead`, `PatchEmbedding1D` |
| `utils/tel_dataset.py` | `F1TelemetryDataset` (PyArrow-backed) |
| `utils/schema.py` | Bronze/Silver PySpark schemas |
| `utils/tel_schema.py` | Telemetry Bronze/Silver schemas |
| `utils/weights.py` | Sample weight computation |
| `utils/transforms.py` | Timedelta → seconds conversion |
