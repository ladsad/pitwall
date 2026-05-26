# Pitwall

F1 race prediction pipeline — PySpark · Databricks · MLlib · PyTorch · Next.js

**Live dashboard**: [pitwall-f1-six.vercel.app](https://pitwall-f1-six.vercel.app)

## Stack

- **Ingestion**: FastF1 Python API → Bronze Parquet (Unity Catalog Volume)
- **Processing**: PySpark cleaning → Silver Delta → Gold Delta with engineered features
- **ML (baseline)**: Spark MLlib GBTClassifier (dual sample weighting: recency decay × session type) + bootstrap uncertainty
- **ML (primary)**: 1D PatchTST-style Masked Autoencoder (PyTorch) pre-trained on 200 Hz lap telemetry, fine-tuned for position prediction
- **Storage**: Databricks Unity Catalog — `/Volumes/workspace/default/pitwall/`
- **Dashboard**: Next.js on Vercel, driven by `dashboard/public/predictions.json`

## Pipeline

| Notebook | Phase |
|---|---|
| `01_ingest.py` | FastF1 → Bronze Parquet (lap times + race results) |
| `02_clean.py` | Clean → Silver Delta |
| `03_features.py` | Feature engineering → Gold Delta |
| `04_eda.py` | EDA + correlation analysis *(dev only)* |
| `05_train.py` | Train RF/GBT → versioned model saved to Volumes |
| `06_predict.py` | RF/GBT predictions + bootstrap uncertainty → predictions.json |
| `07_tel_ingest.py` | FastF1 200 Hz telemetry → Telemetry Bronze Parquet |
| `08_tel_preprocess.py` | Resample to (6, 1024), z-score → Telemetry Silver |
| `09_mae_pretrain.py` | MAE self-supervised pre-training (CE-resumable, per-epoch checkpoint) |
| `10_mae_finetune.py` | Fine-tune encoder for position classification |
| `11_mae_predict.py` | MAE inference → predictions.json *(primary once trained)* |

## Features (Gold Layer)

| Feature | Description |
|---|---|
| `lap_time_delta` | Driver lap time vs. their own session minimum |
| `consistency_score` | Intra-session lap time std dev |
| `best_sector_combo` | Sum of personal best S1+S2+S3 per session |
| `tyre_deg_rate` | Lap time degradation slope per tyre stint |
| `pace_vs_teammate` | Driver best lap vs. team-mate best lap per session |
| `pace_trend` | Recent 2-round avg delta minus prior 4-round avg delta |
| `compound` | Tyre compound (categorical, StringIndexer encoded) |

## MAE Architecture

```
Input (B, 6, 1024)
  │
  PatchEmbedding1D  — 6 parallel Conv1d(1→384, kernel=16, stride=16) summed → (B, 64, 384)
  LearnedPositionalEmbedding
  │
  75% random masking → (B, 16, 384) visible patches
  │
  MAEEncoder  6× TransformerBlock(d=384, h=6)
  │
  MAEDecoder  project d=192 → fill mask tokens → 2× TransformerBlock
  │           → MSE loss (masked patches only) during pre-training
  │
  F1PositionHead  mean-pool → Linear(384, 20) → CrossEntropy
```

Pre-training: 200 epochs, cosine LR, mask_ratio=0.75, batch=128  
Fine-tuning: 10 epochs linear probe → 40 epochs full end-to-end

## Model Versioning

| Version | When | Notebook |
|---|---|---|
| `qualifying_rNN` | Before race — FP + Quali data only | `06_predict.py` |
| `base_rNN` | After race — full labelled set | `06_predict.py` |
| `mae` | After fine-tuning — telemetry-based | `11_mae_predict.py` |

Both RF and MAE predictions share the same `PREDICTIONS_PATH` Delta table (`model_version` column distinguishes them).

## Sample Weighting (RF/GBT)

```
sample_weight = recency_weight × session_weight
recency_weight  = exp(-λ × rounds_ago),  λ = 0.15
session_weights = R:1.0  Q:0.7  SQ:0.6  S:0.5  FP3:0.35  FP2:0.25  FP1:0.15
```

## Storage Layout

```
/Volumes/workspace/default/pitwall/
├── raw/              ← Bronze Parquet (01_ingest.py)
├── clean/            ← Silver Delta  (02_clean.py)
├── features/         ← Gold Delta    (03_features.py)
├── predictions/      ← shared Delta  (06_predict.py + 11_mae_predict.py)
├── models/
│   ├── base_r{N}/            ← RF/GBT Pipeline
│   ├── qualifying_r{N}/      ← RF/GBT Pipeline (pre-race)
│   ├── mae_checkpoint.pt     ← rolling pre-train checkpoint
│   ├── mae_finetuned.pt      ← best fine-tuned model
│   └── mae_train_log.csv
└── telemetry/
    ├── raw/                  ← Telemetry Bronze (07_tel_ingest.py)
    └── clean/silver/         ← Telemetry Silver (08_tel_preprocess.py)
```

## Setup

1. Clone repo; connect VS Code to Databricks workspace (`databricks.yml` pre-configured for `dbc-8cc171c2-6c4e.cloud.databricks.com`)
2. Run RF/GBT pipeline: `01 → 02 → 03 → 05 → 06`
3. Push `dashboard/public/predictions.json` → Vercel auto-deploys within ~30 seconds

```
pip install -r requirements.txt
```

To override the Databricks workspace path:
```
export DATABRICKS_WORKSPACE_ROOT=/Workspace/Repos/your-repo-name
```

## Run Pipelines

```
# Via run_pipeline.py widget on Databricks:
weekend_rf    — RF/GBT race prediction (01→02→03→05→06)
weekend_mae   — MAE race prediction (07→08→11, requires trained model)
tel_backfill  — One-time historical telemetry ingestion (07→08)
mae_pretrain  — MAE pre-training, CE-resumable (09)
mae_finetune  — MAE fine-tuning, CE-resumable (10)
full_mae      — Full MAE setup pipeline (07→08→09→10)
dev_eda       — Development EDA run (01→02→03→04)
```

---

## Roadmap

### Near-term
- Schedule Databricks Jobs for automated race-weekend pipeline (FP1→Quali→Race triggers)
- Post-race retraining job fires automatically when `race_position` labels land
- GitHub Actions push for `predictions.json` (eliminate manual git push step)

### Model improvements
- Extend `pace_trend` lookback beyond 6 rounds; multi-season career data
- Driver profile features: circuit-type performance index, wet-weather history, pitstop consistency
- Qualifying-to-race grid delta as feature
- Circuit DNA embeddings matched to driver strengths

### Research
- Calibrated probability metrics (Brier score, log-loss) alongside top-3 accuracy
- Online/incremental fine-tuning: update MAE weights as each session completes during weekend
