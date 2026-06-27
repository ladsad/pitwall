# Pitwall

F1 race prediction pipeline — PySpark · MLlib · PyTorch · Supabase · Next.js

**Live dashboard**: [pitwall-f1-six.vercel.app](https://pitwall-f1-six.vercel.app)

## Stack

- **Ingestion**: FastF1 Python API → Bronze Parquet (Local/Cloud Storage)
- **Processing**: PySpark cleaning → Silver Parquet → Gold Parquet with engineered features
- **ML (baseline)**: Spark MLlib GBTClassifier (dual sample weighting: recency decay × session type) + bootstrap uncertainty
- **ML (primary)**: 1D PatchTST-style Masked Autoencoder (PyTorch) pre-trained on 200 Hz lap telemetry, fine-tuned for position prediction
- **Storage**: Local filesystem or Cloud Storage via `PITWALL_DATA` environment variable
- **Database**: Supabase (PostgreSQL) for live prediction querying
- **Dashboard**: Next.js on Vercel, querying the Supabase backend via API routes
- **Orchestration**: GitHub Actions (cron-based scheduling)
- **Training**: Google Colab (12-hr GPU sessions for MAE training)

## Pipeline

| Notebook | Phase |
|---|---|
| `01_ingest.py` | FastF1 → Bronze Parquet (lap times + race results) |
| `02_clean.py` | Clean → Silver Parquet |
| `03_features.py` | Feature engineering → Gold Parquet |
| `04_eda.py` | EDA + correlation analysis *(dev only)* |
| `05_train.py` | Train RF/GBT → versioned model saved locally |
| `06_predict.py` | RF/GBT predictions + bootstrap uncertainty → Supabase / predictions.json |
| `07_tel_ingest.py` | FastF1 200 Hz telemetry → Telemetry Bronze Parquet |
| `08_tel_preprocess.py` | Resample to (6, 1024), z-score → Telemetry Silver Parquet |
| `09_mae_pretrain.py` | MAE self-supervised pre-training (Colab-resumable, per-epoch checkpoint) |
| `10_mae_finetune.py` | Fine-tune encoder for position classification |
| `11_mae_predict.py` | MAE inference → Supabase / predictions.json *(primary once trained)* |

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

![MAE Architecture (Dark Mode)](f1mae_diagram_dark.png)

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

Both RF and MAE predictions share the same `predictions` Supabase table (`model_version` column distinguishes them).

## Sample Weighting (RF/GBT)

```
sample_weight = recency_weight × session_weight
recency_weight  = exp(-λ × rounds_ago),  λ = 0.15
session_weights = R:1.0  Q:0.7  SQ:0.6  S:0.5  FP3:0.35  FP2:0.25  FP1:0.15
```

## Storage Layout

The `PITWALL_DATA` environment variable points to the root directory for data (defaults to `./data`).

```
./data/
├── raw/              ← Bronze Parquet (01_ingest.py)
├── clean/            ← Silver Parquet  (02_clean.py)
├── features/         ← Gold Parquet    (03_features.py)
├── predictions/      ← shared Parquet backup  (06_predict.py + 11_mae_predict.py)
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

1. Clone the repository.
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
3. Set up the Supabase database:
   - Create a new project on [supabase.com](https://supabase.com).
   - Run the `supabase_schema.sql` script in the Supabase SQL Editor.
   - Set the `SUPABASE_URL` and `SUPABASE_KEY` environment variables locally and in GitHub Secrets.

To override the local data path:
```bash
export PITWALL_DATA=/path/to/your/custom/data_dir
```

## Run Pipelines

Pipelines are orchestrated via the `PIPELINE` environment variable and `scripts/run_pipeline.py`.

```bash
# Run locally:
PIPELINE=weekend_rf python scripts/run_pipeline.py

# Available Pipelines:
weekend_rf    — RF/GBT race prediction (01→02→03→05→06)
weekend_mae   — MAE race prediction (07→08→11, requires trained model)
tel_backfill  — One-time historical telemetry ingestion (07→08)
mae_pretrain  — MAE pre-training, resumable (09)
mae_finetune  — MAE fine-tuning, resumable (10)
full_mae      — Full MAE setup pipeline (07→08→09→10)
dev_eda       — Development EDA run (01→02→03→04)
```

For Deep Learning (MAE): It's recommended to mount Google Drive in a Google Colab notebook, point `PITWALL_DATA` to your Drive directory, and run `09_mae_pretrain.py` and `10_mae_finetune.py` using their free GPU tier.

---

## Development Highlights & Resolutions

During the project lifecycle, several critical architectural challenges were resolved:
- **Data Pipeline Migration**: Transitioned from Databricks Community Edition to a local/hybrid PySpark setup to handle live telemetry scale and resolve zero-event aggregation anomalies.
- **Data Integrity**: Audited and fixed target variable data leakage in the LightGBM baseline, ensuring future results did not improperly feed back into features.
- **Deep Learning Optimization**: Pre-trained the MAE on a 16GB telemetry dataset via Kaggle/Colab. Resolved initial out-of-memory (OOM) and read-only file system constraints, and scrubbed `NaN` telemetry anomalies that caused early gradient explosions.
- **Model Interpretability**: Scaffolded dynamic, driver-specific occlusion sensitivity for the PyTorch MAE to match the transparency and feature importance reporting of the baseline LightGBM model.
- **Frontend Stability**: Debugged Next.js 500 server crashes caused by deprecated Zustand imports, resolved UI z-index overlaps, and implemented robust SWR/React-Query caching to prevent full-page re-renders during state toggles.

---

## Roadmap

### Near-term
- [x] Migrate from Databricks CE to a local + GitHub Actions CI/CD stack
- [x] Migrate frontend to dynamic Supabase queries
- Auto-trigger retraining when FastF1 `race_position` labels become available

### Model improvements
- Extend `pace_trend` lookback beyond 6 rounds; multi-season career data
- Driver profile features: circuit-type performance index, wet-weather history, pitstop consistency
- Qualifying-to-race grid delta as feature
- Circuit DNA embeddings matched to driver strengths

### Research
- Calibrated probability metrics (Brier score, log-loss) alongside top-3 accuracy
- Online/incremental fine-tuning: update MAE weights as each session completes during weekend
