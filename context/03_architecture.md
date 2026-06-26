# Architecture

## Medallion layers

- **Bronze** — raw FastF1 data, saved as Parquet exactly as received. Partitioned by season/event/session.
- **Silver** — cleaned data in Parquet. Nulls removed, timedeltas converted, outlier laps filtered (107% rule).
- **Gold** — feature store. One row per driver per session per weekend. ML-ready aggregated features.
- **Telemetry Bronze** — raw 200 Hz lap telemetry arrays (speed, throttle, brake, RPM, gear, DRS) from FastF1.
- **Telemetry Silver** — resampled to fixed N=1024 distance axis, per-session z-score normalised, shape (6, 1024).

## Storage layout

```
/Volumes/workspace/default/pitwall/
├── raw/season=…/event=…/session=…/   ← Bronze Parquet (lap times)
├── clean/                              ← Silver Parquet (cleaned laps)
├── features/                           ← Gold Parquet (engineered features)
├── predictions/                        ← shared Parquet (RF model_version & MAE model_version)
├── models/
│   ├── base_r{N}/                      ← RF/GBT MLlib Pipeline (post-race)
│   ├── qualifying_r{N}/                ← RF/GBT MLlib Pipeline (pre-race)
│   ├── mae_checkpoint.pt               ← rolling pre-training checkpoint (per epoch)
│   ├── mae_finetune_checkpoint.pt      ← rolling fine-tune checkpoint (per epoch)
│   ├── mae_finetuned.pt                ← best fine-tuned MAE model (used by 11)
│   └── mae_train_log.csv               ← epoch loss log (append-only)
└── telemetry/
    ├── raw/season=…/event=…/session=… ← Telemetry Bronze Parquet + _SUCCESS flags
    └── clean/silver/season=…/          ← Telemetry Silver Parquet
```

## Dual-pipeline architecture

```
RF/GBT baseline (01–06)              MAE pipeline (07–11)
────────────────────────             ──────────────────────────
01_ingest.py   ──► Bronze            07_tel_ingest.py     ──► Telemetry Bronze
02_clean.py    ──► Silver            08_tel_preprocess.py ──► Telemetry Silver (6×1024)
03_features.py ──► Gold (Parquet)      09_mae_pretrain.py   ──► mae_checkpoint.pt
05_train.py    ──► RF/GBT model      10_mae_finetune.py   ──► mae_finetuned.pt
06_predict.py  ──► Predictions       11_mae_predict.py    ──► Predictions (model_version='mae')
```

Both pipelines share: `config.py` · `utils/spark_session.py` · `utils/predict_utils.py` · same `PREDICTIONS_PATH` Parquet table.

## Model versioning lifecycle

```
base_r{N}          ← RF/GBT trained after round N race completes
qualifying_r{N}    ← RF/GBT base + this weekend's FP + Quali data (pre-race)
mae                ← MAE fine-tuned model; single model_version in predictions Parquet
```

`06_predict.py` auto-selects `qualifying_r{N}` over `base_r{N}` if available.

## Dashboard data flow

```
Local Python pipeline
    → writes predictions.json to Local Storage / Supabase
    → git add dashboard/public/predictions.json && git push
    → Vercel detects push → auto-redeploys
    → pitwall-f1-six.vercel.app updates in ~30 seconds
```
