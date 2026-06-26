# Fin-MAE F1: Implementation Plan

> **Stack**: FastF1 · PySpark · Parquet/Parquet · PyTorch · Local + Supabase + Google Colab  
> **Storage root**: `/Volumes/workspace/default/pitwall/`  
> **Compute**: Single-node CE cluster (K80 or T4 GPU, 15 GB RAM)  
> **Architecture**: 1D PatchTST-style Masked Autoencoder — not GAF/Spectrogram

---

## Overview

```
Existing pipeline (01–06)            New MAE pipeline (07–11)
──────────────────────────           ──────────────────────────
01_ingest.py   ──► Bronze            07_tel_ingest.py     ──► Telemetry Bronze (Parquet)
02_clean.py    ──► Silver            08_tel_preprocess.py ──► Telemetry Silver (Parquet)
03_features.py ──► Gold (Parquet)      09_mae_pretrain.py   ──► mae_checkpoint.pt
05_train.py    ──► RF/GBT model      10_mae_finetune.py   ──► mae_finetuned.pt
06_predict.py  ──► Predictions       11_mae_predict.py    ──► Predictions (Parquet, model_version='mae')
```

Both pipelines share: `config.py` · `utils/spark_session.py` · the same `PREDICTIONS_PATH` Parquet table (different `model_version` values).  
The existing notebooks are **untouched**.

---

## Status

| Phase | File | Status | Notes |
|---|---|---|---|
| 0 — Config | `config.py` | ✅ Done | `TELEMETRY_RAW_PATH`, `TELEMETRY_CLEAN_PATH`, `TEL_SEASONS`, `ENCODER_HPARAMS` added |
| 1 — Bronze | `07_tel_ingest.py` | ✅ Done | Checkpointed `_SUCCESS` flags, `ThreadPoolExecutor(max_workers=2)` |
| 2 — Silver | `08_tel_preprocess.py` | ✅ Done | Distance-axis resampling to N=1024, per-session z-score, label join from Gold |
| 2 — Schemas | `utils/tel_schema.py` | ✅ Done | `TEL_BRONZE_SCHEMA`, `TEL_SILVER_SCHEMA` |
| 3 — Dataset | `utils/tel_dataset.py` | ✅ Done | PyArrow-backed, partition filters, `label_distribution()` |
| 3 — Model | `utils/mae_model.py` | ✅ Done | `PatchEmbedding1D`, `MAEEncoder`, `MAEDecoder`, `F1MAE`, `F1PositionHead` |
| 3 — Pretrain | `09_mae_pretrain.py` | ✅ Done | Cosine LR, per-epoch checkpoint, CSV log, resumes on CE restart |
| 4 — Finetune | `10_mae_finetune.py` | ✅ Done | Two-phase (linear probe → full FT), inverse-freq class weights, top-3 accuracy |
| 5 — Predict | `11_mae_predict.py` | ✅ Done | Parquet write, `model_version='mae'`, comparison vs RF/GBT |

---

## Phase 0 — Config (`config.py`)

```python
# Telemetry paths (new)
TELEMETRY_RAW_PATH   = f"{BASE_PATH}/telemetry/raw"
TELEMETRY_CLEAN_PATH = f"{BASE_PATH}/telemetry/clean"
TEL_SEASONS          = [2024, 2025]

# MAE encoder hyperparameters — single source of truth for 09, 10, 11
# Changing any of these requires deleting mae_checkpoint.pt and retraining.
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
```

> [!IMPORTANT]
> `ENCODER_HPARAMS` is the **single source of truth**. Notebooks 09, 10, and 11 all import it. Never define it locally in a notebook.

---

## Phase 1 — Telemetry Ingestion (`07_tel_ingest.py`)

**Goal**: Fetch raw 200 Hz lap telemetry for `TEL_SEASONS`, write to Bronze Parquet.

### Key design decisions vs `01_ingest.py`

| | `01_ingest.py` | `07_tel_ingest.py` |
|---|---|---|
| `session.load()` | `telemetry=False` | `telemetry=True` |
| Unit of write | Session-level (all drivers) | Session-level (all drivers in one Parquet) |
| Checkpoint | None | `_SUCCESS` flag per session via `dbutils.fs.put` |
| Schema | `BRONZE_SCHEMA` (lap times) | `TEL_BRONZE_SCHEMA` (raw arrays) |
| Parallelism | None | `ThreadPoolExecutor(max_workers=2)` — OOM-safe on CE |

### Output path
```
{TELEMETRY_RAW_PATH}/season={year}/event={gp}/session={code}/
    ├── part-00000.parquet
    └── _SUCCESS    ← written after successful Parquet write; skipped if present on restart
```

### Bronze schema (`utils/tel_schema.py` → `TEL_BRONZE_SCHEMA`)

| Column | Type | Notes |
|---|---|---|
| `lap_id` | string | `{year}_{gp}_{driver}_{lap_n}` — globally unique |
| `driver` | string | 3-letter code |
| `season` | int | |
| `event` | string | |
| `session_type` | string | `FP1/FP2/FP3/Q/R` |
| `lap_number` | int | |
| `distance` | float[] | metres, raw 200 Hz — variable length |
| `speed` | float[] | km/h |
| `throttle` | float[] | 0–100 |
| `brake` | bool[] | |
| `rpm` | float[] | |
| `gear` | int[] | |
| `drs` | int[] | 0/8/10/12/14 |

---

## Phase 2 — Preprocessing & Normalisation (`08_tel_preprocess.py`)

**Goal**: Resample all laps to fixed length N=1024 on the distance axis, normalise, filter invalids, write Silver tensors, join race-position labels from the existing Gold Parquet table.

### Lap filter criteria

| Filter | Threshold | Rationale |
|---|---|---|
| Too few samples | `< 100` raw telemetry points | Bad FastF1 data |
| Monotonic distance | Drop duplicate distance values | FastF1 occasionally emits them at lap start |
| High extrapolation | `> 20%` of N=1024 target points outside original range | Lap is too short to resample reliably |

> [!NOTE]
> Safety-car lap filtering is deferred to Silver read time (the Gold lap-time data needed for median comparison is available in `FEATURES_PATH`). The Silver Parquet retains these laps; downstream callers can filter by joining `label_position` and applying a lap time delta threshold.

### Normalisation

- **Axis**: distance-normalised (not time) — preserves circuit geometry independent of pace
- **Scope**: per-session stats (mean/std computed over all valid laps in a given event+session)
- **Result**: (6, 1024) float32 stored as `array<array<float>>` in Silver

### Silver schema (`utils/tel_schema.py` → `TEL_SILVER_SCHEMA`)

| Column | Type | Notes |
|---|---|---|
| `lap_id` | string | |
| `driver` | string | |
| `season` | int | partition key |
| `event` | string | |
| `session_type` | string | |
| `lap_number` | int | |
| `channels` | array\<array\<float\>\> | shape (6, 1024) — outer=channels, inner=distance bins |
| `label_position` | int (nullable) | race finishing position from Gold; null mid-weekend |

### Output path
```
{TELEMETRY_CLEAN_PATH}/silver/season={year}/
```

---

## Phase 3 — Dataset & Model (`utils/`)

### `utils/tel_dataset.py` — `F1TelemetryDataset`

PyArrow-backed `torch.utils.data.Dataset`. No Spark dependency.

```python
dataset = F1TelemetryDataset(
    silver_path   = SILVER_PATH,
    seasons       = TEL_SEASONS,
    session_types = None,       # None = all sessions (pre-training)
    labelled_only = False,      # True = only laps with race_position (fine-tuning)
)
x, y = dataset[0]
# x: FloatTensor (6, 1024)
# y: 0-indexed class (race_pos - 1), or -1 if unlabelled
```

`label_distribution()` returns `{race_position: count}` for inverse-frequency class weighting in `10_mae_finetune.py`.

### `utils/mae_model.py` — architecture

```
Input (B, 6, 1024)
  │
  PatchEmbedding1D
  │  6 parallel Conv1d(1→384, kernel=16, stride=16) — one per channel
  │  outputs summed → (B, 64, 384)
  │
  LearnedPositionalEmbedding  (B, 64, 384)
  │
  75% random masking → (B, 16, 384) visible patches
  │
  MAEEncoder  6× TransformerBlock(d=384, h=6)
  │           → (B, 16, 384)
  │
  MAEDecoder  project to d=192 → fill mask tokens → 2× TransformerBlock
  │           → pred (B, 48, 16×6=96) patch values
  │
  MSE loss vs. original patchified values (masked patches only)
```

**At fine-tuning / inference**: masking is disabled. Full 64 patches pass through the encoder. The mean-pooled (B, 384) vector feeds `F1PositionHead`.

---

## Phase 3 — MAE Pre-training (`09_mae_pretrain.py`)

### Hyperparameters

| Param | Value | Source |
|---|---|---|
| `d_model` | 384 | `config.ENCODER_HPARAMS` |
| `n_heads` | 6 | `config.ENCODER_HPARAMS` |
| `encoder_layers` | 6 | `config.ENCODER_HPARAMS` |
| `decoder_d_model` | 192 | `config.ENCODER_HPARAMS` |
| `decoder_layers` | 2 | `config.ENCODER_HPARAMS` |
| `patch_stride` | 16 → 64 patches | `config.ENCODER_HPARAMS` |
| `mask_ratio` | 0.75 → 16 visible / 48 masked | `config.ENCODER_HPARAMS` |
| `batch_size` | 128 (K80) / 256 (T4) | `TRAIN_CFG` in notebook |
| `lr` | 1.5e-4 peak, cosine decay | `TRAIN_CFG` |
| `warmup_epochs` | 10 | `TRAIN_CFG` |
| `total_epochs` | 200 | `TRAIN_CFG` |
| `weight_decay` | 0.05 | `TRAIN_CFG` |
| `grad_clip` | 1.0 | in `train_one_epoch()` |

### CE-safe checkpoint pattern

Checkpoint is written **after every epoch** (not just on improvement). On CE cluster restart, `09_mae_pretrain.py` detects the checkpoint and resumes from `epoch + 1` with exact scheduler and optimizer state restored.

```
{BASE_PATH}/models/mae_checkpoint.pt   ← rolling (overwrites each epoch)
{BASE_PATH}/models/mae_train_log.csv   ← appended each epoch; survives restarts
```

### Estimated training time

| GPU | Throughput | Time/epoch | 200 epochs |
|---|---|---|---|
| K80 | ~350 laps/s | ~11 min | ~38 hrs (across CE sessions) |
| T4  | ~900 laps/s | ~4.5 min | ~15 hrs (across CE sessions) |

---

## Phase 4 — Fine-tuning (`10_mae_finetune.py`)

### Two-phase schedule

| Phase | Epochs | Encoder | LR | Purpose |
|---|---|---|---|---|
| Linear probe | 0–9 | Frozen | 1e-3 | Train head only; verify encoder learned useful features |
| Full fine-tuning | 10–49 | Unfrozen | 1e-4 + cosine | End-to-end adaptation |

### Loss

`CrossEntropyLoss` with inverse-frequency class weights (computed via `dataset.label_distribution()`). `ignore_index=-1` handles unlabelled laps safely even if they slip through.

### Evaluation metrics

- **Exact accuracy**: predicted position == actual position
- **Top-3 accuracy**: `|predicted - actual| <= 2` — primary metric (race finishing order is noisy)

### Checkpoint files

```
{BASE_PATH}/models/mae_finetune_checkpoint.pt   ← rolling per-epoch (for CE restart safety)
{BASE_PATH}/models/mae_finetuned.pt             ← best top-3-accuracy snapshot (used by 11)
```

---

## Phase 5 — Inference (`11_mae_predict.py`)

### Input
Silver Parquet filtered to `season=SEASON, event=EVENT`. One prediction per driver using their first (pace-sorted) valid lap.

### Output
Written to the **same `PREDICTIONS_PATH` Parquet table** as `06_predict.py`, with `model_version = 'mae'`. This allows direct comparison in the same table and keeps the dashboard queryable without schema changes.

```sql
-- Query both models for the same event
SELECT driver, model_version, predicted_position, win_probability
FROM pitwall.predictions
WHERE season = 2025 AND event = 'Chinese Grand Prix'
ORDER BY model_version, predicted_position
```

### Comparison block
After writing, `11_mae_predict.py` reads the existing RF/GBT predictions (using Parquet format, `model_version LIKE 'base_r%'`) and prints a side-by-side exact/top-3 accuracy table if post-race Gold data is available.

---

## Storage Layout

```
/Volumes/workspace/default/pitwall/
├── raw/                          ← existing (01_ingest.py output)
├── clean/                        ← existing (02_clean.py output)
├── features/                     ← existing Gold Parquet (03_features.py output)
├── models/
│   ├── qualifying_r##/           ← existing RF/GBT models (05_train.py)
│   ├── base_r##/                 ← existing RF/GBT models (05_train.py)
│   ├── mae_checkpoint.pt         ← rolling pre-training checkpoint (per epoch)
│   ├── mae_finetune_checkpoint.pt← rolling fine-tuning checkpoint (per epoch)
│   ├── mae_finetuned.pt          ← best fine-tuned model (used by 11)
│   └── mae_train_log.csv         ← epoch loss log (append-only)
├── predictions/                  ← Parquet table shared by 06 and 11
└── telemetry/
    ├── raw/                      ← Bronze Parquet (07_tel_ingest.py)
    │   └── season={y}/event={gp}/session={s}/
    │       ├── part-*.parquet
    │       └── _SUCCESS
    └── clean/
        └── silver/               ← Silver Parquet (08_tel_preprocess.py)
            └── season={y}/
```

---

## File Inventory

```
f1-pyspark-analytics/
├── config.py                     ✅ ENCODER_HPARAMS + telemetry paths added
├── FIN_MAE_PLAN.md               ✅ this file
├── notebooks/
│   ├── 01_ingest.py              ✅ unchanged
│   ├── 02_clean.py               ✅ unchanged
│   ├── 03_features.py            ✅ unchanged
│   ├── 04_eda.py                 ✅ unchanged
│   ├── 05_train.py               ✅ unchanged
│   ├── 06_predict.py             ✅ unchanged
│   ├── 07_tel_ingest.py          ✅ implemented
│   ├── 08_tel_preprocess.py      ✅ implemented
│   ├── 09_mae_pretrain.py        ✅ implemented — imports ENCODER_HPARAMS from config
│   ├── 10_mae_finetune.py        ✅ implemented — imports ENCODER_HPARAMS from config
│   └── 11_mae_predict.py         ✅ implemented — Parquet write, correct RF comparison
└── utils/
    ├── __init__.py               ✅ unchanged
    ├── mae_model.py              ✅ implemented
    ├── schema.py                 ✅ unchanged
    ├── spark_session.py          ✅ unchanged
    ├── tel_dataset.py            ✅ implemented (was missing — now created)
    ├── tel_schema.py             ✅ implemented
    ├── transforms.py             ✅ unchanged
    └── weights.py                ✅ unchanged
```

---

## Dependencies

```
# requirements.txt additions needed
scipy    # 08_tel_preprocess.py — distance-axis resampling
torch    # 09, 10, 11 — MAE training and inference
# pyarrow — already ships with Local Python Environment; no install needed
# fastf1  — already in requirements.txt from 01_ingest.py
```

---

## Execution Order on Local/GitHub Actions CE

```
── One-time historical backfill (~2 seasons) ──────────────────────────────────
07_tel_ingest.py        restartable via _SUCCESS flags   ~6–10 hrs across sessions
08_tel_preprocess.py    processes all Bronze at once     ~1–2 hrs

── One-time model training ────────────────────────────────────────────────────
09_mae_pretrain.py      CE-resumable via per-epoch ckpt  ~15–38 hrs across sessions
10_mae_finetune.py      CE-resumable via per-epoch ckpt  ~2–4 hrs

── Per race weekend (same cadence as 01 → 06) ─────────────────────────────────
07_tel_ingest.py        set TEL_SEASONS = [SEASON] in config temporarily
08_tel_preprocess.py    same
11_mae_predict.py       reads mae_finetuned.pt; writes to PREDICTIONS_PATH Parquet
```

> [!TIP]
> For the per-race weekend runs, override `TEL_SEASONS` at the top of the notebook cell rather than modifying `config.py`, to avoid accidentally overwriting the historical backfill flag.
