# Pipeline Phases

## Phase 1 — Ingestion (`01_ingest.py`)
- **Input**: FastF1 API
- **Sessions ingested**: FP1, FP2, FP3, Qualifying, Sprint Qualifying, Sprint, Race — ALL types
- **Output**: `dbfs:/pitwall/raw/season=…/event=…/session=…/` (Parquet)
- **Key steps**:
  - `fastf1.get_session(year, event, session)` + `session.load()`
  - `pick_quicklaps()` to filter telemetry-broken laps at API level
  - Convert timedelta columns to float seconds (PySpark can't read timedeltas)
  - Add `session_type` column — flows through all downstream stages
  - Define schema explicitly — never infer
  - Write partitioned Parquet to Bronze

## Phase 2 — Cleaning (`02_clean.py`)
- **Input**: Bronze Parquet
- **Output**: `dbfs:/pitwall/clean/` (Silver Delta)
- **Key steps**:
  - Drop rows with nulls in critical columns (LapTime, Driver, Team)
  - Filter laps > 107% of session best (F1 qualifying rule — removes unrepresentative laps)
  - Cast and validate all column types
  - Preserve `session_type` column
  - Write Silver Delta table

## Phase 3 — Feature Engineering (`03_features.py`)
- **Input**: Silver Delta
- **Output**: `dbfs:/pitwall/features/` (Gold Delta)
- **Structure**: One row per driver per lap — full lap granularity preserved. Features are lap-level metrics. Dual weights applied at training time, not here.
- **Features engineered**:
  - `lap_time_delta` — delta to session best lap (seconds)
  - `consistency_score` — std deviation of driver's lap times that session
  - `best_sector_combo` — theoretical best lap from best individual sectors
  - `tyre_deg_rate` — pace loss per lap on same tyre set
  - `pace_vs_teammate` — delta to teammate's session best
  - `pace_trend` — avg pace last 2 rounds vs previous 4 rounds (captures upgrade momentum)
  - `session_type` — carried through for weighting

## Phase 4 — EDA (`04_eda.py`)
- **Input**: Gold Delta
- Spark SQL correlation analysis
- Validate which session types have strongest correlation to race results
- Circuit-specific patterns
- Team vs driver pace gap analysis

## Phase 5 — Training (`05_train.py`)
- **Input**: Gold Delta + race results as labels
- **Output**: `dbfs:/pitwall/models/base_r{N}/` or `qualifying_r{N}/`
- **Steps**:
  - Compute `sample_weight` via `utils/weights.py` (recency × session_type)
  - Build MLlib Pipeline: VectorAssembler → StringIndexer → GBTClassifier(weightCol="sample_weight")
  - Train/validation split
  - Save versioned Pipeline to DBFS

## Phase 6 — Prediction + Export (`06_predict.py`)
- **Input**: Trained model + Gold features for current weekend
- **Output**: `dbfs:/pitwall/predictions/` + `dashboard/public/predictions.json`
- **Steps**:
  - Load versioned model (base or qualifying)
  - `.transform()` → extract `probability[1]` as win confidence
  - Bootstrap resampling (N=20) → compute ±% uncertainty band
  - Export ranked JSON → commit to GitHub → Vercel auto-redeploys
