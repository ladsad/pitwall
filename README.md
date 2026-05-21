
# Pitwall

F1 race prediction pipeline — PySpark · Databricks · MLlib · Next.js

**Live dashboard**: [pitwall-f1-six.vercel.app](https://pitwall-f1-six.vercel.app)

## Stack

- **Ingestion**: FastF1 Python API → Bronze Parquet (DBFS Unity Catalog Volume)
- **Processing**: PySpark cleaning → Silver Delta → Gold Delta with engineered features
- **ML**: Spark MLlib `RandomForestClassifier` (100 trees, depth 4) with dual sample weighting (recency decay × session type) + Logistic Regression baseline
- **Uncertainty**: Bootstrap resampling (N=20, 80% fraction) per driver
- **Storage**: Databricks Unity Catalog — `workspace.default.pitwall` volume
- **Dashboard**: Next.js on Vercel, driven by `dashboard/public/predictions.json`

## Pipeline

| Notebook         | Phase                                                              |
| ---------------- | ------------------------------------------------------------------ |
| `01_ingest.py`   | Pull FastF1 → Bronze Parquet (raw laps + results)                 |
| `02_clean.py`    | Validate & clean → Silver Delta                                   |
| `03_features.py` | Feature engineering → Gold Delta (lap-level, partitioned by event)|
| `04_eda.py`      | EDA + correlation analysis                                         |
| `05_train.py`    | Train RF + LR baseline → versioned model saved to DBFS            |
| `06_predict.py`  | Predictions + bootstrap uncertainty → predictions.json            |

## Features (Gold Layer)

| Feature            | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `lap_time_delta`   | Driver lap time vs. their own session minimum            |
| `consistency_score`| Intra-session lap time std dev                           |
| `best_sector_combo`| Sum of personal best S1 + S2 + S3 per session           |
| `tyre_deg_rate`    | Lap time degradation slope per tyre stint                |
| `pace_vs_teammate` | Driver best lap vs. team-mate best lap per session       |
| `pace_trend`       | Recent 2-round avg delta minus prior 4-round avg delta   |
| `compound`         | Tyre compound (categorical, StringIndexer encoded)       |

## Model Versioning

- **`qualifying_rNN`** — trained before race results are available (FP + Quali data only)
- **`base_rNN`** — retrained after race results land (full labelled set)

Predict notebook auto-selects the best available version for the current round.

## Sample Weighting

```
sample_weight = recency_weight × session_weight

recency_weight  = exp(-λ × rounds_ago),  λ = 0.15
session_weights = R:1.0  Q:0.7  SQ:0.6  S:0.5  FP3:0.35  FP2:0.25  FP1:0.15
```

## Output Schema (`predictions.json`)

```
model_version, generated_at, event, round, season,
sessions_used, season_accuracy, recency_lambda,
predictions[]: driver, team, predicted_position,
               win_probability, uncertainty, trend, sessions{},
feature_importance[], history[]
```

## Setup

1. Clone repo, connect VS Code to Databricks workspace (`databricks.yml` pre-configured for `dbc-8cc171c2-6c4e.cloud.databricks.com`)
2. Run notebooks in order: `01 → 02 → 03 → 04 → 05 → 06`
3. Push `dashboard/public/predictions.json` → Vercel auto-deploys within ~30 seconds

```
pip install -r requirements.txt
```

---

## Roadmap

### Phase 1 — Automated Race-Weekend Pipeline (Databricks Jobs)
- Schedule Databricks Jobs to trigger each notebook automatically as the race weekend progresses (FP1 → FP2 → FP3 → Quali → Sprint → Race)
- Post-race retraining job fires when `race_position` labels land in the results store
- Git-synced `predictions.json` push via Databricks Repos API or GitHub Actions, eliminating the manual push step
- Add a `prod` target to `databricks.yml` with job definitions for the full pipeline DAG

### Phase 2 — Richer Driver Profiles & Extended Lookback
- Extend `pace_trend` lookback beyond 6 rounds — use full season history and multi-season career data
- Build **driver profile features**: career avg finishing position by circuit type (street, high-speed, technical), wet-weather performance index, safety car restart history, pitstop consistency
- Add **team/car performance trajectory**: constructors' championship momentum, recent upgrade cadence
- Include **qualifying-to-race delta** as a feature (grid position vs. predicted race pace)
- Incorporate **circuit DNA** embeddings (track characteristics matched to driver strengths)

### Phase 3 — Continuously Learning Transformer / LLM Model
- Explore a sequence model (Transformer encoder) treating each driver's lap sequence as a time series, enabling attention over sector splits and stint patterns within a session
- Pre-train on historical F1 lap data (2018–present via FastF1) to learn general racing dynamics; fine-tune per-weekend
- Investigate online/incremental learning: update model weights as each session completes during a race weekend, without full retraining
- Evaluate against the RF baseline using calibrated probability metrics (Brier score, log-loss) in addition to top-3 accuracy
- Potential integration with an LLM layer for natural-language race commentary and prediction explanation (e.g. "Verstappen is favoured due to a 3-session pace trend improvement on this circuit type")
