# Architecture

## Medallion layers
- **Bronze** — raw FastF1 data, saved as Parquet exactly as received. Partitioned by season/event/session.
- **Silver** — cleaned data in Delta Lake. Nulls removed, timedeltas converted, outlier laps filtered (107% rule).
- **Gold** — feature store. One row per driver per session per weekend. ML-ready aggregated features.

## DBFS storage layout
```
dbfs:/pitwall/raw/season=…/event=…/session=…/   ← Bronze Parquet
dbfs:/pitwall/clean/                              ← Silver Delta
dbfs:/pitwall/features/                           ← Gold Delta
dbfs:/pitwall/models/base_r{N}/                  ← trained MLlib Pipeline
dbfs:/pitwall/models/qualifying_r{N}/            ← mid-weekend model
dbfs:/pitwall/predictions/                        ← output tables
```

## Training data structure
- **One row per driver per session per weekend** (not per lap, not per weekend)
- ~20 weekends × 20 drivers × 5-6 sessions = ~2,000–2,400 rows per season
- Label = race finishing position for that weekend (always from the Race session)
- Features = per-session metrics (lap delta, consistency, sector times, etc.)

## Model versioning lifecycle
```
base_r{N}          ← trained after round N race completes. Foundation for next weekend.
qualifying_r{N}    ← base + this weekend's FP + Quali data. Used for pre-race predictions.
                      Superseded by base_r{N+1} once race finishes.
```
Post-race model IS the new base. No separate "post-race" version.

## Dashboard data flow
```
Databricks pipeline
    → exports predictions.json
    → committed to dashboard/public/ in GitHub repo
    → Vercel detects push → auto-redeploys
    → pitwall.vercel.app updates in ~30 seconds
```
