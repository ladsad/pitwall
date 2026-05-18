# Repository Structure

```
pitwall/
├── config.py                          # Central config — season, paths, lambda, session weights
├── run_pipeline.py                    # Master notebook — chains all stages via %run
│
├── notebooks/
│   ├── 01_ingest.py                   # Pull FastF1 (all session types) → Bronze Parquet
│   ├── 02_clean.py                    # Clean, cast types, filter outliers → Silver Delta
│   ├── 03_features.py                 # Engineer per-driver per-session features → Gold Delta
│   ├── 04_eda.py                      # Spark SQL analysis, correlation checks
│   ├── 05_train.py                    # Dual-weighted MLlib Pipeline training → versioned model
│   └── 06_predict.py                  # Predictions + confidence scores → predictions.json
│
├── utils/
│   ├── spark_session.py               # SparkSession helper (works locally + on Databricks)
│   ├── schema.py                      # All schema definitions (never infer schema)
│   ├── transforms.py                  # Reusable DataFrame transformations
│   └── weights.py                     # Recency decay + session type weight computation
│
├── dashboard/                         # Next.js app
│   ├── app/
│   │   └── page.js                    # Main dashboard page
│   ├── components/                    # React components (charts, driver cards, etc.)
│   ├── public/
│   │   └── predictions.json           # ← pipeline writes here, Vercel reads this
│   ├── styles/
│   └── package.json
│
├── requirements.txt                   # Python deps: fastf1, pyspark, etc.
└── README.md
```

## Key conventions
- Never use `spark.read.csv(..., inferSchema=True)` — always define schema in `utils/schema.py`
- All notebooks are independently runnable — no hidden state dependencies between them
- Model paths always include version: `/Volumes/workspace/default/pitwall/models/base_r05/`
- `predictions.json` is the single handoff point between Databricks and the dashboard
