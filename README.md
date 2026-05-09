
# Pitwall

F1 race prediction pipeline — PySpark · Databricks · MLlib · Next.js

## Stack

- Data: FastF1 → Bronze Parquet → Silver Delta → Gold Delta (Databricks DBFS)
- ML: Spark MLlib GBTClassifier with dual sample weighting (recency × session type)
- Dashboard: Next.js on Vercel, reads `dashboard/public/predictions.json`

## Pipeline

| Notebook       | Phase                              |
| -------------- | ---------------------------------- |
| 01_ingest.py   | Pull FastF1 → Bronze Parquet      |
| 02_clean.py    | Clean → Silver Delta              |
| 03_features.py | Feature engineering → Gold Delta  |
| 04_eda.py      | EDA + correlation analysis         |
| 05_train.py    | Train MLlib GBT → versioned model |
| 06_predict.py  | Predictions → predictions.json    |

## Setup

See docs for Databricks cluster setup and VS Code extension config.
