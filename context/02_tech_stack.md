# Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.10+ | Primary language |
| Compute | Apache Spark / PySpark | Distributed processing |
| Platform | Local + Google Colab | Free managed Spark + Parquet |
| Storage | Local Filesystem + Parquet (Bronze) + Parquet (Silver/Gold) | Medallion architecture |
| Data source | FastF1 Python library | Free, live + historical F1 timing |
| ML framework | Spark MLlib | GBTClassifier, MLlib Pipeline API |
| IDE | VS Code + Local/GitHub Actions extension | Local edit, remote cluster execution |
| Version control | Git + GitHub (pitwall repo) | Also triggers Vercel dashboard deploy |
| Dashboard | Next.js + Recharts + Tailwind | Custom F1-style UI |
| Dashboard hosting | Vercel (free Hobby plan) | Auto-deploys on GitHub push |
| Future: weather | OpenWeatherMap API (free tier) | Rain/temp/wind per circuit |

## Key constraints
- Everything must be **free** — no paid tiers anywhere
- Local + Google Colab cluster auto-terminates after 2 hours idle
- Data is saved to Local Filesystem as Parquet/Parquet, not kept in memory
- Dashboard reads a static `predictions.json` file — no live DB connection needed
