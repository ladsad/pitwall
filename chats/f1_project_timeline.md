# F1 PySpark Analytics — Detailed Development Timeline & Bug Log

This comprehensive timeline is compiled directly from the development chat logs, tracking granular engineering challenges, infrastructure migrations, and machine learning model iterations.

## Phase 1: Foundation & Codebase Refactoring (May 26, 2026)
- **Codebase Restructuring:** Transitioned away from fragmented notebook scripts into a singular, cohesive pipeline to lock in the methodology for the rest of the F1 season.
- **Data Standardization:** Implemented the Medallion architecture (Bronze/Silver/Gold) for data storage (`/pitwall/telemetry/raw` to `clean/silver`) and debugged initial zero-event aggregation issues.

## Phase 2: Local Migration & Pipeline Hardening (June 21, 2026)
- **Databricks Deprecation:** Identified severe data processing anomalies and limitations in Databricks Community Edition; began migration to a local/online-hybrid architecture for live F1 telemetry retrieval.
- **Supabase Integration:** Set up Supabase as the primary remote database. Configured `.env.local` API keys to handle continuous data ingestion without manual JSON drops.
- **Pipeline Debugging:** Handled Python environment complexities and pushed the foundational codebase migration commits to the `migrate` branch.

## Phase 3: LightGBM Finalization & Deployment Fixes (June 23 - June 24, 2026)
- **Vercel CI/CD Routing:** Configured Vercel deployments to point dynamically to the migration branch to test the dashboard live.
- **Feature Enhancements:** Modified the LightGBM model to store session contribution data, ensuring predictions considered granular driver performance.
- **Historical Backtesting:** Built scripts to automate training and inference across all completed historical races, ensuring the dashboard supported dynamic switching.

## Phase 4: Masked Autoencoder (MAE) Transition (June 24, 2026)
- **Architecture Upgrade:** After maximizing the LightGBM baseline, initiated the design of a Masked Autoencoder (MAE) to extract deeper telemetry representations.
- **Kaggle Training Pipeline:** Shifted data generation and model pre-training to Kaggle to handle the 16GB telemetry dataset. Overcame out-of-memory errors (OOM) and read-only file system issues by restructuring paths to `/kaggle/working/data/`.
- **Model Validation:** Fine-tuned the pre-trained MAE, achieving a best top-3 accuracy of 0.981 over 199 epochs (running for ~5.7 hours).
- **Inference Pipeline:** Updated the `11_mae_predict.py` inference script to push MAE predictions to Supabase for all 2026 races.

## Phase 5: Interpretability & Final Polish (June 24 - June 26, 2026)
- **Model Toggles:** Updated the dashboard UI to allow seamless toggling between the baseline Random Forest model and the new MAE model.
- **Dynamic History Architecture:** Eliminated the heavy backend `race_history` upload scripts by rewriting the dashboard API (`route.js`) to dynamically join real-world actual winners against the model's predictions stored in Supabase on the fly.

## Phase 6: Live Season Execution (June 28, 2026)
- **Continuous Learning:** Scaled up MAE epoch limits (`09_mae_pretrain.py` and `10_mae_finetune.py`) to resume from checkpoints and ingest fresh Austrian Grand Prix telemetry without triggering catastrophic forgetting.
- **Idempotent Ingestion:** Validated that downstream Supabase upserts and PySpark Parquet overwrites inherently prevent duplication when regenerating inference for the active race week.
- **Production Monitoring:** Integrated Vercel Analytics into the Next.js layout to track real-time user engagement, page views, and traffic demographics for the live deployment.

## Phase 7: Baseline Hardening & Live Telemetry Fixes (June 27, 2026)
- **RF Baseline Overhaul:** Fixed a structural flaw in the Random Forest baseline model where it was blind to `team` and `driver` identifiers. Explicitly encoded them using `StringIndexer` to allow the model to recognize inherent car/driver pace disparities and increased maxDepth/numTrees to better capture complex interactions.
- **Glory Run Mitigation:** Re-wrote the prediction aggregation (`06_predict.py`) to use a session-weighted average for win probability instead of a raw maximum, eliminating the anomaly where low-fuel Free Practice "glory runs" unfairly skewed race predictions.
- **Context-Aware Ingestion:** Refactored `07_tel_ingest.py` to automatically detect pipeline modes. It now only ingests the active race weekend (e.g., Austrian GP) during standard runs, drastically cutting down ingestion time while preserving full historical backfill capabilities.

## Phase 8: UI/UX Dashboard Polish (June 27, 2026)
- **Minimalist Branding:** Generated and integrated a scalable, borderless vector-style logo and favicon to match the application's clean, high-tech aesthetic, replacing older generic elements.
- **Deep Learning Telemetry Panel:** Created `MaeTelemetryPanel` to replace the empty session panel when the MAE model is active. This dynamically extracts and displays F1-specific prediction statistics (Win Probability, Uncertainty, Momentum, Expected Finish, and Key Factor).
- **Smooth Transitions:** Implemented `framer-motion`'s `AnimatePresence` to orchestrate clean, sliding crossfade animations when toggling between the Baseline Session Panel and the MAE Telemetry Panel.
- **Styling Enhancements:** Applied custom CSS to ensure a sleek, dark-themed scrollbar and refined dropdown selector padding for better typography and layout breathing room.

---

## 🐛 Comprehensive Bug & Issue Resolution Log

Throughout the project's lifecycle, the following critical issues were identified and successfully resolved:

### Data Pipeline & Infrastructure
1. **Databricks Aggregation Failures:** Telemetry processing was failing silently (0 events added to Gold layer). Resolved by migrating pipeline logic off Databricks and standardizing local PySpark extraction.
2. **Supabase Upsert Rejections:** Intermittent failures writing to Supabase. Fixed by verifying API/Service-role keys in `.env` and correcting schema type definitions for the `race_history` table.
3. **Missing Grand Prix Data:** Identified holes in historical API data (specifically missing Canadian GP round 5 and Barcelona round 7) requiring custom API pulls to patch the database.
4. **Ubuntu APT Misconfiguration:** Encountered broken package managers (`sources.list` entry misspelt for `r2u.stat.illinois.edu`). Bypassed via containerized environment fixes.
5. **Kaggle Cloudflare IP Blocking:** `fastf1` telemetry ingestion crashed (`DataNotLoadedError`) on Kaggle because the F1 Livetime API (via Cloudflare) actively rejects Datacenter IPs. Mitigated by wrapping `fastf1.load` in graceful exception handlers and moving the active weekend ingestion to a local machine environment before uploading it as a dataset to Kaggle.

### Machine Learning (LightGBM, MAE, & RF Baseline)
6. **Target Variable Data Leakage:** Discovered that the LightGBM model was inadvertently feeding predicted future results back into its own features. Refactored the training split to strictly isolate past sequences from future inferences.
7. **MAE Training `NaN` Gradients:** Loss exploded to `NaN` immediately (Epochs 0-23). Diagnosed that 57 out of 70,000+ rows contained `NaN` or `Inf` telemetry. Scrubbed these rows via PySpark transformations and lowered the learning rate to 1.50e-06 to stabilize gradients.
8. **Kaggle OOM & Read-Only Errors:** Attempted to write to `/kaggle/input/` causing read-only (`failed to create symbolic link`) crashes. Further crashed due to memory limits (615 MB output after 721 seconds). Fixed by redirecting all tensor outputs to `/kaggle/working/` and optimizing memory buffers.
9. **Static Feature Importance:** Discovered the model was outputting identical Feature Importance metrics across all drivers and races. Re-engineered the inference script to calculate driver-specific occlusion sensitivity dynamically.
10. **Missing Qualifying Data Handlers:** Drivers without qualifying session data caused model hallucinations. Implemented strict exclusion logic in the pipeline to filter out incomplete drivers.
11. **Random Forest "Glory Run" Hallucinations:** The baseline RF model predicted extremely unlikely winners because it took the maximum win probability across all sessions and was blind to driver/team identity. Fixed by adding categorical splits for driver/team and replacing the max aggregation with a session-weighted average.
12. **VectorAssembler Stage Mismatch:** Pipeline crashed trying to run the Logistic Regression baseline because it shared an assembler with the RF model but lacked the newly added `team_indexer` and `driver_indexer` stages. Re-aligned the `lr_pipeline` to ensure it executed the exact same feature engineering path.

### Frontend Dashboard UI (Next.js)
13. **Zustand 500 Server Crashes:** Next.js deployment crashed entirely (`Status 500`) due to a deprecated Zustand default export. Fixed by replacing `import create` with `import { create } from 'zustand'`.
14. **Z-Index Visual Overlaps:** The win-probability progress bars were being obscured behind other DOM elements. Fixed by correcting CSS z-indexing layers.
15. **Vercel Serverless File Reads:** The dashboard API crashed because Vercel serverless functions could not reliably read `public/predictions.json` via `fs.readFileSync(process.cwd())`. Resolved by fetching the file dynamically over HTTP (`fetch(origin + '/predictions.json')`).
16. **API ReferenceErrors:** A silent `ReferenceError` was crashing the API route's `try...catch` block during fallback execution, resulting in empty Historical Accuracy tables. Fixed by properly scoping variables and decoupling the dynamic history generation from the JSON payload fetch.
