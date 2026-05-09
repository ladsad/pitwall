# ML Design

## Model
- **Algorithm**: GBTClassifier (Gradient Boosted Trees) via Spark MLlib
- **Pipeline**: VectorAssembler → StringIndexer → GBTClassifier
- **Baseline**: Logistic Regression also trained for comparison
- **Task**: Classify/rank drivers by predicted race finishing position

## Dual sample weighting
Every training row gets a `sample_weight` = recency_weight × session_weight.

### Recency decay
```python
# Lambda controls how fast old data fades. Higher = more aggressive recency bias.
LAMBDA = 0.15
recency_weight = exp(LAMBDA * (round_number - max_round_number))
# e.g. last race = 1.0, 3 rounds ago ≈ 0.64, 10 rounds ago ≈ 0.22
```

### Session type weights
```python
SESSION_WEIGHTS = {
    "R":   1.0,    # Race — ground truth
    "Q":   0.7,    # Qualifying — pure pace
    "SQ":  0.6,    # Sprint qualifying
    "S":   0.5,    # Sprint
    "FP3": 0.35,   # Race sim data
    "FP2": 0.25,   # Mixed programme
    "FP1": 0.15,   # Least representative
}
```

### Final weight
```python
sample_weight = recency_weight * session_weight
```

## Why full retrain each time (not incremental)
Recency weights shift every round. A race from round N-1 becomes further in the past after round N, so its weight must decrease. This can't be applied to an existing model — a full refit with updated weights is required. Data size (~2,400 rows) means retraining takes seconds.

## Confidence scores
- **Win probability**: `probability[1]` from GBTClassifier `.transform()` output
- **Uncertainty band**: bootstrap resampling — train N=20 models on random subsamples, variance in predictions = ±% uncertainty
- Display format: `84% ± 3%` — high variance means treat with caution

## Label
Race finishing position for that round. All session rows for a given round share the same label (the race result from that weekend).

## Handling the evolving field
- Recency decay handles mid-season upgrades implicitly — recent races reflect new car performance
- `pace_trend` feature (engineered): delta between driver's avg pace last 2 races vs previous 4 — captures upgrade momentum
- New regulation era (e.g. 2026): reset training data or add `regulation_era` feature flag
