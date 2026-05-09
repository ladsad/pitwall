# Data Schema

## Bronze — raw lap data (from FastF1)
All timedelta columns converted to float seconds before saving.

| Column | Type | Notes |
|---|---|---|
| Driver | String | Three-letter code e.g. VER, HAM |
| DriverNumber | String | Car number |
| Team | String | Constructor name |
| LapNumber | Integer | |
| LapTime | Float | Seconds |
| Sector1Time | Float | Seconds |
| Sector2Time | Float | Seconds |
| Sector3Time | Float | Seconds |
| SpeedI1 | Float | km/h — speed trap 1 |
| SpeedI2 | Float | km/h — speed trap 2 |
| SpeedFL | Float | km/h — finish line |
| SpeedST | Float | km/h — speed trap |
| Compound | String | SOFT / MEDIUM / HARD / INTER / WET |
| TyreLife | Float | Laps on current set |
| FreshTyre | String | True/False |
| PitInTime | Float | Seconds (null if no pit) |
| PitOutTime | Float | Seconds (null if no pit) |
| IsPersonalBest | String | True/False |
| session_type | String | FP1/FP2/FP3/Q/SQ/S/R — added on ingest |
| season | Integer | Partition column |
| event | String | Partition column |
| session | String | Partition column |

## Gold — feature store (one row per driver per lap, all sessions)

| Column | Type | Notes |
|---|---|---|
| driver | String | |
| team | String | |
| round_number | Integer | Race round in season |
| event | String | |
| season | Integer | |
| session_type | String | FP1/FP2/FP3/Q/SQ/S/R |
| lap_time_delta | Float | Delta to session best (seconds) |
| consistency_score | Float | Std dev of lap times |
| best_sector_combo | Float | Theoretical best lap |
| tyre_deg_rate | Float | Pace loss per lap |
| pace_vs_teammate | Float | Delta to teammate best |
| pace_trend | Float | Last 2 rounds avg vs previous 4 |
| race_position | Integer | **Label** — race finishing position that weekend |
| recency_weight | Float | Computed by utils/weights.py |
| session_weight | Float | Computed by utils/weights.py |
| sample_weight | Float | recency_weight × session_weight |

## predictions.json structure
```json
{
  "model_version": "qualifying_r06",
  "generated_at": "2025-05-10T14:32:00Z",
  "event": "Monaco Grand Prix",
  "round": 6,
  "predictions": [
    {
      "driver": "VER",
      "team": "Red Bull Racing",
      "win_probability": 0.42,
      "uncertainty": 0.03,
      "predicted_position": 1
    }
  ]
}
```
