# Gridlock Hackathon 2.0 — Traffic Demand Prediction

**Competition:** HackerEarth Gridlock Hackathon 2.0 · Phase 1  
**Task:** Predict normalized traffic demand (0–1) at 15-minute intervals across geohash locations  
**Metric:** R² × 100 (higher = better)

---

## Approach

### Model
LightGBM regression (L2 objective) with 5-fold TimeSeriesSplit cross-validation. Training is strictly temporal — validation always follows training in time, mirroring the train/test split.

### Feature Engineering

| Category | Features |
|---|---|
| **Time** | hour, minute, day-of-week, is_weekend, cyclical sin/cos encodings, 5-bucket time-of-day |
| **Spatial** | Geohash decoded to lat/lon; prefix-4/5 region IDs (multi-resolution) |
| **Lags** | Per-geohash lag_1/2/4/96/192 (15-min steps) + rolling mean/std over 4 and 96 steps |
| **Target Encodings (LOO)** | geohash, geohash×hour, geohash×15min-slot, geohash×is_weekend, prefix5×hour, prefix5×tod_bucket, NumberOfLanes×hour, LargeVehicles×hour — **mean and median** |
| **Smoothed TE** | RoadType (smoothed group mean, alpha=10) |
| **Stat features** | Per-geohash training row count (density signal), per-geohash×hour demand std |
| **Interactions** | RoadType×hour, Weather×hour (label-encoded composites) |

### Leakage Prevention
- All target encodings use **exact leave-one-out** (current row excluded from its own group statistic). For n=1 groups, falls back to global mean/median — no self-inclusion.
- Lag features are computed on the combined train+test timeline with **test demand set to NaN**, so no future demand can propagate backward.
- Rolling windows use `shift(1)` before the rolling call — current row never included.

---

## Results

| Run | Description | CV R² (full) | CV R² (ablation) | LB |
|---|---|---|---|---|
| 1 | LightGBM baseline | 64.82 | — | — |
| 2 | + Lag/rolling features | 67.19 | — | — |
| 3 | + LOO target encodings + interaction features | 89.65 | 85.68 | — |
| 4 | + Multi-resolution prefix TEs, tod_bucket, stat features | 88.77 | **86.73** | — |

*Ablation = short lags (lag_1/2/4) removed — honest leaderboard proxy since these are mostly NaN during CV but valid at test time.*

---

## Project Structure

```
scripts/
  features.py   — Feature engineering (base features + target encodings)
  cv.py         — TimeSeriesSplit fold generator
  train.py      — LightGBM training: full + ablation CV, saves fold models
  predict.py    — Loads fold models, averages predictions, writes submission.csv
  verify_lags.py — Leak-free verification for lag features
  eda.py        — Exploratory data analysis
dataset/
  train.csv           — 77,299 rows × 11 columns
  test.csv            — 41,778 rows × 10 columns
  sample_submission.csv
models/             — Saved fold models (fold_0.lgb … fold_4.lgb)
submission.csv      — Final predictions
experiments.md      — Full experiment log
```

## How to Reproduce

```bash
pip install lightgbm scikit-learn pandas numpy pygeohash

# Train (saves models/ and prints CV R²)
python -m scripts.train

# Generate submission
python -m scripts.predict
```

---

## Stack
- Python 3.x, LightGBM, scikit-learn, pandas, numpy, pygeohash
