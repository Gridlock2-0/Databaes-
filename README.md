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

## Setup for Teammates

### 1. Clone the repo
```bash
git clone https://github.com/Gridlock2-0/Databaes-.git
cd Databaes-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
The dataset is not in the repo (competition data). Get the CSVs from HackerEarth and place them here:
```
dataset/
  train.csv
  test.csv
  sample_submission.csv
```

### 4. You now have two options:

**Option A — Just generate predictions** (models are already trained and committed):
```bash
python -m scripts.predict
# Writes submission.csv
```

**Option B — Re-train from scratch** (takes ~5–6 minutes):
```bash
python -m scripts.train
# Saves fold_0.lgb … fold_4.lgb into models/
# Prints CV R² for full + ablation runs

python -m scripts.predict
# Writes submission.csv
```

### 5. Verify lag features are leak-free (optional)
```bash
python -m scripts.verify_lags
```

---

## File Reference

| File | What it does |
|---|---|
| `scripts/features.py` | All feature engineering — base features, lags, LOO target encodings |
| `scripts/cv.py` | TimeSeriesSplit fold generator |
| `scripts/train.py` | Trains LightGBM with 5-fold CV, runs ablation, saves models |
| `scripts/predict.py` | Loads fold models, averages predictions, writes `submission.csv` |
| `scripts/verify_lags.py` | 4-check proof that lag features have no data leakage |
| `scripts/eda.py` | Exploratory data analysis |
| `models/fold_*.lgb` | Trained fold models (ready to use — no re-training needed) |
| `experiments.md` | Log of every experiment: what changed, CV R², notes |
| `submission.csv` | Current best submission file |

---

## Stack
- Python 3.x, LightGBM, scikit-learn, pandas, numpy, pygeohash, pytorch
