# Gridlock Hackathon 2.0 — Traffic Demand Prediction

**Competition:** HackerEarth Gridlock Hackathon 2.0 · Phase 1  
**Task:** Predict normalized traffic demand (0–1) at 15-minute intervals across geohash locations  
**Metric:** R² × 100 (higher = better)  
**Best LB Score:** 86.01 (`submission_calibrated.csv`)

---

## Approach

### Model
LightGBM regression (L2 objective) with 5-fold `TimeSeriesSplit` cross-validation. Training is strictly temporal — validation always follows training in time, mirroring the train/test split.

**Key hyperparameters:** `num_leaves=127`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `min_child_samples=20`, early stopping at 100 rounds.

### Feature Engineering (57 features)

| Category | Features | Count |
|---|---|---|
| **Time** | hour, minute, time_of_day (0–95 slots), day_mod7, is_weekend, tod_bucket, sin/cos cyclicals for hour, tod, day-of-week | 13 |
| **Spatial** | lat, lon (decoded from geohash), geohash_le, prefix4_le, prefix5_le | 5 |
| **Road/traffic** | RoadType_le, NumberofLanes, LargeVehicles_le, Landmarks_le, Weather_le, Temperature, roadtype_hour_le, weather_hour_le, roadtype_was_missing | 9 |
| **Lag features** | lag_1/2/4 (short-term), lag_96 (24h same time yesterday), lag_192 (48h) | 5 |
| **Rolling** | roll_mean/std × windows 4 and 96 per geohash | 4 |
| **LOO Target Encodings** | geohash, geohash×hour, geohash×tod, geohash×is_weekend, prefix5×hour, prefix5×tod_bucket, NumberofLanes×hour, LargeVehicles×hour — **mean + median** | 16 |
| **Smoothed TE** | RoadType (Bayesian smoothed, alpha=10) | 1 |
| **Stat features** | gh_count (data density), gh_hr_std (demand variability per geohash×hour) | 2 |
| **Cross-day** | lag96_dev (lag_96 − te_gh_tod_mean), gh_day49_scale (day49/day48 demand ratio) | 2 |

### Leakage Prevention
- All LOO target encodings **exclude the current row** from its group statistic. n=1 groups fall back to the global mean — no self-inclusion.
- Lags computed on combined train+test timeline with **test demand = NaN** — no future values leak backward.
- Rolling windows use `shift(1)` before the rolling call.

### Post-processing Calibration
Multiplying model predictions by a per-geohash day49/day48 scale factor (alpha=0.2) corrects for the Sunday→Monday distribution shift, giving the best LB:

```
pred_calibrated = pred_raw × (0.2 × gh_day49_scale + 0.8)
```

---

## Results

| Run | Description | CV R² (full) | CV R² (ablation) | LB |
|---|---|---|---|---|
| 1 | LightGBM baseline | 64.82 | — | — |
| 2 | + Lag/rolling features | 67.19 | — | — |
| 3 | + LOO target encodings | 89.65 | 85.68 | — |
| 4 | + Prefix TEs, tod_bucket, stat features | 88.77 | 86.73 | — |
| 5 | + RoadType×time TEs (regression) | 79.87 | 78.30 | — |
| 6 | Restored Run 4 + RoadType NaN imputation | 90.82 | 87.68 | 85.04 |
| 7 | Run 6 + cross-day calibration (alpha=0.2) | 90.82 | **87.68** | **86.01** ← best |

*Ablation = lag_1/2/4 removed — honest LB proxy (short lags inflate CV by ~3pt since they're mostly NaN at test time).*

**What did NOT help:** XGBoost/CatBoost ensembles (error correlation 0.937 with LGB), seed averaging, autoregressive lag filling, lag_96 post-processing (scale estimated from 0:00–2:00 doesn't transfer to rush hours), global RoadType×time TEs (training/validation inconsistency cost ~4pt), num_leaves=255+ (overfits small early folds).

---

## Project Structure

```
scripts/
  features.py              — Feature engineering (base features + LOO target encodings)
  cv.py                    — TimeSeriesSplit fold generator
  train.py                 — LightGBM training: full + ablation CV, saves fold models
  predict.py               — Loads fold models, averages predictions, writes submission.csv
  eda.py                   — Exploratory data analysis
dataset/
  train.csv                — 77,299 rows × 11 columns
  test.csv                 — 41,778 rows × 10 columns
  sample_submission.csv
models/
  fold_0.lgb … fold_4.lgb  — Trained fold models (ready to use)
  oof_preds.npy             — Out-of-fold predictions (full)
  oof_ablation_preds.npy    — Out-of-fold predictions (ablation)
  feat_importance.npy       — Mean gain importances
gridlock_solution.ipynb    — Clean end-to-end notebook reproducing the best submission
submission.csv             — Raw model predictions (LB 85.04)
submission_calibrated.csv  — Calibrated predictions (LB 86.01) ← SUBMIT THIS
experiments.md             — Full experiment log
```

---

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
Competition data is not in the repo. Download from HackerEarth and place here:
```
dataset/
  train.csv
  test.csv
  sample_submission.csv
```

### 4. Generate predictions

**Option A — Use pre-trained models** (fastest, no retraining needed):
```bash
python -m scripts.predict
# Writes submission.csv (raw model, LB 85.04)
```

Then apply calibration to get the best submission:
```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from scripts.features import build_base_features, compute_day49_scale

train_df    = pd.read_csv('dataset/train.csv')
test_df     = pd.read_csv('dataset/test.csv')
train_feat, test_feat, _, _ = build_base_features(train_df, test_df)
day49_scale = compute_day49_scale(train_feat, train_feat['demand'].values)

sub   = pd.read_csv('submission.csv')
scale = test_feat['geohash'].map(day49_scale).fillna(float(day49_scale.mean())).values
sub['demand'] = np.clip(sub['demand'] * (0.2 * scale + 0.8), 0.0, 1.0)
sub.to_csv('submission_calibrated.csv', index=False)
# → LB 86.01 — best submission
```

**Option B — Retrain from scratch** (~3 minutes):
```bash
python -m scripts.train
# Saves fold_0.lgb … fold_4.lgb, prints CV R²

python -m scripts.predict
# Writes submission.csv
```

**Option C — Run the notebook** (full reproducible pipeline):
```bash
jupyter notebook gridlock_solution.ipynb
# Run all cells top-to-bottom — generates both submissions
```

---

## File Reference

| File | Purpose |
|---|---|
| `scripts/features.py` | All feature engineering — base features, lags, LOO target encodings, cross-day features |
| `scripts/cv.py` | TimeSeriesSplit fold generator |
| `scripts/train.py` | Trains LightGBM 5-fold CV, runs full + ablation, saves models |
| `scripts/predict.py` | Loads fold models, generates `submission.csv` |
| `scripts/eda.py` | Exploratory data analysis |
| `gridlock_solution.ipynb` | End-to-end notebook: data → features → training → submission |
| `models/fold_*.lgb` | Trained LightGBM fold models (57 features each) |
| `experiments.md` | Full log: every run, what changed, CV R², LB score, notes |
| `submission.csv` | Raw model output (LB 85.04) |
| `submission_calibrated.csv` | Calibrated output — **best submission (LB 86.01)** |

---

## Stack
Python 3.x · LightGBM · scikit-learn · pandas · numpy · pygeohash · optuna
