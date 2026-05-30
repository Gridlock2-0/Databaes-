# Experiment Log — Gridlock Hackathon 2.0

## Format
| # | Date | Description | CV R² (×100) | LB Score | Notes |
|---|------|-------------|-------------|----------|-------|

---

## Runs

| # | Date | Description | CV R² | LB Score | Notes |
|---|------|-------------|--------|----------|-------|
| 1 | 2026-05-30 | LightGBM baseline — 5-fold TimeSeriesSplit. Features: hour/minute/day cyclical + geohash lat/lon + prefix5 + label-encoded cats + TE(geohash, geohash×hour, RoadType). num_leaves=127, lr=0.05, early_stop=100. | 64.82 (covered 83.3% of rows) | — | Fold R²: 63.49 / 73.76 / 61.57 / 47.31 / 68.47. Fold 4 (14:45–22:00) worst. Early stopping at 120–255 iters. |
| 2 | 2026-05-30 | Add lag/rolling features — lag_1/2/4/96/192 + roll_mean/std(4,96) per geohash. Exact-time merge for lags, shift(1)+rolling for windows. Test demand=NaN prevents cross-test leakage. Same LightGBM params as run 1. | 67.19 (covered 83.3% of rows) | — | Fold R²: 61.71 / 79.80 / 61.93 / 46.90 / 75.97. +2.4pt gain. lag_1 r=0.971, lag_96 r=0.792 in train. lag_96/192 are NaN in CV folds (no day47 data) but valid for all test rows — actual LB gain may exceed CV gain. Early stopping at 54–121 iters, models still small. |
| 3 | 2026-05-30 | LOO target encodings (geohash, geohash×hour, geohash×tod — mean+median) + interaction features (RoadType×hour, Weather×hour). Fixed exact LOO median (n=1 groups use global_median, not own y_i). Ablation run removes lag_1/2/4 for honest LB proxy. | Full: 89.65 / Ablation: 85.68 (covered 83.3%) | — | Fold R² (full): 79.56 / 95.60 / 92.07 / 84.29 / 92.09. +22.5pt jump from Run 2. Top features: RoadType_le (55.8%), te_gh_hr_mean (16.3%), te_roadtype (9.4%), lag_1 (6.8%). Ablation honest LB proxy = 85.68; short-lag inflation = +3.97pt. Models now large (up to 2892 iters). |
| 4 | 2026-05-30 | Stronger location signal: prefix4 region categorical + tod_bucket (5-bucket time), 5 new LOO TEs (prefix5×hour, prefix5×tod_bucket, geohash×is_weekend, NumberofLanes×hour, LargeVehicles×hour), stat features gh_count + gh_hr_std. 54 features total. | Full: 88.77 / Ablation: 86.73 (covered 83.3%) | — | Fold R² (full): 80.84 / 94.95 / 90.54 / 83.79 / 89.10. Ablation +1.05pt vs Run 3 (honest proxy improved). Full -0.88pt vs Run 3 — models stopping early (fold 5: 107 iters). RoadType_le still 56.9%, te_gh_hr_mean 18.7%. New features show modest gain; neighbourhood TEs help in ablation more than with lags. Short-lag inflation dropped to +2.04pt. |

---

## Known issues / next steps

- OOF covers only 83.3% of train rows (first fold's train rows have no OOF preds) — acceptable
- Fold 4 (14:45–22:00) consistently worst; test is 02:15–13:45, more similar to folds 1–3
- lag_96/lag_192 are NaN for 91.7%/100% of train rows; all test rows have valid lag_96 — LB gain may exceed CV gain
- RoadType_le dominating at ~57% importance — may be a proxy for a latent variable (location type), not necessarily miscalibrated
- Models stopping early (< 500 iters on most folds) despite 3000 round budget — early stopping at 100 is too tight; models are under-fit
- Top priority improvements:
  1. **Tune LightGBM**: lower lr (0.02), more rounds (5000), larger num_leaves (255), WIDER early stopping (200-300) — current models clearly under-fit
  2. **Submit Run 3 or 4** — ablation honest LB = 85.68/86.73; calibrate CV-to-LB gap before further tuning
  3. **Seed averaging** (3–5 seeds) for cheap R² stability
  4. **Add XGBoost / CatBoost** for ensemble
  5. **Log1p target transform** experiment (demand skewed)
