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
| 5 | 2026-05-30 | RoadType NaN imputation (geohash mode), roadtype_was_missing flag, 3 new LOO TEs (RoadType×hour, RoadType×tod_bucket, RoadType×is_weekend) with hierarchical fallback to te_roadtype. Optuna 35-trial search (fold 5 only, lr=0.05). 61 features total. | Full: 79.87 / Ablation: 78.30 (covered 83.3%) | — | REGRESSION vs Run 4. Fold R² (full): 79.02 / 82.94 / 81.60 / 74.84 / 75.57. Two compounding issues: (1) RoadType×time TEs still hurt early folds even with hierarchical fallback (fold training covers only 4 hrs, val covers unseen hours — fallback gives RoadType all-hour avg, not hour-specific avg); (2) Optuna on fold 5 only chose lambda_l1=0.5 + num_leaves=255 that over-regularizes folds 1-4, fold 1 hits 8000 round limit while fold 5 stops at 57 iters. Test max prediction = 0.87 (was 1.0 — Highway demand underpredicted). RoadType_le now 31.8% (down from 56.9%) — imputation + new TEs doing their job conceptually, but net effect negative. Best submission remains Run 4. |
| 6 | 2026-05-30 | Restored Run 4 feature set + Run 4 params (SKIP_OPTUNA=True). Kept RoadType NaN imputation (geohash mode) + roadtype_was_missing flag from Run 5 — dropped the harmful RoadType×time LOO TEs and the Optuna search. 55 features total. | Full: 90.82 / Ablation: 87.68 (covered 83.3%) | 85.04 | Fold R² (full): 87.93 / 94.48 / 92.24 / 85.26 / 89.91. +2.05pt full / +0.95pt ablation vs Run 4. RoadType imputation alone explains most of the gain — filling NaN RoadType from geohash mode restored highway-level signal. RoadType_le back to 60.3% importance. Ablation inflation = +3.15pt (down from +2.04pt in Run 4; larger models needed). Short-lag inflation unchanged. Best submission so far. |
| 7 | 2026-05-30 | **Post-processing calibration** (submission_calibrated.csv): Run 6 model preds × (0.2 × gh_day49_scale + 0.8). Per-geohash day49/day48 scale factor (mean=2.07) estimated from day49 training rows. Alpha=0.2 → multiplier mean=1.22, pred mean 0.101→0.123. | Ablation: 87.68 (no model change) | **86.01** | +1pt over Run 6 raw. Root cause discovered: ALL 5 CV fold models train exclusively on day48 rows (day49 rows only appear in fold 5 VALIDATION). LGB never sees lag_96 as a valid feature in training → can't learn its coefficient. Scale estimated from day49 0:00–2:00 window; may not transfer to test 2:15–13:45 (rush hours). |

---

## Known issues / next steps

- **Best submission so far: Run 7** (LB 86.01, submission_calibrated.csv) — top 100 = 92.26, top 50 = 100.00
- **Root cause of gap**: CV trains exclusively on day48 rows; lag_96 (r=0.792) is NEVER valid in any fold's training set (day49 rows only in fold 5 validation). LGB treats lag_96 as ~constant (fills missing = te_gh_tod_mean). For test (all day49), lag_96 IS valid for 89% of rows but model can't use it.
- **Distribution shift**: scale factor estimated from day49 0:00–2:00 (mean ratio 2.07) may not transfer to test 2:15–13:45 (rush hours may have different scale). Calibration alpha=0.2 gives only +1pt on LB (expected more given day49 OOF +3.82pt).
- OOF covers only 83.3% of train rows — acceptable
- lag_96 linear fit on day49 training: R²=62.8% (slope=1.255, intercept=0.032)
- Stacking (model+lag96 on day49 OOF): R²=92.99% but with negative lag96 coeff → suspect collinearity

**Ready submissions to try (by priority):**
1. **submission_lag96linear.csv** (mean=0.158): lag_96 × 1.255 + 0.032, directly uses actual lag_96 per test row, bypasses scale estimation issue
2. **submission_calib_a05.csv** (mean=0.154): model × (0.5 × scale + 0.5), more aggressive calibration
3. **submission_lag96_direct_v2.csv** (mean=0.182): lag_96 × per-geohash median ratio, R²=83.59% on day49 training
4. **submission_stacked_lag96.csv** (mean=0.112): stacked model+lag96 from day49 OOF

**Priority next improvements:**
1. **Larger model** (num_leaves=255-511, lr=0.02, 6000 rounds) — current underfitting on day48 validation folds
2. **Multi-fold Optuna** (SKIP_OPTUNA=False) — tune structural params; careful not to overfit to day48
3. **Alternative CV scheme**: include some day49 rows in fold training so lag_96 gets used properly
4. **Geohash×timestamp TE** at exact 15-min resolution (more granular than te_gh_hr_mean)
