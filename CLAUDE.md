# CLAUDE.md — Gridlock Hackathon 2.0 · Phase 1: Traffic Demand Prediction

## Mission
Maximize the leaderboard score on the **Traffic demand prediction** regression problem with a clean, reproducible pipeline we can defend and build on in later phases.
Targets: clear the ~92.7 R² cluster, then contend for #1 (current top ≈ 93.13).

---

## The metric — read this first, it drives every decision
```
score = max(0, 100 * r2_score(actual, predicted))     # sklearn R²
```
- The score is **R² × 100**. R² = 1 − SSE/SST, so **maximizing R² is exactly equivalent to minimizing mean squared error (MSE).**
- ⇒ **Train with a plain squared-error objective** (LightGBM `objective="regression"`, XGBoost `reg:squarederror`, CatBoost `RMSE`). The default L2 loss optimizes the metric directly. No custom loss needed.
- R² is dominated by your **largest errors** (they're squared). On demand data that's the **high-traffic peaks** — getting busy geohashes / rush hours right is worth far more than shaving error on quiet rows. Always inspect residuals on the top-demand decile.
- The leaderboard is tight: top ≈ 0.9313, cluster ≈ 0.9274 (~0.004 apart). Wins come from small, *real* RMSE reductions ⇒ **a CV that tracks the leaderboard is our single most important asset.**
- **Always score CV on the real metric**: `r2_score` in the **original demand scale**, never on a transformed target.

---

## Data
- Location: **`./dataset/`** (all three CSVs live here)
- `train.csv` — 77,299 × 11 (10 features + target)
- `test.csv` — 41,778 × 10 (features only)
- `sample_submission.csv` — format template (5-row preview; the real submission must be 41,778 × 2)
- **Target:** `demand` — continuous float in **[0, 1]** (already normalised). Clip predictions to [0, 1], not just ≥ 0.
- **Full feature columns (confirmed by EDA):** `Index` (id), `geohash` (6-char, all start "qp"), `day` (integer ordinal 48/49 — NOT a calendar date), `timestamp` (H:MM, 15-min intervals), `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature` (float, 3.2% missing), `Weather`
- **`Temperature`** is the hidden 11th column (float, continuous). Missing values: Temperature 3.2%, Weather 1.0%, RoadType 0.8% — impute with train median.
- `day` is an integer ordinal; only values 48 and 49 appear. Do NOT parse as calendar date. Use `day % 7` as day-of-week proxy.
- Train/test split: strictly temporal. Train = day 48 (full) + day 49 (0:00–2:00). Test = day 49 (2:15–13:45).

---

## Submission format (exact)
- Two columns: `Index`, `demand` — use the **exact names** from `sample_submission.csv`.
- Exactly **41,778 rows**, one per test `Index`, in the test file's index order.
- `.csv` only. **Clip predictions to ≥ 0** (demand can't be negative).
- The judge also requires uploading your source code as **`.ipynb`** (+ optional presentation). Keep a clean final notebook that reproduces the best submission top-to-bottom.
- **50 submissions total** for the whole challenge — treat them as scarce.

---

## Workflow — do these in order, STOP after each for review
1. **EDA.** Shapes, dtypes, missing values, full column list. Profile `demand` (skew? zeros? integer or continuous? max?). Parse `day` + `timestamp` → real datetime and show samples. Profile `geohash` (length, # unique, prefix clustering, decode one to lat/long). **Characterize the train/test split** — temporal (test = future window) or random? Are test geohashes present in train? This decision dictates the CV scheme. Output a short findings summary + recommended CV.
2. **Validation harness.** Build a **leak-free CV that mirrors the train/test split**: time-based fold if temporal; `GroupKFold` on geohash if rows repeat per location and test reuses locations; plain `KFold` only if rows are independent. Print folds. Wire a function that returns OOF R². Do not proceed until CV is sane and stable.
3. **Baseline.** LightGBM (L2 objective, RMSE eval), minimal features (raw numerics + label-encoded categoricals + basic datetime parts). Report CV R², save OOF preds + a submission. Submit once to calibrate the **CV↔LB** relationship.
4. **Feature engineering — one batch at a time, re-run CV after each, keep only what helps.** (see below)
5. **Models + ensemble.** Tune LightGBM; add XGBoost and CatBoost (CatBoost with native categoricals); blend (simple average / CV-weighted, then consider stacking). Add **seed averaging** (3–5 seeds) for cheap R² stability.
6. **Post-processing.** Clip to ≥ 0 (test clipping to train max too); test rounding if `demand` is integer-valued. Accept only if CV R² improves.

---

## Feature engineering for THIS data
- **Datetime:** parse `day` + `timestamp` → datetime. Extract hour, minute, day-of-week, is_weekend, day-of-month, month. **Cyclical-encode** hour & day-of-week (sin/cos). Demand is almost certainly periodic (rush hours, weekday/weekend) — these are strong.
- **Geohash (spatial):** decode → latitude/longitude numeric features (`pygeohash` / `python-geohash`; fallback: integer-encode the geohash + its prefixes if the lib is unavailable). Geohash **prefixes** encode nested regions — build region IDs at several prefix lengths (e.g. first 4, 5, 6 chars) for grouping/encoding.
- **Target / aggregation encoding (leak-free, out-of-fold):** mean/median `demand` per geohash, per geohash×hour, per geohash×day-of-week, per RoadType, per region-prefix. Compute **inside CV folds** (fit on train part only) or via smoothed OOF target encoding to avoid leakage. Usually the **single biggest lever** on spatial-temporal demand.
- **Lags / rolling (only if temporal & repeated per location):** if each geohash is observed over a time sequence, add lagged demand + rolling mean/std per geohash. Powerful — but valid only if computable for test from past data. Verify against the split before using.
- **Categoricals:** `RoadType`, `Weather`, `LargeVehicles`, `Landmarks` — CatBoost handles natively; for LightGBM use `categorical_feature` or target/frequency encoding. `NumberofLanes` numeric. Try interactions like RoadType×hour, Weather×hour.

---

## Target-transform experiment
`demand` is likely right-skewed. Run **both** and pick by CV R² (measured in original scale):
(a) train on raw `demand`; (b) train on `log1p(demand)`, predict, `expm1`.
Log-target optimizes relative error and can lift R² on skewed targets — but it can also hurt. **Let CV decide; don't assume.**

---

## Submission discipline
- **Decide what to submit from CV, not vibes.** 50 submissions is the whole budget.
- Log every experiment in `experiments.md`: what changed, CV R², and (when submitted) the LB score. This table is how we navigate.
- Watch the CV↔LB gap. If Phase-1 selection may re-score on a private split, **trust CV over the public LB** and don't tune to the public number.
- Track two end-state candidates: best-CV and best-LB. If they diverge, prefer **best-CV** unless you understand exactly why.

---

## Engineering hygiene
- Develop in modular scripts (`data.py`, `features.py`, `cv.py`, `train.py`, `predict.py`) — easier to iterate and keep CV honest. Generate the clean final **`.ipynb`** for the judge at the end.
- Set all random seeds; pin library versions; cache parsed/engineered features so loops are fast.
- **Never** fit any statistic (scaling, target encoding, imputation) on test or on the validation fold.

---

## First three things to run
1. EDA: load all three files, print schema + `demand` stats, parse datetime, profile geohash, characterize the train/test split → short findings summary.
2. Define the CV scheme from (1) and print the folds.
3. LightGBM baseline → CV R² + a first submission to calibrate CV↔LB.
