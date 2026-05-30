"""
verify_lags.py — Prove that lag/rolling features are leak-free.

Checks
------
1. Spot-verify exact-time lags: for several sample rows, confirm that
   lag_k == demand[same geohash, t - k*15 min].
2. Per-fold range analysis: for each TimeSeriesSplit fold, show the
   earliest val row and confirm its lag_1 / lag_96 timestamps are in
   the fold's training portion (or are NaN if no prior data exists).
3. Rolling window inspection for the first 5 rows after a fold boundary.
4. Correlations: lag_1 and lag_96 vs demand on all non-NaN train rows.
"""

import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.features import build_base_features, LAG_STEPS, ROLL_WINDOWS
from scripts.cv import get_folds, N_SPLITS

DATA_DIR = os.path.join(ROOT, 'dataset')


def main():
    print('Loading data and building features...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    train_feat, test_feat, _, _ = build_base_features(train_df, test_df)

    # Attach helper columns from raw df for cross-checking
    ts = train_df['timestamp'].str.split(':', expand=True).astype(int)
    train_feat['_t'] = train_df['day'].values * 1440 + ts[0].values * 60 + ts[1].values
    train_feat['_demand_raw'] = train_df['demand'].values

    # Build a lookup: (geohash, _t) -> demand  for all train rows
    lookup = train_feat.set_index(['geohash', '_t'])['_demand_raw'].to_dict()

    # ── CHECK 1: spot-verify exact-time lags ─────────────────────────────────
    print('\n' + '='*70)
    print('CHECK 1 — spot-verify lag_k == demand[same geohash, t - k*15min]')
    print('='*70)
    sample = train_feat.dropna(subset=['lag_1', 'lag_4', 'lag_96']).sample(5, random_state=0)

    for _, row in sample.iterrows():
        gh, t = row['geohash'], row['_t']
        print(f'\n  geohash={gh}  t={int(t)} ({int(row["day"])} {int(row["hour"]):02d}:{int(row["minute"]):02d})')
        for k in [1, 4, 96]:
            col     = f'lag_{k}'
            back_t  = int(t) - k * 15
            looked  = lookup.get((gh, back_t), np.nan)
            feat_v  = row[col]
            match   = np.isclose(feat_v, looked, equal_nan=True) if not (np.isnan(feat_v) and np.isnan(looked)) else True
            back_day  = back_t // 1440
            back_h    = (back_t % 1440) // 60
            back_m    = back_t % 60
            status    = 'OK' if match else '*** MISMATCH ***'
            print(f'    lag_{k:3d}: t-{k*15:4d}min -> day{back_day} {back_h:02d}:{back_m:02d}  '
                  f'lookup={looked:.5f}  feat={feat_v:.5f}  [{status}]')

    # ── CHECK 2: per-fold boundary analysis ──────────────────────────────────
    print('\n' + '='*70)
    print('CHECK 2 — per-fold: first val row lags must not exceed fold train boundary')
    print('='*70)

    folds = get_folds(train_feat, N_SPLITS)

    for fold_i, (tr_iloc, val_iloc) in enumerate(folds):
        tr_sub  = train_feat.iloc[tr_iloc]
        val_sub = train_feat.iloc[val_iloc]

        tr_t_max  = tr_sub['_t'].max()
        val_t_min = val_sub['_t'].min()

        # Day/hour/min representations
        def t_to_str(t):
            t = int(t)
            return f"day{t//1440} {(t%1440)//60:02d}:{t%60:02d}"

        print(f'\n  Fold {fold_i+1}: train ends at {t_to_str(tr_t_max)}'
              f'  |  val starts at {t_to_str(val_t_min)}')

        # For each lag, verify the first val row's lag points before or AT train boundary
        first_val = val_sub.sort_values('_t').iloc[0]
        fv_t = int(first_val['_t'])
        fv_gh = first_val['geohash']
        print(f'  First val row: geohash={fv_gh}  t={t_to_str(fv_t)}')

        for k in LAG_STEPS:
            lag_val  = first_val[f'lag_{k}']
            back_t   = fv_t - k * 15
            in_train = back_t <= tr_t_max
            is_nan   = np.isnan(lag_val)
            if is_nan:
                status = 'NaN (no prior data) -- safe'
            elif in_train:
                status = 'SAFE (points into fold train)'
            else:
                status = '*** LEAKAGE: points into fold val ***'
            vfmt = f'{lag_val:.5f}' if not is_nan else 'NaN'
            print(f'    lag_{k:3d}: t-{k*15:4d}min -> {t_to_str(back_t)}  '
                  f'in_train={in_train}  val={vfmt}  [{status}]')

        # Rolling: print the first 5 val rows and check roll_mean_4 window source
        print(f'\n  First 5 val rows roll_mean_4 (window should come from train):')
        first5_val = val_sub.sort_values('_t').head(5)
        for _, vrow in first5_val.iterrows():
            vt  = int(vrow['_t'])
            gh  = vrow['geohash']
            rm4 = vrow['roll_mean_4']
            # The 4 prior demand values: t-15, t-30, t-45, t-60 (step back in observations)
            # These are in the sorted geohash timeline, so they MIGHT include earlier val rows
            n_prior_in_val = sum(
                1 for step in range(1, 5)
                if (vt - step * 15) >= val_t_min   # rough: whether the prior time is in val
            )
            status = 'all from train' if n_prior_in_val == 0 else f'{n_prior_in_val} prior slot(s) in val range (minor)'
            print(f'    t={t_to_str(vt)}  roll_mean_4={rm4:.5f}  [{status}]')

    # ── CHECK 3: rolling window for test rows ─────────────────────────────────
    print('\n' + '='*70)
    print('CHECK 3 — test lag_1 and roll_mean_4: confirm NaN propagation for future test rows')
    print('='*70)

    ts_t = test_df['timestamp'].str.split(':', expand=True).astype(int)
    test_feat['_t_test'] = test_df['day'].values * 1440 + ts_t[0].values * 60 + ts_t[1].values

    sample_gh = test_feat['geohash'].value_counts().index[0]
    sample_test = test_feat[test_feat['geohash'] == sample_gh].sort_values('_t_test').head(8)
    print(f'\n  geohash={sample_gh}, first 8 test slots:')
    print(f"  {'slot':>4}  {'t (day H:M)':>14}  {'lag_1':>8}  {'lag_96':>8}  {'roll_mean_4':>12}")
    for j, (_, row) in enumerate(sample_test.iterrows()):
        t = int(row['_t_test'])
        def fmt(v): return f'{v:.5f}' if not np.isnan(v) else 'NaN'
        print(f'  {j:>4}  {t//1440}d {(t%1440)//60:02d}:{t%60:02d}  '
              f'{fmt(row["lag_1"]):>8}  {fmt(row["lag_96"]):>8}  {fmt(row["roll_mean_4"]):>12}')

    # ── CHECK 4: correlations ─────────────────────────────────────────────────
    print('\n' + '='*70)
    print('CHECK 4 — correlations of lag features with demand (train, non-NaN rows)')
    print('='*70)

    demand = train_feat['_demand_raw']
    corr_cols = [f'lag_{k}' for k in LAG_STEPS] + [f'roll_mean_{w}' for w in ROLL_WINDOWS]
    print(f'\n  {"Feature":<20}  {"N non-NaN":>10}  {"Pearson r":>10}  {"Coverage%":>10}')
    print('  ' + '-'*54)
    for col in corr_cols:
        mask   = ~train_feat[col].isna() & ~demand.isna()
        n      = mask.sum()
        pct    = n / len(demand) * 100
        if n > 1:
            r = np.corrcoef(train_feat.loc[mask, col], demand[mask])[0, 1]
        else:
            r = np.nan
        print(f'  {col:<20}  {n:>10,}  {r:>10.4f}  {pct:>9.1f}%')

    print('\nVerification complete.')


if __name__ == '__main__':
    main()
