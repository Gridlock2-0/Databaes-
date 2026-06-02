"""
features.py — Feature engineering for the Gridlock traffic demand pipeline.

Public API
----------
build_base_features(train_df, test_df)
    -> (train_feat, test_feat, NON_TE_COLS, encoders)
    Computes BASE_COLS + LAG_COLS (no target encodings yet).

add_all_encodings(tr_feat, y_tr, val_feat, test_feat=None, global_rt_stats=None)
    -> (tr_out, val_out, test_out)
    Adds LOO TEs for training rows; regular group TEs for val/test.
    Adds smoothed TE for RoadType.
    If global_rt_stats provided, appends 2 global RoadType×time TE columns.

Module-level constants
----------------------
BASE_COLS       27  (26 original + roadtype_was_missing)
LAG_COLS         9  (lag_1/2/4/96/192 + roll mean/std × 4/96)
SHORT_LAG_COLS   3  (lag_1/2/4  — inflated CV, mostly NaN at test time)
TE_COLS         19  (16 LOO×mean/med + 1 smoothed roadtype + 2 stat cols)
NON_TE_COLS     36
ALL_COLS        55

Lag / rolling leakage contract
-------------------------------
Same as before: all lags are exact-time-merge (strictly backward);
rolling uses shift(1)+transform; test demand forced to NaN.
"""

import os, sys
import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── column definitions ────────────────────────────────────────────────────────

CAT_COLS = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
NUM_COLS = ['NumberofLanes', 'Temperature']
LE_COLS  = CAT_COLS + ['geohash', 'prefix5', 'prefix4']

LAG_STEPS     = [1, 2, 4, 96, 192]
SHORT_LAG_COLS = ['lag_1', 'lag_2', 'lag_4']   # CV-inflating: mostly NaN at test time
ROLL_WINDOWS  = [4, 96]
LAG_COLS = (
    [f'lag_{k}' for k in LAG_STEPS]
    + [f'roll_mean_{w}' for w in ROLL_WINDOWS]
    + [f'roll_std_{w}'  for w in ROLL_WINDOWS]
)  # 9 columns

# LOO target encodings: (group_cols, col_prefix) → te_{prefix}_mean + te_{prefix}_med
LOO_TE_SPECS = [
    (['geohash'],                'te_gh'),
    (['geohash', 'hour'],        'te_gh_hr'),
    (['geohash', 'time_of_day'], 'te_gh_tod'),
    (['geohash', 'is_weekend'],  'te_gh_we'),
    (['prefix5', 'hour'],        'te_p5_hr'),
    (['prefix5', 'tod_bucket'],  'te_p5_todb'),
    (['NumberofLanes', 'hour'],  'te_nl_hr'),
    (['LargeVehicles', 'hour'],  'te_lv_hr'),
]
LOO_TE_COLS = [f'{p}_mean' for _, p in LOO_TE_SPECS] + [f'{p}_med' for _, p in LOO_TE_SPECS]
# 8 specs × 2 = 16 columns

# Smoothed TE (road type — low-cardinality, smoothing preferred over LOO)
SMOOTHED_TE_SPECS = [(['RoadType_le'], 'te_roadtype')]
SMOOTHED_TE_COLS  = [tc for _, tc in SMOOTHED_TE_SPECS]

# Non-TE stat features (computed from fold training data, applied as lookup to val/test)
STAT_COLS = ['gh_count', 'gh_hr_std']

# Cross-day deviation features: lag96_dev = lag_96 - te_gh_tod_mean
# NaN for day48 rows (lag96 unavailable), valid for day49/test rows.
# Encodes "how much did yesterday's actual demand deviate from its historical average?"
# gh_day49_scale: per-geohash demand ratio day49-morning / day48-morning
# (only valid for day49/test; computed from full training data, not per-fold)
CROSSDAY_COLS = ['lag96_dev', 'gh_day49_scale']

TE_COLS  = LOO_TE_COLS + SMOOTHED_TE_COLS + STAT_COLS   # 16 + 1 + 2 = 19
TE_ALPHA = 10

# Global TEs: computed once from full train, LOO-corrected per training row
GLOBAL_RT_SPECS = [
    (['RoadType', 'hour'],        'te_rt_hr_g'),
    (['RoadType', 'time_of_day'], 'te_rt_tod_g'),
]
GLOBAL_RT_COLS = [f'{p}_mean' for _, p in GLOBAL_RT_SPECS]  # 2 columns, mean only

BASE_COLS = [
    'day', 'hour', 'minute', 'time_of_day', 'day_mod7', 'is_weekend', 'tod_bucket',
    'sin_hour', 'cos_hour', 'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow',
    'lat', 'lon',
    'NumberofLanes', 'Temperature',
    'RoadType_le', 'Weather_le', 'LargeVehicles_le', 'Landmarks_le',
    'geohash_le', 'prefix5_le', 'prefix4_le',
    'roadtype_hour_le', 'weather_hour_le',
    'roadtype_was_missing',
]  # 27 columns

NON_TE_COLS  = BASE_COLS + LAG_COLS                        # 36
ALL_COLS     = BASE_COLS + LAG_COLS + TE_COLS              # 55
ALL_COLS_V2  = BASE_COLS + LAG_COLS + TE_COLS + CROSSDAY_COLS  # 57

CAT_FEATURE_NAMES = [
    'RoadType_le', 'Weather_le', 'LargeVehicles_le', 'Landmarks_le',
    'geohash_le', 'prefix5_le', 'prefix4_le', 'day_mod7', 'is_weekend', 'tod_bucket',
    'roadtype_hour_le', 'weather_hour_le',
]

# ── private helpers ───────────────────────────────────────────────────────────

def _build_roadtype_map(train_df: pd.DataFrame) -> dict:
    """Return {geohash: RoadType} for geohashes with exactly 1 unique RoadType in train."""
    rt_known   = train_df[train_df['RoadType'].notna()][['geohash', 'RoadType']]
    gh_nunique = rt_known.groupby('geohash')['RoadType'].nunique()
    gh_mode    = rt_known.groupby('geohash')['RoadType'].agg(lambda x: x.mode().iloc[0])
    return gh_mode[gh_nunique == 1].to_dict()


def _impute_roadtype(df: pd.DataFrame, rt_map: dict) -> pd.DataFrame:
    """Fill NaN RoadType from geohash map; truly-ambiguous geohashes → 'Missing'."""
    df = df.copy()
    df['roadtype_was_missing'] = df['RoadType'].isna().astype(np.int8)
    nan_mask = df['RoadType'].isna()
    df.loc[nan_mask, 'RoadType'] = df.loc[nan_mask, 'geohash'].map(rt_map)
    df['RoadType'] = df['RoadType'].fillna('Missing')
    return df


_GH_CACHE: dict = {}

def _decode_gh(gh: str):
    if gh not in _GH_CACHE:
        try:
            ll = pgh.decode(gh)
            _GH_CACHE[gh] = (ll.latitude, ll.longitude)
        except Exception:
            _GH_CACHE[gh] = (np.nan, np.nan)
    return _GH_CACHE[gh]


def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour']        = ts[0]
    df['minute']      = ts[1]
    df['time_of_day'] = df['hour'] * 60 + df['minute']  # 0..1425 (96 slots)
    df['day_mod7']    = (df['day'] % 7).astype(np.int8)
    df['is_weekend']  = df['day_mod7'].isin([5, 6]).astype(np.int8)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_tod']  = np.sin(2 * np.pi * df['time_of_day'] / 1440)
    df['cos_tod']  = np.cos(2 * np.pi * df['time_of_day'] / 1440)
    df['sin_dow']  = np.sin(2 * np.pi * df['day_mod7'] / 7)
    df['cos_dow']  = np.cos(2 * np.pi * df['day_mod7'] / 7)
    # 5-bucket time-of-day: night(0-4), morning(5-8), rush(9-11), midday(12-14), evening(15+)
    df['tod_bucket'] = np.digitize(df['hour'].values, bins=[5, 9, 12, 15]).astype(np.int8)
    return df


def _add_spatial(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = [_decode_gh(g) for g in df['geohash']]
    df['lat']     = [p[0] for p in pairs]
    df['lon']     = [p[1] for p in pairs]
    df['prefix4'] = df['geohash'].str[:4]
    df['prefix5'] = df['geohash'].str[:5]
    return df


# ── lag / rolling computation (unchanged from previous version) ───────────────

def _abs_min(day: pd.Series, ts: pd.DataFrame) -> pd.Series:
    return day * 1440 + ts[0] * 60 + ts[1]


def _compute_lags(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Leak-free lag and rolling features on combined train+test timeline.
    Test demand is set to NaN; all lags use strictly past timestamps.
    Returns (train_lag_df, test_lag_df) containing only LAG_COLS.
    """
    tr = train_df[['geohash', 'day', 'timestamp', 'demand']].copy()
    te = test_df[['geohash', 'day', 'timestamp']].copy()
    te['demand'] = np.nan

    tr['_src'] = 0; tr['_row'] = np.arange(len(train_df), dtype=np.int32)
    te['_src'] = 1; te['_row'] = np.arange(len(test_df),  dtype=np.int32)

    comb = pd.concat([tr, te], ignore_index=True)
    ts_s = comb['timestamp'].str.split(':', expand=True).astype(int)
    comb['_t'] = _abs_min(comb['day'], ts_s)
    comb = comb.sort_values(['geohash', '_t']).reset_index(drop=True)

    lookup = comb[['geohash', '_t', 'demand']].copy()

    for k in LAG_STEPS:
        col    = f'lag_{k}'
        back_t = (comb['_t'] - k * 15).values
        left   = pd.DataFrame({'geohash': comb['geohash'].values, '_t_back': back_t})
        right  = lookup.rename(columns={'_t': '_t_back', 'demand': col})
        merged = left.merge(right, on=['geohash', '_t_back'], how='left')
        comb[col] = merged[col].values

    comb['_ds1'] = comb.groupby('geohash', sort=False)['demand'].shift(1)
    for w in ROLL_WINDOWS:
        comb[f'roll_mean_{w}'] = comb.groupby('geohash', sort=False)['_ds1'].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )
        comb[f'roll_std_{w}'] = comb.groupby('geohash', sort=False)['_ds1'].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).std()
        )
    comb.drop(columns=['_ds1'], inplace=True)

    tr_out = (comb[comb['_src'] == 0]
              .set_index('_row').loc[np.arange(len(train_df))][LAG_COLS]
              .reset_index(drop=True))
    te_out = (comb[comb['_src'] == 1]
              .set_index('_row').loc[np.arange(len(test_df))][LAG_COLS]
              .reset_index(drop=True))
    return tr_out, te_out


# ── public API: base features ─────────────────────────────────────────────────

def build_base_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Parse time, decode geohash, label-encode categoricals, add interactions,
    compute lag/rolling features.
    Returns (train_feat, test_feat, NON_TE_COLS, encoders).
    """
    print('  Parsing time features...')
    train = _parse_time(train_df)
    test  = _parse_time(test_df)

    print('  Decoding geohash + spatial...')
    train = _add_spatial(train)
    test  = _add_spatial(test)

    # Impute RoadType NaN from geohash mode (unambiguous geohashes only → else 'Missing')
    print('  Imputing RoadType from geohash map...')
    rt_map = _build_roadtype_map(train_df)
    train  = _impute_roadtype(train, rt_map)
    test   = _impute_roadtype(test,  rt_map)
    print(f'    train NaN filled: {train["roadtype_was_missing"].sum()}  '
          f'test NaN filled: {(test["roadtype_was_missing"].sum() - (test["RoadType"] == "Missing").sum())}  '
          f'remain Missing: {(train["RoadType"] == "Missing").sum() + (test["RoadType"] == "Missing").sum()}')

    # Numeric NaN → train median
    medians = {}
    for col in NUM_COLS:
        med = float(train[col].median())
        medians[col] = med
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

    # Label encode (fit on union)
    encoders = {'_medians': medians}
    for col in LE_COLS:
        le = LabelEncoder()
        vals = pd.concat([
            train[col].astype(str).fillna('_NA_'),
            test[col].astype(str).fillna('_NA_'),
        ])
        le.fit(vals)
        train[f'{col}_le'] = le.transform(train[col].astype(str).fillna('_NA_'))
        test[f'{col}_le']  = le.transform(test[col].astype(str).fillna('_NA_'))
        encoders[col] = le

    # Interaction features: RoadType×hour and Weather×hour
    train['roadtype_hour_le'] = train['RoadType_le'] * 24 + train['hour']
    test['roadtype_hour_le']  = test['RoadType_le']  * 24 + test['hour']
    train['weather_hour_le']  = train['Weather_le']  * 24 + train['hour']
    test['weather_hour_le']   = test['Weather_le']   * 24 + test['hour']

    # Lag / rolling (computed jointly on train+test for correct lookback)
    print('  Computing lag / rolling features...')
    train_lags, test_lags = _compute_lags(train_df, test_df)
    for col in LAG_COLS:
        train[col] = train_lags[col].values
        test[col]  = test_lags[col].values

    lag_nan_pct = train[LAG_COLS].isna().mean() * 100
    print('  Lag NaN% in train: ' +
          ', '.join(f'{c}:{p:.0f}%' for c, p in lag_nan_pct.items()))

    return train, test, NON_TE_COLS, encoders


# ── public API: target encodings ─────────────────────────────────────────────

def _apply_te_map(df, group_cols, te_map, fallback):
    """Apply a precomputed group->value map; fill unknowns with scalar fallback."""
    if len(group_cols) == 1:
        mapped = df[group_cols[0]].map(te_map)
    else:
        keys   = list(zip(*[df[c] for c in group_cols]))
        mapped = pd.Series(keys, index=df.index).map(te_map)
    return mapped.fillna(fallback).to_numpy(dtype=np.float64)


def _apply_te_map_with_row_fallback(df, group_cols, te_map, fallback_arr, global_fallback):
    """
    Like _apply_te_map but uses a per-row fallback array for missing keys.
    This prevents high-magnitude errors when a group key is unseen in the training fold
    but the per-row fallback carries the right signal (e.g. RoadType-level TE).

    fallback_arr: np.ndarray aligned to df rows.
    global_fallback: scalar used if fallback_arr is also NaN.
    """
    if len(group_cols) == 1:
        mapped = df[group_cols[0]].map(te_map).to_numpy(dtype=np.float64)
    else:
        keys   = list(zip(*[df[c] for c in group_cols]))
        mapped = pd.Series(keys, index=df.index).map(te_map).to_numpy(dtype=np.float64)
    na_mask = np.isnan(mapped)
    if na_mask.any():
        fb = np.where(np.isnan(fallback_arr), global_fallback, fallback_arr)
        mapped[na_mask] = fb[na_mask]
    return mapped


def _loo_stats(tr_feat, y_tr, group_cols):
    """
    Compute group-level statistics on training data.
    Returns (loo_mean_arr, group_med_arr, mean_map, median_map)
    where loo_mean_arr and group_med_arr are aligned to tr_feat rows.
    """
    global_mean   = float(y_tr.mean())
    global_median = float(np.median(y_tr.values))

    tmp = tr_feat[group_cols].copy()
    tmp['_y'] = y_tr.values

    # Within-group transforms (aligned to tr_feat)
    g_sum   = tmp.groupby(group_cols)['_y'].transform('sum').values
    g_count = tmp.groupby(group_cols)['_y'].transform('count').values
    yi      = y_tr.values

    # Exact LOO mean: avoid divide-by-zero for n=1 groups
    denom    = np.where(g_count > 1, g_count - 1, 1.0)
    loo_mean = np.where(g_count > 1, (g_sum - yi) / denom, global_mean)

    # Exact LOO median: exclude current row from its own group
    # For n=1 groups, transform('median') returns y_i (pure leakage) — use global_median instead.
    # For large groups (n > 20) the single-element LOO bias is negligible; use group median for speed.
    def _make_loo_median(gmed, exact_threshold=20):
        def fn(group):
            arr = group.values
            n = len(arr)
            if n == 1:
                return pd.Series([gmed], index=group.index)
            if n <= exact_threshold:
                result = np.empty(n)
                for i in range(n):
                    rest = np.concatenate([arr[:i], arr[i + 1:]])
                    result[i] = np.median(rest)
            else:
                result = np.full(n, np.median(arr))
            return pd.Series(result, index=group.index)
        return fn

    group_med = (
        tmp.groupby(group_cols, group_keys=False)['_y']
        .apply(_make_loo_median(global_median))
        .values
    )

    # Build lookup dicts for val / test
    stats = tmp.groupby(group_cols)['_y'].agg(['sum', 'count', 'median']).reset_index()
    if len(group_cols) == 1:
        col = group_cols[0]
        mean_map   = (stats['sum'] / stats['count']).set_axis(stats[col]).to_dict()
        median_map = stats['median'].set_axis(stats[col]).to_dict()
    else:
        mean_map   = {tuple(row[c] for c in group_cols): row['sum'] / row['count']
                      for _, row in stats.iterrows()}
        median_map = {tuple(row[c] for c in group_cols): row['median']
                      for _, row in stats.iterrows()}

    return loo_mean, group_med, mean_map, median_map, global_mean, global_median


def _smoothed_te_map(tr_feat, y_tr, group_cols, global_mean, alpha=TE_ALPHA):
    """Compute smoothed group mean: te = (sum + alpha*global) / (n + alpha)."""
    tmp = tr_feat[group_cols].copy()
    tmp['_y'] = y_tr.values
    stats = tmp.groupby(group_cols)['_y'].agg(['sum', 'count']).reset_index()
    stats['te'] = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)
    if len(group_cols) == 1:
        return stats['te'].set_axis(stats[group_cols[0]]).to_dict()
    return {tuple(row[c] for c in group_cols): row['te'] for _, row in stats.iterrows()}


def compute_day49_scale(train_feat: pd.DataFrame, y_full: np.ndarray) -> pd.Series:
    """
    Compute per-geohash scale = mean(demand on day49 training rows) / mean(demand on
    day48 at the same time slots).  Returns a Series indexed by geohash.

    Encodes "how much higher/lower is day49 demand relative to the same geohash
    on day48?" — a powerful calibration signal for cross-day prediction.

    Only day49 training rows (demand known) are used; no test leakage.
    Clipped to [0.2, 5.0] to suppress extreme ratios from sparse geohashes.
    """
    tmp = train_feat[['geohash', 'day', 'time_of_day']].copy()
    tmp['_y'] = y_full

    day49_mask = tmp['day'] == 49
    day48_mask = tmp['day'] == 48

    day49_rows = tmp[day49_mask]
    day48_rows = tmp[day48_mask]

    # Mean demand per (geohash, time_slot) for each day
    day49_mean = day49_rows.groupby(['geohash', 'time_of_day'])['_y'].mean()
    day48_mean = day48_rows.groupby(['geohash', 'time_of_day'])['_y'].mean()

    # Scale per (geohash, time_slot) — then average over time slots per geohash
    combined = day49_mean.rename('d49').to_frame().join(day48_mean.rename('d48'), how='inner')
    combined['scale'] = (combined['d49'] / combined['d48'].clip(lower=1e-6)).clip(0.2, 5.0)
    gh_scale = combined.groupby('geohash')['scale'].mean()
    return gh_scale


def compute_global_rt_stats(train_feat: pd.DataFrame, y_full: np.ndarray) -> dict:
    """
    Build global RoadType×time stats from ALL training data.
    Returns dict: {prefix: {key_tuple: (group_sum, group_count), ...}, ...}
    """
    stats = {}
    for group_cols, col_prefix in GLOBAL_RT_SPECS:
        tmp = train_feat[group_cols].copy()
        tmp['_y'] = y_full
        agg = tmp.groupby(group_cols)['_y'].agg(['sum', 'count']).reset_index()
        key_dict = {}
        for _, row in agg.iterrows():
            key = tuple(row[c] for c in group_cols)
            key_dict[key] = (float(row['sum']), int(row['count']))
        stats[col_prefix] = key_dict
    return stats


def add_all_encodings(
    tr_feat: pd.DataFrame, y_tr: pd.Series,
    val_feat: pd.DataFrame, test_feat=None,
    global_rt_stats=None,
    day49_scale: pd.Series = None,
):
    """
    Compute all target encodings and return (tr_out, val_out, test_out).

    Smoothed TEs are computed first (RoadType) so they can serve as per-row
    fallbacks for unseen groups.

    Training rows → LOO mean + LOO median for LOO_TE_SPECS.
    Val / test   → group mean + group median from training fold.

    If global_rt_stats is provided (dict from compute_global_rt_stats), appends
    2 global RoadType×time TE columns using LOO correction for training rows.
    """
    global_mean = float(y_tr.mean())

    tr_out   = tr_feat.copy()
    val_out  = val_feat.copy()
    test_out = test_feat.copy() if test_feat is not None else None

    # ── Smoothed TEs FIRST ───────────────────────────────────────────────────
    for group_cols, te_col in SMOOTHED_TE_SPECS:
        te_map = _smoothed_te_map(tr_feat, y_tr, group_cols, global_mean)
        for out_df, src_df in [(tr_out,  tr_feat),
                               (val_out, val_feat),
                               *([(test_out, test_out)] if test_out is not None else [])]:
            out_df[te_col] = _apply_te_map(src_df, group_cols, te_map, global_mean)

    # ── LOO TEs ──────────────────────────────────────────────────────────────
    for group_cols, col_prefix in LOO_TE_SPECS:
        loo_mean, grp_med, mean_map, median_map, g_mean, g_med = _loo_stats(
            tr_feat, y_tr, group_cols
        )

        tr_out[f'{col_prefix}_mean'] = loo_mean
        tr_out[f'{col_prefix}_med']  = grp_med

        val_out[f'{col_prefix}_mean'] = _apply_te_map(val_feat, group_cols, mean_map,   g_mean)
        val_out[f'{col_prefix}_med']  = _apply_te_map(val_feat, group_cols, median_map, g_med)
        if test_out is not None:
            test_out[f'{col_prefix}_mean'] = _apply_te_map(test_out, group_cols, mean_map,   g_mean)
            test_out[f'{col_prefix}_med']  = _apply_te_map(test_out, group_cols, median_map, g_med)

    # ── Global RoadType×time TEs (Stage 2+, optional) ────────────────────────
    if global_rt_stats is not None:
        # Compute global mean for fallback
        y_arr = y_tr.values  # training fold y (for LOO correction)
        g_mean_all = float(y_tr.mean())  # fold-level mean for fallback

        for group_cols, col_prefix in GLOBAL_RT_SPECS:
            col_name  = f'{col_prefix}_mean'
            key_dict  = global_rt_stats[col_prefix]

            # Global mean (for unseen keys)
            total_sum   = sum(v[0] for v in key_dict.values())
            total_count = sum(v[1] for v in key_dict.values())
            g_global    = total_sum / total_count if total_count > 0 else g_mean_all

            # Training rows: LOO correction using global stats
            tr_keys = list(zip(*[tr_feat[c] for c in group_cols]))
            tr_vals = np.empty(len(tr_feat), dtype=np.float64)
            for i, (key, yi) in enumerate(zip(tr_keys, y_arr)):
                if key in key_dict:
                    gs, gc = key_dict[key]
                    tr_vals[i] = (gs - yi) / (gc - 1) if gc > 1 else g_global
                else:
                    tr_vals[i] = g_global
            tr_out[col_name] = tr_vals

            # Val rows: global mean (no LOO needed)
            val_keys = list(zip(*[val_feat[c] for c in group_cols]))
            val_vals = np.array(
                [key_dict[k][0] / key_dict[k][1] if k in key_dict else g_global
                 for k in val_keys], dtype=np.float64
            )
            val_out[col_name] = val_vals

            # Test rows
            if test_out is not None:
                te_keys  = list(zip(*[test_out[c] for c in group_cols]))
                te_vals  = np.array(
                    [key_dict[k][0] / key_dict[k][1] if k in key_dict else g_global
                     for k in te_keys], dtype=np.float64
                )
                test_out[col_name] = te_vals

    # ── Stat features (no target, computed from fold training data) ───────────
    # gh_count: data-density signal — how many training observations this geohash has
    gh_count_map  = tr_feat.groupby('geohash').size().to_dict()
    count_fallback = float(np.mean(list(gh_count_map.values())))
    for out_df, src_df in [(tr_out, tr_feat), (val_out, val_feat),
                           *( [(test_out, test_out)] if test_out is not None else [] )]:
        out_df['gh_count'] = src_df['geohash'].map(gh_count_map).fillna(count_fallback).to_numpy(dtype=np.float64)

    # gh_hr_std: demand variability per geohash×hour in training fold
    tmp_std = tr_feat[['geohash', 'hour']].copy()
    tmp_std['_y'] = y_tr.values
    gh_hr_std_series = tmp_std.groupby(['geohash', 'hour'])['_y'].std()
    gh_hr_std_map  = gh_hr_std_series.to_dict()
    std_fallback   = float(y_tr.std())

    def _apply_std(df):
        keys = list(zip(df['geohash'].values, df['hour'].values))
        return pd.Series(keys).map(gh_hr_std_map).fillna(std_fallback).to_numpy(dtype=np.float64)

    tr_out['gh_hr_std']  = _apply_std(tr_feat)
    val_out['gh_hr_std'] = _apply_std(val_feat)
    if test_out is not None:
        test_out['gh_hr_std'] = _apply_std(test_out)

    # ── Cross-day deviation features ─────────────────────────────────────────
    # lag96_dev = lag_96 - te_gh_tod_mean: how much actual yesterday deviates
    # from the historical geohash×slot average.
    # Day48 rows: lag96_dev = NaN (lag_96 is NaN, no yesterday in dataset).
    # Day49/test rows: lag96_dev = actual lag_96 - te_gh_tod_mean (non-zero).
    for out_df in [tr_out, val_out] + ([test_out] if test_out is not None else []):
        if 'lag_96' in out_df.columns and 'te_gh_tod_mean' in out_df.columns:
            out_df['lag96_dev'] = (
                out_df['lag_96'].values - out_df['te_gh_tod_mean'].values
            )
        else:
            out_df['lag96_dev'] = np.nan

    # gh_day49_scale: per-geohash calibration ratio (day49 morning / day48 morning).
    # Day48 rows → 1.0 (neutral; scale not yet known at day48 time).
    # Day49/test rows → actual ratio from full training data.
    # Always non-NaN so LGBM can split on the contrast between Sunday (1.0) and Monday (>1).
    global_scale = float(day49_scale.mean()) if day49_scale is not None else 1.0
    for out_df in [tr_out, val_out] + ([test_out] if test_out is not None else []):
        if day49_scale is not None:
            raw_scale = out_df['geohash'].map(day49_scale).fillna(global_scale).to_numpy(dtype=np.float64)
            # Day48 rows (day < 49) get neutral scale=1.0; day49 rows get actual scale
            is_day48 = (out_df['day'].values < 49)
            raw_scale[is_day48] = 1.0
            out_df['gh_day49_scale'] = raw_scale
        else:
            out_df['gh_day49_scale'] = 1.0  # neutral fallback when no scale provided

    return tr_out, val_out, test_out
