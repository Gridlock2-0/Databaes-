"""
cv.py — Validation scheme for the Gridlock traffic demand pipeline.

Split logic
-----------
The train/test split is strictly temporal:
  train  = day 48 (full, 0:00–23:45) + day 49 (0:00–2:00)
  test   = day 49 (2:15–13:45)

We use TimeSeriesSplit(n_splits=5) on rows sorted by (day, hour, minute).
Validation sets are always strictly after their training set in time — this
mirrors the actual forward-looking test gap.

Run as script to inspect fold ranges.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N_SPLITS = 5


def _sort_key(df: pd.DataFrame) -> np.ndarray:
    """Integer sort key encoding temporal order: day * 10000 + hour * 100 + minute."""
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    return (df['day'].values * 10_000 + ts[0].values * 100 + ts[1].values).astype(np.int64)


def get_folds(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[tuple]:
    """
    Return list of (train_idx, val_idx) integer position arrays (iloc-style).
    Rows are sorted by temporal key before splitting.
    """
    key = _sort_key(df)
    sorted_pos = np.argsort(key, kind='stable')   # positions in sorted order

    tss = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_pos, val_pos in tss.split(sorted_pos):
        folds.append((sorted_pos[tr_pos], sorted_pos[val_pos]))
    return folds


def print_fold_info(df: pd.DataFrame, folds: list[tuple]) -> None:
    """Print fold sizes and the (day, hour:minute) range of each split."""
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df2 = df[['day']].copy()
    df2['hour']   = ts[0].values
    df2['minute'] = ts[1].values

    def _fmt(sub: pd.DataFrame) -> str:
        mn = sub.sort_values(['day', 'hour', 'minute']).iloc[0]
        mx = sub.sort_values(['day', 'hour', 'minute']).iloc[-1]
        return (f"day{int(mn.day)} {int(mn.hour):02d}:{int(mn.minute):02d}"
                f" -> day{int(mx.day)} {int(mx.hour):02d}:{int(mx.minute):02d}")

    header = f"{'Fold':>4}  {'Train rows':>11}  {'Val rows':>9}  Train range                      Val range"
    print(header)
    print('-' * len(header))
    for i, (tr_iloc, val_iloc) in enumerate(folds):
        tr_sub  = df2.iloc[tr_iloc]
        val_sub = df2.iloc[val_iloc]
        print(f"{i+1:>4}  {len(tr_iloc):>11,}  {len(val_iloc):>9,}  "
              f"{_fmt(tr_sub):<32}   {_fmt(val_sub)}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA = os.path.join(ROOT, 'dataset')
    train = pd.read_csv(os.path.join(DATA, 'train.csv'))

    print(f'TimeSeriesSplit with n_splits={N_SPLITS}')
    print(f'Total train rows: {len(train):,}\n')

    folds = get_folds(train, N_SPLITS)
    print_fold_info(train, folds)

    print('\nNote: test window = day49 02:15 -> day49 13:45  (strictly after training ends)')
