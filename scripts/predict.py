"""
predict.py — Generate submission.csv from saved fold models.

Recomputes base features and target encodings (on full training set),
loads all fold models, averages their predictions, clips to [0, 1],
and writes submission.csv.
"""

import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.features import (
    build_base_features, add_all_encodings, ALL_COLS, CAT_FEATURE_NAMES,
)
from scripts.cv import N_SPLITS

DATA_DIR   = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
OUT_DIR    = ROOT


def main():
    # ── load data ─────────────────────────────────────────────────────────────
    print('Loading data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    sub_tmpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    print(f'  sample_submission columns: {list(sub_tmpl.columns)}')

    # ── features ──────────────────────────────────────────────────────────────
    print('Building features...')
    train_feat, test_feat, _, _ = build_base_features(train_df, test_df)
    y_train = train_feat['demand']

    # Target encodings fitted on full training set; applied to test rows
    print('Computing target encodings on full training set...')
    _, test_te, _ = add_all_encodings(train_feat, y_train, test_feat, None)

    X_test = test_te[ALL_COLS].values
    cat_idx = [ALL_COLS.index(c) for c in CAT_FEATURE_NAMES if c in ALL_COLS]

    # ── load models and predict ───────────────────────────────────────────────
    test_preds_list = []
    for fold_i in range(N_SPLITS):
        model_path = os.path.join(MODELS_DIR, f'fold_{fold_i}.lgb')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}. Run train.py first.')
        model = lgb.Booster(model_file=model_path)
        pred  = model.predict(X_test)
        test_preds_list.append(pred)
        print(f'  fold {fold_i+1}: pred mean={pred.mean():.5f}  min={pred.min():.5f}  max={pred.max():.5f}')

    test_preds = np.mean(test_preds_list, axis=0)
    test_preds = np.clip(test_preds, 0.0, 1.0)
    print(f'\nAveraged: mean={test_preds.mean():.5f}  min={test_preds.min():.5f}  max={test_preds.max():.5f}')

    # ── build submission ──────────────────────────────────────────────────────
    sub = pd.DataFrame({
        'Index':  test_df['Index'].values,
        'demand': test_preds,
    })

    assert sub.shape == (41778, 2), f'Expected (41778, 2), got {sub.shape}'
    assert list(sub.columns) == ['Index', 'demand'], f'Bad columns: {sub.columns.tolist()}'
    assert sub['Index'].equals(test_df['Index']), 'Index mismatch vs test file'

    out_path = os.path.join(OUT_DIR, 'submission.csv')
    sub.to_csv(out_path, index=False)
    print(f'\nSubmission written -> {out_path}')
    print(f'Shape: {sub.shape}')
    print('\nHead:')
    print(sub.head(10).to_string(index=False))

    return sub


if __name__ == '__main__':
    main()
