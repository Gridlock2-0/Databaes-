"""
predict.py — Generate submission.csv from saved fold models (LGB + CatBoost + XGBoost).

Loads whichever model families are present in models/, averages fold predictions
within each family, then blends families with equal weight (or custom weights via
BLEND_WEIGHTS dict). Clips final predictions to [0, 1].
"""

import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.features import (
    build_base_features, add_all_encodings, compute_day49_scale,
    ALL_COLS_V2, CAT_FEATURE_NAMES,
)
from scripts.cv import N_SPLITS

DATA_DIR   = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
OUT_DIR    = ROOT

# Optional per-family weights; None = equal weight for all present families
BLEND_WEIGHTS = {'lgb': 1.0}   # LGB only — CB (79% CV) and XGB (stale) hurt the blend


def _build_test_features(train_df, test_df):
    train_feat, test_feat, _, _ = build_base_features(train_df, test_df)
    y_train = train_feat['demand']
    day49_scale = compute_day49_scale(train_feat, y_train.values)
    _, test_te, _ = add_all_encodings(
        train_feat, y_train, test_feat, None, day49_scale=day49_scale)
    return test_te


def _predict_lgb(X_test, cat_idx):
    preds = []
    for fold_i in range(N_SPLITS):
        path = os.path.join(MODELS_DIR, f'fold_{fold_i}.lgb')
        if not os.path.exists(path):
            return None
        model = lgb.Booster(model_file=path)
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)


def _predict_catboost(X_test_df, cat_names):
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError:
        return None
    preds = []
    for fold_i in range(N_SPLITS):
        path = os.path.join(MODELS_DIR, f'fold_{fold_i}_cb.cbm')
        if not os.path.exists(path):
            return None
        model = CatBoostRegressor()
        model.load_model(path)
        pool = Pool(X_test_df, cat_features=cat_names)
        preds.append(model.predict(pool))
    return np.mean(preds, axis=0)


def _predict_xgboost(X_test):
    try:
        import xgboost as xgb
    except ImportError:
        return None
    preds = []
    for fold_i in range(N_SPLITS):
        path = os.path.join(MODELS_DIR, f'fold_{fold_i}_xgb.json')
        if not os.path.exists(path):
            return None
        model = xgb.Booster()
        model.load_model(path)
        dtest = xgb.DMatrix(X_test.astype(np.float32))
        try:
            preds.append(model.predict(dtest))
        except Exception as e:
            print(f'  XGB fold {fold_i} predict failed ({e}) — skipping XGB')
            return None
    return np.mean(preds, axis=0)


def main():
    print('Loading data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    sub_tmpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    print(f'  sample_submission columns: {list(sub_tmpl.columns)}')

    print('Building features...')
    test_te   = _build_test_features(train_df, test_df)
    cat_idx   = [ALL_COLS_V2.index(c) for c in CAT_FEATURE_NAMES if c in ALL_COLS_V2]
    cat_names = [ALL_COLS_V2[i] for i in cat_idx]
    X_test    = test_te[ALL_COLS_V2].values
    # CatBoost needs cat columns as strings
    X_test_cb = test_te[ALL_COLS_V2].copy()
    for c in cat_names:
        X_test_cb[c] = X_test_cb[c].astype(int).astype(str)

    family_preds = {}

    print('\nLoading LightGBM fold models...')
    lgb_pred = _predict_lgb(X_test, cat_idx)
    if lgb_pred is not None:
        family_preds['lgb'] = lgb_pred
        print(f'  LGB: mean={lgb_pred.mean():.5f}  min={lgb_pred.min():.5f}  max={lgb_pred.max():.5f}')
    else:
        print('  LGB models not found — skipping')

    print('Loading CatBoost fold models...')
    cb_pred = _predict_catboost(X_test_cb, cat_names)
    if cb_pred is not None:
        family_preds['cb'] = cb_pred
        print(f'  CB:  mean={cb_pred.mean():.5f}  min={cb_pred.min():.5f}  max={cb_pred.max():.5f}')
    else:
        print('  CatBoost models not found — skipping')

    print('Loading XGBoost fold models...')
    xgb_pred = _predict_xgboost(X_test)
    if xgb_pred is not None:
        family_preds['xgb'] = xgb_pred
        print(f'  XGB: mean={xgb_pred.mean():.5f}  min={xgb_pred.min():.5f}  max={xgb_pred.max():.5f}')
    else:
        print('  XGBoost models not found — skipping')

    if not family_preds:
        raise RuntimeError('No models found. Run train.py / train_catboost.py / train_xgboost.py first.')

    # When BLEND_WEIGHTS is given, restrict to families listed there
    if BLEND_WEIGHTS is not None:
        family_preds = {k: v for k, v in family_preds.items() if k in BLEND_WEIGHTS}
    families_used = list(family_preds.keys())
    print(f'\nBlending: {families_used}')

    if BLEND_WEIGHTS is not None:
        weights = np.array([BLEND_WEIGHTS[k] for k in families_used], dtype=np.float64)
        weights /= weights.sum()
        test_preds = sum(w * family_preds[k] for w, k in zip(weights, families_used))
        print(f'  Weights: {dict(zip(families_used, weights.round(4)))}')
    else:
        test_preds = np.mean(list(family_preds.values()), axis=0)
        w = 1.0 / len(families_used)
        print(f'  Equal weights: {w:.4f} each')

    test_preds = np.clip(test_preds, 0.0, 1.0)
    print(f'\nFinal blend: mean={test_preds.mean():.5f}  min={test_preds.min():.5f}  max={test_preds.max():.5f}')

    sub = pd.DataFrame({
        'Index':  test_df['Index'].values,
        'demand': test_preds,
    })

    assert sub.shape == (41778, 2), f'Expected (41778, 2), got {sub.shape}'
    assert list(sub.columns) == ['Index', 'demand']
    assert sub['Index'].equals(test_df['Index'])

    out_path = os.path.join(OUT_DIR, 'submission.csv')
    sub.to_csv(out_path, index=False)
    print(f'\nSubmission written -> {out_path}')
    print(f'Shape: {sub.shape}')
    print('\nHead:')
    print(sub.head(10).to_string(index=False))

    return sub


if __name__ == '__main__':
    main()
