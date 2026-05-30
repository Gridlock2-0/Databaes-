"""
train.py — LightGBM training with 5-fold TimeSeriesSplit CV.

Runs TWO CV loops:
  1. Full features (all lags including short lag_1/2/4)
  2. Ablation: short lags removed  ← honest leaderboard proxy

Outputs
-------
models/fold_{i}.lgb          fold models trained on full feature set
models/oof_preds.npy          OOF predictions (full feature set)
models/test_preds.npy         averaged test predictions (full feature set)
models/feat_importance.npy    mean gain importance aligned to ALL_COLS
"""

import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)

from scripts.features import (
    build_base_features, add_all_encodings,
    ALL_COLS, SHORT_LAG_COLS, CAT_FEATURE_NAMES,
)
from scripts.cv import get_folds, print_fold_info, N_SPLITS

DATA_DIR   = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
SEED       = 42

LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'num_leaves':        127,
    'learning_rate':     0.05,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'min_child_samples': 20,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
    'verbose':           -1,
    'seed':              SEED,
    'n_jobs':            -1,
}
NUM_ROUNDS     = 3000
EARLY_STOPPING = 100
LOG_EVAL       = 500


def run_cv_loop(train_feat, test_feat, y, folds, use_cols, cat_idx,
                save_models=False, label=''):
    """
    Run one full CV loop and return (oof_preds, test_preds_avg, importances).
    Only saves .lgb files when save_models=True.
    """
    oof_preds      = np.full(len(train_feat), np.nan)
    test_preds_list = []
    importance_acc  = np.zeros(len(use_cols))

    for fold_i, (tr_iloc, val_iloc) in enumerate(folds):
        tr_raw  = train_feat.iloc[tr_iloc].reset_index(drop=True)
        val_raw = train_feat.iloc[val_iloc].reset_index(drop=True)
        y_tr    = pd.Series(y[tr_iloc])
        y_val   = pd.Series(y[val_iloc])

        # All encodings: LOO for train; group TE for val/test
        tr_enc, val_enc, test_enc = add_all_encodings(tr_raw, y_tr, val_raw, test_feat)

        X_tr   = tr_enc[use_cols].values
        X_val  = val_enc[use_cols].values
        X_test = test_enc[use_cols].values

        lgb_tr  = lgb.Dataset(X_tr,  label=y_tr.values,
                               categorical_feature=cat_idx, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, label=y_val.values,
                               categorical_feature=cat_idx, free_raw_data=False,
                               reference=lgb_tr)

        model = lgb.train(
            LGB_PARAMS, lgb_tr,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[lgb_tr, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(LOG_EVAL),
            ],
        )

        best_iter = model.best_iteration
        val_pred  = model.predict(X_val, num_iteration=best_iter)
        fold_r2   = r2_score(y_val.values, val_pred)
        print(f'  [{label}] fold {fold_i+1}  best_iter={best_iter:4d}  '
              f'val R2={fold_r2*100:.4f}')

        oof_preds[val_iloc] = val_pred
        test_preds_list.append(model.predict(X_test, num_iteration=best_iter))
        importance_acc += model.feature_importance(importance_type='gain')

        if save_models:
            model.save_model(os.path.join(MODELS_DIR, f'fold_{fold_i}.lgb'))

    nan_mask       = np.isnan(oof_preds)
    oof_r2         = r2_score(y[~nan_mask], oof_preds[~nan_mask])
    test_preds_avg = np.clip(np.mean(test_preds_list, axis=0), 0.0, 1.0)
    importance_avg = importance_acc / len(folds)

    return oof_preds, test_preds_avg, oof_r2, importance_avg


def main():
    t0 = time.time()

    # ── load data ─────────────────────────────────────────────────────────────
    print('Loading data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    print(f'  train {train_df.shape}  test {test_df.shape}')

    # ── build non-TE features ─────────────────────────────────────────────────
    print('Building base features...')
    train_feat, test_feat, _, _ = build_base_features(train_df, test_df)
    y = train_feat['demand'].values.astype(np.float64)

    # ── CV folds ──────────────────────────────────────────────────────────────
    print(f'\nCV folds (TimeSeriesSplit n={N_SPLITS}):')
    folds = get_folds(train_feat, N_SPLITS)
    print_fold_info(train_feat, folds)

    # ── feature column sets ───────────────────────────────────────────────────
    # Full feature set
    cat_idx_full = [ALL_COLS.index(c) for c in CAT_FEATURE_NAMES if c in ALL_COLS]
    # Ablation: drop short lags (lag_1/2/4)
    ablation_cols = [c for c in ALL_COLS if c not in SHORT_LAG_COLS]
    cat_idx_ablation = [ablation_cols.index(c) for c in CAT_FEATURE_NAMES
                        if c in ablation_cols]

    print(f'\nFull feature set   : {len(ALL_COLS)} features')
    print(f'Ablation (no short lags): {len(ablation_cols)} features')
    print(f'All features: {ALL_COLS}')

    # ── RUN 1: full features ──────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('RUN 1 — full feature set (including lag_1/2/4)')
    print('='*65)
    oof1, test_preds, r2_full, imp_full = run_cv_loop(
        train_feat, test_feat, y, folds,
        use_cols=ALL_COLS, cat_idx=cat_idx_full,
        save_models=True, label='full',
    )
    nan_mask = np.isnan(oof1)
    print(f'\nRUN 1 OOF R2 (covered {(~nan_mask).mean()*100:.1f}%): '
          f'{r2_full*100:.4f}')

    # ── RUN 2: ablation (no short lags) ──────────────────────────────────────
    print(f'\n{"="*65}')
    print('RUN 2 — ablation: lag_1/2/4 REMOVED  (honest LB proxy)')
    print('='*65)
    oof2, _, r2_ablation, imp_ablation = run_cv_loop(
        train_feat, test_feat, y, folds,
        use_cols=ablation_cols, cat_idx=cat_idx_ablation,
        save_models=False, label='ablation',
    )
    print(f'\nRUN 2 OOF R2 (no short lags): {r2_ablation*100:.4f}')

    # ── feature importances (full model) ─────────────────────────────────────
    print(f'\n{"="*65}')
    print('FEATURE IMPORTANCES — full model (mean gain across 5 folds, top 25)')
    print('='*65)
    imp_df = pd.DataFrame({
        'feature':    ALL_COLS,
        'importance': imp_full,
    }).sort_values('importance', ascending=False)
    total = imp_df['importance'].sum()
    imp_df['pct'] = imp_df['importance'] / total * 100
    print(imp_df.head(25).to_string(index=False))

    # ── save artifacts ────────────────────────────────────────────────────────
    np.save(os.path.join(MODELS_DIR, 'oof_preds.npy'),       oof1)
    np.save(os.path.join(MODELS_DIR, 'test_preds.npy'),      test_preds)
    np.save(os.path.join(MODELS_DIR, 'train_index.npy'),     train_df['Index'].values)
    np.save(os.path.join(MODELS_DIR, 'feat_importance.npy'), imp_full)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.0f}s')
    print(f'RUN 1 OOF R2 (with short lags)  : {r2_full*100:.4f}')
    print(f'RUN 2 OOF R2 (no short lags)    : {r2_ablation*100:.4f}  <- honest LB proxy')
    print(f'Inflation from short lags        : {(r2_full - r2_ablation)*100:+.4f}')

    return r2_full, r2_ablation


if __name__ == '__main__':
    main()
