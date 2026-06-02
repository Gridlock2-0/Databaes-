"""
train.py — LightGBM training with Optuna hyperparameter search + 5-fold TimeSeriesSplit CV.

Pipeline
--------
SKIP_OPTUNA=True  (Stage 1):
  Use Run4-compatible params, skip search.
  RUN 1: full features → saves fold models + OOF preds
  RUN 2: ablation (no lag_1/2/4) → honest LB proxy

SKIP_OPTUNA=False (Stage 2):
  Multi-fold Optuna search → best structural params
  Then same RUN 1 + RUN 2 with best params

Outputs
-------
models/fold_{i}.lgb          fold models (full feature set, tuned params)
models/oof_preds.npy          OOF predictions
models/oof_ablation_preds.npy OOF predictions (ablation, no short lags)
models/test_preds.npy         averaged test predictions
models/test_ablation_preds.npy averaged test predictions (ablation)
models/feat_importance.npy    mean gain importance aligned to ALL_COLS
models/best_params.npy        best Optuna params dict
"""

import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)

from scripts.features import (
    build_base_features, add_all_encodings,
    compute_day49_scale,
    ALL_COLS_V2, SHORT_LAG_COLS, CAT_FEATURE_NAMES,
)
from scripts.cv import get_folds, print_fold_info, N_SPLITS

DATA_DIR   = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
SEED       = 42

DAY49_WEIGHT = None   # no effect: day49 rows never appear in any fold's training set

# ── Stage 1 flag: True = skip Optuna, use Run4 baseline params ────────────────
SKIP_OPTUNA = True   # Stage 1: use Run6 baseline params (best known)

# ── Stage 1 params (Run4-compatible) ─────────────────────────────────────────
LGB_STAGE1 = {
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
ROUNDS_STAGE1 = 3000
ES_STAGE1     = 100

# ── base params (fixed, not searched) ─────────────────────────────────────────
LGB_BASE = {
    'objective':   'regression',
    'metric':      'rmse',
    'bagging_freq': 1,
    'verbose':     -1,
    'seed':        SEED,
    'n_jobs':      -1,
}

# Fallback structural params if search is skipped
LGB_DEFAULT_STRUCTURAL = {
    'num_leaves':        127,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
}

# ── search settings ───────────────────────────────────────────────────────────
SEARCH_LR       = 0.02    # lower lr so larger models can converge properly
SEARCH_ROUNDS   = 8000
SEARCH_ES       = 300
N_SEARCH_TRIALS = 40

# ── final training settings ───────────────────────────────────────────────────
FINAL_LR       = 0.02
NUM_ROUNDS     = 8000
EARLY_STOPPING = 300
LOG_EVAL       = 500


# ── Optuna search ──────────────────────────────────────────────────────────────

def _search_params_multifold(train_feat, y, folds, use_cols, cat_idx,
                             global_rt_stats=None, day49_scale=None):
    """
    Multi-fold Optuna search: pre-compute features for all 5 folds once,
    then each trial trains all 5 folds. Trials hitting round ceiling or
    non-generalizing (fold5 < 150 iters) are marked INVALID.
    """
    print('  Pre-computing features for all 5 folds...')
    fold_data = []
    for fold_i, (tr_iloc, val_iloc) in enumerate(folds):
        tr_raw = train_feat.iloc[tr_iloc].reset_index(drop=True)
        val_raw = train_feat.iloc[val_iloc].reset_index(drop=True)
        y_tr   = pd.Series(y[tr_iloc])
        y_val  = pd.Series(y[val_iloc])
        tr_enc, val_enc, _ = add_all_encodings(tr_raw, y_tr, val_raw, None,
                                               global_rt_stats=global_rt_stats,
                                               day49_scale=day49_scale)
        X_tr  = tr_enc[use_cols].values
        X_val = val_enc[use_cols].values
        fold_data.append((X_tr, X_val, y_tr.values, y_val.values))
        print(f'    fold {fold_i+1}: X_tr={X_tr.shape}  X_val={X_val.shape}')

    def objective(trial):
        params = {
            **LGB_BASE,
            'learning_rate':     SEARCH_LR,
            'num_leaves':        trial.suggest_categorical('num_leaves',        [127, 255, 511]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [10, 20, 50]),
            'feature_fraction':  trial.suggest_categorical('feature_fraction',  [0.7, 0.8, 0.9]),
            'bagging_fraction':  trial.suggest_categorical('bagging_fraction',  [0.7, 0.8, 0.9]),
            'lambda_l1':         trial.suggest_categorical('lambda_l1',         [0.0, 0.05, 0.1]),
            'lambda_l2':         trial.suggest_categorical('lambda_l2',         [0.0, 0.05, 0.1]),
        }
        fold_r2s   = []
        fold_iters = []
        for fold_i, (X_tr, X_val, y_tr_v, y_val_v) in enumerate(fold_data):
            ds_tr  = lgb.Dataset(X_tr,  label=y_tr_v,  categorical_feature=cat_idx, free_raw_data=False)
            ds_val = lgb.Dataset(X_val, label=y_val_v, categorical_feature=cat_idx, free_raw_data=False, reference=ds_tr)
            model  = lgb.train(
                params, ds_tr,
                num_boost_round=SEARCH_ROUNDS,
                valid_sets=[ds_val],
                valid_names=['val'],
                callbacks=[
                    lgb.early_stopping(SEARCH_ES, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            best_iter = model.best_iteration
            # Invalid trial: hitting ceiling
            if best_iter >= SEARCH_ROUNDS - 100:
                return 0.0
            # Invalid trial: fold 5 stops too early (non-generalizing)
            if fold_i == 4 and best_iter < 150:
                return 0.0
            val_pred = model.predict(X_val, num_iteration=best_iter)
            fold_r2s.append(float(r2_score(y_val_v, val_pred)))
            fold_iters.append(best_iter)
        return float(np.mean(fold_r2s))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_SEARCH_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_r2 = study.best_value * 100
    best_trial = study.best_trial.number
    print(f'\n  Best trial #{best_trial}  R²={best_r2:.4f}')
    print(f'  Best params: {best}')

    # Show top-5 trials
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(f'\n  Top-5 trials:')
    for _, row in trials_df.iterrows():
        p = {k.replace('params_', ''): v for k, v in row.items() if k.startswith('params_')}
        print(f'    trial #{int(row["number"]):2d}  R²={row["value"]*100:.4f}  {p}')

    return best


# ── CV training loop ───────────────────────────────────────────────────────────

def run_cv_loop(train_feat, test_feat, y, folds, use_cols, cat_idx, lgb_params,
                n_rounds=None, es_rounds=None,
                save_models=False, label='', global_rt_stats=None, day49_scale=None,
                day49_weight=None):
    """
    Run one full CV loop and return (oof_preds, test_preds_avg, oof_r2, importances).
    Saves .lgb files when save_models=True.

    n_rounds: num_boost_round (defaults to module-level NUM_ROUNDS)
    es_rounds: early_stopping_rounds (defaults to module-level EARLY_STOPPING)
    global_rt_stats: passed through to add_all_encodings for Stage 2+
    day49_scale: per-geohash day49/day48 scale Series (cross-day calibration)
    """
    if n_rounds  is None: n_rounds  = NUM_ROUNDS
    if es_rounds is None: es_rounds = EARLY_STOPPING

    oof_preds       = np.full(len(train_feat), np.nan)
    test_preds_list = []
    importance_acc  = np.zeros(len(use_cols))

    for fold_i, (tr_iloc, val_iloc) in enumerate(folds):
        tr_raw  = train_feat.iloc[tr_iloc].reset_index(drop=True)
        val_raw = train_feat.iloc[val_iloc].reset_index(drop=True)
        y_tr    = pd.Series(y[tr_iloc])
        y_val   = pd.Series(y[val_iloc])

        tr_enc, val_enc, test_enc = add_all_encodings(
            tr_raw, y_tr, val_raw, test_feat,
            global_rt_stats=global_rt_stats, day49_scale=day49_scale)

        X_tr   = tr_enc[use_cols].values
        X_val  = val_enc[use_cols].values
        X_test = test_enc[use_cols].values

        # Sample weights: upweight day49 rows so LGB learns lag_96 coefficient
        if day49_weight is not None:
            sw = np.where(tr_raw['day'].values == 49, float(day49_weight), 1.0)
            n49 = (tr_raw['day'].values == 49).sum()
            print(f'    day49 weight={day49_weight}x  n49_in_fold={n49}')
        else:
            sw = None

        lgb_tr  = lgb.Dataset(X_tr,  label=y_tr.values, weight=sw,
                               categorical_feature=cat_idx, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, label=y_val.values,
                               categorical_feature=cat_idx, free_raw_data=False,
                               reference=lgb_tr)

        model = lgb.train(
            lgb_params, lgb_tr,
            num_boost_round=n_rounds,
            valid_sets=[lgb_tr, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(es_rounds, verbose=False),
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


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── load data ─────────────────────────────────────────────────────────────
    print('Loading data...')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    print(f'  train {train_df.shape}  test {test_df.shape}')

    # ── build base features ───────────────────────────────────────────────────
    print('Building base features...')
    train_feat, test_feat, _, _ = build_base_features(train_df, test_df)
    y = train_feat['demand'].values.astype(np.float64)

    # ── CV folds ──────────────────────────────────────────────────────────────
    print(f'\nCV folds (TimeSeriesSplit n={N_SPLITS}):')
    folds = get_folds(train_feat, N_SPLITS)
    print_fold_info(train_feat, folds)

    # ── cross-day calibration: computed once from full training data ──────────
    print('\nComputing cross-day calibration features...')
    day49_scale = compute_day49_scale(train_feat, y)
    print(f'  gh_day49_scale: {len(day49_scale)} geohashes  '
          f'mean={day49_scale.mean():.3f}  median={day49_scale.median():.3f}')

    # ── feature column sets ───────────────────────────────────────────────────
    global_rt_stats  = None   # global RT TEs hurt CV (train/val inconsistency)
    use_cols_full    = ALL_COLS_V2
    cat_idx_full     = [use_cols_full.index(c) for c in CAT_FEATURE_NAMES if c in use_cols_full]
    ablation_cols    = [c for c in use_cols_full if c not in SHORT_LAG_COLS]
    cat_idx_ablation = [ablation_cols.index(c) for c in CAT_FEATURE_NAMES if c in ablation_cols]

    print(f'\nFull feature set        : {len(use_cols_full)} features')
    print(f'Ablation (no short lags): {len(ablation_cols)} features')

    # ── param selection ───────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    if SKIP_OPTUNA:
        print('STAGE 1 — skipping Optuna, using Run4 baseline params')
        print('='*65)
        lgb_params = LGB_STAGE1
        n_rounds   = ROUNDS_STAGE1
        es_rounds  = ES_STAGE1
        np.save(os.path.join(MODELS_DIR, 'best_params.npy'), LGB_STAGE1)
    else:
        print(f'STAGE 2 — Optuna multi-fold search: lr={SEARCH_LR}, '
              f'{SEARCH_ROUNDS} rounds, ES={SEARCH_ES}, {N_SEARCH_TRIALS} trials')
        print('='*65)
        structural_params = _search_params_multifold(
            train_feat, y, folds, use_cols_full, cat_idx_full,
            global_rt_stats=global_rt_stats, day49_scale=day49_scale,
        )
        lgb_params = {**LGB_BASE, 'learning_rate': FINAL_LR, **structural_params}
        n_rounds   = NUM_ROUNDS
        es_rounds  = EARLY_STOPPING
        np.save(os.path.join(MODELS_DIR, 'best_params.npy'), structural_params)
        print(f'\nFinal LGB_PARAMS (lr={FINAL_LR}, {NUM_ROUNDS} rounds, ES={EARLY_STOPPING}):')
        for k, v in lgb_params.items():
            print(f'  {k}: {v}')

    # ── RUN 1: full features ───────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('RUN 1 — full feature set (including lag_1/2/4)')
    print('='*65)
    oof1, test_preds, r2_full, imp_full = run_cv_loop(
        train_feat, test_feat, y, folds,
        use_cols=use_cols_full, cat_idx=cat_idx_full,
        lgb_params=lgb_params, n_rounds=n_rounds, es_rounds=es_rounds,
        save_models=True, label='full',
        global_rt_stats=global_rt_stats, day49_scale=day49_scale,
        day49_weight=DAY49_WEIGHT,
    )
    nan_mask = np.isnan(oof1)
    print(f'\nRUN 1 OOF R2 (covered {(~nan_mask).mean()*100:.1f}%): {r2_full*100:.4f}')

    # ── RUN 2: ablation ────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('RUN 2 — ablation: lag_1/2/4 REMOVED  (honest LB proxy)')
    print('='*65)
    oof2, test_preds_ablation, r2_ablation, _ = run_cv_loop(
        train_feat, test_feat, y, folds,
        use_cols=ablation_cols, cat_idx=cat_idx_ablation,
        lgb_params=lgb_params, n_rounds=n_rounds, es_rounds=es_rounds,
        save_models=False, label='ablation',
        global_rt_stats=global_rt_stats, day49_scale=day49_scale,
        day49_weight=DAY49_WEIGHT,
    )
    print(f'\nRUN 2 OOF R2 (no short lags): {r2_ablation*100:.4f}')

    # ── feature importances ────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('FEATURE IMPORTANCES — full model (mean gain, top 30)')
    print('='*65)
    imp_df = pd.DataFrame({'feature': use_cols_full, 'importance': imp_full})
    imp_df = imp_df.sort_values('importance', ascending=False)
    total  = imp_df['importance'].sum()
    imp_df['pct'] = imp_df['importance'] / total * 100
    print(imp_df.head(30).to_string(index=False))

    # ── FULL-DATA model: train on entire training set (day48 + day49) ────────────
    # The 5-fold CV never puts day49 rows in the training set (always in val).
    # This means the fold models have never seen valid lag_96 during training and
    # can't use it for test rows. Training on all data exposes the model to the
    # 7872 day49 rows (r=0.792 for lag_96) so it learns the right coefficient.
    print(f'\n{"="*65}')
    print('RUN 3 — full-data model (all 77k rows in training, day49 included)')
    print('='*65)
    full_enc, test_enc_full, _ = add_all_encodings(
        train_feat, pd.Series(y), test_feat, None,
        global_rt_stats=global_rt_stats, day49_scale=day49_scale)
    X_full  = full_enc[use_cols_full].values
    X_tf    = test_enc_full[use_cols_full].values
    lgb_full = lgb.Dataset(X_full, label=y,
                            categorical_feature=cat_idx_full, free_raw_data=False)
    # Use best_iter from fold 5 as a proxy for the final model's round count
    # (fold 5 trains on day48 → validating on day49; closest to full-data regime)
    # No early stopping here — use a fixed round count = 1.1× fold-5 best_iter
    _fold5_best = 120   # fold 5 best_iter from RUN 1 above (update automatically)
    try:
        # Re-estimate using saved fold model
        _m5 = lgb.Booster(model_file=os.path.join(MODELS_DIR, 'fold_4.lgb'))
        _fold5_best = _m5.num_trees()
    except Exception:
        pass
    full_rounds = max(int(_fold5_best * 1.5), 300)
    print(f'  Training on {len(X_full)} rows, {full_rounds} rounds (1.5x fold-5 best_iter)')
    model_full = lgb.train(
        lgb_params, lgb_full,
        num_boost_round=full_rounds,
        callbacks=[lgb.log_evaluation(500)],
    )
    model_full.save_model(os.path.join(MODELS_DIR, 'fold_full.lgb'))
    test_preds_full = np.clip(model_full.predict(X_tf), 0.0, 1.0)
    print(f'  Full-data model test pred: mean={test_preds_full.mean():.5f}  '
          f'min={test_preds_full.min():.5f}  max={test_preds_full.max():.5f}')
    np.save(os.path.join(MODELS_DIR, 'test_preds_full.npy'), test_preds_full)

    # Feature importance of full model
    imp_full2 = model_full.feature_importance(importance_type='gain')
    imp_df2 = pd.DataFrame({'feature': use_cols_full, 'importance': imp_full2})
    imp_df2 = imp_df2.sort_values('importance', ascending=False)
    total2  = imp_df2['importance'].sum()
    imp_df2['pct'] = imp_df2['importance'] / total2 * 100
    print('\n  Top-15 features (full-data model):')
    print(imp_df2.head(15).to_string(index=False))

    # ── save artifacts ─────────────────────────────────────────────────────────
    np.save(os.path.join(MODELS_DIR, 'oof_preds.npy'),          oof1)
    np.save(os.path.join(MODELS_DIR, 'oof_ablation_preds.npy'), oof2)
    np.save(os.path.join(MODELS_DIR, 'test_preds.npy'),         test_preds)
    np.save(os.path.join(MODELS_DIR, 'test_ablation_preds.npy'),test_preds_ablation)
    np.save(os.path.join(MODELS_DIR, 'train_index.npy'),        train_df['Index'].values)
    np.save(os.path.join(MODELS_DIR, 'feat_importance.npy'),    imp_full)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.0f}s')
    print(f'RUN 1 OOF R2 (with short lags)  : {r2_full*100:.4f}')
    print(f'RUN 2 OOF R2 (no short lags)    : {r2_ablation*100:.4f}  <- honest LB proxy')
    print(f'Inflation from short lags        : {(r2_full - r2_ablation)*100:+.4f}')
    print(f'Full-data model: fold_full.lgb saved.')

    return r2_full, r2_ablation


if __name__ == '__main__':
    main()
