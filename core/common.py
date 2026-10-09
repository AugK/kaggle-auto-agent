"""Shared data loading, fold freezing, CV runner and experiment log.

Everything competition-specific (paths, column names, positive label) lives in
DataConfig; this module contains no literal from any particular competition.

Frozen folds and the per-fold gate are the two core disciplines of this
framework:
- Frozen folds: split once, persist, reuse forever. Comparisons are only
  meaningful under the same split, otherwise +/-0.000x differences are noise.
- Fold gate: after each fold, abort when the running score is clearly worse
  than the reference -- a knowingly-worse config must not burn remaining folds.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_LOG = os.path.join(ROOT, 'logs', 'experiments.jsonl')


@dataclass
class DataConfig:
    """All data conventions for one competition (or one experiment run).

    id_col/target_col/pos_label decide how data is read;
    num_cols/cat_cols decide feature engineering (binning only for num_cols,
    TE keys derived from cat_cols + bins).
    """
    name: str
    train_path: str
    test_path: str
    id_col: str = 'id'
    target_col: str = 'target'
    pos_label: object = 1
    num_cols: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)


def load_config(name):
    """Load a DataConfig from configs/<name>.json."""
    path = os.path.join(ROOT, 'configs', f'{name}.json')
    with open(path, encoding='utf-8') as f:
        return DataConfig(**json.load(f))


def load_data(cfg):
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    train['y'] = (train[cfg.target_col] == cfg.pos_label).astype(np.int8)
    return train, test


def _fold_file(cfg):
    return os.path.join(ROOT, 'data', f'{cfg.name}_fold_assignment.csv')


def get_folds(train, cfg, n_splits=5, seed=42):
    """Frozen StratifiedKFold assignment, persisted to disk (per config)."""
    ffile = _fold_file(cfg)
    if os.path.exists(ffile):
        fold_map = pd.read_csv(ffile, index_col=cfg.id_col)['fold']
        train['fold'] = train[cfg.id_col].map(fold_map)
        assert train['fold'].notna().all() and train['fold'].max() == n_splits - 1
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train['fold'] = -1
        for i, (_, val_idx) in enumerate(skf.split(train, train['y'])):
            train.iloc[val_idx, train.columns.get_loc('fold')] = i
        train[[cfg.id_col, 'fold']].to_csv(ffile, index=False)
    return train


def log_experiment(name, params, fold_scores, oof_auc, submission_file=None,
                   lb_score=None, notes=None):
    record = {
        'timestamp': datetime.now().isoformat(),
        'name': name,
        'params': params,
        'fold_scores': [round(float(s), 5) for s in fold_scores],
        'fold_std': round(float(np.std(fold_scores)), 5),
        'oof_auc': round(float(oof_auc), 5),
        'submission_file': submission_file,
        'public_lb': lb_score,
        'notes': notes,
    }
    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    with open(EXPERIMENT_LOG, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record


class FoldGateAbort(Exception):
    """Raised when cumulative fold scores fall too far below the gate reference."""


def run_cv(model_fn, train, X_test, feature_cols, name='model', params=None,
           n_splits=5, cat_features=None, notes=None, save_oof=True, callbacks=None,
           gate_ref_folds=None, gate_margin=0.0005, probe_folds=None):
    """Unified k-fold CV with frozen folds. Returns (oof, test_pred).

    gate_ref_folds: reference model's per-fold scores (same frozen folds). After
    each fold, if the running mean is worse than the reference running mean by
    more than gate_margin, raise FoldGateAbort -- caller keeps no artifacts.

    probe_folds: probe mode -- run only the first k folds. OOF is scored on
    covered rows only and NOT cached (a partial OOF would pollute the reuse
    pool); the log entry is marked probe. For corner screening before a
    full-parameter sweep.
    """
    n_used = n_splits if probe_folds is None else min(int(probe_folds), n_splits)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(X_test))
    fold_scores = []
    covered = np.zeros(len(train), dtype=bool)

    for fold_idx in range(n_used):
        tr_mask = train['fold'] != fold_idx
        va_mask = train['fold'] == fold_idx
        X_tr, y_tr = train.loc[tr_mask, feature_cols], train.loc[tr_mask, 'y']
        X_va, y_va = train.loc[va_mask, feature_cols], train.loc[va_mask, 'y']

        model = model_fn()
        fit_kwargs = {}
        if cat_features:
            fit_kwargs['categorical_feature'] = cat_features
        if callbacks:  # lgb only; xgb/catboost reject this kwarg in recent versions
            fit_kwargs['callbacks'] = callbacks
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], **fit_kwargs)
        except TypeError:
            # sklearn estimators (e.g. HistGradientBoosting) take X_val/y_val,
            # or do their own internal early stopping -- fit without eval_set.
            model.fit(X_tr, y_tr, **fit_kwargs)
        oof[va_mask.values] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_test[feature_cols])[:, 1] / n_used
        covered[va_mask.values] = True
        s = roc_auc_score(y_va, oof[va_mask.values])
        fold_scores.append(s)
        print(f"  fold {fold_idx}: {s:.5f}", flush=True)

        if gate_ref_folds is not None:
            cur = float(np.mean(fold_scores))
            ref = float(np.mean(gate_ref_folds[:len(fold_scores)]))
            if cur < ref - gate_margin:
                msg = (f'fold {fold_idx}: cum {cur:.5f} < ref {ref:.5f} - {gate_margin}'
                       f' -> ABORT (clearly worse than reference)')
                print(f"  [gate] {msg}", flush=True)
                raise FoldGateAbort(msg)

    oof_auc = roc_auc_score(train.loc[covered, 'y'], oof[covered])
    print(f"  OOF AUC: {oof_auc:.5f} (std {np.std(fold_scores):.5f})"
          + (f" [probe {n_used}/{n_splits} folds]" if probe_folds is not None else ""),
          flush=True)

    if probe_folds is not None:
        notes = f'{notes or ""} [probe {n_used}/{n_splits} folds, not cached]'.strip()
    log_experiment(name, params or {}, fold_scores, oof_auc, notes=notes)
    if save_oof and probe_folds is None:
        os.makedirs(os.path.join(ROOT, 'oof'), exist_ok=True)
        np.save(os.path.join(ROOT, 'oof', f'{name}_oof.npy'), oof)
        np.save(os.path.join(ROOT, 'oof', f'{name}_test.npy'), test_pred)
    return oof, test_pred


def make_submission(test, pred, filename, cfg):
    sub = pd.DataFrame({cfg.id_col: test[cfg.id_col], cfg.target_col: pred})
    assert len(sub) == len(test) and sub[cfg.target_col].between(0, 1).all()
    path = os.path.join(ROOT, 'submissions', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"saved {path}: mean={pred.mean():.5f} min={pred.min():.5f} max={pred.max():.5f}")
    return path
