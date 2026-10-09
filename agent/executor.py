"""ExperimentExecutor -- leak-safe experiment execution engine.

Responsibilities:
- run experiments through the frozen-fold protocol (core/common.run_cv)
- protocolized: baseline / interactions / target encoding / multi-seed / model matrix
- cache: same (name, featset, seed) reuses OOF directly, no recompute
- log: appends to logs/experiments.jsonl (shared with core.common)
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np

from core.common import ROOT, run_cv, log_experiment, make_submission
from core.dataset import build_dataset
# cfg is injected by the caller (agent orchestration layer); no global data config here


def _cache_paths(name):
    return (os.path.join(ROOT, 'oof', f'{name}_oof.npy'),
            os.path.join(ROOT, 'oof', f'{name}_test.npy'))


class ExperimentExecutor:
    def __init__(self, cfg, n_jobs: int = 20, n_splits: int = 5, use_cache: bool = True):
        self.cfg = cfg
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.use_cache = use_cache
        self.n_folds_used = n_splits  # may drop to 3 under the minimum-cost protocol

    # ---- model factory ----
    def _lgb(self, seed, lr=0.03, num_leaves=63):
        # strictly same config as prior runs (single-variable control); param upgrades are separate deep-dive experiments
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'n_jobs': self.n_jobs,
            'n_estimators': 3000, 'learning_rate': lr, 'num_leaves': num_leaves,
            'max_depth': -1, 'min_child_samples': 50,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'colsample_bytree': 0.8, 'subsample': 0.8, 'subsample_freq': 1,
            'random_state': seed,
        }
        return lgb.LGBMClassifier(**params), params, []

    def _lgb_nb(self, seed):
        """Tuned LGB params: depth5/leaves247/colsample0.30/mcw10/max_bin1024/lr0.02.
        Same params as a reference notebook (seed 60) measured 0.94548."""
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'n_jobs': self.n_jobs,
            'n_estimators': 20000, 'learning_rate': 0.02,
            'max_depth': 5, 'num_leaves': 247, 'min_child_samples': 10,
            'colsample_bytree': 0.3029300829885024,
            'reg_alpha': 0.07094285437903122, 'reg_lambda': 2.033039097703242495,
            'max_bin': 1024, 'feature_pre_filter': False,
            'random_state': seed,
        }
        callbacks = [lgb.early_stopping(500, verbose=False)]
        return lgb.LGBMClassifier(**params), params, callbacks

    def _xgb(self, seed):
        import xgboost as xgb
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': self.n_jobs,
            'n_estimators': 20000, 'learning_rate': 0.02, 'max_depth': 7,
            'min_child_weight': 5, 'reg_alpha': 0.3, 'reg_lambda': 5.0,
            'colsample_bytree': 0.7, 'subsample': 0.8, 'random_state': seed,
            'early_stopping_rounds': 200,
        }
        return xgb.XGBClassifier(**params), params, None

    def _cat(self, seed):
        from catboost import CatBoostClassifier
        params = {
            'iterations': 20000, 'learning_rate': 0.02, 'depth': 8, 'l2_leaf_reg': 5.0,
            'random_seed': seed, 'verbose': False, 'eval_metric': 'AUC',
            'early_stopping_rounds': 200, 'thread_count': self.n_jobs,
            'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'rsm': 0.7,
        }
        return CatBoostClassifier(**params), params, None

    def _xgb_nb1(self, seed):
        """nb1 recipe: depth6 / mcw30 / lambda8 / lr0.01 / 20k trees, early stopping 200."""
        import xgboost as xgb
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': self.n_jobs,
            'n_estimators': 20000, 'learning_rate': 0.01, 'max_depth': 6,
            'min_child_weight': 30, 'reg_alpha': 0.05, 'reg_lambda': 8.0,
            'colsample_bytree': 0.8, 'subsample': 0.8, 'tree_method': 'hist',
            'random_state': seed, 'early_stopping_rounds': 200,
        }
        return xgb.XGBClassifier(**params), params, None

    def _hgbc(self, seed):
        """nb2 recipe: lr0.2 / max_leaf15 / min_samples160."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = {
            'learning_rate': 0.2, 'max_leaf_nodes': 15, 'min_samples_leaf': 160,
            'max_iter': 2000, 'l2_regularization': 1.0, 'early_stopping': True,
            'validation_fraction': 0.1, 'n_iter_no_change': 50, 'random_state': seed,
        }
        return HistGradientBoostingClassifier(**params), params, None

    def _svc(self, seed):
        """Optional plugin: stratified-subsample RBF SVC (core/hetero.py, not bundled).

        Heterogeneous models (SVM/KNN family) are the usual counterexample to
        the "diversity needs strength underneath" principle: low correlation
        with the tree stack does not imply incremental value -- it must pass
        the gate. Full implementation and measurements live in the system
        layer; see docs/tutorial.md step 3.
        """
        try:
            from core.hetero import SubsampledSVC
        except ImportError as e:
            raise ImportError('hetero plugin not installed: optional extension, see module docstring') from e
        params = {'n_sub': 40_000, 'pca_components': 20, 'C': 1.0, 'seed': seed}
        return SubsampledSVC(**params), params, None

    def _knn(self, seed):
        """Optional plugin: PCA + distance-weighted KNN (core/hetero.py, not bundled)."""
        try:
            from core.hetero import PCAKNN
        except ImportError as e:
            raise ImportError('hetero plugin not installed: optional extension, see module docstring') from e
        params = {'pca_components': 15, 'k': 300, 'chunk': 512, 'seed': seed}
        return PCAKNN(**params), params, None

    def model_factory(self, kind, seed, param_overrides=None):
        """Return (fresh-model factory, params, callbacks) for run_cv to instantiate per fold.

        param_overrides: sweep variants. The factory builds a base instance per
        fold and applies overrides via set_params; the returned params already
        include them, so the log shows exactly what ran. Custom models (svc/knn)
        without set_params raise explicitly instead of silently ignoring.
        """
        builder = {'lgb': self._lgb, 'lgb_nb': self._lgb_nb,
                   'xgb': self._xgb, 'cat': self._cat,
                   'xgb_nb1': self._xgb_nb1, 'hgbc': self._hgbc,
                   'svc': self._svc, 'knn': self._knn}[kind]
        _, params, callbacks = builder(seed)
        if not param_overrides:
            return (lambda: builder(seed)[0]), params, callbacks
        params = {**params, **param_overrides}

        def factory():
            model = builder(seed)[0]
            try:
                model.set_params(**param_overrides)
            except (AttributeError, TypeError) as e:
                raise ValueError(f'{kind} does not support param_overrides: {e}') from e
            return model

        return factory, params, callbacks

    # ---- single-variable control protocols ----
    def protocol_v2_batch(self, featset, kind='lgb', seed=42,
                          param_overrides=None, name=None, notes=None, probe_folds=None):
        """Single-variable control: train a model on a feature set (OOF cache reused).

        param_overrides: sweep variants. When name is omitted, a stable hash suffix
        is derived from the overrides to avoid cache-name collisions.
        probe_folds: probe mode, first k folds only, nothing cached.
        """
        import json as _json
        import zlib

        factory, params, cb = self.model_factory(kind, seed, param_overrides)
        if name is None:
            name = f'{kind}_{featset}_s{seed}'
            if param_overrides:
                payload = _json.dumps(sorted(param_overrides.items()), sort_keys=True)
                name += f'_o{zlib.crc32(payload.encode()):08x}'
        return self.train_eval(name, factory, params, cb, featset,
                               notes=notes or f'{featset} {kind}',
                               probe_folds=probe_folds)

    # ---- cache ----
    def _load_cache(self, name):
        oof_p, test_p = _cache_paths(name)
        if self.use_cache and os.path.exists(oof_p) and os.path.exists(test_p):
            return np.load(oof_p), np.load(test_p)
        return None

    # ---- core execution ----
    def train_eval(self, name, model_fn, params, callbacks, featset,
                   notes=None, save_submission=False, test_df=None, probe_folds=None):
        """Single-model k-fold CV (with cache). Returns dict(oof, test_pred, oof_auc).

        probe_folds: probe mode passed through to run_cv (first k folds only,
        nothing cached). Probe results still land in experiments.jsonl (notes marked probe).
        """
        cached = self._load_cache(name)
        train, test, feats = build_dataset(self.cfg, featset)
        if cached is not None:
            oof, test_pred = cached
            from sklearn.metrics import roc_auc_score
            oof_auc = roc_auc_score(train['y'], oof)
            print(f'  [cache] {name}: OOF {oof_auc:.5f}', flush=True)
            return {'name': name, 'oof': oof, 'test_pred': test_pred,
                    'oof_auc': oof_auc, 'cached': True}
        oof, test_pred = run_cv(model_fn, train, test, feats, name=name,
                                params=params, n_splits=self.n_splits,
                                notes=notes, callbacks=callbacks,
                                probe_folds=probe_folds)
        from sklearn.metrics import roc_auc_score
        if probe_folds is not None:
            # probe mode: OOF covers only the first k folds, score on covered rows only
            cov = (train['fold'] < int(probe_folds)).to_numpy()
            oof_auc = roc_auc_score(train['y'].to_numpy()[cov], oof[cov])
        else:
            oof_auc = roc_auc_score(train['y'], oof)
        if save_submission:
            make_submission(test, test_pred, f'sub_{name}.csv', self.cfg)
        return {'name': name, 'oof': oof, 'test_pred': test_pred,
                'oof_auc': oof_auc, 'cached': False}

    # ---- protocols ----
    def protocol_baseline(self, featset='raw', seed=42):
        """Phase-1 baseline: single model on raw features."""
        factory, params, cb = self.model_factory('lgb', seed)
        return self.train_eval(f'lgb_{featset}_s{seed}', factory, params, cb,
                               featset, notes=f'baseline seed{seed}')

    def protocol_interaction(self, featset='inter', seed=42, base_result=None):
        """Interaction features as a whole-group A/B (guards against gain misjudgment)."""
        factory, params, cb = self.model_factory('lgb', seed)
        name = f'lgb_{featset}_s{seed}'
        res = self.train_eval(name, factory, params, cb, featset,
                              notes='interaction group ablation')
        delta = self._delta(base_result, res)
        return res, delta

    def protocol_te(self, featset='te', seed=42, base_result=None, smoothing=30):
        """Target encoding protocol (fold-safe OOF)."""
        factory, params, cb = self.model_factory('lgb', seed)
        name = f'lgb_{featset}_s{seed}'
        res = self.train_eval(name, factory, params, cb, featset,
                              notes=f'TE smoothing={smoothing}')
        delta = self._delta(base_result, res)
        return res, delta

    def protocol_multi_seed(self, featset='combo', seeds=(42, 123, 456), tag='ms'):
        """Multi-seed equal-weight ensemble (average of OOF and test)."""
        oofs, preds, name = [], [], f'lgb_{featset}_{tag}'
        for s in seeds:
            factory, params, cb = self.model_factory('lgb', s)
            r = self.train_eval(f'lgb_{featset}_s{s}', factory, params, cb, featset,
                                notes=f'multi-seed member {s}')
            oofs.append(r['oof'])
            preds.append(r['test_pred'])
        oof, test_pred = np.mean(oofs, axis=0), np.mean(preds, axis=0)
        np.save(_cache_paths(name)[0], oof)
        np.save(_cache_paths(name)[1], test_pred)
        log_experiment(name, {'seeds': list(seeds)}, [], 0.0, notes='multi-seed mean (auc filled by evaluator)')
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(build_dataset(self.cfg, featset)[0]['y'], oof)
        print(f'  multi-seed {name}: OOF {auc:.5f}', flush=True)
        return {'name': name, 'oof': oof, 'test_pred': test_pred, 'oof_auc': auc}

    @staticmethod
    def _delta(base, res):
        if base is None or base.get('oof_auc') in (None, 0.0):
            return None
        return res['oof_auc'] - base['oof_auc']
