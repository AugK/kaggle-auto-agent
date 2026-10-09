"""Feature set assembly: raw / inter / te / combo, all config-driven.

The four feature sets carry the single-variable-control discipline -- each
tier adds exactly one group of features so score changes can be attributed to
that group (docs/principles.md #2). Any new feature idea should first answer:
which tier does it belong to, and can it be validated as its own tier?
"""
import pandas as pd

from core.common import DataConfig, load_data, get_folds
from core.features import (add_interaction_features, add_quantile_bins,
                           add_te_features, ordinal_encode)


def numeric_feats(df, cfg):
    skip = {cfg.id_col, 'y', 'fold', cfg.target_col}
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in 'ifub']


def build_dataset(cfg, kind='combo'):
    """Returns (train, test, feature_cols). kind in {raw, inter, te, combo}."""
    train, test = load_data(cfg)
    train = get_folds(train, cfg)

    if kind == 'raw':
        tr, te = ordinal_encode(train, test, cfg.cat_cols)
        return tr, te, numeric_feats(tr, cfg)

    if kind == 'inter':
        tr, te, combo_cols = add_interaction_features(train, test, cfg.cat_cols)
        tr, te = ordinal_encode(tr, te, cfg.cat_cols + combo_cols)
        return tr, te, numeric_feats(tr, cfg)

    if kind == 'te':
        tr, te, bin_cols = add_quantile_bins(train, test, cfg.num_cols)
        te_keys = cfg.cat_cols[:3] + bin_cols[:3]   # example keys: low-card cats + bins
        tr, te, te_cols = add_te_features(tr, te, te_keys)
        tr, te = ordinal_encode(tr, te, cfg.cat_cols)
        return tr, te, numeric_feats(tr, cfg)

    if kind == 'combo':
        tr, te, combo_cols = add_interaction_features(train, test, cfg.cat_cols)
        tr, te, bin_cols = add_quantile_bins(tr, te, cfg.num_cols)
        te_keys = cfg.cat_cols[:3] + bin_cols[:3] + combo_cols[:3]
        tr, te, te_cols = add_te_features(tr, te, te_keys)
        tr, te = ordinal_encode(tr, te, cfg.cat_cols + combo_cols)
        return tr, te, numeric_feats(tr, cfg)

    raise ValueError(kind)
