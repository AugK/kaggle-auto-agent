"""Generic feature utilities: ordinal encode, interactions, quantile bins, fold-safe TE.

Every function here only depends on DataConfig column conventions -- no
literal from any particular competition.

Fold-safe TE is where leakage happens most easily: encoding statistics
must be computed on each fold's training rows only. add_te_features
encodes per fold via the passed fold column and refuses to run without
one, rather than falling back to full-data statistics (full-data TE =
feeding validation answers into the features).
"""
import numpy as np
import pandas as pd


def ordinal_encode(train, test, cols):
    """Category -> int, categories resolved on train+test union, unseen -> -1."""
    tr, te = train.copy(), test.copy()
    for c in cols:
        cats = sorted(set(tr[c].astype(str).unique()) | set(te[c].astype(str).unique()))
        mapping = {v: i for i, v in enumerate(cats)}
        tr[c] = tr[c].astype(str).map(mapping).fillna(-1).astype(np.int16)
        te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(np.int16)
    return tr, te


def add_interaction_features(train, test, cat_cols, max_pairs=10):
    """Pairwise category concatenation (a_b) as combo keys; pairs ranked by
    chi-square/mutual information on train, top max_pairs kept.

    This simplified implementation pairs columns in order. In a real
    competition, picking the top-K pairs by mutual information is worth it --
    which pairs to pick is itself a judgment-layer decision, see
    docs/principles.md #2 (single-variable control).
    """
    tr, te = train.copy(), test.copy()
    pairs = [(a, b) for i, a in enumerate(cat_cols) for b in cat_cols[i + 1:]]
    pairs = pairs[:max_pairs]
    for a, b in pairs:
        name = f'{a}__{b}'
        tr[name] = tr[a].astype(str) + '_' + tr[b].astype(str)
        te[name] = te[a].astype(str) + '_' + te[b].astype(str)
    return tr, te, [f'{a}__{b}' for a, b in pairs]


def add_quantile_bins(train, test, num_cols, n_bins=10):
    """Quantile bins for numeric columns (edges computed on train, reused on test)."""
    tr, te = train.copy(), test.copy()
    bin_cols = []
    for c in num_cols:
        name = f'{c}_bin'
        _, edges = pd.qcut(tr[c].rank(method='first'), n_bins, retbins=True, duplicates='drop')
        edges[0], edges[-1] = -np.inf, np.inf
        tr[name] = pd.cut(tr[c], bins=edges, labels=False, include_lowest=True).astype('Int16')
        te[name] = pd.cut(te[c], bins=edges, labels=False, include_lowest=True).astype('Int16')
        bin_cols.append(name)
    return tr, te, bin_cols


def _te_frame(tr_part, va_part, te, cols, smoothing):
    """Fit TE on tr_part, apply to va_part/te. smoothing: number of phantom rows m."""
    prior = tr_part['y'].mean()
    stats = tr_part.groupby(cols).agg(sum_y=('y', 'sum'), n=('y', 'size'))
    enc = (stats['sum_y'] + prior * smoothing) / (stats['n'] + smoothing)
    key = lambda df: df[cols].astype(str).agg('_'.join, axis=1)
    out_va = key(va_part).map(enc).fillna(prior)
    out_te = key(te).map(enc).fillna(prior)
    return out_va, out_te, prior


def add_te_features(train, test, te_cols, fold_col='fold', smoothing=30, prefix='te'):
    """Fold-safe target encoding: per-fold statistics come only from that fold's
    training rows.

    te_cols: a single column name (str) or a list (combo-key TE).
    Appends a <prefix>__<cols> column; test is encoded with full-train stats.
    """
    cols = [te_cols] if isinstance(te_cols, str) else list(te_cols)
    cname = f'{prefix}__' + '_'.join(cols)
    assert fold_col in train.columns, 'TE needs a fold column (leak-safe encoding); run get_folds first'
    train[cname] = np.nan
    for f in sorted(train[fold_col].unique()):
        tr_part = train[train[fold_col] != f]
        va_part = train[train[fold_col] == f]
        out_va, _, _ = _te_frame(tr_part, va_part, test, cols, smoothing)
        train.loc[va_part.index, cname] = out_va.values
    _, out_te, _ = _te_frame(train, train.head(0), test, cols, smoothing)
    test[cname] = out_te.values
    return train, test, [cname]
