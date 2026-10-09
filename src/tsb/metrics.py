"""TS-Arena scoring, re-implemented from DAG-UPB/ts-arena-backend (MIT).

Sources mirrored (read 2026-10-07, backend HEAD 2863da2):
  api-portal/app/services/score_evaluation_service.py  (arena MASE = MAE_model / MAE_persistence)
  api-portal/app/services/forecast_metrics.py          (SQL, fev quantile loss, same scale)
  api-portal/app/services/elo_ranking_service.py       (per-round averaging, bootstrapped Elo, K=4)

Arena MASE is NOT Hyndman-Koehler MASE: the denominator is the MAE of the flat last-context-value
forecast over the evaluated horizon timestamps.  Both are lower-is-better.
"""
from __future__ import annotations
import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QKEYS = tuple(f"q_{l}" for l in LEVELS)


def quantile_loss(y, q_pred, level):
    """fev form: 2*|(y-q) * (1(y<=q) - level)|."""
    y = np.asarray(y, float); q_pred = np.asarray(q_pred, float)
    return 2.0 * np.abs((y - q_pred) * ((y <= q_pred).astype(float) - level))


def arena_series_scores(y_true, y_point, q_pred, naive_last):
    """One (round, model, series). q_pred: array [9, T] or None (point-only => degenerate deciles).
    Returns dict(mase, sql, mae, mae_naive).  Points with non-finite pred/truth are dropped by caller."""
    y_true = np.asarray(y_true, float); y_point = np.asarray(y_point, float)
    mae = float(np.mean(np.abs(y_true - y_point)))
    mae_naive = float(np.mean(np.abs(y_true - naive_last)))
    if mae_naive <= 0:
        return dict(mase=np.nan, sql=np.nan, mae=mae, mae_naive=mae_naive)
    if q_pred is None:
        q_pred = np.tile(y_point, (len(LEVELS), 1))
    else:
        q_pred = np.sort(np.asarray(q_pred, float), axis=0)  # platform repairs crossings by sorting
    sql = float(np.mean([np.mean(quantile_loss(y_true, q_pred[i], l)) / mae_naive for i, l in enumerate(LEVELS)]))
    return dict(mase=mae / mae_naive, sql=sql, mae=mae, mae_naive=mae_naive)


def context_scale_mase(y_true, y_point, context_values, m=1):
    """Hyndman-Koehler MASE with the in-sample naive lag-m scale of the served context (the new,
    not-yet-ranked 'forecast_scores' pipeline).  Reported as a secondary view only."""
    c = np.asarray(context_values, float)
    c = c[np.isfinite(c)]
    if len(c) <= m:
        return np.nan
    scale = float(np.mean(np.abs(c[m:] - c[:-m])))
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_point, float))) / scale) if scale > 0 else np.nan


def elo_bootstrap(matrix, n_boot=500, k=4.0, base=1000.0, seed=0):
    """Replicates EloRankingService._run_single_bootstrap.
    matrix: [n_rounds, n_models] of per-round mean metric (NaN = absent).  Returns [n_boot, n_models]."""
    rng = np.random.default_rng(seed)
    M = np.asarray(matrix, float)
    n_rounds, n_models = M.shape
    out = np.zeros((n_boot, n_models))
    valid = ~np.isnan(M)
    for b in range(n_boot):
        r = np.full(n_models, base)
        for ri in rng.permutation(n_rounds):
            idx = np.flatnonzero(valid[ri])
            if len(idx) < 2:
                continue
            v = M[ri, idx]
            ra = r[idx]
            diff = v[:, None] - v[None, :]
            outcome = np.where(diff < 0, 1.0, np.where(diff == 0, 0.5, 0.0))
            expected = 1.0 / (1.0 + 10.0 ** ((ra[None, :] - ra[:, None]) / 400.0))
            np.fill_diagonal(outcome, 0.0); np.fill_diagonal(expected, 0.0)
            r[idx] += k * (outcome.sum(1) - expected.sum(1))
        out[b] = r
    return out


def elo_summary(matrix, names, **kw):
    boots = elo_bootstrap(matrix, **kw)
    med = np.median(boots, 0); lo = np.percentile(boots, 2.5, 0); hi = np.percentile(boots, 97.5, 0)
    n = np.sum(~np.isnan(np.asarray(matrix, float)), 0)
    import pandas as pd
    return pd.DataFrame(dict(model=names, elo=med, lo=lo, hi=hi, rounds=n)).sort_values("elo", ascending=False).reset_index(drop=True)


def bootstrap_mean_ci(x, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(a, b, n_boot=2000, seed=0, alpha=0.05):
    """CI for mean(a-b) over paired rounds (negative => a better for lower-is-better metrics)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))), float((d < 0).mean())
