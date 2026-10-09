"""Inference core shared by the faithful backtest and the ts-arena-models service (numpy + torch/onnx session).

forecast(ts, vals, horizon, freq_minutes) works on ONE series given as it arrives from the platform:
timestamps (epoch seconds, UTC platform clock), values (NaN allowed), possibly irregular/duplicated.
"""
import numpy as np
from tsb import feats

class Predictor:
    """Wraps a callable q = f(ctx, lag, step, base) over numpy arrays."""
    def __init__(self, fn):
        self.fn = fn

def regularize(ts, vals, step_s, L):
    """Place points on a regular grid ending at the last timestamp; returns (W[L], last_ts). NaN where missing."""
    ts = np.asarray(ts, dtype=np.int64); vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts, vals = ts[order], vals[order]
    fin = np.isfinite(vals)
    if not fin.any():
        return None, None
    last_idx = np.flatnonzero(fin)[-1]            # origin = last finite value
    ts, vals = ts[: last_idx + 1], vals[: last_idx + 1]
    last_ts = int(ts[-1])
    k = np.rint((ts - last_ts) / step_s).astype(np.int64)          # <= 0
    W = np.full(L, np.nan, dtype=np.float32)
    keep = k >= -(L - 1)
    W[(L - 1) + k[keep]] = vals[keep]                               # later duplicates overwrite earlier
    return W, last_ts

def forecast_series(predict_np, calendar, spec, ts, vals, horizon):
    """-> (future_ts [horizon] int64, q [horizon, 9] float64 absolute units) or None if the context is unusable."""
    W, last_ts = regularize(ts, vals, spec.step_s, spec.L)
    if W is None:
        return None
    H = spec.H
    pos_ts = last_ts + (np.arange(spec.L + H, dtype=np.int64) - (spec.L - 1)) * spec.step_s
    cal = {k: v[None] for k, v in calendar.features(pos_ts).items()}
    Wfull = np.concatenate([W, np.full(H, np.nan, np.float32)])[None]
    f = feats.build(spec, Wfull, cal, with_targets=False)
    if not f["ok"][0]:
        return None
    q = predict_np(f["ctx"], f["lag"], f["step"], f["base"])[0]          # [H,9] value-1 units
    q = (q + 1.0) * float(f["mu"][0])
    fut = last_ts + np.arange(1, horizon + 1, dtype=np.int64) * spec.step_s
    return fut, q[:horizon].astype(np.float64)
