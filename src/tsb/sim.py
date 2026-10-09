"""Simulated (final-data) rounds on the platform-clock bank, scored exactly like the arena, + baselines."""
import numpy as np, pandas as pd
from tsb import feats, bank, cal_np, metrics, ROOT

LEV = np.array(metrics.LEVELS)

class Bank:
    def __init__(self, freq, no_hol=False, spec_kwargs=None):
        b = bank.load(freq)
        self.freq = freq; self.X = b["X"]; self.t0 = int(b["t0"]); self.step = int(b["step"])
        self.N = self.X.shape[1]
        z = np.load(f"{ROOT}/data/cache/hol_table.npz")
        tab = np.zeros_like(z["table"]) if no_hol else z["table"]
        self.C = cal_np.Calendar(int(z["day0"]), tab)
        self.P = self.t0 + np.arange(self.N, dtype=np.int64) * self.step
        self.cg = self.C.features(self.P)
        self.spec = feats.Spec(freq, **(spec_kwargs or {}))

    def idx_of(self, iso):
        return int((int(pd.Timestamp(iso).timestamp()) - self.t0) // self.step)

    def window(self, s, o):
        sp = self.spec; a = o - sp.L + 1; b = o + sp.H + 1
        return self.X[s, a:b]

    def cal_window(self, o):
        sp = self.spec; a = o - sp.L + 1; b = o + sp.H + 1
        return {k: v[a:b] for k, v in self.cg.items()}

    def batch(self, ss, oo, with_targets=True):
        sp = self.spec
        W = np.stack([self.window(s, o) for s, o in zip(ss, oo)])
        cal = {k: np.stack([self.cg[k][o - sp.L + 1: o + sp.H + 1] for o in oo]) for k in self.cg}
        return feats.build(sp, W, cal, with_targets=with_targets)

def round_origins(bk, start, end, phase_hhmm):
    """origin indices (platform clock) for daily rounds start..end (inclusive dates) at hh:mm."""
    days = pd.date_range(start, end, freq="D", tz="UTC")
    hh, mm = map(int, phase_hhmm.split(":"))
    return [bk.idx_of(d + pd.Timedelta(hours=hh, minutes=mm)) for d in days]

def predict(model, bk, ss, oo, device="cpu"):
    """-> q [n, H, 9] in absolute units, plus feature dict."""
    import torch
    f = bk.batch(ss, oo, with_targets=False)
    with torch.no_grad():
        q = model(torch.from_numpy(f["ctx"]), torch.from_numpy(f["lag"]), torch.from_numpy(f["step"]), torch.from_numpy(f["base"])).numpy()
    mu = f["mu"][:, None, None]
    return (q + 1.0) * mu, f

def score_rounds(bk, origins, predict_fn):
    """origins: list of origin idx (one per round). predict_fn(ss, oo) -> q [n,H,9] abs units. Returns DataFrame per (round, series)."""
    S = bk.X.shape[0]; H = bk.spec.H
    rows = []
    ss = np.repeat(np.arange(S), len(origins)); oo = np.tile(np.array(origins), S)
    q = predict_fn(ss, oo)
    for i, (s, o) in enumerate(zip(ss, oo)):
        y = bk.X[s, o + 1:o + H + 1]
        if not np.isfinite(y).all() or not np.isfinite(bk.X[s, o]):
            continue
        sc = metrics.arena_series_scores(y, q[i, :, 4], q[i].T, bk.X[s, o])
        cover = float(((y >= q[i, :, 0]) & (y <= q[i, :, 8])).mean())
        rows.append((int(o), int(s), sc["mase"], sc["sql"], cover))
    return pd.DataFrame(rows, columns=["origin", "series", "mase", "sql", "cover80"])

def baseline_predict(bk, kind):
    P = bk.spec.P; H = bk.spec.H
    def fn(ss, oo):
        out = np.zeros((len(ss), H, 9), np.float32)
        for i, (s, o) in enumerate(zip(ss, oo)):
            x = bk.X[s]
            if kind == "persist":
                m = np.full(H, x[o])
            elif kind == "lag7":
                m = x[o + 1 - 7 * P: o + 1 - 7 * P + H] if H <= 7 * P else None
            elif kind == "lagmean":
                idx = np.arange(1, H + 1)
                vals = np.stack([x[o + idx - P * d] for d in range(1, 8) if True], 0) if True else None
                dmin = np.ceil(idx / P).astype(int)
                vals = np.stack([x[o + idx - P * (dmin + k)] for k in range(7)], 0)
                m = np.nanmedian(vals, 0)
            m = np.nan_to_num(m, nan=float(x[o]))
            out[i] = m[:, None]
        return out
    return fn
