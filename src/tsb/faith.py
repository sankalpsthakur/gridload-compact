"""Platform-faithful scoring of our model on archive rounds: served contexts, first-published truth, per-series anchors."""
import numpy as np, pandas as pd
from tsb import archive, metrics, infer, ROOT

def load_rounds(defn, day_from=None, day_to=None):
    rd = archive.round_dirs(defn)
    out = []
    for day, sub in sorted(rd.items()):
        d = day[:10]
        if day_from and d < day_from: continue
        if day_to and d > day_to: continue
        ctx, rid = archive.context(defn, sub)
        tr = archive.truth(defn, sub)
        out.append(dict(day=d, round_id=rid, ctx=ctx, truth=tr))
    return out

def score_model(rounds, forecast_fn, horizon, step_s):
    """forecast_fn(ts[int64 s], vals[float]) -> (future_ts[H], q[H,9]) or None.  Returns per-(round,series) DataFrame."""
    rows = []
    for r in rounds:
        for uid, c in r["ctx"].items():
            ts = np.asarray(c.index.as_unit("s").asi8, dtype=np.int64); vals = c.to_numpy(np.float64)
            res = forecast_fn(ts, vals)
            y = r["truth"][uid]
            naive = float(vals[np.flatnonzero(np.isfinite(vals))[-1]])
            if res is None:
                rows.append((r["day"], r["round_id"], archive.UID2REG[uid], np.nan, np.nan, 0, np.nan)); continue
            fut, q = res
            f = pd.DataFrame(q, index=pd.to_datetime(fut, unit="s", utc=True))
            j = f.join(y.rename("y"), how="inner").dropna()
            if len(j) == 0:
                rows.append((r["day"], r["round_id"], archive.UID2REG[uid], np.nan, np.nan, 0, np.nan)); continue
            qq = j.drop(columns="y").to_numpy().T
            sc = metrics.arena_series_scores(j.y.to_numpy(), qq[4], qq, naive)
            cover = float(((j.y.to_numpy() >= qq[0]) & (j.y.to_numpy() <= qq[8])).mean())
            rows.append((r["day"], r["round_id"], archive.UID2REG[uid], sc["mase"], sc["sql"], len(j), cover))
    return pd.DataFrame(rows, columns=["day", "round_id", "series", "mase", "sql", "n", "cover80"])

def field_matrix(defn, metric="mase", days=None, extra=None):
    """Per-round mean-over-series matrix [rounds x models] from the official-style field cache (+ extra columns {name: Series by day})."""
    f = pd.read_parquet(f"{ROOT}/data/cache/field_def{defn}.parquet")
    if metric == "sql":
        f = f[f.has_q]
    g = f.groupby(["day", "model"])[metric].mean().unstack("model")
    if days is not None:
        g = g.loc[[d for d in g.index if d[:10] in set(days)]]
    if extra:
        for name, s in extra.items():
            g[name] = s.reindex(g.index.str[:10]).to_numpy()
    return g
