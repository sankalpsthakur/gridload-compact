"""Tables for a faithful window: our model(s) inserted into the real archive field, TS-Arena style."""
import numpy as np, pandas as pd
from tsb import metrics, ROOT

def week_of(day):
    d = pd.Timestamp(day[:10])
    return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"

def block_ci(values_by_day, n_boot=2000, seed=0):
    """Block bootstrap over ISO weeks: values_by_day = Series(index=day, values). Returns mean, lo, hi."""
    s = values_by_day.dropna()
    wk = s.groupby([week_of(d) for d in s.index]).apply(lambda x: x.to_numpy())
    blocks = list(wk.values)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        means.append(np.concatenate([blocks[i] for i in pick]).mean())
    return float(s.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def paired_block_diff(a, b, n_boot=2000, seed=0):
    """a, b: Series by day (lower is better). Returns mean(a-b), lo, hi, win-rate of a over b."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (j.a - j.b)
    mean, lo, hi = block_ci(d, n_boot, seed)
    return mean, lo, hi, float((d < 0).mean()) + 0.5 * float((d == 0).mean())

def field_by_round(defn, days, metric):
    f = pd.read_parquet(f"{ROOT}/data/cache/field_def{defn}.parquet")
    f["d"] = f.day.str[:10]
    f = f[f.d.isin(set(days))]
    if metric == "sql":
        f = f[f.has_q]
    return f.groupby(["d", "model"])[metric].mean().unstack("model")

def build_table(defn, days, ours, metric="mase", n_boot=2000, top=12, elo_boot=500):
    """ours: {name: DataFrame(day, series, mase, sql, ...) from faith.score_model}. Returns (table DataFrame, elo DataFrame, matrix)."""
    M = field_by_round(defn, days, metric)
    for name, df in ours.items():
        s = df.groupby("day")[metric].mean()
        M[name] = s.reindex(M.index).to_numpy()
    mean_by_model = M.mean().sort_values()
    rows = []
    ref = [c for c in M.columns if c not in ours]
    best = M[ref].mean().sort_values().index[0]
    for m in list(mean_by_model.index):
        mean, lo, hi = block_ci(M[m], n_boot)
        row = dict(model=m, mean=mean, lo=lo, hi=hi, rounds=int(M[m].notna().sum()))
        if m != best:
            dm, dlo, dhi, win = paired_block_diff(M[m], M[best], n_boot)
            row.update(diff_vs_best_ref=dm, d_lo=dlo, d_hi=dhi, win_rate_vs_best_ref=win)
        rows.append(row)
    tab = pd.DataFrame(rows)
    elo = metrics.elo_summary(M.to_numpy(), list(M.columns), n_boot=elo_boot, seed=0)
    return tab, elo, M, best
