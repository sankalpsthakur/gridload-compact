"""Replica of the TS-Arena Arena Score (graded Bradley-Terry, naive-anchored) on the Archive TEST-F window.
Read-only analysis of the track's own cached evaluation tables. numpy/scipy/pandas only; no models.
Outcome for A vs B on a (round, series) task: e_B / (e_A + e_B); weight 1/(k-1); score = 100 * (theta - theta_naive).
"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
d = int(sys.argv[1]) if len(sys.argv) > 1 else 2
METRIC = sys.argv[2] if len(sys.argv) > 2 else "mase"
field = pd.read_parquet(f"{ROOT}/bench-timeseries-a2/fieldcache/field_def{d}.parquet")[["day", "round_id", "model", "series", "mase", "sql", "has_q"]]
ours = pd.read_parquet(f"{ROOT}/bench-timeseries-b/out/faith/faith_final_def{d}_test_d{d}_service_int8_2026-02-08_2026-03-31.parquet")[["day", "round_id", "series", "mase", "sql"]].copy()
ours["model"] = "GRIDLOAD-COMPACT"
field["day"] = field["day"].str[:10]
ours["has_q"] = True
df = pd.concat([field, ours[["day", "round_id", "model", "series", "mase", "sql", "has_q"]]], ignore_index=True)
if METRIC == "sql":
    df = df[df["has_q"] | (df["model"] == "naive-forecast")]
df["mase"] = df[METRIC]
df = df[np.isfinite(df["mase"]) & (df["mase"] > 0)]
df["day"] = pd.to_datetime(df["day"])
ANCHOR = "naive-forecast"


def build_pairs(sub):
    models = sorted(sub["model"].unique())
    idx = {m: i for i, m in enumerate(models)}
    I, J, Y, W = [], [], [], []
    for _, g in sub.groupby(["round_id", "series"]):
        m = g["model"].map(idx).to_numpy()
        e = g["mase"].to_numpy()
        k = len(m)
        if k < 2:
            continue
        a, b = np.triu_indices(k, 1)
        I.append(m[a]); J.append(m[b])
        Y.append(e[b] / (e[a] + e[b]))
        W.append(np.full(len(a), 1.0 / (k - 1)))
    return models, np.concatenate(I), np.concatenate(J), np.concatenate(Y), np.concatenate(W)


def fit(models, I, J, Y, W):
    n = len(models)
    def f(th):
        z = th[I] - th[J]
        p = 1.0 / (1.0 + np.exp(-z))
        ll = W * (Y * np.log(p + 1e-300) + (1 - Y) * np.log(1 - p + 1e-300))
        r = W * (Y - p)
        g = -np.bincount(I, weights=r, minlength=n) + np.bincount(J, weights=r, minlength=n)
        return -ll.sum(), g
    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B", options=dict(maxiter=500, gtol=1e-9))
    th = res.x - res.x[models.index(ANCHOR)]
    return pd.Series(100 * th, index=models)


def window(name, lo, hi):
    sub = df[(df["day"] >= lo) & (df["day"] <= hi)]
    models, I, J, Y, W = build_pairs(sub)
    s = fit(models, I, J, Y, W).sort_values(ascending=False)
    ntask = sub.groupby(["round_id", "series"]).ngroups
    print(f"\n=== def {d} | {name} | {lo.date()}..{hi.date()} | tasks={ntask} | models={len(models)}")
    top = s.head(8)
    for m, v in top.items():
        print(f"  {v:7.1f}  {m}")
    rk = list(s.index).index("GRIDLOAD-COMPACT") + 1
    print(f"  ... GRIDLOAD-COMPACT: {s['GRIDLOAD-COMPACT']:.1f} (place {rk} of {len(s)}); best reference: {s.drop('GRIDLOAD-COMPACT').index[0]} {s.drop('GRIDLOAD-COMPACT').iloc[0]:.1f}")
    gap = s["GRIDLOAD-COMPACT"] - s.drop("GRIDLOAD-COMPACT").iloc[0]
    print(f"  gap to best reference: {gap:+.1f} points")
    # cheap paired bootstrap of the gap over ISO-week blocks: mean over tasks of 100*ln(e_ref/e_ours)
    best = s.drop("GRIDLOAD-COMPACT").index[0]
    pv = sub[sub["model"].isin(["GRIDLOAD-COMPACT", best])].pivot_table(index=["round_id", "series", "day"], columns="model", values="mase").dropna().reset_index()
    pv["lr"] = 100 * np.log(pv[best] / pv["GRIDLOAD-COMPACT"])
    pv["wk"] = pv["day"].dt.isocalendar().week.astype(int)
    wk = pv.groupby("wk")["lr"].agg(["sum", "count"])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(wk), len(wk))
        bs.append(wk["sum"].to_numpy()[pick].sum() / wk["count"].to_numpy()[pick].sum())
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    print(f"  paired mean log-ratio gap vs {best}: {pv['lr'].mean():+.1f} pts, week-block 95% CI [{lo_:+.1f}, {hi_:+.1f}] (n tasks={len(pv)}, weeks={len(wk)})")
    return s


T = pd.Timestamp
window("TEST-F full", T("2026-02-08"), T("2026-03-31"))
window("TEST-F first 35 rounds", T("2026-02-08"), T("2026-03-14"))
window("TEST-F late March (archive-only)", T("2026-03-15"), T("2026-03-31"))
s = window("VAL-F + TEST-F whole archive Q1", T("2026-01-01"), T("2026-03-31"))
