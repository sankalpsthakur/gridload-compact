"""The same Archive TEST-F comparison under the metric of our public texts and under the Arena Score's per-task metric (notes/arena_score_vs_mase.md).
Read-only analysis of the cached evaluation tables (numpy / pandas); no models.   usage (bin/guard.sh): python def5_metric_compare.py <2|5>

For the best reference model of each window (lowest mean old-style arena MASE, as in the public tables) it prints:
  A. old-style arena MASE (mae / flat-naive mae; per-round mean over series, then mean over rounds): means, per-round difference, ISO-week block interval, rounds won
  B. the same comparison per task in log-ratio form (Arena Score style): 100 * mean ln(err_ref / err_ours), ISO-week block interval, tasks won; and the Arena-style number
     of the two arithmetic means, 100 * ln(mean_ref / mean_ours), to separate "different weighting" from "different interval"
  C. per series (definition 5, 35 rounds) and by error tercile of the reference: where the log-ratio difference comes from"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TRACK, "kernels", "shadow"))
import shadow_lib as L  # noqa: E402

d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ROOT = os.path.join(TRACK, "results")
field = pd.read_parquet(f"{ROOT}/bench-timeseries-a2/fieldcache/field_def{d}.parquet")[["day", "round_id", "model", "series", "mase", "sql", "has_q"]]
ours = pd.read_parquet(f"{ROOT}/bench-timeseries-b/out/faith/faith_final_def{d}_test_d{d}_service_int8_2026-02-08_2026-03-31.parquet")[["day", "round_id", "series", "mase", "sql"]].copy()
ours["model"] = "OURS"
field["day"] = pd.to_datetime(field["day"].astype(str).str[:10])
ours["day"] = pd.to_datetime(ours["day"].astype(str).str[:10])
field = field[field["model"] != "naive-forecast"]
EPOCH = pd.Timestamp("1970-01-01")


def window(lo, hi):
    f = field[(field["day"] >= lo) & (field["day"] <= hi)]
    o = ours[(ours["day"] >= lo) & (ours["day"] <= hi)]
    return f, o


def report(label, lo, hi, metric, ref=None):
    f, o = window(pd.Timestamp(lo), pd.Timestamp(hi))
    f = f[np.isfinite(f[metric]) & (f[metric] > 0)]
    if metric == "sql":
        f = f[f["has_q"]]
    rounds_used = sorted(set(o["round_id"]))
    f = f[f["round_id"].isin(rounds_used)]
    # best reference: lowest mean over rounds of the per-round mean over series, among models present in at least 90 % of the tasks
    ntask = f.groupby(["round_id", "series"]).ngroups
    cnt = f.groupby("model").size()
    cand = [m for m in cnt.index if cnt[m] >= 0.9 * ntask]
    pr = f[f["model"].isin(cand)].groupby(["model", "round_id"])[metric].mean().groupby("model").mean()
    best = ref or pr.idxmin()
    a = o[["round_id", "series", "day", metric]].rename(columns={metric: "ours"})
    b = f[f["model"] == best][["round_id", "series", metric]].rename(columns={metric: "ref"})
    m = a.merge(b, on=["round_id", "series"]).dropna()
    m = m[(m["ours"] > 0) & (m["ref"] > 0)]
    # A. per-round arithmetic means over the common series
    g = m.groupby("round_id").agg(ours=("ours", "mean"), ref=("ref", "mean"), day=("day", "first"))
    diff = (g["ours"] - g["ref"]).to_numpy()
    wk = L.iso_week_of_day((g["day"] - EPOCH).dt.days.to_numpy())
    am, alo, ahi = L.week_blocks_bootstrap(diff, wk)
    # B. per task log-ratio
    lr = 100 * np.log(m["ref"] / m["ours"]).to_numpy()
    wk2 = L.iso_week_of_day((m["day"] - EPOCH).dt.days.to_numpy())
    lm, llo, lhi = L.week_blocks_bootstrap(lr, wk2)
    mean_ref, mean_ours = g["ref"].mean(), g["ours"].mean()
    print(f"\n--- def {d} | {label} | {metric} | best reference by mean {best} | {len(g)} rounds, {len(m)} tasks")
    print(f"  A. old-style: ours {mean_ours:.4f}, {best} {mean_ref:.4f}; per-round difference {am:+.4f} [{alo:+.4f}, {ahi:+.4f}] (ISO-week blocks); rounds won {np.mean(diff < 0):.2f}")
    print(f"  B. per task : 100*mean ln(ref/ours) = {lm:+.2f} [{llo:+.2f}, {lhi:+.2f}] (ISO-week blocks); tasks won {np.mean(lr > 0):.2f}; "
          f"median per-task {np.median(lr):+.2f}; 100*ln(mean_ref/mean_ours) = {100 * np.log(mean_ref / mean_ours):+.2f}")
    return m, best


for metric in ("mase", "sql"):
    for label, lo, hi in (("TEST-F full (52 rounds)", "2026-02-08", "2026-03-31"), ("first 35 rounds (02-08..03-14)", "2026-02-08", "2026-03-14"),
                          ("late March, archive-only (03-15..03-31)", "2026-03-15", "2026-03-31")):
        m, best = report(label, lo, hi, metric)
        if d == 5 and metric == "mase" and label.startswith("first 35"):
            print("  C1. per series (log-ratio, 35 rounds, vs " + best + "):")
            for s, gg in m.groupby("series"):
                x = 100 * np.log(gg["ref"] / gg["ours"])
                print(f"      {s:24s} mean {x.mean():+6.2f}   tasks won {np.mean(x > 0):.2f}   ours {gg['ours'].mean():.3f} vs {gg['ref'].mean():.3f}")
            print("  C2. by error tercile of the reference (log-ratio, 35 rounds):")
            q = m["ref"].quantile([1 / 3, 2 / 3]).to_numpy()
            for name, sel in (("low-error tasks", m["ref"] <= q[0]), ("mid", (m["ref"] > q[0]) & (m["ref"] <= q[1])), ("high-error tasks", m["ref"] > q[1])):
                gg = m[sel]
                x = 100 * np.log(gg["ref"] / gg["ours"])
                print(f"      {name:18s} tasks {len(gg):3d}  mean log-ratio {x.mean():+6.2f}   tasks won {np.mean(x > 0):.2f}   mean abs difference of MASE {np.mean(gg['ours'] - gg['ref']):+.4f}")
