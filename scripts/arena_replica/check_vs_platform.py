"""Cross-check of the scout's Arena Score replica (bt_replica2.py) against the platform's own engine (vendored arena_score.py, TEST-L1 item 5),
on the Archive TEST-F tables the replica uses.  Read-only analysis of the track's cached evaluation tables; numpy / pandas only; no models.

usage (through bin/guard.sh):  python check_vs_platform.py <def 2|5> <mase|sql> [n_boot]
Per window it prints, for the best reference model of the replica (top non-ours BT score) and for the fixed comparators:
  BT gap (platform engine, joint fit)   |   replica BT gap   |   BT-refit interval, 7-day circular blocks (the TEST-L1 interval)   |   the platform's 28-day blocks
The Archive's naive-forecast rows (old arena MASE exactly 1) play the platform's internal naive; n_points is taken as equal for all rows (the
like-for-like check of TEST-LOG post-run note 8 found no mismatch)."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TRACK, "kernels", "shadow"))
import shadow_lib as L  # noqa: E402

arena = L.arena
d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
METRIC = sys.argv[2] if len(sys.argv) > 2 else "mase"
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 500
ROOT = os.path.join(TRACK, "results")
field = pd.read_parquet(f"{ROOT}/bench-timeseries-a2/fieldcache/field_def{d}.parquet")[["day", "round_id", "model", "series", "mase", "sql", "has_q"]]
ours = pd.read_parquet(f"{ROOT}/bench-timeseries-b/out/faith/faith_final_def{d}_test_d{d}_service_int8_2026-02-08_2026-03-31.parquet")[["day", "round_id", "series", "mase", "sql"]].copy()
ours["model"] = "GRIDLOAD-COMPACT"
ours["has_q"] = True
field["day"] = field["day"].astype(str).str[:10]
df = pd.concat([field, ours[["day", "round_id", "model", "series", "mase", "sql", "has_q"]]], ignore_index=True)
if METRIC == "sql":
    df = df[df["has_q"] | (df["model"] == "naive-forecast")]
df["err"] = df[METRIC]
df = df[np.isfinite(df["err"]) & (df["err"] > 0)]
df["day"] = pd.to_datetime(df["day"])
names = sorted(df["model"].unique())
code = {m: i + 1 for i, m in enumerate(names)}
code["naive-forecast"] = L.NAIVE_ID
code["GRIDLOAD-COMPACT"] = L.OURS_ID
series_code = {s: i for i, s in enumerate(sorted(df["series"].unique()))}
inv = {v: k for k, v in code.items()}
COMP = ["visiontspp-large", "chronos-2", "flowstate", "moirai-2-small"]


def run(label, lo, hi):
    sub = df[(df["day"] >= lo) & (df["day"] <= hi)]
    rows = (sub["round_id"].astype(np.int64).to_numpy(), sub["series"].map(series_code).astype(np.int64).to_numpy(),
            sub["model"].map(code).astype(np.int64).to_numpy(), sub["err"].to_numpy(dtype=float), np.ones(len(sub), dtype=np.int64),
            (sub["day"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy().astype(np.int64))
    t0 = time.time()
    ps = arena.pair_sums(*rows)
    ids, theta, aux = L.fit_theta(ps)
    score = {int(m): 100.0 * float(t) for m, t in zip(ids, theta)}
    refs = sorted((m for m in score if m not in (L.OURS_ID, L.NAIVE_ID)), key=lambda m: -score[m])
    best = refs[0]
    tasks = sub.groupby(["round_id", "series"]).ngroups
    print(f"\n=== def {d} | {METRIC} | {label} | tasks {tasks} | models {len(refs) + 1} | fit {time.time() - t0:.1f}s | max grad {aux[4]:.1e}")
    print(f"  platform-engine scores: ours {score[L.OURS_ID]:.1f}; best reference {inv[best]} {score[best]:.1f}")
    pairs = [(L.OURS_ID, best)] + [(L.OURS_ID, code[c]) for c in COMP if c in code and code[c] != best]
    t0 = time.time()
    c7 = L.gap_bootstrap(ps, pairs, 7, NB)
    c28 = L.gap_bootstrap(ps, pairs, 28, max(100, NB // 5))
    for a, b in pairs:
        g = score[a] - score[b]
        q7, q28 = c7[(a, b)], c28[(a, b)]
        print(f"  gap vs {inv[b]:18s} {g:+6.1f}   7-day blocks [{q7['lo']:+.1f}, {q7['hi']:+.1f}] ({q7['draws']} refits)   28-day (platform) [{q28['lo']:+.1f}, {q28['hi']:+.1f}]")
    print(f"  bootstrap {time.time() - t0:.1f}s")


T = pd.Timestamp
run("TEST-F full (52 rounds)", T("2026-02-08"), T("2026-03-31"))
run("first 35 rounds (02-08..03-14)", T("2026-02-08"), T("2026-03-14"))
run("late March, archive-only (03-15..03-31)", T("2026-03-15"), T("2026-03-31"))
