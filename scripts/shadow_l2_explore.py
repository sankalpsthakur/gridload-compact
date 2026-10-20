#!/usr/bin/env python3
"""EXPLORATORY follow-up of TEST-L2 (not pre-registered; it changes no verdict).  The TEST-L2 result showed fallbacks only in the first half of the window
(platform data gaps inside the context windows, Sep 3 .. Sep 23); this prints the same Arena gaps (V1) for (a) the rounds registered from 2026-09-24 on, when no
series was routed to the fallback, and (b) all rounds but only the tasks the specialist served.   usage (bin/guard.sh): python shadow_l2_explore.py <out_dir> <res_dir>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TRACK, "kernels", "shadow"))
import numpy as np  # noqa: E402

import shadow_lib as L  # noqa: E402

L.V2_COVERAGE = 0.0
L.MIN_REPRO_ROWS_OTHER = 10
out_dir, res_dir = sys.argv[1], sys.argv[2]
names = {int(k): v for k, v in json.load(open(os.path.join(out_dir, "model_names.json"))).items()}
LATE_FROM = (np.datetime64("2026-09-24") - np.datetime64("1970-01-01")).astype(int)
result = {}
lines = ["# TEST-L2 exploratory follow-up (not pre-registered; changes no verdict)\n",
         "Same engine, V1 contexts, intervals = 7-day circular blocks, 1000 refits.  (a) rounds registered from 2026-09-24 on; (b) all rounds, tasks the specialist served (no fallback).\n"]
for d in (2, 5):
    S = L.read_table(os.path.join(out_dir, f"scores_def{d}.csv.gz"))
    T = L.read_table(os.path.join(out_dir, f"tasks_def{d}.csv.gz"))
    r0 = L.repro_stats(S, T)
    usable = {c for c, v in r0["comparators"].items() if v["reproduced"]}
    el = T["eligible"] == 1
    subsets = {"(a) rounds from 2026-09-24": el & (T["day"] >= LATE_FROM), "(b) specialist-served tasks only": el & (T["path_v1"] != "fallback")}
    for label, keep in subsets.items():
        T2 = dict(T)
        T2["eligible"] = keep.astype(float)
        for metric in ("prob", "point"):
            ps = L.pair_sums_of(L.build_pair_rows(S, T2, metric, "v1"))
            ids, theta, aux = L.fit_theta(ps)
            score = {int(m): 100.0 * float(t) for m, t in zip(ids, theta)}
            order = sorted((m for m in score if m != L.NAIVE_ID), key=lambda m: -score[m])
            pairs = [(L.OURS_ID, c) for c in L.COMPARATORS if c in usable and c in score]
            ci = L.gap_bootstrap(ps, pairs, 7, 1000)
            tasks_n = int(keep.sum())
            rnd = {int(r) for r, k in zip(T["round_id"], keep) if k}
            row = {}
            for c in pairs:
                q = ci[c]
                row[L.COMPARATORS[c[1]]] = dict(gap=score[L.OURS_ID] - score[c[1]], lo=q["lo"], hi=q["hi"])
            result[f"def{d} {label} {metric}"] = dict(tasks=tasks_n, rounds=len(rnd), ours=score[L.OURS_ID], place=order.index(L.OURS_ID) + 1, of=len(order), gaps=row)
            lines.append(f"- def {d}, {label}, {'SQL' if metric == 'prob' else 'MAE'}: {tasks_n} tasks in {len(rnd)} rounds, ours {score[L.OURS_ID]:.1f} (place {order.index(L.OURS_ID) + 1} of {len(order)}); gaps: " +
                         "; ".join(f"{k} {v['gap']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]" for k, v in row.items()))
text = "\n".join(lines)
print(text)
os.makedirs(res_dir, exist_ok=True)
open(os.path.join(res_dir, "explore_l2.md"), "w").write(text + "\n")
json.dump(result, open(os.path.join(res_dir, "explore_l2.json"), "w"), indent=1)
