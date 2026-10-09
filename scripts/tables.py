#!/usr/bin/env python3
"""Markdown tables for SUBMISSION.md from the kernel outputs (json only).  usage: python scripts/tables.py results/bench-timeseries-b/out [valdry|test]"""
import glob, json, os, statistics, sys

d = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else "test"
NAMES = [("service_int8", "service code path, int8 (headline)"), ("ens3_torch_fp32", "ensemble, torch fp32"), ("ens3_ort_fp32", "ensemble, onnxruntime fp32"),
         ("ens3_ort_int8", "ensemble, onnxruntime int8"), ("single_s0", "single net, seed 0"), ("single_s1", "single net, seed 1"), ("single_s2", "single net, seed 2"),
         ("nohol_s0", "single net, seed 0, no holiday features")]
for dfn, label in ((2, "definition 2 (24 h at 15 min)"), (5, "definition 5 (72 h at 1 h)")):
    print(f"\n#### {label}\n")
    print("| variant | MASE | SQL | cover 80 % | rounds | missing | routing |")
    print("|---|---|---|---|---|---|---|")
    singles = []
    for key, text in NAMES:
        p = f"{d}/{sub}/d{dfn}_{key}.json"
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["summary"]
        if key.startswith("single_s"):
            singles.append((s["mase"], s["sql"]))
        print(f"| {text} | {s['mase']:.4f} | {s['sql']:.4f} | {s['cover80']:.3f} | {s['rounds']} | {s['n_missing']} | {s.get('routing', '')} |")
    if len(singles) > 1:
        print(f"\nSeeds (single nets): MASE {statistics.mean(x[0] for x in singles):.4f} +- {statistics.stdev(x[0] for x in singles):.4f} (sd over {len(singles)} seeds), "
              f"SQL {statistics.mean(x[1] for x in singles):.4f} +- {statistics.stdev(x[1] for x in singles):.4f}")
