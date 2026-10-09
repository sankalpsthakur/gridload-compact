#!/usr/bin/env python3
"""Print the key numbers of a kernel output directory (json only; local, no compute).  usage: python scripts/summarize.py results/bench-timeseries-a2/out"""
import glob, json, os, sys

d = sys.argv[1]
def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
for st in sorted(glob.glob(f"{d}/status_*.json")):
    s = jl(st)
    print("==", os.path.basename(st), "total", s.get("total_secs"), "s")
    for k, v in s["stages"].items():
        print(f"  {k:22s} ok={v.get('ok')} {v.get('secs')}s {v.get('error','')[:160]}")
sel = jl(f"{d}/selection.json")
if sel:
    print("== selection")
    for fk, v in sel.items():
        print(f" freq {fk}: selected {v['selected']} | gate_failed={v['gate_failed']} | best ref {v['best_ref']}")
        for c in v["table"]:
            print(f"   hid {c['hid']:3d} params {c['params']:6d} VAL-S {json.dumps({k: round(x[0], 4) for k, x in c['val_s'].items()})} rel {json.dumps({k: round(x, 3) for k, x in c['rel_to_best'].items()})} "
                  f"VAL-F mase {c['val_f_mase']:.4f} sql {c['val_f_sql']:.4f} qualifies={c['qualifies']} gate={c['gate_pass']} secs={c.get('train_seconds', 0):.0f}")
sw = jl(f"{d}/sweep_results.json")
if sw:
    for fk, v in sw.items():
        for i in v.get("informational", []):
            print(f" info {fk}: {json.dumps(i)[:300]}")
for p in sorted(glob.glob(f"{d}/spread_def*.json")):
    j = jl(p); print("== spread", os.path.basename(p), "chosen", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in j["chosen"].items() if k in ("scale", "mase", "sql", "cover80")})
    for r in j["rows"]:
        print(f"   scale {r['scale']:.2f} MASE {r['mase']:.4f} SQL {r['sql']:.4f} cover80 {r['cover80']:.3f}")
for sub in ("valdry", "test"):
    fs = sorted(glob.glob(f"{d}/{sub}/*.json"))
    if fs:
        print("==", sub)
        for p in fs:
            j = jl(p)["summary"]
            print(f"   {os.path.basename(p)[:-5]:26s} MASE {j['mase']:.4f} SQL {j['sql']:.4f} cover80 {j['cover80']:.3f} rounds {j['rounds']} missing {j['n_missing']} {j.get('routing','')}")
fz = jl(f"{d}/freeze_candidate.json")
if fz:
    print("== freeze candidate", json.dumps({k: fz[k] for k in ("selection", "spread", "seeds")}, indent=1))
