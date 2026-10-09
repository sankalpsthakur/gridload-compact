#!/usr/bin/env python3
"""Turn the a2 kernel's freeze_candidate.json into configs/final.json (the frozen configuration the TEST-F kernel verifies).
usage: python scripts/freeze.py results/bench-timeseries-a2/out/freeze_candidate.json
Only json handling; run after reading the a2 results.  Commit configs/final.json, then add the freeze note to TEST-LOG.md, then build the b kernel."""
import json, sys, time

ROOT = __file__.rsplit("/scripts/", 1)[0]
fz = json.load(open(sys.argv[1]))
out = dict(frozen_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), source="a2 kernel freeze_candidate.json (selection by the pre-registered rule in configs/prereg.json)",
           a2_kernel=fz.get("kernel_a2"), a2_commit=fz.get("commit_of_a2"), selection=fz["selection"], spread=fz["spread"], seeds=fz["seeds"], cutoffs=fz["cutoffs"],
           test_stage_ckpt_sha256=fz["test_stage_ckpt_sha256"], val_stage_ckpt_sha256=fz["val_stage_ckpt_sha256"])
json.dump(out, open(f"{ROOT}/configs/final.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("selection", "spread", "seeds")}, indent=1))
print(len(out["test_stage_ckpt_sha256"]), "test-stage checkpoints hashed")

# ---- freeze note (printed; paste into TEST-LOG.md before building the b kernel) -------------------------------------------------------
sel = out["selection"]; sp = out["spread"]
print("\n### FREEZE NOTE (draft)\n")
print(f"- frozen {out['frozen_utc']} from the a2 kernel ({out['a2_kernel']}, code commit {str(out['a2_commit'])[:10]}); `configs/final.json` is the frozen configuration.")
for fk, name in (("15", "def 2, 15 min x 96"), ("60", "def 5, 1 h x 72")):
    s = sel[fk]
    print(f"- {name}: E2 features J={s['J']} dmeans={s['dmeans']}, hid {s['hid']} emb {s['emb']}, {s['params']} parameters per net, 3 seeds {out['seeds']}, steps {s['steps']}; "
          f"spread {sp[fk]}; gate_failed={s['gate_failed']} ({s['reason']}).")
print("- test-stage checkpoints (cutoff " + out["cutoffs"]["test"] + ") SHA256 prefixes: " + ", ".join(f"{k}={v[:12]}" for k, v in sorted(out["test_stage_ckpt_sha256"].items())))
print("- headline variant per definition: service code path, int8 (Addendum 2); the b kernel verifies the hashes above before any TEST-F evaluation and stops otherwise.")
