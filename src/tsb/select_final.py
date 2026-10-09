"""Apply the pre-registered selection rule (configs/prereg.json, TEST-LOG.md Addendum 1) to the sweep results.
python -m tsb.select_final sweep_results.json out/selection.json
sweep_results.json: {"15": {"best_ref": {"model": str, "val_f_mase": float}, "cands": [{"hid", "emb", "params", "val_s": {phase: [mase, sql, cover]}, "val_f_mase", "val_f_sql", ...}]}, "60": {...}}"""
import json, sys
from tsb import ROOT

def select_freq(cands, best_ref_mase, tol):
    phases = list(cands[0]["val_s"].keys())
    best = {ph: min(c["val_s"][ph][0] for c in cands) for ph in phases}
    for c in cands:
        c["rel_to_best"] = {ph: c["val_s"][ph][0] / best[ph] - 1.0 for ph in phases}
        c["qualifies"] = all(c["val_s"][ph][0] <= (1.0 + tol) * best[ph] for ph in phases)
        c["gate_pass"] = bool(c["val_f_mase"] < best_ref_mase)
    qual = sorted([c for c in cands if c["qualifies"]], key=lambda c: c["params"])
    for c in qual:
        if c["gate_pass"]:
            return c, dict(gate_failed=False, reason="smallest qualifying candidate that passes the VAL-F gate")
    pool = qual or cands
    c = min(pool, key=lambda c: c["val_f_mase"])
    return c, dict(gate_failed=True, reason="no qualifying candidate passes the VAL-F gate; lowest VAL-F MASE among qualifying")

def main(res_path, out_path, prereg_path=f"{ROOT}/configs/prereg.json"):
    pre = json.load(open(prereg_path)); res = json.load(open(res_path))
    out = {}
    for fk, r in res.items():
        c, info = select_freq(r["cands"], r["best_ref"]["val_f_mase"], pre["tolerance"])
        out[fk] = dict(selected=dict(hid=c["hid"], emb=c["emb"], params=c["params"], J=c["J"], dmeans=c["dmeans"], steps=pre["steps"]),
                       best_ref=r["best_ref"], table=r["cands"], **info)
        print(fk, "->", out[fk]["selected"], info, flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    return out

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
