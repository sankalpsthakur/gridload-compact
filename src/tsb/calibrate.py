"""Global decile-spread scalar from VAL-F (pre-registered grid and objective: configs/prereg.json).
python -m tsb.calibrate --def 2 --ckpt runs/a.pt runs/b.pt runs/c.pt --days 2026-01-01 2026-02-07 --out out/spread_def2.json"""
import argparse, json
from tsb import evalf, faith, feats, infer, ROOT

def scan(ckpts, defn, d0, d1, grid, rounds=None):
    models, freq = evalf.load_models(ckpts)
    pn = evalf.make_predict_np(models)
    sp = feats.Spec(freq, **evalf.SPEC_KW[ckpts[0]])
    C = evalf.calendar(no_hol=evalf.NO_HOL[ckpts[0]])
    rounds = rounds if rounds is not None else faith.load_rounds(defn, d0, d1)
    cache = []
    def ff(ts, vals):
        r = infer.forecast_series(pn, C, sp, ts, vals, sp.H); cache.append(r); return r
    faith.score_model(rounds, ff, sp.H, sp.step_s)
    rows = []
    for s in grid:
        it = iter(cache)
        def fs(ts, vals, s=s):
            r = next(it)
            if r is None:
                return None
            fut, q = r; med = q[:, 4:5]
            return fut, med + (q - med) * s
        df = faith.score_model(rounds, fs, sp.H, sp.step_s)
        rows.append(dict(scale=s, **evalf.summarize(df, f"scale {s}")))
    return rows

def choose(rows):
    """minimum mean SQL; ties -> smaller scale"""
    return sorted(rows, key=lambda r: (round(r["sql"], 9), r["scale"]))[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="defn", type=int, required=True); ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--days", nargs=2, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grid = json.load(open(f"{ROOT}/configs/prereg.json"))["spread"]["grid"]
    rows = scan(a.ckpt, a.defn, a.days[0], a.days[1], grid)
    best = choose(rows)
    for r in rows:
        print(f"scale {r['scale']:.2f}: MASE {r['mase']:.4f} SQL {r['sql']:.4f} cover80 {r['cover80']:.3f}", flush=True)
    print("chosen", best, flush=True)
    json.dump(dict(def_=a.defn, ckpts=[c.split('/')[-1] for c in a.ckpt], rows=rows, chosen=best), open(a.out, "w"), indent=1)
