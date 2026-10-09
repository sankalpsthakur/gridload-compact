"""Markdown evidence for one faithful window: our variants inserted into the real archive field.
python -m tsb.make_evidence --def 2 --days 2026-02-08 2026-03-31 --label TEST-F --ours "gridload fp32=runs/a.parquet" "gridload int8=runs/b.parquet" --out reports/test_f_def2.md"""
import argparse, json, os
import numpy as np, pandas as pd
from tsb import report, ROOT

def model_sizes():
    try:
        rows = json.load(open(f"{ROOT}/data/raw/models.json"))
        return {r["name"]: r.get("model_size") for r in rows}
    except Exception:
        return {}

def fmt_ci(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="defn", type=int, required=True)
    ap.add_argument("--days", nargs=2, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ours", nargs="+", required=True, help='"name=path.parquet" (per-round per-series scores from tsb.evalf)')
    ap.add_argument("--params", nargs="*", default=[], help='"name=params_in_millions" for our variants')
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.days[0], a.days[1])]
    ours = {}
    for kv in a.ours:
        k, v = kv.split("=", 1)
        ours[k] = pd.read_parquet(v if os.path.isabs(v) else f"{ROOT}/{v}")
    psz = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.params}
    sizes = model_sizes()
    out = [f"# {a.label}: SMARD Net Load definition {a.defn}, rounds {a.days[0]} .. {a.days[1]}\n"]
    for metric in ("mase", "sql"):
        tab, elo, M, best = report.build_table(a.defn, days, ours, metric=metric, n_boot=a.boot, elo_boot=500)
        n_rounds = int(M.shape[0])
        out.append(f"\n## {metric.upper()} (arena definition; lower is better), {n_rounds} rounds, 95% block-bootstrap CI over ISO weeks\n")
        out.append(f"Best reference model in this window: **{best}**.  Diff = mean(model - {best}) per round with CI; win-rate = share of rounds with lower score than {best}.\n")
        out.append("| rank | model | params (M) | mean [95% CI] | diff vs best ref [95% CI] | win-rate vs best ref | Elo [95% CI] |")
        out.append("|---|---|---|---|---|---|---|")
        eloi = elo.set_index("model")
        shown = list(tab.model.head(a.top)) + [o for o in ours if o not in list(tab.model.head(a.top))]
        for _, r in tab[tab.model.isin(shown)].iterrows():
            name = r["model"]
            ps = psz.get(name, sizes.get(name))
            ps = "-" if ps is None or (isinstance(ps, float) and np.isnan(ps)) else (f"{ps:.3f}" if ps < 1 else f"{ps:g}")
            rank = int(tab.index[tab.model == name][0]) + 1
            d = "-" if pd.isna(r.get("diff_vs_best_ref", np.nan)) else f"{r['diff_vs_best_ref']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]"
            w = "-" if pd.isna(r.get("win_rate_vs_best_ref", np.nan)) else f"{100*r['win_rate_vs_best_ref']:.0f}%"
            e = eloi.loc[name]
            out.append(f"| {rank} | {'**'+name+'**' if name in ours else name} | {ps} | {fmt_ci(r['mean'], r['lo'], r['hi'])} | {d} | {w} | {e['elo']:.0f} [{e['lo']:.0f}, {e['hi']:.0f}] |")
        out.append(f"\nField for this metric: {M.shape[1]-len(ours)} models + ours; Elo = platform procedure (K=4, base 1000, 500 bootstraps, per-round mean over series).")
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    open(os.path.join(ROOT, a.out), "w").write("\n".join(out) + "\n")
    print("\n".join(out))

if __name__ == "__main__":
    main()
