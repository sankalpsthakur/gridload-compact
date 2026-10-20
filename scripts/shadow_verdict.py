#!/usr/bin/env python3
"""TEST-L1 verdict, computed locally from the small tables fetched from kernel bench-timeseries-shadow (pre-registration: TEST-LOG.md, TEST-L1, Amendments 1-2).

Order of reading (pre-registered): 1. integrity of the fetched files against the kernel's manifest; 2. R0 per definition, before any statistic involving our
model; 3. when R0 holds for definition 2: the Arena Score point estimates and the decision rule, recomputed here with the same code as in the kernel
(`kernels/shadow/shadow_analysis.py`) and compared with the kernel's own `analysis.json`; the intervals (block bootstraps) are the kernel's.

usage (through bin/guard.sh, -t 60 -m 512):  python scripts/shadow_verdict.py <out_dir> [--defs 2,5] [--write <result_dir>]
Without --write it prints only.  Never edits the fetched files."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TRACK, "kernels", "shadow"))
import numpy as np  # noqa: E402

import shadow_analysis as A  # noqa: E402
import shadow_lib as L  # noqa: E402

args = sys.argv[1:]
out_dir = args[0]
defs = [int(x) for x in (args[args.index("--defs") + 1] if "--defs" in args else "2,5").split(",")]
write_dir = args[args.index("--write") + 1] if "--write" in args else None


def f1(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def ci_txt(c, nd=1):
    return "n/a" if not c or c[0] is None else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


# ------------------------------------------------------------------------------------------------------------ 1. integrity
man = json.load(open(os.path.join(out_dir, "manifest.json")))
bad = []
for fn, meta in man["files"].items():
    p = os.path.join(out_dir, fn)
    if not os.path.exists(p) or os.path.getsize(p) != meta["bytes"] or L.sha256_file(p) != meta["sha256"]:
        bad.append(fn)
print(f"1. fetched files: {len(man['files'])} in the kernel manifest, missing or different: {bad or 'none'}; kernel ok={man.get('ok')} error={man.get('error')}")
print(f"   build commit {man.get('build', {}).get('commit')}, kernel seconds {man.get('seconds')}, api calls {man.get('api', {}).get('calls')}, retries {man.get('api', {}).get('retries')}, api errors {man.get('api', {}).get('n_errors')}")
if bad or not man.get("ok"):
    print("STOP: the fetch is incomplete or the kernel failed; nothing is computed.")
    sys.exit(2)
smoke = json.load(open(os.path.join(out_dir, "smoke.json")))
print(f"   smoke on out-of-window rounds: pass={smoke.get('pass')}")
names = {int(k): v for k, v in json.load(open(os.path.join(out_dir, "model_names.json"))).items()}
kern = json.load(open(os.path.join(out_dir, "analysis.json")))

# ------------------------------------------------------------------------------------------------------------ 2. R0 first
tabs, r0s = {}, {}
for d in defs:
    S = L.read_table(os.path.join(out_dir, f"scores_def{d}.csv.gz"))
    T = L.read_table(os.path.join(out_dir, f"tasks_def{d}.csv.gz"))
    tabs[d] = (S, T)
    r0s[d] = L.repro_stats(S, T)
    r = r0s[d]
    c131 = r["comparators"][L.R0_COMPARATOR]
    print(f"2. R0 definition {d}: visiontspp-large informative rows {c131['n_cond']} of {c131['n']} stored final rows (rmse reproduced in {c131['share_rmse']:.3f}), "
          f"structure share {c131['share_cond']:.4f}, V2 built for {r['v2_coverage']:.3f} of eligible tasks -> R0 {'HOLDS' if r['r0_pass'] else 'FAILS'}")
    print(f"   all models: informative {r['all_models']['n_cond']}, structure share {r['all_models']['share_cond']:.4f}; tasks with the served scale reproduced exactly (f = 1): {r['scale_factor']['share_exact']:.3f}; "
          f"tasks with |f-1| > 1e-3: {r['scale_factor']['share_over_1e3']:.3f}")
    for c, v in r["comparators"].items():
        sc_txt = "n/a" if v["share_cond"] is None else format(v["share_cond"], ".4f")
        print(f"   comparator {v['name']:28s} rows {v['n']:4d} informative {v['n_cond']:4d} structure share {sc_txt:>7s} -> {'reproduced' if v['reproduced'] else 'DROPPED'}")
    by = r["per_model"].get(L.R0_COMPARATOR, {}).get("by_age", {})
    print("   visiontspp-large rmse reproduced by round age:", {k: (v["n"], round(v["share_rmse"], 3)) for k, v in by.items()})
if (2 in r0s) and (not r0s[2]["r0_pass"]):
    print("\nVERDICT: NOT EVALUABLE (R0 fails in definition 2): no statistic involving our model is computed or quoted.")
    res_json = dict(verdict="NOT EVALUABLE (R0 fails in definition 2)", r0={str(d): r0s[d] for d in r0s})
    if write_dir:
        os.makedirs(write_dir, exist_ok=True)
        json.dump(res_json, open(os.path.join(write_dir, "verdict.json"), "w"), indent=1, default=str)
    sys.exit(0)

# ------------------------------------------------------------------------------------------------------------ 3. statistics
loc = {}
for d in defs:
    S, T = tabs[d]
    loc[d] = A.analyze_definition(d, S, T, names, ci=False, platform=False)
    print(f"3. definition {d}: analysis status {loc[d]['status']}")


def close(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


mism = []
for d in defs:
    if loc[d]["status"] != "ok":
        continue
    kd = kern[f"def{d}"]
    for metric in ("point", "prob"):
        for variant in ("v2", "v1"):
            for subset, blk in loc[d][metric][variant].items():
                kb = kd[metric][variant].get(subset)
                if kb is None or "score_ours" not in kb:
                    mism.append((d, metric, variant, subset, "missing in kernel"))
                    continue
                if not close(blk["score_ours"], kb["score_ours"]):
                    mism.append((d, metric, variant, subset, "score_ours", blk["score_ours"], kb["score_ours"]))
                for c, g in blk["gaps"].items():
                    if not close(g.get("gap"), kb["gaps"][c].get("gap")):
                        mism.append((d, metric, variant, subset, f"gap {g['name']}", g.get("gap"), kb["gaps"][c].get("gap")))
print(f"   local recomputation of every point estimate against the kernel's analysis.json: {'IDENTICAL (1e-6)' if not mism else 'MISMATCH ' + str(mism[:5])}")

# merge the kernel's intervals into the local result; decide with the local point estimates
for d in defs:
    if loc[d]["status"] != "ok":
        continue
    kd = kern[f"def{d}"]
    for metric in ("point", "prob"):
        for variant in ("v2", "v1"):
            for subset, blk in loc[d][metric][variant].items():
                kb = kd[metric][variant].get(subset, {})
                for c, g in blk["gaps"].items():
                    for key in ("ci7", "ci14", "ci28", "ci7_draws", "ci14_draws", "ci28_draws"):
                        if key in kb.get("gaps", {}).get(c, {}):
                            g[key] = kb["gaps"][c][key]
                if "platform_style" in kb:
                    blk["platform_style"] = kb["platform_style"]
dec = A.decide(loc[2], loc.get(5))
print("\nDECISION (rule of TEST-L1 as amended):")
print(json.dumps({k: v for k, v in dec.items() if k != "rule"}, indent=1, default=str))
print("\nVERDICT:", dec["verdict"], "| V1 alone would change it:", dec.get("v1_would_change_verdict"), "| fragile:", dec.get("fragile"))

# ------------------------------------------------------------------------------------------------------------ report
md = []
md.append("# TEST-L1: shadow check of gridload-compact v1 on completed live TS-Arena rounds\n")
md.append(f"Kernel `sankalpsthakur/bench-timeseries-shadow` (build commit `{man.get('build', {}).get('commit', '')[:10]}`), window 2026-08-24 .. 2026-10-18, definitions 2 (24 h at 15 min) and 5 (72 h at 1 h).  "
          f"Self-evaluated shadow check on public data; contexts reconstructed (V2 = the served context, V1 = current data); not an official ranking.\n")
md.append(f"**Verdict: {dec['verdict']}**  (gate variant V2; V1 alone would change the verdict: {dec.get('v1_would_change_verdict')}; fragile under stale-free / V1 subsets: {dec.get('fragile')}).\n")
if "R1_gap" in dec:
    md.append("| condition | value | threshold | met |\n|---|---|---|---|")
    md.append(f"| R0 reproduction of visiontspp-large (definition 2) | {r0s[2]['comparators'][L.R0_COMPARATOR]['n_cond']} informative rows, structure share {r0s[2]['comparators'][L.R0_COMPARATOR]['share_cond']:.4f}, V2 coverage {r0s[2]['v2_coverage']:.3f} | >= 100 rows, >= 0.95, >= 0.90 | {dec['R0']} |")
    md.append(f"| R1 definition 2, SQL gap vs visiontspp-large | {f1(dec['R1_gap'])} {ci_txt(dec['R1_ci7'])} ({dec['R1_common_tasks']} common tasks) | >= +5.0 (point) | {dec['R1']} |")
    md.append(f"| R2 definition 2, SQL gap vs SCCH/PCFM-FT | {f1(dec['R2_gap'])} {ci_txt(dec['R2_ci7'])} ({dec['R2_common_tasks']} common tasks) | >= -5.0 (point) | {dec['R2']} |")
    md.append(f"| R3 definition 2 fallback share (V2) | {dec['R3_fallback_share_def2']:.4f} (V1 {dec['fallback_share_def2_v1']:.4f}) | <= 0.05 | {dec['R3']} |\n")
for d in defs:
    r = loc[d]
    if r["status"] != "ok":
        md.append(f"## Definition {d}\nR0 failed: not evaluated.\n")
        continue
    ts = r["tasks"]
    md.append(f"## Definition {d}\n")
    md.append(f"Tasks {ts['n_tasks']} ({ts['n_rounds']} rounds), eligible {ts['n_eligible']}, dropped {ts['drop_reasons']}; stale rounds (> 6 h): {ts['stale_rounds']}.  "
              f"Fallback share V2 {ts['fallback_share_v2']:.4f}, V1 {ts['fallback_share_v1']:.4f}.  V2 built for {ts['v2_applied_share']:.3f} of the eligible tasks; the last season differs between served and current values on "
              f"{100 * (ts['served_diff_share_mean'] or 0):.1f} % of its points on average ({100 * (ts['served_diff_share_any'] or 0):.0f} % of tasks have at least one differing point); "
              f"V1 scale differs from the served scale by more than 1e-3 in {100 * (ts['scale_v1_vs_served_share_over_1e3'] or 0):.0f} % of the tasks.  Latency {ts['latency_ms_per_series_p50']:.2f} ms per series (p50, Kaggle CPU, 1 thread).\n")
    for metric, title in (("prob", "probabilistic ranking (SQL)"), ("point", "point ranking (MAE)")):
        for variant in ("v2", "v1"):
            blk = r[metric][variant].get("all")
            if not blk:
                continue
            md.append(f"**{title}, context {variant.upper()}**: ours {blk['score_ours']:.1f} points (place {blk['rank_ours']} of {blk['n_models']} fitted models), {blk['n_tasks']} tasks.\n")
            md.append("| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks |\n|---|---|---|---|---|---|")
            for c, g in blk["gaps"].items():
                if g.get("gap") is None:
                    md.append(f"| {g['name']} | n/a ({g.get('dropped', 'not in the fit')}) | | | | {g['common_tasks']} |")
                else:
                    md.append(f"| {g['name']} | {f1(g['gap'])} | {ci_txt(g.get('ci7'))} | {ci_txt(g.get('ci14'))} | {ci_txt(g.get('ci28'))} | {g['common_tasks']} |")
            md.append("")
            if variant == "v2":
                md.append("top 10 of the fit: " + "; ".join(f"{t['name']} {t['score']:.1f}" for t in blk["top10"]) + "\n")
                ps = blk.get("platform_style")
                if ps:
                    md.append("platform-style score, interval (28-day blocks + weekly floor) and rank range: " + "; ".join(
                        f"{names.get(int(k), k)} {v['score']:.1f} [{v['ci_lower']:.1f}, {v['ci_upper']:.1f}] ranks {(str(v['rank_best']) + "-" + str(v['rank_worst'])) if v.get('rank_best') is not None else 'n/a (provisional)'}" for k, v in ps.items() if v.get('ci_lower') is not None) + "\n")
    stale = r["prob"]["v2"].get("nonstale")
    if stale:
        md.append("without the stale rounds (V2, SQL): " + "; ".join(f"{g['name']} {f1(g['gap'])} {ci_txt(g.get('ci7'))}" for g in stale["gaps"].values() if g.get("gap") is not None) + "\n")
    md.append("paired log-ratio of per-task errors, 100 * mean ln(err_comparator / err_ours), ISO-week block interval, tasks won (V2):\n")
    md.append("| comparator | metric | mean | 95 % interval | tasks | share won |\n|---|---|---|---|---|---|")
    for key, v in r["paired_logratio"].items():
        metric, variant, c = key.split(":")
        if variant != "v2" or v is None:
            continue
        md.append(f"| {L.COMPARATORS[int(c)]} | {metric} | {f1(v['mean'], 2)} | [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v['n_tasks']} | {v['win_share']:.2f} |")
    md.append("\nold-style arena MASE of our public texts (mae / flat-naive mae, mean over series then over rounds), ours minus comparator, ISO-week bootstrap (V2):\n")
    md.append("| comparator | ours | comparator | difference | 95 % interval | rounds | rounds won |\n|---|---|---|---|---|---|---|")
    for key, v in r["old_arena_mase"].items():
        variant, c = key.split(":")
        if variant != "v2" or v is None:
            continue
        md.append(f"| {L.COMPARATORS[int(c)]} | {v['ours']:.4f} | {v['comp']:.4f} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['n_rounds']} | {v['round_win_share']:.2f} |")
    md.append("")
text = "\n".join(md)
if write_dir:
    os.makedirs(write_dir, exist_ok=True)
    json.dump(dict(verdict=dec["verdict"], decision=dec, r0={str(d): r0s[d] for d in r0s}, local_vs_kernel_mismatches=mism,
                   tasks={str(d): loc[d].get("tasks") for d in loc}), open(os.path.join(write_dir, "verdict.json"), "w"), indent=1, default=str)
    open(os.path.join(write_dir, "verdict.md"), "w").write(text + "\n")
    print(f"\nwrote {write_dir}/verdict.json and verdict.md")
