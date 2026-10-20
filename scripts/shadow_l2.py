#!/usr/bin/env python3
"""TEST-L2 analysis (TEST-LOG.md, "TEST-L2"): the fetched TEST-L1 tables read under the original context variant V1 (current-data context).
Same engine and statistics as TEST-L1 (kernels/shadow/shadow_lib.py, shadow_analysis.py: the platform's vendored arena_score.py); R0 without the V2-coverage clause.

usage (every call through bin/guard.sh -t 60 -m 512):
  python scripts/shadow_l2.py points <out_dir> <res_dir>                                     R0 first, then point estimates, both readings of the verdict, paired statistics
  python scripts/shadow_l2.py ci <out_dir> <res_dir> <def> <metric> <subset> <block> <nboot>  one bootstrap (7 / 14 / 28-day blocks) of the gaps -> <res_dir>/ci/
  python scripts/shadow_l2.py report <out_dir> <res_dir>                                      merge points and intervals -> <res_dir>/verdict_l2.{json,md}
Never edits the fetched files."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TRACK, "kernels", "shadow"))
import numpy as np  # noqa: E402

import shadow_analysis as A  # noqa: E402
import shadow_lib as L  # noqa: E402

# TEST-L2 constants (L2-2 and L2-4)
L.V2_COVERAGE = 0.0           # no V2 clause
L.MIN_REPRO_ROWS_OTHER = 10   # other comparators: at least 10 informative rows
FLOOR_STRICT = 140            # B3: common tasks for a gate comparator
FLOOR_LITERAL = 30            # "rounds they played": at least 3 rounds
VARIANT = "v1"

stage = sys.argv[1]
out_dir = sys.argv[2]
res_dir = sys.argv[3]
os.makedirs(res_dir, exist_ok=True)


def f1(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def ci_txt(c, nd=1):
    return "n/a" if not c or c[0] is None else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def integrity():
    man = json.load(open(os.path.join(out_dir, "manifest.json")))
    bad = [fn for fn, meta in man["files"].items()
           if not os.path.exists(os.path.join(out_dir, fn)) or os.path.getsize(os.path.join(out_dir, fn)) != meta["bytes"]
           or L.sha256_file(os.path.join(out_dir, fn)) != meta["sha256"]]
    if bad or not man.get("ok"):
        print("STOP: fetched files missing or different:", bad, "kernel ok:", man.get("ok"))
        sys.exit(2)
    return man


def load(d):
    return L.read_table(os.path.join(out_dir, f"scores_def{d}.csv.gz")), L.read_table(os.path.join(out_dir, f"tasks_def{d}.csv.gz"))


def prepare(d):
    S, T = load(d)
    r0 = L.repro_stats(S, T)
    usable = {c for c, v in r0["comparators"].items() if v["reproduced"]}
    ts = A.task_summary(T)
    return S, T, r0, usable, ts


def v1_vs_served(S, T):
    """How far the V1 (current-data) context is from the served one, in context-scale terms, over tasks with a defined factor f
    (informative field rows: rmse reproduced within 1e-6; f = median of mase_recomputed / mase_stored, so the stored scale is f * scale_used)."""
    field = (S["variant"] == "field") & (S["stored_final"] == 1)
    r_, s_ = S["rmse"], S["stored_rmse"]
    inf = field & np.isfinite(r_) & np.isfinite(s_) & (np.abs(r_ - s_) <= L.REPRO_RMSE_TOL * np.abs(s_))
    ratio = S["mase_ctx"] / S["stored_mase"]
    key = list(zip(S["round_id"].astype(int).tolist(), S["series_id"].astype(int).tolist()))
    fs = {}
    for i in np.flatnonzero(inf & np.isfinite(ratio)):
        fs.setdefault(key[i], []).append(ratio[i])
    tidx = {(int(r), int(s)): i for i, (r, s) in enumerate(zip(T["round_id"], T["series_id"]))}
    dev = []
    for k, v in fs.items():
        i = tidx.get(k)
        if i is None or not np.isfinite(T["scale_v1"][i]) or not np.isfinite(T["scale"][i]):
            continue
        dev.append(abs(T["scale_v1"][i] / (float(np.median(v)) * T["scale"][i]) - 1.0))
    dev = np.array(dev)
    if not len(dev):
        return None
    return dict(n_tasks=int(len(dev)), share_exact=float(np.mean(dev <= 1e-9)), share_over_1e3=float(np.mean(dev > 1e-3)), share_over_1e2=float(np.mean(dev > 1e-2)),
                median=float(np.median(dev)), p90=float(np.percentile(dev, 90)), max=float(dev.max()))


names = {int(k): v for k, v in json.load(open(os.path.join(out_dir, "model_names.json"))).items()}

# ---------------------------------------------------------------------------------------------------------------- stage: ci
if stage == "ci":
    d, metric, subset, block, nboot = int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7]), int(sys.argv[8])
    integrity()
    S, T, r0, usable, ts = prepare(d)
    if not r0["r0_pass"]:
        print("R0 fails: no interval computed")
        sys.exit(0)
    keep = None if subset == "all" else {int(r) for r in T["round_id"]} - set(ts["stale_rounds"])
    ps = L.pair_sums_of(L.build_pair_rows(S, T, metric, VARIANT, keep_rounds=keep))
    ids = set(int(x) for x in np.unique(np.r_[ps.i, ps.j]))
    pairs = [(L.OURS_ID, c) for c in L.COMPARATORS if c in usable and c in ids]
    res = L.gap_bootstrap(ps, pairs, block, nboot)
    os.makedirs(os.path.join(res_dir, "ci"), exist_ok=True)
    path = os.path.join(res_dir, "ci", f"ci_d{d}_{metric}_{subset}_b{block}.json")
    json.dump({str(c): dict(lo=v["lo"], hi=v["hi"], draws=v["draws"], failed=v["failed"], block_days=v["block_days"], n_boot=v["n_boot"]) for (_, c), v in res.items()},
              open(path, "w"), indent=1)
    print("wrote", path)
    sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------- stage: platform (added after the run; a reported item of TEST-L1 item 7a)
if stage == "platform":
    integrity()
    outp = {}
    for d in (2, 5):
        S, T, r0, usable, ts = prepare(d)
        if not r0["r0_pass"]:
            continue
        for metric in ("prob", "point"):
            ps = L.pair_sums_of(L.build_pair_rows(S, T, metric, VARIANT))
            tab = L.platform_style_table(ps)
            outp[f"d{d}_{metric}"] = {str(m): tab[m] for m in ([L.OURS_ID] + list(L.COMPARATORS)) if m in tab}
    json.dump(outp, open(os.path.join(res_dir, "platform_style.json"), "w"), indent=1, default=str)
    print("wrote platform_style.json")
    sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------- stage: points
if stage == "points":
    man = integrity()
    print(f"1. fetched files verified against the kernel manifest ({len(man['files'])} files); build commit {man.get('build', {}).get('commit')}")
    prep = {d: prepare(d) for d in (2, 5)}
    for d in (2, 5):
        r0 = prep[d][2]
        c = r0["comparators"][L.R0_COMPARATOR]
        print(f"2. R0 definition {d} (TEST-L2): visiontspp-large informative rows {c['n_cond']}, structure share {c['share_cond']:.4f} -> {'HOLDS' if r0['r0_pass'] else 'FAILS'}; "
              f"other comparators reproduced: " + ", ".join(f"{v['name']} {'yes' if v['reproduced'] else 'NO'} ({v['n_cond']} rows)" for k, v in r0['comparators'].items() if k != L.R0_COMPARATOR))
    res = {"r0": {str(d): prep[d][2] for d in prep}, "tasks": {str(d): prep[d][4] for d in prep}, "blocks": {}, "paired": {}, "old_mase": {}}
    if not prep[2][2]["r0_pass"]:
        res["verdict"] = "NOT EVALUABLE (R0 fails in definition 2)"
        json.dump(res, open(os.path.join(res_dir, "l2_points.json"), "w"), indent=1, default=str)
        print("VERDICT: NOT EVALUABLE (R0 fails in definition 2); nothing else computed.")
        sys.exit(0)
    for d in (2, 5):
        S, T, r0, usable, ts = prep[d]
        if not r0["r0_pass"]:
            res["blocks"][str(d)] = "R0 fails: not evaluated"
            continue
        stale = set(ts["stale_rounds"])
        res["blocks"][str(d)] = {}
        for metric in ("prob", "point"):
            res["blocks"][str(d)][metric] = {}
            for subset in ("all", "nonstale"):
                if subset == "nonstale" and not stale:
                    continue
                res["blocks"][str(d)][metric][subset] = A._fit_block(S, T, metric, VARIANT, subset, stale, names, ci=False, platform_table=False, full_ci=False, usable=usable)
        res["v1_vs_served"] = res.get("v1_vs_served", {})
        res["v1_vs_served"][str(d)] = v1_vs_served(S, T)
        res["paired"][str(d)] = {f"{m}:{c}": L.paired_logratio(S, T, m, VARIANT, c) for c in sorted(usable) for m in ("point", "prob")}
        res["old_mase"][str(d)] = {str(c): L.old_arena_mase(S, T, VARIANT, c) for c in sorted(usable)}
    # decision, both readings
    ts2 = prep[2][4]
    g = res["blocks"]["2"]["prob"]["all"]["gaps"]
    g1, g2 = g[str(131)], g[str(4240)]
    dec = dict(R0=True, lost_share_def2=ts2["lost_share"], R3_fallback_share_def2_v1=ts2["fallback_share_v1"], R3=bool(ts2["fallback_share_v1"] is not None and ts2["fallback_share_v1"] <= A.GO_R3),
               R1_gap=g1.get("gap"), R1_common_tasks=g1["common_tasks"], R2_gap=g2.get("gap"), R2_common_tasks=g2["common_tasks"], R2_dropped=g2.get("dropped"))
    dec["R1"] = bool(g1.get("gap") is not None and g1["gap"] >= A.GO_R1)
    dec["R2"] = bool(g2.get("gap") is not None and g2["gap"] >= A.GO_R2)
    if ts2["lost_share"] is not None and ts2["lost_share"] > A.LOST_MAX:
        dec["strict"] = dec["literal"] = "NOT EVALUABLE (more than 5 % of definition-2 tasks lost to fetch or run errors)"
    else:
        dec["strict"] = ("NOT EVALUABLE (SCCH/PCFM-FT has %d common tasks with ours in definition 2, under the 140 of B3)" % g2["common_tasks"]) if (g2.get("dropped") or g2["common_tasks"] < FLOOR_STRICT or g1["common_tasks"] < FLOOR_STRICT) \
            else ("GO" if (dec["R1"] and dec["R2"] and dec["R3"]) else "NO-GO")
        dec["literal"] = ("NOT EVALUABLE (SCCH/PCFM-FT has %d common tasks, under 30)" % g2["common_tasks"]) if (g2.get("dropped") or g2["common_tasks"] < FLOOR_LITERAL or g1["common_tasks"] < FLOOR_LITERAL) \
            else ("GO" if (dec["R1"] and dec["R2"] and dec["R3"]) else "NO-GO")
    frag = {}
    blk = res["blocks"]["2"]["prob"].get("nonstale")
    if blk:
        frag["nonstale"] = dict(R1_gap=blk["gaps"][str(131)].get("gap"), R2_gap=blk["gaps"][str(4240)].get("gap"),
                                R1=bool((blk["gaps"][str(131)].get("gap") or -99) >= A.GO_R1), R2=bool((blk["gaps"][str(4240)].get("gap") or -99) >= A.GO_R2))
    dec["variants"] = frag
    dec["fragile"] = any((v["R1"] != dec["R1"]) or (v["R2"] != dec["R2"]) for v in frag.values())
    res["decision"] = dec
    json.dump(res, open(os.path.join(res_dir, "l2_points.json"), "w"), indent=1, default=str)
    print("\nDECISION (TEST-L2): " + json.dumps({k: v for k, v in dec.items()}, indent=1, default=str))
    sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------- stage: report
if stage == "report":
    man = integrity()
    P = json.load(open(os.path.join(res_dir, "l2_points.json")))
    cis = {}
    for fp in sorted(glob.glob(os.path.join(res_dir, "ci", "ci_*.json"))):
        base = os.path.basename(fp)[3:-5]          # d2_prob_all_b7
        d_, metric, subset, b = base.split("_")
        cis[(d_[1:], metric, subset, int(b[1:]))] = json.load(open(fp))
    md = ["# TEST-L2 verdict: gridload-compact v1 against the live field (current-data contexts, V1)\n"]
    dec = P.get("decision")
    if dec is None:
        md.append(f"**Verdict: {P['verdict']}**\n")
    else:
        g1ci = cis.get(("2", "prob", "all", 7), {})
        md.append(f"Window 2026-08-24 .. 2026-10-18 (completed live rounds), kernel build `{man.get('build', {}).get('commit', '')[:10]}`; self-evaluated shadow check on public data, contexts reconstructed from the current data "
                  f"(consistent with the revised truth used for scoring, unlike the first-published tails of the field's served contexts: probably optimistic for a live entry, by an amount not measured); not an official ranking.\n")
        md.append(f"**Strict reading (B3, at least 140 common tasks per gate comparator): {dec['strict']}**  \n**Literal-brief reading (SCCH/PCFM-FT on the rounds it played, at least 30 tasks): {dec['literal']}**\n")
        if not (dec["R1"] and dec["R3"]):
            md.append("**No reading can give GO: " + ", ".join(n for n, ok in (("R1", dec["R1"]), ("R3", dec["R3"])) if not ok) + " fail" + ("s" if (dec["R1"] != dec["R3"]) else "") + " on their own, whatever SCCH/PCFM-FT does.**\n")
        md.append("| condition | value | threshold | met |\n|---|---|---|---|")
        md.append(f"| R1 definition 2, SQL gap vs visiontspp-large | {f1(dec['R1_gap'])} {ci_txt([g1ci.get('131', {}).get('lo'), g1ci.get('131', {}).get('hi')]) if g1ci else ''} ({dec['R1_common_tasks']} common tasks) | >= +5.0 (point) | {dec['R1']} |")
        md.append(f"| R2 definition 2, SQL gap vs SCCH/PCFM-FT | {f1(dec['R2_gap'])} {ci_txt([g1ci.get('4240', {}).get('lo'), g1ci.get('4240', {}).get('hi')]) if g1ci else ''} ({dec['R2_common_tasks']} common tasks) | >= -5.0 (point) | {dec['R2']} |")
        md.append(f"| R3 definition 2 fallback share (V1) | {dec['R3_fallback_share_def2_v1']:.4f} | <= 0.05 | {dec['R3']} |")
        md.append(f"| stale-free subset flips R1 or R2 | {dec['variants']} | - | fragile = {dec['fragile']} |\n")
    ps_path = os.path.join(res_dir, "platform_style.json")
    PS = json.load(open(ps_path)) if os.path.exists(ps_path) else {}
    for d in ("2", "5"):
        blocks = P.get("blocks", {}).get(d)
        if not isinstance(blocks, dict):
            md.append(f"## Definition {d}\n{blocks}\n")
            continue
        ts = P["tasks"][d]
        r0 = P["r0"][d]
        md.append(f"## Definition {d}\n")
        md.append(f"Tasks {ts['n_tasks']} ({ts['n_rounds']} rounds), eligible {ts['n_eligible']}, dropped {ts['drop_reasons']}; stale rounds (> 6 h): {ts['stale_rounds']}.  Fallback share (V1) {ts['fallback_share_v1']:.4f}; "
                  f"latency {ts['latency_ms_per_series_p50']:.2f} ms per series (p50, Kaggle CPU, 1 thread).  R0: visiontspp-large {r0['comparators']['131']['n_cond']} informative rows, structure share "
                  f"{r0['comparators']['131']['share_cond']:.4f}; the stored scores of recent rounds drift (rmse reproduced in {100 * r0['comparators']['131']['share_rmse']:.0f} % of its rows overall).  "
                  f"V1 context scale versus the stored (served) scale over {(P['v1_vs_served'].get(d) or {}).get('n_tasks', 0)} tasks with a defined factor: exact in {100 * (P['v1_vs_served'].get(d) or {}).get('share_exact', float('nan')):.0f} %, off by more than 1e-3 in {100 * (P['v1_vs_served'].get(d) or {}).get('share_over_1e3', float('nan')):.0f} % and by more than 1 % in {100 * (P['v1_vs_served'].get(d) or {}).get('share_over_1e2', float('nan')):.0f} % (median {100 * (P['v1_vs_served'].get(d) or {}).get('median', float('nan')):.2f} %, p90 {100 * (P['v1_vs_served'].get(d) or {}).get('p90', float('nan')):.2f} %): V1 is probably optimistic against the served context (not measured).\n")
        for metric, title in (("prob", "probabilistic ranking (SQL)"), ("point", "point ranking (MAE)")):
            for subset in ("all", "nonstale"):
                blk = blocks.get(metric, {}).get(subset)
                if not blk:
                    continue
                md.append(f"**{title}, context V1, {'all rounds' if subset == 'all' else 'stale rounds left out'}**: ours {blk['score_ours']:.1f} points (place {blk['rank_ours']} of {blk['n_models']} fitted models), {blk['n_tasks']} tasks.\n")
                md.append("| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |\n|---|---|---|---|---|---|---|")
                for c, g in blk["gaps"].items():
                    if g.get("gap") is None:
                        md.append(f"| {g['name']} | n/a ({g.get('dropped', 'not in the fit')}) | | | | {g['common_tasks']} | |")
                        continue
                    cc = {b: cis.get((d, metric, subset, b), {}).get(c) for b in (7, 14, 28)}
                    cell = lambda b: ci_txt([cc[b]["lo"], cc[b]["hi"]]) if cc[b] else "n/a"
                    md.append(f"| {g['name']} | {f1(g['gap'])} | {cell(7)} | {cell(14)} | {cell(28)} | {g['common_tasks']} | {g.get('common_rounds')} |")
                md.append("")
                if subset == "all":
                    md.append("top 10 of the fit: " + "; ".join(f"{t['name']} {t['score']:.1f}" for t in blk["top10"]) + "\n")
                    ps_ = PS.get(f"d{d}_{metric}")
                    if ps_:
                        md.append("platform-style score [interval of the 28-day bootstrap with the weekly floor], rank range: " + "; ".join(f"{names.get(int(k), k)} {v['score']:.1f} [{v['ci_lower']:.1f}, {v['ci_upper']:.1f}] ranks {(str(v['rank_best']) + '-' + str(v['rank_worst'])) if v.get('rank_best') is not None else 'n/a'}" for k, v in ps_.items() if v.get("ci_lower") is not None) + "\n")
        md.append("paired log ratio of per-task errors, 100 * mean ln(err_comparator / err_ours), ISO-week block interval, tasks won:\n")
        md.append("| comparator | metric | mean | 95 % interval | tasks | share won |\n|---|---|---|---|---|---|")
        for key, v in P["paired"][d].items():
            metric, c = key.split(":")
            if v is None:
                continue
            md.append(f"| {L.COMPARATORS[int(c)]} | {metric} | {f1(v['mean'], 2)} | [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v['n_tasks']} | {v['win_share']:.2f} |")
        md.append("\nold-style arena MASE of our public texts (mae / flat-naive mae; mean over series, then rounds), ours minus comparator, ISO-week bootstrap:\n")
        md.append("| comparator | ours | comparator | difference | 95 % interval | rounds | rounds won |\n|---|---|---|---|---|---|---|")
        for c, v in P["old_mase"][d].items():
            if v is None:
                continue
            md.append(f"| {L.COMPARATORS[int(c)]} | {v['ours']:.4f} | {v['comp']:.4f} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['n_rounds']} | {v['round_win_share']:.2f} |")
        md.append("")
    text = "\n".join(md)
    open(os.path.join(res_dir, "verdict_l2.md"), "w").write(text + "\n")
    json.dump(dict(decision=dec, ci_files=sorted(os.path.basename(f) for f in glob.glob(os.path.join(res_dir, "ci", "ci_*.json")))), open(os.path.join(res_dir, "verdict_l2.json"), "w"), indent=1, default=str)
    print("wrote verdict_l2.md and verdict_l2.json")
    sys.exit(0)

print("unknown stage", stage)
sys.exit(2)
