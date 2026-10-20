"""TEST-L1 analysis and decision rule (pre-registered in TEST-LOG.md, TEST-L1 items 4-8 and Amendment 1).  Shared by the kernel and scripts/shadow_verdict.py.

Order of reading: R0 (reproduction of the stored scores of the comparators) is evaluated first; when it fails nothing involving our model is computed.
Gate variant: V2 (the served context, reconstructed with the platform's Seasonal-Naive stored forecast); V1 (current-data context) is reported as the optimistic bound.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

import shadow_lib as L

GO_R1 = 5.0      # def 2, prob: gap(ours, visiontspp-large) >= +5
GO_R2 = -5.0     # def 2, prob: gap(ours, SCCH/PCFM-FT) >= -5
GO_R3 = 0.05     # def 2: fallback share <= 5 %
LOST_MAX = 0.05  # share of tasks lost to fetch / run errors
GATE_VARIANT = "v2"


def task_summary(T: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = len(T["round_id"])
    elig = T["eligible"] == 1
    reasons = Counter(str(r) for r, e in zip(T["drop_reason"], elig) if not e)
    lost = sum(v for k, v in reasons.items() if k in ("fetch_error", "run_error", "ours_incomplete"))
    stale = {}
    for r, h in zip(T["round_id"].astype(int), T["stale_h"]):
        if math.isfinite(h):
            stale[r] = max(stale.get(r, -1e9), float(h))
    stale_rounds = sorted(r for r, h in stale.items() if h > L.STALE_H)
    ve = T["ms_v1"][elig]
    n_el = int(elig.sum())
    p1 = Counter(str(p) for p, e in zip(T["path_v1"], elig) if e)
    p2 = Counter(str(p) for p, e in zip(T["path_v2"], elig) if e)
    sn_ok = (T["sn_present"] == 1) & elig
    v2 = (T["v2_applied"] == 1) & elig
    sc1, sc2 = T["scale_v1"][v2], T["scale_v2"][v2]
    rel = np.abs(sc1 / sc2 - 1.0)
    return dict(
        n_tasks=n, n_eligible=n_el, n_rounds=len(set(T["round_id"].astype(int))), drop_reasons=dict(reasons),
        lost_share=lost / n if n else None, stale_rounds=stale_rounds,
        max_stale_h_by_round={int(k): round(v, 2) for k, v in stale.items() if v > L.STALE_H},
        fallback_v1=int(p1.get("fallback", 0)), fallback_v2=int(p2.get("fallback", 0)),
        fallback_share_v1=(p1.get("fallback", 0) / n_el) if n_el else None, fallback_share_v2=(p2.get("fallback", 0) / n_el) if n_el else None,
        latency_ms_per_series_p50=float(np.nanmedian(ve)) if len(ve) else None, latency_ms_per_series_p95=float(np.nanpercentile(ve, 95)) if len(ve) else None,
        sn_present_share=float(sn_ok.sum() / n_el) if n_el else None,
        v2_applied=int(v2.sum()), v2_applied_share=float(v2.sum() / n_el) if n_el else None,
        served_diff_share_mean=float(np.nanmean(T["served_diff_share"][v2])) if v2.any() else None,
        served_diff_share_any=float(np.mean(T["served_diff_share"][v2] > 0)) if v2.any() else None,
        served_diff_rel_p50=float(np.nanmedian(T["served_diff_rel"][v2])) if v2.any() else None,
        served_diff_rel_p95=float(np.nanpercentile(T["served_diff_rel"][v2], 95)) if v2.any() else None,
        scale_v1_vs_served_share_over_1e3=float(np.mean(rel > 1e-3)) if v2.any() else None,
        scale_v1_vs_served_rel_p50=float(np.nanmedian(rel)) if v2.any() else None, scale_v1_vs_served_rel_max=float(np.nanmax(rel)) if v2.any() else None,
    )


def _fit_block(S, T, metric: str, variant: str, subset: str, stale_rounds: set, names: Dict[int, str], ci: bool, platform_table: bool,
               full_ci: bool, usable: set) -> Dict[str, Any]:
    all_rounds = {int(r) for r in T["round_id"]}
    keep = None if subset == "all" else all_rounds - stale_rounds
    rows = L.build_pair_rows(S, T, metric, variant, keep_rounds=keep)
    if len(rows[0]) == 0:
        return dict(error="no rows")
    ps = L.pair_sums_of(rows)
    ids, theta, aux = L.fit_theta(ps)
    score = {int(m): 100.0 * float(t) for m, t in zip(ids, theta)}
    models = [m for m in score if m != L.NAIVE_ID]
    order = sorted(models, key=lambda m: -score[m])
    rank_ours = order.index(L.OURS_ID) + 1 if L.OURS_ID in score else None
    rnd, ser, mid = rows[0], rows[1], rows[2]
    tasks_of: Dict[int, set] = {}
    for r, s, m in zip(rnd, ser, mid):
        tasks_of.setdefault(int(m), set()).add((int(r), int(s)))
    gaps: Dict[str, Any] = {}
    pairs = [(L.OURS_ID, c) for c in L.COMPARATORS if c in score and c in usable]
    cis: Dict[int, Dict[Any, Any]] = {}
    if ci and L.OURS_ID in score:
        blocks = [(7, 1000), (14, 500), (28, 200)] if full_ci else [(7, 1000)]
        for bd, nb in blocks:
            cis[bd] = L.gap_bootstrap(ps, pairs, bd, nb)
    for c in L.COMPARATORS:
        if c not in usable:
            gaps[str(c)] = dict(name=L.COMPARATORS[c], gap=None, common_tasks=0, dropped="stored scores not reproduced or too few stored rows")
            continue
        if c not in score or L.OURS_ID not in score:
            gaps[str(c)] = dict(name=L.COMPARATORS[c], gap=None, common_tasks=0)
            continue
        common = tasks_of.get(L.OURS_ID, set()) & tasks_of.get(c, set())
        g = dict(name=L.COMPARATORS[c], gap=score[L.OURS_ID] - score[c], score_comp=score[c], common_tasks=len(common),
                 common_rounds=len({r for r, _ in common}), n_tasks_comp=len(tasks_of.get(c, ())))
        for bd, d in cis.items():
            q = d.get((L.OURS_ID, c))
            if q:
                g[f"ci{bd}"] = [q["lo"], q["hi"]]
                g[f"ci{bd}_draws"] = q["draws"]
        gaps[str(c)] = g
    out = dict(n_models=len(models), n_tasks=len({(int(r), int(s)) for r, s in zip(rnd, ser)}), n_pairs=int(ps.n_pairs), skipped_points=int(ps.skipped_points),
               skipped_zero=len(ps.skipped_zero), fit_max_grad=float(aux[4]), score_ours=score.get(L.OURS_ID), rank_ours=rank_ours,
               top10=[dict(id=m, name=names.get(m, str(m)), score=round(score[m], 3)) for m in order[:10]], gaps=gaps,
               comparator_scores={str(c): score.get(c) for c in L.COMPARATORS})
    if platform_table:
        try:
            tab = L.platform_style_table(ps)
            out["platform_style"] = {str(m): tab[m] for m in ([L.OURS_ID] + list(L.COMPARATORS)) if m in tab}
        except Exception as e:  # reported, never fatal
            out["platform_style_error"] = f"{type(e).__name__}: {e}"
    return out


def analyze_definition(defn: int, S, T, names: Optional[Dict[int, str]] = None, ci: bool = True, platform: bool = True) -> Dict[str, Any]:
    names = names or {}
    res: Dict[str, Any] = dict(defn=defn, tasks=task_summary(T))
    res["r0"] = L.repro_stats(S, T)
    if not res["r0"]["r0_pass"]:
        res["status"] = "NOT_EVALUABLE_R0"
        return res
    stale_rounds = set(res["tasks"]["stale_rounds"])
    usable = {c for c, v in res["r0"]["comparators"].items() if v["reproduced"]}
    res["usable_comparators"] = sorted(usable)
    res["status"] = "ok"
    for metric in ("point", "prob"):
        res[metric] = {}
        for variant in (GATE_VARIANT, "v1"):
            res[metric][variant] = {}
            for subset in ("all", "nonstale"):
                if subset == "nonstale" and not stale_rounds:
                    continue
                full = (variant == GATE_VARIANT and subset == "all")
                res[metric][variant][subset] = _fit_block(S, T, metric, variant, subset, stale_rounds, names, ci=ci, platform_table=(full and platform), full_ci=full, usable=usable)
    res["paired_logratio"], res["old_arena_mase"] = {}, {}
    for variant in (GATE_VARIANT, "v1"):
        for c in sorted(usable):
            for metric in ("point", "prob"):
                res["paired_logratio"][f"{metric}:{variant}:{c}"] = L.paired_logratio(S, T, metric, variant, c)
            res["old_arena_mase"][f"{variant}:{c}"] = L.old_arena_mase(S, T, variant, c)
    return res


def decide(A2: Dict[str, Any], A5: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered rule (Amendment 1: gate variant V2).  Returns the verdict and every condition with its value."""
    d: Dict[str, Any] = dict(rule="GO iff R0 and R1 (def2 prob gap vs visiontspp-large >= +5) and R2 (gap vs SCCH/PCFM-FT >= -5) and R3 (def2 fallback share <= 5 %), all on the V2 (served-context) variant",
                             gate_variant=GATE_VARIANT)
    ts = A2["tasks"]
    d["R0"] = bool(A2["r0"]["r0_pass"])
    d["lost_share_def2"] = ts["lost_share"]
    d["R3_fallback_share_def2"] = ts[f"fallback_share_{GATE_VARIANT}"]
    d["R3"] = bool(d["R3_fallback_share_def2"] is not None and d["R3_fallback_share_def2"] <= GO_R3)
    d["fallback_share_def2_v1"] = ts["fallback_share_v1"]
    if A5 is not None:
        d["fallback_share_def5_v2"] = A5["tasks"]["fallback_share_v2"]
        d["fallback_share_def5_v1"] = A5["tasks"]["fallback_share_v1"]
    if not d["R0"]:
        d["verdict"] = "NOT EVALUABLE (R0: the stored scores of visiontspp-large are not reproduced in definition 2)"
        return d
    if ts["lost_share"] is not None and ts["lost_share"] > LOST_MAX:
        d["verdict"] = "NOT EVALUABLE (more than 5 % of definition-2 tasks lost to fetch or run errors)"
        return d
    g = A2["prob"][GATE_VARIANT]["all"]["gaps"]
    id1, id2 = L.GATE_COMPARATORS
    g1, g2 = g[str(id1)], g[str(id2)]
    if g1.get("dropped") or g2.get("dropped"):
        d["verdict"] = "NOT EVALUABLE (a gate comparator's stored scores were not reproduced or it has too few stored rows in definition 2)"
        return d
    d["R1_gap"], d["R1_common_tasks"], d["R1_ci7"] = g1.get("gap"), g1["common_tasks"], g1.get("ci7")
    d["R2_gap"], d["R2_common_tasks"], d["R2_ci7"] = g2.get("gap"), g2["common_tasks"], g2.get("ci7")
    if g1["common_tasks"] < L.MIN_COMMON_TASKS or g2["common_tasks"] < L.MIN_COMMON_TASKS or g1.get("gap") is None or g2.get("gap") is None:
        d["verdict"] = "NOT EVALUABLE (a gate comparator has fewer than 140 common tasks with ours in definition 2)"
        return d
    d["R1"], d["R2"] = bool(g1["gap"] >= GO_R1), bool(g2["gap"] >= GO_R2)
    d["verdict"] = "GO" if (d["R1"] and d["R2"] and d["R3"]) else "NO-GO"
    # fragility: the same two conditions on V1 (optimistic bound) and on the stale-free subsets (reported; never changes the verdict)
    frag = {}
    for variant, subset in (("v1", "all"), (GATE_VARIANT, "nonstale"), ("v1", "nonstale")):
        blk = A2["prob"].get(variant, {}).get(subset)
        if blk and "gaps" in blk and blk["gaps"][str(id1)].get("gap") is not None and blk["gaps"][str(id2)].get("gap") is not None:
            frag[f"{variant}:{subset}"] = dict(R1_gap=blk["gaps"][str(id1)]["gap"], R2_gap=blk["gaps"][str(id2)]["gap"],
                                               R1=bool(blk["gaps"][str(id1)]["gap"] >= GO_R1), R2=bool(blk["gaps"][str(id2)]["gap"] >= GO_R2))
    d["variants"] = frag
    d["fragile"] = any((v["R1"] != d["R1"]) or (v["R2"] != d["R2"]) for k, v in frag.items() if not k.startswith("v1:all"))
    d["v1_would_change_verdict"] = bool(frag.get("v1:all") and ((frag["v1:all"]["R1"] != d["R1"]) or (frag["v1:all"]["R2"] != d["R2"])))
    # claim rule wording (7-day-block interval of the gap)
    claims = {}
    for c, gg in g.items():
        ci7 = gg.get("ci7")
        if gg.get("gap") is None or not ci7 or ci7[0] is None:
            claims[c] = "no interval"
        elif ci7[0] > 0:
            claims[c] = "ahead"
        elif ci7[1] < 0:
            claims[c] = "behind"
        else:
            claims[c] = "not distinguishable"
    d["claims_def2_prob"] = {g[c]["name"]: v for c, v in claims.items()}
    return d
