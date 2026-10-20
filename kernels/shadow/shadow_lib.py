"""TEST-L1 shadow check: shared code of the Kaggle kernel and of scripts/shadow_verdict.py (numpy + standard library only).

Scoring and the Arena Score engine are the platform's own MIT code, vendored unmodified in ./vendor (see TEST-LOG, TEST-L1 items 3 and 5):
  vendor/forecast_metrics.py  compute_score_fields, context_scale         (ts-arena-backend main, last change 2026-10-15)
  vendor/arena_score.py       pair_sums, rate_scope, _newton, ...          (ts-arena-backend main, b8babf7)
This file only fetches, aligns, builds the rows the platform's arena_score_service builds, and adds the pre-registered paired bootstrap of gaps.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import arena_score as arena  # noqa: E402
import forecast_metrics as fm  # noqa: E402

BASE = "https://ts-arena.live/api/v1"
UA = "bench-timeseries-shadow/1 (read-only research shadow check of a public model; github.com/sankalpsthakur)"

DEFS = {
    2: dict(freq="15min", step=900, H=96, P=96, m=96, name="SMARD net load, 24 h at 15 min"),
    5: dict(freq="h", step=3600, H=72, P=24, m=24, name="SMARD net load, 72 h at 1 h"),
}
OURS_ID = 10 ** 9           # model id of gridload-compact v1 inside the fits
NAIVE_ID = -1               # the platform's internal naive forecast
COMPARATORS = {131: "visiontspp-large", 4240: "SCCH/PCFM-FT", 2328: "farang/fx2-medium-thinking", 107: "chronos-2"}
GATE_COMPARATORS = (131, 4240)
SN_READABLE = "Statistical/Seasonal-Naive"
MIN_COMMON_TASKS = 140     # a gate comparator needs this many common tasks with ours for its gap to be evaluable (Amendment 2: a deliberate two-week floor)
R0_COMPARATOR = 131        # R0 = the stored scores of this model are reproduced (per definition)
MIN_REPRO_ROWS = 100       # informative stored rows of the R0 comparator needed in a definition (Amendment 2)
MIN_REPRO_ROWS_OTHER = 20  # rows needed for another comparator to count as reproduced or not
REPRO_TOL = 1e-3
REPRO_SHARE = 0.95
REPRO_FACTOR_TOL = 1e-6   # tolerance of the one-factor-per-task structure test
REPRO_RMSE_TOL = 1e-6     # a row is informative only when the scale-free rmse is reproduced this closely (inputs identical)
V2_COVERAGE = 0.90        # R0 also needs the served context (V2) built for this share of the eligible tasks
STALE_H = 6.0

SCORE_COLS = ["round_id", "series_id", "model_id", "variant", "has_q", "n_points", "mae", "rmse", "mase_ctx", "mase_old", "sql",
              "stored_mase", "stored_sql", "stored_rmse", "stored_final", "stored_has_q"]
TASK_COLS = ["round_id", "series_id", "defn", "day", "registered", "ctx_start", "ctx_end", "n_ctx", "stale_h", "H", "n_truth",
             "eligible", "drop_reason", "scale", "scale_v1", "scale_v2", "naive_value", "naive_mae", "sn_present", "served_diff_share",
             "served_diff_rel", "v2_applied", "path_v1", "path_v2", "ms_v1"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- HTTP
class Api:
    """Read-only client of the public dashboard API (no key); retries with backoff; optional on-disk cache (tests only)."""

    def __init__(self, base: str = BASE, cache_dir: Optional[str] = None, pause: float = 0.05, tries: int = 7, timeout: int = 180):
        self.base, self.cache_dir, self.pause, self.tries, self.timeout = base, cache_dir, pause, tries, timeout
        self.lock = threading.Lock()
        self.calls = self.retries = self.nbytes = 0
        self.errors: List[str] = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        cpath = None
        if self.cache_dir:
            cpath = os.path.join(self.cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".json.gz")
            if os.path.exists(cpath):
                with gzip.open(cpath, "rb") as f:
                    return json.loads(f.read())
        last = None
        for attempt in range(self.tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                obj = json.loads(raw)
                with self.lock:
                    self.calls += 1
                    self.nbytes += len(raw)
                if cpath:
                    with gzip.open(cpath, "wb") as f:
                        f.write(raw)
                time.sleep(self.pause)
                return obj
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in (408, 425, 429, 500, 502, 503, 504):
                    break
            except Exception as e:  # network trouble, truncated body, bad JSON
                last = f"{type(e).__name__}: {e}"
            with self.lock:
                self.retries += 1
            time.sleep(min(60.0, 2.0 ** attempt))
        with self.lock:
            self.errors.append(f"{path} {params or ''}: {last}")
        raise RuntimeError(f"GET {url} failed: {last}")


def iso_to_epoch(s: str) -> int:
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def epoch_to_iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


# --------------------------------------------------------------------------------------------------- one task
def split_context_truth(points: List[Dict[str, Any]], ctx_end: int, step: int, H: int):
    """/data points -> (context timestamps, values, truth dict for ctx_end + k*step, k = 1..H)."""
    ts = np.array([iso_to_epoch(p["ts"]) for p in points], dtype=np.int64)
    vals = np.array([_f(p.get("value")) for p in points], dtype=np.float64)
    c = ts <= ctx_end
    expected = {ctx_end + k * step for k in range(1, H + 1)}
    truth = {int(t): float(v) for t, v in zip(ts[~c], vals[~c]) if int(t) in expected and math.isfinite(v)}
    return ts[c], vals[c], truth


def context_scale_of(ts: np.ndarray, vals: np.ndarray, step: int, m: int):
    stamps = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    return fm.context_scale(stamps, [None if not math.isfinite(v) else float(v) for v in vals], timedelta(seconds=step), m)[0]


def score_forecast(truth: Dict[int, float], f_ts: Sequence[int], f_y: Sequence[float], f_ci: Sequence[Optional[dict]],
                   scale: Optional[float], naive_value: Optional[float]) -> Optional[Dict[str, Any]]:
    """The platform's score fields for one (model, series); points without truth or with a non-finite forecast are dropped."""
    yt, yp, pv = [], [], []
    for t, y, ci in zip(f_ts, f_y, f_ci):
        v = truth.get(int(t))
        yy = _f(y)
        if v is None or not math.isfinite(yy):
            continue
        yt.append(v)
        yp.append(yy)
        pv.append(ci if isinstance(ci, dict) and ci else None)
    if not yt:
        return None
    return fm.compute_score_fields(np.array(yt), np.array(yp), pv, scale, naive_value)


def score_row(round_id: int, series_id: int, model_id: int, variant: str, sc: Dict[str, Any], stored: Optional[Dict[str, Any]]) -> list:
    mase = sc.get("mase")
    naive_mae = sc.get("naive_mae")
    mae = sc["mae"]
    return [round_id, series_id, model_id, variant, int(bool(sc.get("has_quantiles"))), int(sc["n_points"]), mae, sc.get("rmse", ""),
            "" if mase is None else mase,
            "" if (not naive_mae) else mae / naive_mae,
            "" if sc.get("sql_score") is None else sc["sql_score"],
            "" if not stored or stored.get("mase") is None else stored["mase"],
            "" if not stored or stored.get("sql_score") is None else stored["sql_score"],
            "" if not stored or stored.get("rmse") is None else stored["rmse"],
            "" if not stored else int(bool(stored.get("is_final"))),
            "" if not stored else int(bool(stored.get("has_quantiles")))]


def ours_forecast_to_points(res: Dict[str, Any]):
    """One element of GridLoadCompact.predict(...)['series'] -> timestamps, median, per-point decile dicts."""
    q = res["quantiles"]
    n = len(res["ts"])
    ci = [{f"q_{lv}": float(q[lv][i]) for lv in q} for i in range(n)]
    return list(res["ts"]), list(res["forecasts"]), ci


def v2_context(ts: np.ndarray, vals: np.ndarray, sn: Optional[List[Dict[str, Any]]], step: int, P: int, ctx_len: int = 1000):
    """V2 context = the context the platform SERVED: the last P points of the current context replaced, by position, with the stored forecast
    of the platform's Statistical/Seasonal-Naive model.  That model repeats the last season of the history it received (forecast step k = history
    point n-P+k-1), so its first P forecast values ARE the last P served context points, first-published values included.
    Applied only when the current context has exactly ctx_len points (so the served point set is the one we hold), the stored forecast starts one step after
    the context end, has at least P finite values, and the replaced values stay within 5 % of the level of the current ones (a sanity check, not an
    identity test: revised points are exactly what V2 restores).
    Returns (values, applied, diff_share, diff_abs_mean): the share of the P points where served and current values differ by more than 1e-6
    relative, and their mean absolute difference."""
    nope = (vals, False, float("nan"), float("nan"))
    if not sn or len(sn) < P or len(ts) != ctx_len or len(ts) < P:
        return nope
    if iso_to_epoch(sn[0]["ts"]) != int(ts[-1]) + step:
        return nope
    y = np.array([_f(p.get("y")) for p in sn[:P]], dtype=np.float64)
    if not np.isfinite(y).all():
        return nope
    cur = vals[-P:]
    fin = np.isfinite(cur)
    if fin.sum() < 0.9 * P:
        return nope
    diffs = np.abs(y[fin] - cur[fin])
    level = float(np.mean(np.abs(cur[fin])))
    dshare = float(np.mean(diffs > 1e-6 * np.maximum(1.0, np.abs(cur[fin]))))
    if not (level > 0 and float(diffs.mean()) <= 0.05 * level):
        return vals, False, dshare, float(diffs.mean())
    new = vals.copy()
    new[-P:] = y
    return new, True, dshare, float(diffs.mean())


# --------------------------------------------------------------------------------------------------- tables
def write_csv_gz(path: str, header: List[str], rows: Iterable[list]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_table(path: str) -> Dict[str, np.ndarray]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        cols: Dict[str, list] = {h: [] for h in head}
        for row in rd:
            for h, v in zip(head, row):
                cols[h].append(v)
    out: Dict[str, np.ndarray] = {}
    for h, v in cols.items():
        try:
            out[h] = np.array([float(x) if x != "" else np.nan for x in v], dtype=np.float64)
        except ValueError:
            out[h] = np.array(v, dtype=object)
    return out


# --------------------------------------------------------------------------------------------------- reproduction (R0)
def repro_stats(S: Dict[str, np.ndarray], T: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Reproduction of the stored scores of the field (Amendment 2: scale-free, conditional on unchanged inputs).
    Rows: stored final rows of eligible tasks.  A row is INFORMATIVE when the scale-free `rmse` is reproduced within REPRO_TOL (same forecasts and
    the same truth as when the platform scored); rows whose rmse differs are inputs that changed since (truth revised after the evaluation, or
    fewer points scored then) and are only reported, by round age.  On informative rows the structure of `mase`, `sql_score` and `has_quantiles`
    must be reproduced up to ONE factor per task, the ratio of the stored context scale to ours (f_task = median over informative rows of
    mase_recomputed / mase_stored): |ratio / f_task - 1| <= REPRO_FACTOR_TOL for both columns.  The factor itself is a diagnostic (the share of
    tasks with f_task = 1 is the share whose served context is reproduced exactly); Arena Score gaps do not depend on it.
    R0 holds iff the R0 comparator has at least MIN_REPRO_ROWS informative rows with a share >= REPRO_SHARE and V2 was built for at least
    V2_COVERAGE of the eligible tasks."""
    elig = eligible_set(T)
    field = S["variant"] == "field"
    out: Dict[str, Any] = {"per_model": {}}
    mid, rnd, ser = S["model_id"].astype(np.int64), S["round_id"].astype(np.int64), S["series_id"].astype(np.int64)
    keep = np.array([(r, s) in elig for r, s in zip(rnd, ser)]) & field & (S["stored_final"] == 1)
    v2 = {(int(r), int(s)): (v == 1) for r, s, v in zip(T["round_id"], T["series_id"], T["v2_applied"])}
    dayof = {(int(r), int(s)): int(d) for r, s, d in zip(T["round_id"], T["series_id"], T["day"])}
    last_day = int(np.nanmax(T["day"])) if len(T["day"]) else 0
    key = list(zip(rnd.tolist(), ser.tolist()))
    age = np.array([last_day - dayof.get(k, last_day) for k in key])
    r_, s_ = S["rmse"], S["stored_rmse"]
    informative = keep & np.isfinite(r_) & np.isfinite(s_) & (np.abs(r_ - s_) <= REPRO_RMSE_TOL * np.abs(s_))
    ratio_m = S["mase_ctx"] / S["stored_mase"]
    ratio_s = S["sql"] / S["stored_sql"]
    # one factor per task, from the informative rows of all models
    fs: Dict[tuple, list] = {}
    for i in np.flatnonzero(informative & np.isfinite(ratio_m)):
        fs.setdefault(key[i], []).append(ratio_m[i])
    f_task = {k: float(np.median(v)) for k, v in fs.items()}
    f_row = np.array([f_task.get(k, np.nan) for k in key])
    okm = np.isfinite(ratio_m) & np.isfinite(f_row) & (np.abs(ratio_m / f_row - 1.0) <= REPRO_FACTOR_TOL)
    oks = (np.isfinite(ratio_s) & np.isfinite(f_row) & (np.abs(ratio_s / f_row - 1.0) <= REPRO_FACTOR_TOL)) | (~np.isfinite(S["sql"]) & ~np.isfinite(S["stored_sql"]))
    okq = S["has_q"] == S["stored_has_q"]
    ok = okm & oks & okq
    buckets = [("0-6 d", 0, 6), ("7-13 d", 7, 13), ("14-27 d", 14, 27), ("28+ d", 28, 10 ** 6)]
    tot_cond = tot_ok = 0
    for m in np.unique(mid[keep]):
        sel = keep & (mid == m)
        cond = informative & (mid == m)
        by_age = {}
        for name, lo, hi in buckets:
            sa = sel & (age >= lo) & (age <= hi)
            if sa.any():
                by_age[name] = dict(n=int(sa.sum()), share_rmse=float(informative[sa].mean()), share_struct=float(ok[sa & informative].mean()) if (sa & informative).any() else None)
        out["per_model"][int(m)] = dict(n=int(sel.sum()), share_rmse=float(informative[sel].mean()), n_cond=int(cond.sum()), n_cond_ok=int((ok & cond).sum()),
                                        share_cond=float((ok & cond).sum() / cond.sum()) if cond.any() else float("nan"), by_age=by_age)
        tot_cond += int(cond.sum())
        tot_ok += int((ok & cond).sum())
    out["all_models"] = dict(n=int(keep.sum()), n_cond=tot_cond, n_cond_ok=tot_ok, share_cond=(tot_ok / tot_cond) if tot_cond else float("nan"),
                             share_rmse=float(informative[keep].mean()) if keep.any() else float("nan"))
    # scale diagnostics: f_task per task (1 = the served context is reproduced exactly)
    fv = np.array(list(f_task.values())) if f_task else np.array([])
    out["scale_factor"] = dict(n_tasks=int(len(fv)), share_exact=float(np.mean(np.abs(fv - 1) <= 1e-9)) if len(fv) else None,
                               share_over_1e3=float(np.mean(np.abs(fv - 1) > 1e-3)) if len(fv) else None,
                               max_abs_dev=float(np.max(np.abs(fv - 1))) if len(fv) else None, p95_abs_dev=float(np.percentile(np.abs(fv - 1), 95)) if len(fv) else None)
    res = {}
    for c, name in COMPARATORS.items():
        r = out["per_model"].get(c)
        need = MIN_REPRO_ROWS if c == R0_COMPARATOR else MIN_REPRO_ROWS_OTHER
        res[c] = dict(name=name, n=0 if r is None else r["n"], n_cond=0 if r is None else r["n_cond"], share_cond=None if r is None else r["share_cond"],
                      share_rmse=None if r is None else r["share_rmse"], reproduced=bool(r is not None and r["n_cond"] >= need and r["share_cond"] >= REPRO_SHARE))
    out["comparators"] = res
    n_el = len(elig)
    n_v2 = sum(1 for k in elig if v2.get(k, False))
    out["v2_coverage"] = (n_v2 / n_el) if n_el else float("nan")
    out["r0_pass"] = bool(res[R0_COMPARATOR]["reproduced"] and out["v2_coverage"] >= V2_COVERAGE)
    return out


# --------------------------------------------------------------------------------------------------- Arena Score rows and fits
def eligible_set(T: Dict[str, np.ndarray]) -> set:
    return {(int(r), int(s)) for r, s, e in zip(T["round_id"], T["series_id"], T["eligible"]) if e == 1}


def build_pair_rows(S, T, metric: str, ours_variant: Optional[str], keep_rounds: Optional[set] = None,
                    drop_models: Iterable[int] = ()):
    """Rows of arena_score_service.summarize_definition: field models + (ours) + the naive row per task.  metric: point | prob."""
    elig = eligible_set(T)
    tinfo = {(int(r), int(s)): (float(n), float(sc), int(d)) for r, s, n, sc, d in zip(T["round_id"], T["series_id"], T["naive_mae"], T["scale"], T["day"])}
    variant = S["variant"]
    sel = (variant == "field")
    if ours_variant:
        sel = sel | (variant == ("ours_" + ours_variant))
    rnd, ser, mid = S["round_id"].astype(np.int64)[sel], S["series_id"].astype(np.int64)[sel], S["model_id"].astype(np.int64)[sel]
    mid = np.where(S["variant"][sel] == "field", mid, OURS_ID)
    npt = S["n_points"].astype(np.int64)[sel]
    if metric == "point":
        err = S["mae"][sel]
        ok = np.isfinite(err) & (err >= 0)
    else:
        err = S["sql"][sel]
        ok = np.isfinite(err) & (err >= 0) & (S["has_q"][sel] == 1)
    ok &= np.array([(r, s) in elig for r, s in zip(rnd, ser)])
    if keep_rounds is not None:
        ok &= np.array([r in keep_rounds for r in rnd])
    drop = set(int(x) for x in drop_models)
    if drop:
        ok &= ~np.isin(mid, list(drop))
    rnd, ser, mid, err, npt = rnd[ok], ser[ok], mid[ok], err[ok], npt[ok]
    seen, first = set(), []
    for k, key in enumerate(zip(rnd.tolist(), ser.tolist(), mid.tolist())):   # one row per (round, series, model), as the platform's table has
        if key not in seen:
            seen.add(key)
            first.append(k)
    if len(first) != len(rnd):
        first = np.array(first, dtype=np.int64)
        rnd, ser, mid, err, npt = rnd[first], ser[first], mid[first], err[first], npt[first]
    day = np.array([tinfo[(int(r), int(s))][2] for r, s in zip(rnd, ser)], dtype=np.int64)
    # naive row per task (from the fullest row of the task), as in arena_score_service.naive_rows
    best: Dict[Tuple[int, int], int] = {}
    for r, s, n in zip(rnd, ser, npt):
        k = (int(r), int(s))
        if n > best.get(k, -1):
            best[k] = int(n)
    nr, ns, ne, nn, nd = [], [], [], [], []
    for (r, s), n in best.items():
        nm, sc, d = tinfo[(r, s)]
        e = nm if metric == "point" else (nm / sc if sc and math.isfinite(sc) else float("nan"))
        if math.isfinite(e) and e >= 0:
            nr.append(r); ns.append(s); ne.append(e); nn.append(n); nd.append(d)
    return (np.r_[rnd, nr], np.r_[ser, ns], np.r_[mid, np.full(len(nr), NAIVE_ID)], np.r_[err, ne], np.r_[npt, nn], np.r_[day, nd])


def pair_sums_of(rows) -> "arena.PairSums":
    return arena.pair_sums(*rows)


def platform_style_table(ps: "arena.PairSums", n_boot: int = 200, block_days: int = 28) -> Dict[int, Dict[str, Any]]:
    """The platform's own rate_scope (naive = zero point); block bootstrap 28 days, 200 refits, weekly floor.  Per model: score, ci."""
    ids = np.unique(np.r_[ps.i, ps.j])
    models = [int(m) for m in ids if int(m) != NAIVE_ID]
    covered = {m: arena.Coverage(True, 1.0) for m in models}
    old = arena.BLOCK_DAYS
    arena.BLOCK_DAYS = block_days
    try:
        rating = arena.rate_scope(ps, models, covered, n_boot=n_boot, seed=0, scope="shadow", reference=NAIVE_ID)
    finally:
        arena.BLOCK_DAYS = old
    out = {}
    for r in rating.models:
        out[int(r.model_id)] = dict(score=r.score, ci_lower=r.ci_lower, ci_upper=r.ci_upper, weeks=r.weeks, evidence=r.evidence,
                                    rank_best=r.rank_best, rank_worst=r.rank_worst, provisional=r.provisional)
    return out


def _prep_fit(ps: "arena.PairSums"):
    ids = np.unique(np.r_[ps.i, ps.j])
    n = len(ids)
    ii, jj = np.searchsorted(ids, ps.i), np.searchsorted(ids, ps.j)
    key, row_pair = np.unique(ii * n + jj, return_inverse=True)
    row_pair = row_pair.ravel()
    return ids, n, key // n, key % n, row_pair


def fit_theta(ps: "arena.PairSums"):
    """Joint Bradley-Terry fit with the platform's Newton; returns ids and strengths (naive = 0), and the prepared structure."""
    ids, n, pi, pj, row_pair = _prep_fit(ps)
    w = np.bincount(row_pair, ps.w, len(pi))
    wy = np.bincount(row_pair, ps.wy, len(pi))
    theta, gmax = arena._newton(pi, pj, w, wy, n, np.zeros(n))
    zero = theta[int(np.searchsorted(ids, NAIVE_ID))]
    return ids, theta - zero, (n, pi, pj, row_pair, gmax)


def gap_bootstrap(ps: "arena.PairSums", pairs: List[Tuple[int, int]], block_days: int, n_boot: int, seed: int = 0) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Paired 95 % percentile interval of 100*(theta_A - theta_B) under the platform's circular block bootstrap of the days of the window
    with the block length set to block_days (whole fit refitted per draw, warm start from the full fit)."""
    ids, n, pi, pj, row_pair = _prep_fit(ps)
    w0 = np.bincount(row_pair, ps.w, len(pi))
    wy0 = np.bincount(row_pair, ps.wy, len(pi))
    theta0, _ = arena._newton(pi, pj, w0, wy0, n, np.zeros(n))
    n_days = int(ps.last_day - ps.first_day + 1)
    circle = block_days * -(-n_days // block_days)
    starts = np.random.default_rng(seed).integers(0, circle, size=(n_boot, circle // block_days))
    old = arena.BLOCK_DAYS
    arena.BLOCK_DAYS = block_days
    try:
        counts = arena._block_counts(starts, circle)
    finally:
        arena.BLOCK_DAYS = old
    row_day = (ps.day - ps.first_day).astype(np.int64)
    idset = set(int(x) for x in ids.tolist())
    pos = {int(m): int(np.searchsorted(ids, m)) for pair in pairs for m in pair if int(m) in idset}
    draws = {p: [] for p in pairs}
    failed = 0
    for b in range(n_boot):
        c = counts[b][row_day]
        wb = np.bincount(row_pair, c * ps.w, len(pi))
        yb = np.bincount(row_pair, c * ps.wy, len(pi))
        keep = wb > 0
        present = (np.bincount(pi[keep], minlength=n) + np.bincount(pj[keep], minlength=n)) > 0
        try:
            th, _ = arena._newton(pi[keep], pj[keep], wb[keep], yb[keep], n, theta0)
        except arena.ArenaScoreFitError:
            failed += 1
            continue
        for a_, b_ in pairs:
            if a_ in pos and b_ in pos and present[pos[a_]] and present[pos[b_]]:
                draws[(a_, b_)].append(100.0 * (th[pos[a_]] - th[pos[b_]]))
    out = {}
    for p, d in draws.items():
        d = np.array(d)
        out[p] = dict(lo=float(np.percentile(d, 2.5)) if len(d) >= 20 else None, hi=float(np.percentile(d, 97.5)) if len(d) >= 20 else None,
                      draws=int(len(d)), failed=int(failed), block_days=block_days, n_boot=n_boot)
    return out


# --------------------------------------------------------------------------------------------------- paired descriptive statistics
def week_blocks_bootstrap(values: np.ndarray, weeks: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean of values with a 95 % ISO-week block bootstrap (resample weeks, ratio of sums, as the scout's replica)."""
    uw = np.unique(weeks)
    sums = np.array([values[weeks == w].sum() for w in uw])
    cnt = np.array([(weeks == w).sum() for w in uw])
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uw), len(uw))
        bs[b] = sums[pick].sum() / cnt[pick].sum()
    return float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def iso_week_of_day(day: np.ndarray) -> np.ndarray:
    base = datetime(1970, 1, 1)
    return np.array([(base + timedelta(days=int(d))).isocalendar()[0] * 100 + (base + timedelta(days=int(d))).isocalendar()[1] for d in day])


def paired_logratio(S, T, metric: str, ours_variant: str, comp: int) -> Optional[Dict[str, Any]]:
    """100*mean ln(err_comp / err_ours) over eligible tasks both have, ISO-week block interval, task win share."""
    col = "mae" if metric == "point" else "sql"
    elig = eligible_set(T)
    dayof = {(int(r), int(s)): int(d) for r, s, d in zip(T["round_id"], T["series_id"], T["day"])}
    f = (S["variant"] == "field") & (S["model_id"] == comp)
    o = S["variant"] == "ours_" + ours_variant
    e_c = {(int(r), int(s)): (v, q) for r, s, v, q in zip(S["round_id"][f], S["series_id"][f], S[col][f], S["has_q"][f])}
    e_o = {(int(r), int(s)): (v, q) for r, s, v, q in zip(S["round_id"][o], S["series_id"][o], S[col][o], S["has_q"][o])}
    keys = [k for k in e_o if k in e_c and k in elig and math.isfinite(e_o[k][0]) and math.isfinite(e_c[k][0]) and e_o[k][0] > 0 and e_c[k][0] > 0
            and (metric == "point" or (e_c[k][1] == 1 and e_o[k][1] == 1))]
    if not keys:
        return None
    lr = np.array([100.0 * math.log(e_c[k][0] / e_o[k][0]) for k in keys])
    wk = iso_week_of_day(np.array([dayof[k] for k in keys]))
    mean, lo, hi = week_blocks_bootstrap(lr, wk)
    return dict(n_tasks=len(keys), n_rounds=len({k[0] for k in keys}), mean=mean, lo=lo, hi=hi, win_share=float((lr > 0).mean()))


def old_arena_mase(S, T, ours_variant: str, comp: int) -> Optional[Dict[str, Any]]:
    """The metric of our public texts: per round the mean over series of mae / naive_mae, then the mean over rounds; per-round difference
    (ours - comparator) over common series, ISO-week block bootstrap of the mean difference; round win share."""
    elig = eligible_set(T)
    dayof = {(int(r), int(s)): int(d) for r, s, d in zip(T["round_id"], T["series_id"], T["day"])}
    f = (S["variant"] == "field") & (S["model_id"] == comp)
    o = S["variant"] == "ours_" + ours_variant
    mc = {(int(r), int(s)): v for r, s, v in zip(S["round_id"][f], S["series_id"][f], S["mase_old"][f])}
    mo = {(int(r), int(s)): v for r, s, v in zip(S["round_id"][o], S["series_id"][o], S["mase_old"][o])}
    by_round: Dict[int, list] = {}
    for k in mo:
        if k in mc and k in elig and math.isfinite(mo[k]) and math.isfinite(mc[k]):
            by_round.setdefault(k[0], []).append((mo[k], mc[k], dayof[k]))
    if not by_round:
        return None
    rids = sorted(by_round)
    ours_m = np.array([np.mean([a for a, _, _ in by_round[r]]) for r in rids])
    comp_m = np.array([np.mean([b for _, b, _ in by_round[r]]) for r in rids])
    diff = ours_m - comp_m
    wk = iso_week_of_day(np.array([by_round[r][0][2] for r in rids]))
    mean, lo, hi = week_blocks_bootstrap(diff, wk)
    return dict(n_rounds=len(rids), ours=float(ours_m.mean()), comp=float(comp_m.mean()), diff=mean, lo=lo, hi=hi, round_win_share=float((diff < 0).mean()))
