#!/usr/bin/env python3
"""TEST-L1 kernel main (private Kaggle CPU kernel bench-timeseries-shadow; pre-registration: TEST-LOG.md, TEST-L1).

Stages: hashes -> service -> smoke (out-of-window rounds 11493 / 11494) -> window fetch + rescoring of every model + our forecasts -> analysis.
Reads only the public dashboard API (no key, no account).  Writes small tables to --out.  Nothing is published.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np

ROOT = os.environ.get("SHADOW_ROOT") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "svc"))
import shadow_lib as L  # noqa: E402
import shadow_analysis as A  # noqa: E402

LEVELS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
SMOKE = {2: 11493, 5: 11494}          # out-of-window rounds (2026-07-20) used only to test the pipeline
OUT = "/kaggle/working/out"
_LOG = None


def log(msg: str) -> None:
    line = f"{L.utc_now()} {msg}"
    print(line, flush=True)
    if _LOG:
        _LOG.write(line + "\n")
        _LOG.flush()


class MockService:
    """Plumbing test only (never used in the kernel): repeats the last season with fixed relative deciles."""
    def predict(self, history, horizon, freq):
        step, P = (900, 96) if freq == "15min" else (3600, 24)
        out = []
        for s in history:
            ts = [L.iso_to_epoch(it["ts"]) for it in s]
            vals = np.array([np.nan if it["value"] is None else it["value"] for it in s], dtype=float)
            last = vals[np.isfinite(vals)][-1]
            season = np.where(np.isfinite(vals[-P:]), vals[-P:], last)
            q = np.array([[season[(k - 1) % P] * (1 + z * 0.03) for z in (-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28)] for k in range(1, horizon + 1)])
            out.append({"ts": [ts[-1] + k * step for k in range(1, horizon + 1)], "forecasts": q[:, 4].tolist(), "path": "specialist",
                        "quantiles": {lv: q[:, i].tolist() for i, lv in enumerate(LEVELS)}})
        return {"series": out, "batch": True}


def assert_manifest() -> Dict[str, str]:
    """Fail closed: every vendored service file must equal the hash in V1-MANIFEST.txt; the int8 files must equal the hashes in the info files."""
    bad, seen = [], {}
    for line in open(os.path.join(ROOT, "V1-MANIFEST.txt")):
        if line.startswith("#") or not line.strip():
            continue
        sha, rel = line.split(None, 1)
        rel = rel.strip()
        p = os.path.join(ROOT, "svc", rel)
        got = L.sha256_file(p) if os.path.exists(p) else None
        seen[rel] = got
        if got != sha:
            bad.append(rel)
    for fm_ in (15, 60):
        info = json.load(open(os.path.join(ROOT, "svc", "weights", f"net{fm_}_info.json")))
        if info["int8_sha256"] != seen.get(f"weights/net{fm_}_int8.onnx"):
            bad.append(f"net{fm_} int8 vs info")
    if bad:
        raise SystemExit("V1 manifest check FAILED: " + ", ".join(bad))
    return seen


def make_service(mock: bool):
    if mock:
        return MockService()
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        log("onnxruntime missing: pip install onnxruntime==1.30.0")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "onnxruntime==1.30.0"])
    from app.model import GridLoadCompact
    return GridLoadCompact(precision="int8")


# ----------------------------------------------------------------------------------------------------------- one round
def process_round(api: L.Api, svc, defn: int, rinfo: Dict[str, Any], workers: int, id_map: Dict[str, tuple], names: Dict[int, str]):
    cfg = L.DEFS[defn]
    step, H, P, m, freq = cfg["step"], cfg["H"], cfg["P"], cfg["m"], cfg["freq"]
    rid = int(rinfo["id"])
    registered = L.iso_to_epoch(rinfo["registration_start"])
    day = registered // 86400
    meta = api.get(f"rounds/{rid}/series")
    lb = api.get(f"rounds/{rid}/leaderboard")
    stored: Dict[tuple, dict] = {}
    for r in lb:
        stored[(int(r["series_id"]), int(r["model_id"]))] = r
        id_map.setdefault(r["readable_id"], (int(r["model_id"]), r["model_name"], r.get("track")))
        names.setdefault(int(r["model_id"]), r["model_name"])

    def fetch(sm):
        sid = int(sm["series_id"])
        try:
            ce, cs = L.iso_to_epoch(sm["context_end_time"]), L.iso_to_epoch(sm["context_start_time"])
            d = api.get(f"rounds/{rid}/series/{sid}/data", {"start_time": L.epoch_to_iso(cs), "end_time": L.epoch_to_iso(ce + H * step)})
            f = api.get(f"rounds/{rid}/series/{sid}/forecasts")
            return sm, d, f
        except Exception as e:  # noqa: BLE001  (recorded as a dropped task)
            return sm, None, None

    with ThreadPoolExecutor(workers) as ex:
        fetched = list(ex.map(fetch, meta))

    tasks: List[dict] = []
    rows: List[list] = []
    pending: List[dict] = []
    for sm, d, f in fetched:
        sid = int(sm["series_id"])
        ce, cs = L.iso_to_epoch(sm["context_end_time"]), L.iso_to_epoch(sm["context_start_time"])
        t = {c: "" for c in L.TASK_COLS}
        t.update(round_id=rid, series_id=sid, defn=defn, day=day, registered=registered, ctx_start=cs, ctx_end=ce, H=H,
                 stale_h=(registered - 7200 - ce) / 3600.0, eligible=0)
        tasks.append(t)
        if d is None or f is None:
            t["drop_reason"] = "fetch_error"
            continue
        ts, vals, truth = L.split_context_truth(d["data"], ce, step, H)
        t["n_ctx"], t["n_truth"] = len(ts), len(truth)
        if not np.isfinite(vals).any():
            t["drop_reason"] = "no_context"
            continue
        if len(truth) < H:
            t["drop_reason"] = "truth_incomplete"
            continue
        fc = f.get("forecasts") or {}
        sn = (fc.get(L.SN_READABLE) or {}).get("data")
        # V2 = the context the platform served (current context with its last season restored from the Seasonal-Naive model's stored forecast);
        # its scale reproduces the stored MASE scale, so every row of the task is scored with it (V1 only when V2 cannot be built).
        v2vals, applied, dshare, dabs = L.v2_context(ts, vals, sn, step, P)
        vals_used = v2vals if applied else vals
        scale_v1 = L.context_scale_of(ts, vals, step, m)
        scale_v2 = L.context_scale_of(ts, v2vals, step, m) if applied else None
        scale = scale_v2 if applied else scale_v1
        if not (scale and math.isfinite(scale) and scale > 0):
            t["drop_reason"] = "scale_undefined"
            continue
        fin = np.flatnonzero(np.isfinite(vals_used))
        naive_value = float(vals_used[fin[-1]])
        naive_mae = float(np.mean([abs(v - naive_value) for v in truth.values()]))
        t.update(scale=scale, scale_v1=scale_v1 if scale_v1 else "", scale_v2=scale_v2 if scale_v2 else "", naive_value=naive_value, naive_mae=naive_mae,
                 sn_present=int(bool(sn)), served_diff_share=dshare, served_diff_rel=(dabs / scale) if math.isfinite(dabs) else "", v2_applied=int(applied))
        for key, entry in fc.items():
            ident = id_map.get(key)
            if ident is None or not entry or not entry.get("data"):
                continue
            mid = ident[0]
            names.setdefault(mid, ident[1])
            pts = entry["data"]
            sc = L.score_forecast(truth, [L.iso_to_epoch(p["ts"]) for p in pts], [p.get("y") for p in pts], [p.get("ci") for p in pts], scale, naive_value)
            if sc is not None:
                rows.append(L.score_row(rid, sid, mid, "field", sc, stored.get((sid, mid))))
        mk = lambda v: [{"ts": L.epoch_to_iso(int(a)), "value": (None if not math.isfinite(b) else float(b))} for a, b in zip(ts, v)]
        pending.append(dict(t=t, truth=truth, scale=scale, naive_value=naive_value, items1=mk(vals), items2=mk(v2vals) if applied else None, sid=sid))

    if pending:
        def run(items_lists):
            """Batch call; when it raises, retry series by series so one bad context costs one task, not the round."""
            try:
                return svc.predict(items_lists, H, freq)["series"]
            except Exception as e:  # noqa: BLE001
                log(f"round {rid}: batch service error {type(e).__name__}: {e}; retrying series by series")
                out = []
                for it in items_lists:
                    try:
                        out.append(svc.predict([it], H, freq)["series"][0])
                    except Exception:  # noqa: BLE001
                        out.append(None)
                return out

        t0 = time.perf_counter()
        res1 = run([p["items1"] for p in pending])
        ms = (time.perf_counter() - t0) * 1000.0 / len(pending)
        ap = [p for p in pending if p["items2"] is not None]
        res2 = {id(p): r for p, r in zip(ap, run([p["items2"] for p in ap]))} if ap else {}
        for p, r1 in zip(pending, res1):
            t = p["t"]
            r2 = res2.get(id(p))
            if r1 is None or (p["items2"] is not None and r2 is None):
                t["drop_reason"] = "run_error"
                continue
            sc1 = L.score_forecast(p["truth"], *L.ours_forecast_to_points(r1), p["scale"], p["naive_value"])
            sc2 = L.score_forecast(p["truth"], *L.ours_forecast_to_points(r2), p["scale"], p["naive_value"]) if r2 is not None else sc1
            if sc1 is None or sc1["n_points"] < H or sc2 is None or sc2["n_points"] < H:
                t["drop_reason"] = "ours_incomplete"
                continue
            t.update(eligible=1, path_v1=r1.get("path", ""), path_v2=(r2 if r2 is not None else r1).get("path", ""), ms_v1=ms)
            rows.append(L.score_row(rid, p["sid"], L.OURS_ID, "ours_v1", sc1, None))
            rows.append(L.score_row(rid, p["sid"], L.OURS_ID, "ours_v2", sc2, None))
    return tasks, rows


def write_tables(out: str, defn: int, tasks: List[dict], rows: List[list]) -> None:
    L.write_csv_gz(os.path.join(out, f"scores_def{defn}.csv.gz"), L.SCORE_COLS, rows)
    L.write_csv_gz(os.path.join(out, f"tasks_def{defn}.csv.gz"), L.TASK_COLS, ([t[c] for c in L.TASK_COLS] for t in tasks))


def smoke(api, svc, workers, id_map, names) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for defn, rid in SMOKE.items():
        t0 = time.time()
        tasks, rows = process_round(api, svc, defn, dict(id=rid, registration_start="2026-07-20T%s:00:00Z" % ("08" if defn == 2 else "09")), workers, id_map, names)
        S = {c: np.array([r[i] for r in rows], dtype=object) for i, c in enumerate(L.SCORE_COLS)}
        num = {}
        for c in L.SCORE_COLS:
            try:
                num[c] = np.array([float(x) if x != "" else np.nan for x in S[c]], dtype=float)
            except ValueError:
                num[c] = S[c]
        field = (num["variant"] == "field") & (num["stored_final"] == 1)
        a, b = num["mase_ctx"][field], num["stored_mase"][field]
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= L.REPRO_TOL * np.abs(b))
        ours = num["variant"] == "ours_v1"
        paths = [t["path_v1"] for t in tasks if t["eligible"] == 1]
        res[f"def{defn}"] = dict(round=rid, n_tasks=len(tasks), eligible=int(sum(t["eligible"] for t in tasks)), field_rows=int(field.sum()),
                                 field_match_share=float(ok.mean()) if field.any() else None, ours_rows=int(ours.sum()),
                                 ours_finite=bool(np.isfinite(num["mae"][ours]).all()) if ours.any() else False,
                                 ours_mean_mase=float(np.nanmean(num["mase_ctx"][ours])) if ours.any() else None,
                                 paths={p: paths.count(p) for p in set(paths)}, seconds=round(time.time() - t0, 1))
    res["pass"] = all(res[k]["eligible"] >= 8 and (res[k]["field_match_share"] or 0) >= 0.9 and res[k]["ours_finite"] for k in ("def2", "def5"))
    return res


class Enc(json.JSONEncoder):
    def default(self, o):  # numpy scalars and arrays
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("SHADOW_OUT", OUT))
    ap.add_argument("--cache", default=None, help="on-disk response cache (local tests only)")
    ap.add_argument("--mock", action="store_true", help="plumbing test with a stand-in service (local tests only)")
    ap.add_argument("--rounds2", default=None, help="override the frozen definition-2 round ids (tests)")
    ap.add_argument("--rounds5", default=None)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--no-ci", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _LOG = open(os.path.join(a.out, "log.txt"), "a")
    t_start = time.time()
    status: Dict[str, Any] = dict(started=L.utc_now(), python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(), mock=a.mock)
    try:
        log(f"TEST-L1 shadow kernel start; python {status['python']} numpy {status['numpy']}")
        build = os.path.join(ROOT, "BUILD.json")
        if os.path.exists(build):
            status["build"] = json.load(open(build))
            log(f"build: {status['build']}")
        seen = assert_manifest()
        status["v1_files_sha256"] = seen
        log("V1 manifest check passed (service files and int8 weights equal the frozen hashes)")
        svc = make_service(a.mock)
        try:
            import onnxruntime
            status["onnxruntime"] = onnxruntime.__version__
        except ImportError:
            pass
        api = L.Api(cache_dir=a.cache)
        models = api.get("models")
        id_map = {m["readable_id"]: (int(m["id"]), m["name"], m.get("track")) for m in models if m.get("readable_id")}
        names: Dict[int, str] = {int(m["id"]): m["name"] for m in models}
        names[L.OURS_ID] = "gridload-compact v1 (int8)"
        names[L.NAIVE_ID] = "naive (internal)"
        status["n_models_listed"] = len(models)
        if not a.no_smoke:
            sm = smoke(api, svc, a.workers, id_map, names)
            json.dump(clean(sm), open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
            log(f"smoke: {json.dumps(clean(sm))}")
            if not sm["pass"]:
                raise SystemExit("smoke on the out-of-window rounds FAILED: stopping before the window")
        frozen = json.load(open(os.path.join(ROOT, "rounds_frozen.json")))
        lists = {2: frozen["def2"], 5: frozen["def5"]}
        if a.rounds2:
            lists[2] = [dict(id=int(x), registration_start=None) for x in a.rounds2.split(",")]
        if a.rounds5:
            lists[5] = [dict(id=int(x), registration_start=None) for x in a.rounds5.split(",")]
        if a.rounds2 or a.rounds5:   # tests: take the registration start from the round listing
            for dfn in (2, 5):
                lst = api.get(f"definitions/{dfn}/rounds", {"status": "completed", "page": 1, "page_size": 100})["items"]
                by = {int(r["id"]): r for r in lst}
                lists[dfn] = [dict(id=r["id"], registration_start=by[r["id"]]["registration_start"][:19] + "Z") for r in lists[dfn] if r["id"] in by]
        for dfn in (2, 5):
            tasks, rows = [], []
            t0 = time.time()
            for i, rinfo in enumerate(lists[dfn]):
                try:
                    tk, rw = process_round(api, svc, dfn, rinfo, a.workers, id_map, names)
                except Exception as e:  # noqa: BLE001  (a round that cannot be processed is counted, not hidden)
                    log(f"def {dfn} round {rinfo['id']}: {type(e).__name__}: {e}")
                    tk, rw = [dict({c: "" for c in L.TASK_COLS}, round_id=rinfo["id"], series_id=-1, defn=dfn, eligible=0, drop_reason="fetch_error")], []
                tasks += tk
                rows += rw
                if i % 5 == 0 or i == len(lists[dfn]) - 1:
                    log(f"def {dfn}: round {i + 1}/{len(lists[dfn])} (id {rinfo['id']}) tasks {len(tasks)} eligible {sum(t['eligible'] for t in tasks)} rows {len(rows)} calls {api.calls} retries {api.retries} errors {len(api.errors)} {time.time() - t0:.0f}s")
            write_tables(a.out, dfn, tasks, rows)
            status[f"def{dfn}_tasks"], status[f"def{dfn}_rows"] = len(tasks), len(rows)
        status["api"] = dict(calls=api.calls, retries=api.retries, mbytes=round(api.nbytes / 1e6, 1), errors=api.errors[:50], n_errors=len(api.errors))
        json.dump(clean({str(k): v for k, v in names.items()}), open(os.path.join(a.out, "model_names.json"), "w"))
        res = {}
        for dfn in (2, 5):
            S = L.read_table(os.path.join(a.out, f"scores_def{dfn}.csv.gz"))
            T = L.read_table(os.path.join(a.out, f"tasks_def{dfn}.csv.gz"))
            t0 = time.time()
            res[dfn] = A.analyze_definition(dfn, S, T, names, ci=not a.no_ci)
            log(f"analysis def {dfn}: status {res[dfn].get('status')} r0 {res[dfn]['r0']['r0_pass']} {time.time() - t0:.0f}s")
        dec = A.decide(res[2], res[5])
        log(f"decision (kernel copy; the verdict is recomputed locally): {dec['verdict']}")
        json.dump(clean({"def2": res[2], "def5": res[5], "decision": dec}), open(os.path.join(a.out, "analysis.json"), "w"), indent=1, cls=Enc)
        status["decision"] = dec["verdict"]
        status["ok"] = True
    except SystemExit as e:
        status["ok"] = False
        status["error"] = str(e)
        log(f"STOP: {e}")
    except Exception as e:  # noqa: BLE001
        status["ok"] = False
        status["error"] = f"{type(e).__name__}: {e}"
        log("FAILED: " + traceback.format_exc())
    status["finished"] = L.utc_now()
    status["seconds"] = round(time.time() - t_start, 1)
    files = {}
    for fn in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, fn)
        if os.path.isfile(p) and fn != "manifest.json":
            files[fn] = dict(bytes=os.path.getsize(p), sha256=L.sha256_file(p))
    status["files"] = files
    json.dump(clean(status), open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
