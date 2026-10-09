#!/usr/bin/env python3
"""Kaggle kernel driver (timeseries track).  Runs inside a private CPU kernel after make_kernel.py extracted the repo to $TSB_ROOT.
plans: a1 (env, public data, caches, v0 repro gate, smoke, service tests)  |  a2 (sweep, selection, val/test-stage training, calibration)
       b (frozen TEST-F evaluation, export, service tests, latency).   Every stage is wrapped: a failure is recorded, later stages still run."""
import concurrent.futures as cf, glob, hashlib, json, os, shutil, subprocess, sys, tarfile, threading, time, traceback

T0 = time.time()
KAGGLE = os.path.exists("/kaggle")
REPO = os.environ.get("TSB_ROOT", "/tmp/repo")
OUT = os.environ.get("TSB_OUT", "/kaggle/working/out" if KAGGLE else f"{REPO}/out")
PREP_OUT = os.environ.get("TSB_PREP_OUT", "/kaggle/working/prepcache" if KAGGLE else f"{REPO}/prepcache")
CKPT_OUT = os.environ.get("TSB_CKPT_OUT", "/kaggle/working/ckpt" if KAGGLE else f"{REPO}/ckpt_out")
PY = sys.executable
sys.path.insert(0, f"{REPO}/src")
os.environ["TSB_ROOT"] = REPO
os.environ["TSB_TESTLOG_OUT"] = f"{OUT}/TEST-LOG-entries.md"
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
STATUS = {"plan": None, "stages": {}}
_lock = threading.Lock()
for d in (OUT, f"{OUT}/logs"):
    os.makedirs(d, exist_ok=True)

def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

def save_status():
    for _ in range(5):
        try:
            with _lock:
                snap = json.dumps(STATUS, indent=1, default=str)
            open(f"{OUT}/status_{STATUS['plan']}.json", "w").write(snap)
            return
        except RuntimeError:                 # another thread changed the dict while it was serialized: retry
            time.sleep(0.05)

def set_stage(name, d):
    with _lock:
        STATUS["stages"][name] = d

def run(cmd, name, env=None, cwd=None, quiet=False):
    """Run a command, stream its output to the kernel log and to out/logs/<name>.log; returns the return code."""
    t = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env or ENV, cwd=cwd or REPO)
    with open(f"{OUT}/logs/{name}.log", "w") as lf:
        for line in p.stdout:
            lf.write(line)
            if not quiet:
                print(f"  [{name}] {line.rstrip()}", flush=True)
    rc = p.wait()
    log(f"{name}: rc={rc} {time.time() - t:.0f}s")
    return rc

def run_chain(name, cmds, quiet=False):
    """Run commands one after the other (stop at the first failure); returns the last return code."""
    rc = 0
    for i, c in enumerate(cmds):
        rc = run(c, f"{name}" if len(cmds) == 1 else f"{name}_{i}", None, None, quiet)
        if rc != 0:
            break
    return rc

def run_chains(chains, workers=4):
    """chains: [(name, [cmd, ...])] -> {name: rc}; at most `workers` chains at once."""
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {n: ex.submit(run_chain, n, cmds) for n, cmds in chains}
        return {n: f.result() for n, f in futs.items()}

def collect_faith():
    """Per-round score files and VAL-S sidecars: /tmp is not saved by Kaggle, out/ is."""
    os.makedirs(f"{OUT}/faith", exist_ok=True)
    for f in glob.glob(f"{REPO}/runs/faith_*.parquet") + glob.glob(f"{REPO}/runs/*.val.json"):
        dst = f"{OUT}/faith/{os.path.basename(f)}"
        if not os.path.exists(dst) or os.path.getmtime(f) > os.path.getmtime(dst):
            shutil.copy(f, dst)

def export_ckpts():
    """Every checkpoint trained so far (+ sidecars) to ckpt/ with a manifest; idempotent, independent of every other stage."""
    os.makedirs(CKPT_OUT, exist_ok=True); man = {}
    for f in sorted(glob.glob(f"{REPO}/runs/*.pt")) + sorted(glob.glob(f"{REPO}/runs/*.pt.val.json")):
        shutil.copy(f, CKPT_OUT); man[os.path.basename(f)] = sha256(f)
    json.dump(man, open(f"{CKPT_OUT}/ckpt_manifest.json", "w"), indent=1)
    return man

def submit_chains(chains, workers=4):
    ex = cf.ThreadPoolExecutor(max_workers=workers)
    return ex, {n: ex.submit(run_chain, n, cmds) for n, cmds in chains}

def run_jobs(jobs, workers=4):
    """jobs: [(name, cmd)] -> {name: rc}; at most `workers` at once; training logs are kept quiet in the kernel stream (see out/logs)."""
    res = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {n: ex.submit(run, c, n, None, None, True) for n, c in jobs}
        for n, f in futs.items():
            res[n] = f.result()
    return res

def guarded(name, fn, *a, **k):
    """Run fn(*a, **k) as a recorded stage: failures are logged to out/errors.txt and do not stop later stages."""
    t = time.time(); log(f"=== stage {name} start")
    set_stage(name, {"ok": None})
    try:
        r = fn(*a, **k)
        set_stage(name, {"ok": True, "secs": round(time.time() - t, 1), "result": r if isinstance(r, (dict, list, str, int, float, bool)) else None})
        log(f"=== stage {name} ok {time.time() - t:.0f}s")
        return r
    except BaseException as e:
        tb = traceback.format_exc()
        set_stage(name, {"ok": False, "secs": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"})
        with open(f"{OUT}/errors.txt", "a") as f:
            f.write(f"--- {name}\n{tb}\n")
        log(f"=== stage {name} FAILED: {type(e).__name__}: {e}\n{tb}")
        return None
    finally:
        save_status()

def stage(name):
    def deco(fn):
        return lambda *a, **k: guarded(name, fn, *a, **k)
    return deco

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def jload(p):
    return json.load(open(p))

def tsb(mod, *args, name=None, quiet=False, env=None):
    return run([PY, "-m", f"tsb.{mod}", *map(str, args)], name or mod, env=env, cwd=f"{REPO}/src", quiet=quiet)

# ----------------------------------------------------------------------------------------------- environment
@stage("env")
def stage_env():
    pins = ["numpy==2.5.3", "pandas==3.0.6", "pyarrow==25.0.1", "holidays==0.106", "onnxruntime==1.30.0"]
    rc = run([PY, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pins], "pip_pins")
    probe = "import sys,numpy,pandas,pyarrow,torch,onnx,onnxruntime,holidays;print(sys.version.split()[0],numpy.__version__,pandas.__version__,pyarrow.__version__,torch.__version__,onnx.__version__,onnxruntime.__version__,holidays.__version__)"
    r = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
    info = dict(pip_rc=rc, probe_rc=r.returncode, versions=r.stdout.strip(), err=r.stderr[-400:])
    if r.returncode != 0:                       # keep going on the stock stack, but say so
        run([PY, "-m", "pip", "install", "-q", "--disable-pip-version-check", "holidays", "onnxruntime"], "pip_fallback")
        r = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
        info.update(fallback=True, versions=r.stdout.strip(), err=r.stderr[-400:])
    cpu = subprocess.run("lscpu | egrep 'Model name|^CPU\\(s\\)|Thread|Flags' | cut -c1-300; nproc; free -g | head -2", shell=True, capture_output=True, text=True).stdout
    info["cpu"] = cpu
    log("env:", info["versions"], "| fallback" if info.get("fallback") else "")
    return info

# ----------------------------------------------------------------------------------------------- data
def find_cache():
    for p in glob.glob("/kaggle/input/**/cache_manifest.json", recursive=True):
        if os.path.exists(os.path.join(os.path.dirname(p), "archive_ctx_truth.tar")):
            return os.path.dirname(p)
    return None

def restore_cache(src, keep_field=False):
    for sub in ("smard", "cache"):
        shutil.copytree(f"{src}/{sub}", f"{REPO}/data/{sub}", dirs_exist_ok=True)
    os.makedirs(f"{REPO}/data/archive", exist_ok=True)
    with tarfile.open(f"{src}/archive_ctx_truth.tar") as t:
        t.extractall(f"{REPO}/data/archive")
    os.makedirs(f"{REPO}/data/raw", exist_ok=True)
    shutil.copy(f"{REPO}/assets/models.json", f"{REPO}/data/raw/models.json")
    if not keep_field:
        for f in glob.glob(f"{REPO}/data/cache/field_def*.parquet") + glob.glob(f"{REPO}/data/cache/rounds_def*.npz"):
            os.remove(f)                  # b restores the field cache from the a2 export
    log("restored data caches from", src, "(field cache kept)" if keep_field else "")

SCHEDULES = "smard_load_challenge_24h_15min,smard_load_challenge_72h_1h"

@stage("fetch")
def stage_fetch():
    """SMARD API (CC BY 4.0) and the TS-Arena-Archive context/ground-truth partitions at the pinned revision (CC BY 4.0), both fetched here.
    (The forecast partitions, 5760 files, are only needed for the field cache: see stage_fetch_forecasts.)"""
    for d in ("raw", "smard", "archive", "cache"):
        os.makedirs(f"{REPO}/data/{d}", exist_ok=True)
    shutil.copy(f"{REPO}/assets/models.json", f"{REPO}/data/raw/models.json")
    shutil.copy(f"{REPO}/assets/clock_ref.npz", f"{REPO}/data/cache/clock_ref.npz")
    env_s = dict(ENV, TSB_REGION_WORKERS="3")
    with cf.ThreadPoolExecutor(2) as ex:
        fs = ex.submit(run, [PY, "-m", "tsb.fetch_smard", "2018-01-01"], "fetch_smard", env_s, f"{REPO}/src")
        fa = ex.submit(run, [PY, "-m", "tsb.fetch_archive", SCHEDULES, "--only", "context_data,ground_truth"], "fetch_archive_ctx", ENV, f"{REPO}/src")
        rs, ra = fs.result(), fa.result()
    n_s = len(glob.glob(f"{REPO}/data/smard/*.parquet"))
    n_c = sum(len(f) for _, _, f in os.walk(f"{REPO}/data/archive/context_data")); n_t = sum(len(f) for _, _, f in os.walk(f"{REPO}/data/archive/ground_truth"))
    if rs != 0 or ra != 0 or n_s != 10 or n_c != 360 or n_t != 205:
        raise RuntimeError(f"fetch incomplete: smard rc={rs} ({n_s}/10 files), archive rc={ra} (context {n_c}/360, truth {n_t}/205)")
    return dict(smard_files=n_s, context_files=n_c, truth_files=n_t)

@stage("fetch_forecasts")
def stage_fetch_forecasts():
    """Forecast partitions of the 32-model field (5760 files): paced to the Hub's anonymous rate limit, about 11 minutes."""
    rc = run([PY, "-m", "tsb.fetch_archive", SCHEDULES, "--only", "forecasts"], "fetch_archive_fc", ENV, f"{REPO}/src", True)
    n = sum(len(f) for _, _, f in os.walk(f"{REPO}/data/archive/forecasts"))
    if rc != 0 or n != 5760:
        raise RuntimeError(f"forecast fetch incomplete: rc={rc}, {n}/5760 files")
    return dict(forecast_files=n)

@stage("tables")
def stage_tables():
    if tsb("make_hol", name="make_hol") != 0 or tsb("bank", name="bank") != 0:
        raise RuntimeError("make_hol/bank failed")
    import numpy as np
    z = np.load(f"{REPO}/data/cache/hol_table.npz")
    h = hashlib.sha256(np.ascontiguousarray(z["table"]).tobytes()).hexdigest()
    exp = jload(f"{REPO}/assets/ref/expected.json") if os.path.exists(f"{REPO}/assets/ref/expected.json") else {}
    shutil.copy(f"{REPO}/data/cache/hol_table.npz", f"{OUT}/hol_table_kernel.npz")
    r = dict(hol_sha256=h, hol_day0=int(z["day0"]), hol_matches_mac=(h == exp.get("hol_table_sha256")) if exp else None)
    if exp and not r["hol_matches_mac"]:
        raise RuntimeError(f"hol_table differs from the Mac table: {h} vs {exp.get('hol_table_sha256')}")
    return r

FROOT = "/tmp/froot"

@stage("field_cache")
def stage_field_cache():
    """Official-style scores of the whole archive field (needs the forecast partitions); the two definitions in parallel.
    Built under its own root (data symlinked) so it never overwrites the cache files other stages are reading."""
    shutil.rmtree(FROOT, ignore_errors=True)
    os.makedirs(f"{FROOT}/data/cache", exist_ok=True)
    for sub in ("smard", "archive"):
        os.symlink(f"{REPO}/data/{sub}", f"{FROOT}/data/{sub}")
    env = dict(ENV, TSB_ROOT=FROOT)
    jobs = [("build_cache_2", [PY, "-m", "tsb.build_cache", "2"]), ("build_cache_5", [PY, "-m", "tsb.build_cache", "5"])]
    with cf.ThreadPoolExecutor(2) as ex:
        fs = [ex.submit(run, c, n, env, f"{REPO}/src", True) for n, c in jobs]
        rcs = [f.result() for f in fs]
    import pandas as pd
    f2 = pd.read_parquet(f"{FROOT}/data/cache/field_def2.parquet"); f5 = pd.read_parquet(f"{FROOT}/data/cache/field_def5.parquet")
    r = dict(rcs=rcs, rows_def2=len(f2), rows_def5=len(f5), counts_match=(len(f2) == 28448 and len(f5) == 28736))
    if rcs != [0, 0] or not r["counts_match"]:
        raise RuntimeError(f"field cache: {r}")
    return r

def cmp_faith(new_path, ref_path):
    import numpy as np, pandas as pd
    a = pd.read_parquet(new_path); b = pd.read_parquet(ref_path)
    m = a.merge(b, on=["day", "round_id", "series"], how="outer", suffixes=("_k", "_m"), indicator=True)
    res = dict(rows_kernel=len(a), rows_mac=len(b), only_one_side=int((m["_merge"] != "both").sum()))
    for c in ["mase", "sql", "cover80", "n"]:
        d = (m[c + "_k"] - m[c + "_m"]).abs().to_numpy(dtype=float)
        res[f"max_abs_diff_{c}"] = float(np.nanmax(d)) if np.isfinite(d).any() else None
        res[f"nan_mismatch_{c}"] = int((m[c + "_k"].isna() != m[c + "_m"].isna()).sum())
    res["passed"] = bool(res["only_one_side"] == 0 and all((res[f"max_abs_diff_{c}"] or 0) <= 1e-4 and res[f"nan_mismatch_{c}"] == 0 for c in ["mase", "sql", "cover80", "n"]))
    return res

@stage("repro_eval")
def stage_repro_eval():
    """Re-evaluate the Mac v0 dev checkpoint on VAL-F def 2 in this kernel; per-(round, series) agreement is the gate."""
    ref = f"{REPO}/assets/ref"
    if not os.path.exists(f"{ref}/v0_15_s0.pt"):
        return dict(skipped="no reference checkpoint in this build")
    rc = tsb("evalf", "--def", 2, "--ckpt", f"{ref}/v0_15_s0.pt", "--days", "2026-01-01", "2026-02-07", "--tag", "v0_15_s0", "--out-json", f"{OUT}/repro_v0_eval.json", name="repro_eval")
    if rc != 0:
        raise RuntimeError("evalf failed")
    new = f"{REPO}/runs/faith_val_def2_v0_15_s0_2026-01-01_2026-02-07.parquet"
    shutil.copy(new, f"{OUT}/faith_val_def2_v0_15_s0_kernel.parquet")
    r = cmp_faith(new, f"{ref}/faith_val_def2_v0_15_s0_2026-01-01_2026-02-07.parquet")
    r["summary_kernel"] = jload(f"{OUT}/repro_v0_eval.json")["summary"]
    log("repro per-pair comparison:", json.dumps(r))
    return r

@stage("repro_table")
def stage_repro_table():
    """Rebuild the dev report (our v0 inside the real 32-model field) and diff it against the committed Mac report."""
    ref = f"{REPO}/assets/ref"
    if not os.path.exists(f"{ref}/dev_val_f_def2.md"):
        return dict(skipped=True)
    rc = tsb("make_evidence", "--def", 2, "--days", "2026-01-01", "2026-02-07", "--label", "VAL-F (dev check)",
             "--ours", f"gridload-v0={REPO}/runs/faith_val_def2_v0_15_s0_2026-01-01_2026-02-07.parquet", "--params", "gridload-v0=0.150",
             "--out", "reports/dev_val_f_def2.md", name="repro_table", quiet=True)
    if rc != 0:
        raise RuntimeError("make_evidence failed")
    shutil.copy(f"{REPO}/reports/dev_val_f_def2.md", f"{OUT}/dev_val_f_def2_kernel.md")
    a = open(f"{REPO}/reports/dev_val_f_def2.md").read().splitlines(); b = open(f"{ref}/dev_val_f_def2.md").read().splitlines()
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    r = dict(lines_kernel=len(a), lines_mac=len(b), differing_lines=len(diff) + abs(len(a) - len(b)), identical=(a == b), first_diffs=[list(d) for d in diff[:6]])
    log("repro report identical:", r["identical"], "diff lines:", r["differing_lines"])
    return r

# ----------------------------------------------------------------------------------------------- smoke
@stage("smoke")
def stage_smoke():
    """E2-shaped tiny nets through torch -> export -> onnxruntime fp32/int8 -> service code, with routing assertions."""
    days = ["2026-01-01", "2026-01-03"]; W = f"{REPO}/smokeW"; os.makedirs(W, exist_ok=True); os.makedirs(f"{REPO}/runs", exist_ok=True)
    shapes = {15: dict(defn=2, dm=10), 60: dict(defn=5, dm=14)}
    jobs = [(f"smoke_train{f}", [PY, "-m", "tsb.train", "--freq", f, "--steps", 30, "--hid", 32, "--emb", 8, "--J", 96, "--dmeans", s["dm"], "--eval-every", 1000000,
                                 "--threads", 1, "--out", f"{REPO}/runs/smoke_{f}.pt"]) for f, s in shapes.items()]
    jobs = [(n, [str(x) for x in c]) for n, c in jobs]
    rcs = run_jobs(jobs, 2)
    res = dict(train_rcs=rcs, checks={})
    for f, s in shapes.items():
        d = s["defn"]; ck = f"{REPO}/runs/smoke_{f}.pt"
        tsb("evalf", "--def", d, "--ckpt", ck, "--spread", 1.1, "--days", *days, "--tag", f"smoke_torch{f}", "--out-json", f"{OUT}/smoke_torch{f}.json", name=f"smoke_torch{f}", quiet=True)
        tsb("export", "--ckpt", ck, "--out", f"{W}/net{f}", "--spread", 1.1, name=f"smoke_export{f}", quiet=True)
        for prec, suf in (("fp32", ""), ("int8", "_int8")):
            tsb("evalf", "--def", d, "--onnx", f"{W}/net{f}{suf}.onnx", "--freq", f, "--days", *days, "--tag", f"smoke_ort{f}{suf}", "--out-json", f"{OUT}/smoke_ort{f}{suf}.json", name=f"smoke_ort{f}{suf}", quiet=True)
    for n in ("hol_table.npz", "clock_ref.npz"):
        shutil.copy(f"{REPO}/data/cache/{n}", f"{W}/{n}")
    for f, s in shapes.items():
        tsb("evalf", "--def", s["defn"], "--service", W, "--precision", "int8", "--days", *days, "--tag", f"smoke_service{f}", "--out-json", f"{OUT}/smoke_service{f}.json", name=f"smoke_service{f}", quiet=True)
    ok = True
    for f, s in shapes.items():
        try:
            t = jload(f"{OUT}/smoke_torch{f}.json")["summary"]; o = jload(f"{OUT}/smoke_ort{f}.json")["summary"]
            q = jload(f"{OUT}/smoke_ort{f}_int8.json")["summary"]; v = jload(f"{OUT}/smoke_service{f}.json")["summary"]
            pairs = 10 * 3
            c = dict(torch_mase=t["mase"], ort_fp32_mase=o["mase"], ort_int8_mase=q["mase"], service_int8_mase=v["mase"], routing=v.get("routing"),
                     torch_vs_ort_fp32=abs(t["mase"] - o["mase"]), int8_vs_service=abs(q["mase"] - v["mase"]),
                     specialist_all=(v.get("routing", {}).get("specialist", 0) == pairs and v.get("routing", {}).get("fallback", 0) == 0), n_missing=[t["n_missing"], o["n_missing"], q["n_missing"], v["n_missing"]])
            c["ok"] = bool(c["torch_vs_ort_fp32"] < 1e-4 and c["int8_vs_service"] < 1e-4 and c["specialist_all"] and max(c["n_missing"]) == 0)
        except Exception as e:
            c = dict(ok=False, error=f"{type(e).__name__}: {e}")
        res["checks"][str(f)] = c; ok &= c["ok"]
    res["ok"] = ok
    log("smoke:", json.dumps(res))
    if not ok:
        raise RuntimeError("smoke checks failed: " + json.dumps(res["checks"]))
    return res

# ----------------------------------------------------------------------------------------------- service tests + harness
SVC_TAM = "/tmp/tam"; SVC_VENV = "/tmp/svcenv"

def svc_prepare():
    """Clone ts-arena-models at the pinned commit and build the service venv with the Docker image's pinned requirements (venv without ensurepip,
    packages installed by the system pip: the Kaggle image has no working ensurepip)."""
    if os.path.exists(f"{SVC_VENV}/.ready"):
        return "already prepared"
    shutil.rmtree(SVC_TAM, ignore_errors=True); shutil.rmtree(SVC_VENV, ignore_errors=True)
    if run(["git", "clone", "-q", "https://github.com/DAG-UPB/ts-arena-models", SVC_TAM], "svc_clone") != 0 or run(["git", "-C", SVC_TAM, "checkout", "-q", "cbbb347"], "svc_checkout") != 0:
        raise RuntimeError("cannot clone ts-arena-models at cbbb347")
    vpy = f"{SVC_VENV}/bin/python"
    if run([PY, "-m", "venv", "--without-pip", SVC_VENV], "svc_venv") != 0:
        raise RuntimeError("venv creation failed")
    req = f"{REPO}/submission/model-services/gridload-compact/requirements.txt"
    if run([PY, "-m", "pip", "--python", vpy, "install", "-q", "--disable-pip-version-check", "-r", req, "pytest", "httpx"], "svc_pip") != 0:
        raise RuntimeError("pip install into the service venv failed")
    freeze = subprocess.run([PY, "-m", "pip", "--python", vpy, "freeze"], capture_output=True, text=True).stdout
    open(f"{OUT}/svc_pip_freeze.txt", "w").write(freeze)
    open(f"{SVC_VENV}/.ready", "w").write("ok")
    return "prepared"

def svc_tests(weights_dir, tag):
    """Service tests, repo contract tests and the live harness against the service code with the given weights."""
    svc_prepare()
    tam, venv = SVC_TAM, SVC_VENV; dest = f"{tam}/model-services/gridload-compact"; vpy = f"{venv}/bin/python"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(f"{REPO}/submission/model-services/gridload-compact", dest, ignore=shutil.ignore_patterns("weights", "__pycache__", ".pytest_cache"))
    shutil.copytree(weights_dir, f"{dest}/weights")
    res = {}
    res["service_tests_rc"] = run([vpy, "-m", "pytest", "-q", f"{dest}/tests"], f"svc_pytest_{tag}", cwd=dest, quiet=True)
    res["repo_contract_tests_rc"] = run([vpy, "-m", "pytest", "-q", f"{tam}/model-services/tests"], f"svc_contract_{tag}", cwd=tam, quiet=True)
    env = dict(os.environ, GRIDLOAD_PRECISION="int8")
    srv = subprocess.Popen([vpy, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"], cwd=dest, env=env, stdout=open(f"{OUT}/logs/svc_uvicorn_{tag}.log", "w"), stderr=subprocess.STDOUT)
    try:
        import urllib.request
        up = False
        for _ in range(60):
            try:
                up = urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=3).status == 200; break
            except Exception:
                time.sleep(1)
        res["server_up"] = up
        for mode, freq, hz in (("single", "15min", 96), ("batch", "15min", 96), ("single", "h", 24), ("batch", "h", 24), ("single", "h", 72), ("batch", "h", 72)):
            res[f"harness_{mode}_{freq}_{hz}"] = run([vpy, f"{tam}/model-services/tests/run_harness.py", "--url", "http://127.0.0.1:8765/predict", "--horizon", str(hz), "--freq", freq, "--mode", mode],
                                                      f"svc_harness_{tag}_{mode}_{freq}_{hz}", cwd=tam, quiet=True)
    finally:
        srv.terminate()
    res["ok"] = all(v == 0 for k, v in res.items() if k.endswith("_rc") or k.startswith("harness_")) and res.get("server_up", False)
    return res

# ----------------------------------------------------------------------------------------------- cache export
@stage("export_cache")
def stage_export_cache():
    import numpy as np, pandas as pd
    os.makedirs(PREP_OUT, exist_ok=True)
    for sub in ("smard", "cache"):
        shutil.copytree(f"{REPO}/data/{sub}", f"{PREP_OUT}/{sub}", dirs_exist_ok=True)
    with tarfile.open(f"{PREP_OUT}/archive_ctx_truth.tar", "w") as t:
        for top in ("context_data", "ground_truth"):
            t.add(f"{REPO}/data/archive/{top}", arcname=top)
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), hf_revision=os.environ.get("TSB_HF_REV", "72c6cb0fe11a7c0349b063c74129ff73a87542cc"),
               files={os.path.relpath(p, PREP_OUT): sha256(p) for p in sorted(glob.glob(f"{PREP_OUT}/cache/*")) + sorted(glob.glob(f"{PREP_OUT}/smard/*"))},
               smard_last_ts={os.path.basename(p)[:-8]: str(pd.to_datetime(pd.read_parquet(p).ts_ms.max(), unit="ms")) for p in sorted(glob.glob(f"{PREP_OUT}/smard/*.parquet"))})
    json.dump(man, open(f"{PREP_OUT}/cache_manifest.json", "w"), indent=1)
    shutil.copy(f"{PREP_OUT}/cache_manifest.json", f"{OUT}/cache_manifest_copy.json")
    return dict(files=len(man["files"]))

def finish():
    STATUS["total_secs"] = round(time.time() - T0, 1)
    try:
        collect_faith()
    except Exception as e:
        log("collect_faith failed", e)
    save_status()
    try:
        shutil.copy(f"{REPO}/TEST-LOG.md", f"{OUT}/TEST-LOG.after-run.md")
    except Exception:
        pass
    log("done:", json.dumps({k: v.get("ok") for k, v in STATUS["stages"].items()}))

# ----------------------------------------------------------------------------------------------- common helpers for a2 / b
PRE = None
def prereg():
    global PRE
    if PRE is None:
        PRE = jload(f"{REPO}/configs/prereg.json")
    return PRE

DEF_OF = {"15": 2, "60": 5}
VALF = ("2026-01-01", "2026-02-07"); TESTF = ("2026-02-08", "2026-03-31")

def train_cmd(freq, hid, emb, J, dmeans, seed, train_end, out, eval_every=1000000, nohol=False, steps=None):
    c = [PY, "-m", "tsb.train", "--freq", freq, "--steps", steps or prereg()["steps"], "--hid", hid, "--emb", emb, "--J", J, "--dmeans", dmeans, "--seed", seed,
         "--train-end", train_end, "--eval-every", eval_every, "--threads", 1, "--out", out]
    return [str(x) for x in c] + (["--no-hol"] if nohol else [])

def evalf_cmd(defn, days, tag, out_json, ckpts=None, onnx=None, freq=None, service=None, precision=None, spread=None, final=False, note=None):
    c = [PY, "-m", "tsb.evalf", "--def", defn, "--days", *days, "--tag", tag, "--out-json", out_json]
    if ckpts: c += ["--ckpt", *ckpts]
    if onnx: c += ["--onnx", onnx, "--freq", freq]
    if service: c += ["--service", service, "--precision", precision]
    if spread is not None: c += ["--spread", spread]
    if final: c += ["--final", "--note", note or tag]
    return [str(x) for x in c]

FIELD_OUT = os.environ.get("TSB_FIELD_OUT", "/kaggle/working/fieldcache" if KAGGLE else f"{REPO}/fieldcache")

def restore_base(keep_field=False):
    """SMARD + banks + hol table + context/truth from the a1 kernel output; falls back to fetching everything (clean-checkout path)."""
    cache = find_cache()
    if cache:
        restore_cache(cache, keep_field)
        n_c = sum(len(f) for _, _, f in os.walk(f"{REPO}/data/archive/context_data")); n_t = sum(len(f) for _, _, f in os.walk(f"{REPO}/data/archive/ground_truth"))
        if (n_c, n_t) != (360, 205):
            log("context/truth incomplete in the cache, repairing:", n_c, n_t)
            run([PY, "-m", "tsb.fetch_archive", SCHEDULES, "--only", "context_data,ground_truth"], "fetch_archive_repair", ENV, f"{REPO}/src")
        return "restored"
    stage_fetch(); stage_tables()
    return "fetched"

@stage("export_field")
def stage_export_field():
    """Export the rebuilt (row-count verified) field cache for b."""
    import pandas as pd
    os.makedirs(FIELD_OUT, exist_ok=True); man = {}
    for f in ("field_def2.parquet", "field_def5.parquet", "rounds_def2.npz", "rounds_def5.npz"):
        shutil.copy(f"{FROOT}/data/cache/{f}", FIELD_OUT); man[f] = sha256(f"{FIELD_OUT}/{f}")
    man["rows_def2"] = len(pd.read_parquet(f"{FIELD_OUT}/field_def2.parquet")); man["rows_def5"] = len(pd.read_parquet(f"{FIELD_OUT}/field_def5.parquet"))
    json.dump(man, open(f"{FIELD_OUT}/field_manifest.json", "w"), indent=1)
    shutil.copy(f"{FIELD_OUT}/field_manifest.json", f"{OUT}/field_manifest_copy.json")
    return man

FIELD_FILES = ("field_def2.parquet", "field_def5.parquet", "rounds_def2.npz", "rounds_def5.npz")

@stage("field_check")
def stage_field_check():
    """The rebuilt field cache (FROOT) against the provisional one restored from a1 (used by the selection): equal on every row both have."""
    import numpy as np, pandas as pd
    res = {}; passed = True
    for d in (2, 5):
        a = pd.read_parquet(f"{FROOT}/data/cache/field_def{d}.parquet"); b = pd.read_parquet(f"{REPO}/data/cache/field_def{d}.parquet")
        m = a.merge(b, on=["day", "round_id", "model", "series"], how="outer", suffixes=("_k", "_p"), indicator=True)
        both = m[m["_merge"] == "both"]
        r = dict(rows_rebuilt=len(a), rows_provisional=len(b), only_in_rebuilt=int((m["_merge"] == "left_only").sum()), only_in_provisional=int((m["_merge"] == "right_only").sum()),
                 first_day_only_in_rebuilt=(str(m[m["_merge"] == "left_only"].day.min()) if (m["_merge"] == "left_only").any() else None))
        ok = len(both) > 0 and r["only_in_provisional"] == 0
        for c in ("mase", "sql", "mae", "mae_naive", "n"):
            diff = (both[c + "_k"] - both[c + "_p"]).abs().to_numpy(dtype=float)
            r[f"max_diff_{c}"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
            r[f"nan_mismatch_{c}"] = int((both[c + "_k"].isna() != both[c + "_p"].isna()).sum())
            ok &= bool(r[f"max_diff_{c}"] <= 1e-9 and r[f"nan_mismatch_{c}"] == 0)
        r["passed"] = bool(ok); passed &= ok; res[str(d)] = r
    res["passed"] = bool(passed)
    return res

def field_pipeline():
    """Forecast partitions -> field scores (own root, complete) -> comparison with the provisional cache -> exported field cache for b.  Runs beside the sweep."""
    stage_fetch_forecasts()
    built = False
    if STATUS["stages"].get("fetch_forecasts", {}).get("ok"):
        stage_field_cache()
        built = bool(STATUS["stages"].get("field_cache", {}).get("ok"))
    if built:
        chk = stage_field_check()
        STATUS["field_rebuild"] = "complete, equals the provisional cache on shared rows" if (chk and chk.get("passed")) else "complete, DIFFERS from the provisional cache: review"
        stage_export_field()
    else:
        STATUS["field_rebuild"] = "failed: b needs the field cache for its evidence tables"
    save_status()

def restore_field():
    for p in glob.glob("/kaggle/input/**/field_manifest.json", recursive=True):
        d = os.path.dirname(p)
        for f in FIELD_FILES:
            shutil.copy(f"{d}/{f}", f"{REPO}/data/cache/{f}")
        STATUS["field_source"] = f"a2 export ({d})"
        return d
    raise RuntimeError("no a2 field export mounted: the evidence tables need the complete field cache")

def restore_ckpts():
    """Checkpoints produced by the a2 kernel (mounted as a kernel source) -> runs/."""
    srcs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/ckpt_manifest.json", recursive=True)]
    if not srcs:
        raise RuntimeError("no a2 checkpoint source mounted")
    os.makedirs(f"{REPO}/runs", exist_ok=True)
    for f in glob.glob(f"{srcs[0]}/*"):
        shutil.copy(f, f"{REPO}/runs/")
    return srcs[0]

def val_days():
    import pandas as pd
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(*VALF)]

# ----------------------------------------------------------------------------------------------- a2: sweep -> selection -> stage-2 training -> calibration
@stage("sweep")
def stage_sweep():
    pre = prereg(); os.makedirs(f"{OUT}/sweep", exist_ok=True); os.makedirs(f"{REPO}/runs", exist_ok=True)
    chains = []
    for hid in (256, 128, 64):                       # longest first
        for fk in ("15", "60"):
            c = pre["candidates"][fk]; tag = f"sweep_{fk}_h{hid}"; ck = f"{REPO}/runs/{tag}.pt"
            chains.append((tag, [train_cmd(fk, hid, c["emb"][str(hid)], c["J"], c["dmeans"], 0, pre["cutoffs"]["sweep"], ck, eval_every=pre["steps"]),
                                 evalf_cmd(DEF_OF[fk], VALF, tag, f"{OUT}/sweep/{tag}_valf.json", ckpts=[ck])]))
    rcs = run_chains(chains, 4)
    export_ckpts(); collect_faith()
    return rcs

@stage("selection")
def stage_selection():
    if not STATUS.get("field_source"):
        raise RuntimeError("no field cache in place: the VAL-F gate needs the reference scores")
    from tsb import report, select_final
    pre = prereg(); days = val_days(); res = {}
    for fk in ("15", "60"):
        c = pre["candidates"][fk]; cands = []
        for hid in c["hids"]:
            tag = f"sweep_{fk}_h{hid}"
            v = jload(f"{REPO}/runs/{tag}.pt.val.json"); f = jload(f"{OUT}/sweep/{tag}_valf.json")["summary"]
            cands.append(dict(tag=tag, hid=hid, emb=c["emb"][str(hid)], params=v["params"], J=c["J"], dmeans=c["dmeans"], val_s=v["val_s"],
                              val_f_mase=f["mase"], val_f_sql=f["sql"], val_f_cover80=f["cover80"], train_seconds=v["train_seconds"]))
        M = report.field_by_round(DEF_OF[fk], days, "mase"); means = M.mean()
        res[fk] = dict(best_ref=dict(model=str(means.idxmin()), val_f_mase=float(means.min())), cands=cands)
    json.dump(res, open(f"{OUT}/sweep_results.json", "w"), indent=1)
    return select_final.main(f"{OUT}/sweep_results.json", f"{OUT}/selection.json")

def calibrate_impl():
    pre = prereg(); chains = []
    for fk in ("15", "60"):
        cks = [f"{REPO}/runs/val_{fk}_s{s}.pt" for s in pre["ensemble_seeds"]]
        chains.append((f"calibrate_{fk}", [[PY, "-m", "tsb.calibrate", "--def", str(DEF_OF[fk]), "--ckpt", *cks, "--days", *VALF, "--out", f"{OUT}/spread_def{DEF_OF[fk]}.json"]]))
    rcs = run_chains(chains, 2)
    spread = {fk: jload(f"{OUT}/spread_def{DEF_OF[fk]}.json")["chosen"] for fk in ("15", "60")}
    json.dump(spread, open(f"{OUT}/spread_chosen.json", "w"), indent=1)
    return dict(rcs=rcs, chosen={k: v["scale"] for k, v in spread.items()})

@stage("stage2")
def stage_stage2():
    """val-stage and test-stage 3-seed ensembles (+ no-holiday ablation, + the informational v0 retrains).  The calibration starts as soon as the
    val-stage members exist, beside the remaining training."""
    pre = prereg(); sel = jload(f"{OUT}/selection.json"); chains = []
    for seed in pre["ensemble_seeds"]:               # val-stage first (calibration needs them)
        for fk in ("15", "60"):
            s = sel[fk]["selected"]
            chains.append((f"val_{fk}_s{seed}", [train_cmd(fk, s["hid"], s["emb"], s["J"], s["dmeans"], seed, pre["cutoffs"]["val"], f"{REPO}/runs/val_{fk}_s{seed}.pt")]))
    for fk in ("15", "60"):                           # informational (not candidates): the Mac dev recipe retrained here, same cutoff as the sweep
        tag = f"info_v0_{fk}"; ck = f"{REPO}/runs/{tag}.pt"
        chains.append((tag, [train_cmd(fk, 256, 64, 48, 0, 0, pre["cutoffs"]["sweep"], ck, eval_every=pre["steps"]),
                             evalf_cmd(DEF_OF[fk], VALF, tag, f"{OUT}/sweep/{tag}_valf.json", ckpts=[ck])]))
    for seed in pre["ensemble_seeds"]:
        for fk in ("15", "60"):
            s = sel[fk]["selected"]
            chains.append((f"test_{fk}_s{seed}", [train_cmd(fk, s["hid"], s["emb"], s["J"], s["dmeans"], seed, pre["cutoffs"]["test"], f"{REPO}/runs/test_{fk}_s{seed}.pt")]))
    for fk in ("15", "60"):
        s = sel[fk]["selected"]; seed = pre["ablation"]["no_holiday_features"]["seed"]
        chains.append((f"test_{fk}_nohol_s{seed}", [train_cmd(fk, s["hid"], s["emb"], s["J"], s["dmeans"], seed, pre["cutoffs"]["test"], f"{REPO}/runs/test_{fk}_nohol_s{seed}.pt", nohol=True)]))
    ex, futs = submit_chains(chains, 4)
    vals = {n: futs[n].result() for n, _ in chains if n.startswith("val_")}
    export_ckpts()
    guarded("calibrate", calibrate_impl)
    rcs = {n: f.result() for n, f in futs.items()}
    ex.shutdown()
    export_ckpts(); collect_faith()
    info = {}
    for fk in ("15", "60"):
        tag = f"info_v0_{fk}"
        try:
            v = jload(f"{REPO}/runs/{tag}.pt.val.json"); f = jload(f"{OUT}/sweep/{tag}_valf.json")["summary"]
            info[fk] = dict(params=v["params"], val_s=v["val_s"], val_f_mase=f["mase"], val_f_sql=f["sql"], train_seconds=v["train_seconds"])
        except Exception as e:
            info[fk] = dict(error=str(e))
    json.dump(info, open(f"{OUT}/info_v0.json", "w"), indent=1)
    return rcs

def assemble_weights(prefix, W, spread):
    """Export the 3-seed ensembles (fp32 + int8, spread baked in) of both frequencies and put the service data files next to them."""
    pre = prereg(); os.makedirs(W, exist_ok=True); info = {}
    for fk in ("15", "60"):
        cks = [f"{REPO}/runs/{prefix}_{fk}_s{s}.pt" for s in pre["ensemble_seeds"]]
        rc = tsb("export", "--ckpt", *cks, "--out", f"{W}/net{fk}", "--spread", spread[fk], name=f"export_{prefix}_{fk}", quiet=True)
        if rc != 0:
            raise RuntimeError(f"export {prefix} {fk} failed")
        info[fk] = jload(f"{W}/net{fk}_info.json")
    for n in ("hol_table.npz", "clock_ref.npz"):
        shutil.copy(f"{REPO}/data/cache/{n}", f"{W}/{n}")
    return info

def variants(prefix, W, spread, window, final, nohol_prefix=None):
    """The eight evaluations per definition (pre-registered list).  Returns [(name, cmd)]."""
    pre = prereg(); jobs = []
    for fk in ("15", "60"):
        d = DEF_OF[fk]; sp = spread[fk]; cks = [f"{REPO}/runs/{prefix}_{fk}_s{s}.pt" for s in pre["ensemble_seeds"]]
        ev = f"{OUT}/{'test' if final else 'valdry'}"; os.makedirs(ev, exist_ok=True)
        def J(tag): return f"{ev}/{tag}.json"
        jobs += [(f"d{d}_ens3_torch_fp32", evalf_cmd(d, window, f"{prefix}_d{d}_ens3_torch_fp32", J(f"d{d}_ens3_torch_fp32"), ckpts=cks, spread=sp, final=final, note=f"def{d} ens3 torch fp32")),
                 (f"d{d}_ens3_ort_fp32", evalf_cmd(d, window, f"{prefix}_d{d}_ens3_ort_fp32", J(f"d{d}_ens3_ort_fp32"), onnx=f"{W}/net{fk}.onnx", freq=fk, final=final, note=f"def{d} ens3 onnxruntime fp32")),
                 (f"d{d}_ens3_ort_int8", evalf_cmd(d, window, f"{prefix}_d{d}_ens3_ort_int8", J(f"d{d}_ens3_ort_int8"), onnx=f"{W}/net{fk}_int8.onnx", freq=fk, final=final, note=f"def{d} ens3 onnxruntime int8")),
                 (f"d{d}_service_int8", evalf_cmd(d, window, f"{prefix}_d{d}_service_int8", J(f"d{d}_service_int8"), service=W, precision="int8", final=final, note=f"def{d} service code path int8"))]
        for s in pre["ensemble_seeds"]:
            jobs.append((f"d{d}_single_s{s}", evalf_cmd(d, window, f"{prefix}_d{d}_single_s{s}", J(f"d{d}_single_s{s}"), ckpts=[f"{REPO}/runs/{prefix}_{fk}_s{s}.pt"], spread=sp, final=final, note=f"def{d} single member seed {s} torch fp32")))
        if nohol_prefix:
            seed = pre["ablation"]["no_holiday_features"]["seed"]
            jobs.append((f"d{d}_nohol_s{seed}", evalf_cmd(d, window, f"{nohol_prefix}_d{d}_nohol_s{seed}", J(f"d{d}_nohol_s{seed}"), ckpts=[f"{REPO}/runs/{nohol_prefix}_{fk}_nohol_s{seed}.pt"], spread=sp, final=final, note=f"def{d} no-holiday ablation seed {seed} torch fp32")))
    return jobs

def evidence(prefix, label, window, final, names, tagbase, params_m=None, src_window=None):
    """Our variants inserted into the real field (MASE and SQL tables, CIs, Elo).  names: {display name: variant tag suffix}."""
    out = []
    for fk in ("15", "60"):
        d = DEF_OF[fk]
        sw = src_window or window     # the per-round file covers the whole evaluated window; sub-window tables are views of it
        ours = [f"{n}={REPO}/runs/faith_{'final' if final else 'val'}_def{d}_{tagbase.get(n, prefix)}_d{d}_{suf}_{sw[0]}_{sw[1]}.parquet" for n, suf in names.items()]
        cmd = [PY, "-m", "tsb.make_evidence", "--def", str(d), "--days", *window, "--label", f"{label} (definition {d}, {window[0]} .. {window[1]})",
               "--ours", *ours, "--out", f"reports/{label}_def{d}_{window[0]}_{window[1]}.md"]
        if params_m:
            cmd += ["--params", *[f"{n}={params_m[fk][n]}" for n in names if n in params_m[fk]]]
        out.append((f"{label}_def{d}_{window[0]}_{window[1]}", cmd))
    return out

@stage("val_dryrun")
def stage_val_dryrun():
    """Final recipe at the validation stage: all evaluation paths on VAL-F (these are real VAL-F numbers: the models stop at 2025-12-31 23:45)."""
    spread = {k: v["scale"] for k, v in jload(f"{OUT}/spread_chosen.json").items()}
    W = f"{REPO}/dryW"; info = assemble_weights("val", W, spread)
    rcs = run_chains([(n, [c]) for n, c in variants("val", W, spread, VALF, False)], 4)
    pm = {fk: {"gridload ens3 fp32": info[fk]["params"] / 1e6, "gridload ens3 service int8": info[fk]["params"] / 1e6, "gridload single s0": info[fk]["params_per_net"] / 1e6} for fk in info}
    ev = evidence("val", "val_f", VALF, False, {"gridload ens3 fp32": "ens3_torch_fp32", "gridload ens3 service int8": "service_int8", "gridload single s0": "single_s0"}, {}, pm)
    rcs.update(run_chains([(n, [c]) for n, c in ev], 2))
    for f in glob.glob(f"{REPO}/reports/val_f_*.md"):
        shutil.copy(f, OUT)
    shutil.copytree(W, f"{OUT}/dry_weights", dirs_exist_ok=True)
    collect_faith()
    return dict(rcs=rcs, weights={k: dict(params=v["params"], fp32_bytes=v["fp32_bytes"], int8_bytes=v["int8_bytes"], spread=v["spread"]) for k, v in info.items()})

@stage("freeze_candidate")
def stage_freeze_candidate():
    pre = prereg(); sel = jload(f"{OUT}/selection.json"); sp = jload(f"{OUT}/spread_chosen.json")
    man = export_ckpts()
    fz = dict(prereg_version=pre["version"], commit_of_a2=os.environ.get("TSB_COMMIT"), kernel_a2=os.environ.get("TSB_KERNEL"),
              selection={fk: sel[fk]["selected"] | {"gate_failed": sel[fk]["gate_failed"], "reason": sel[fk]["reason"]} for fk in sel},
              spread={fk: sp[fk]["scale"] for fk in sp}, seeds=pre["ensemble_seeds"], cutoffs=pre["cutoffs"],
              test_stage_ckpt_sha256={k: v for k, v in man.items() if k.startswith("test_") and k.endswith(".pt")},
              val_stage_ckpt_sha256={k: v for k, v in man.items() if k.startswith("val_") and k.endswith(".pt")})
    json.dump(fz, open(f"{OUT}/freeze_candidate.json", "w"), indent=1)
    return fz

def plan_a2():
    stage_env()
    restore_base(keep_field=True)
    have = all(os.path.exists(f"{REPO}/data/cache/field_def{d}.parquet") for d in (2, 5))
    STATUS["field_source"] = "a1 cache (provisional; valid for the VAL-F rounds, verified against the Mac copy offline)" if have else None
    stage_repro_eval()
    th = threading.Thread(target=lambda: guarded("field_pipeline", field_pipeline)); th.start()
    sv = threading.Thread(target=lambda: guarded("svc_prepare", svc_prepare)); sv.start()
    stage_sweep()
    stage_selection()
    stage_stage2()
    th.join(); sv.join()
    stage_val_dryrun()
    if STATUS["stages"].get("val_dryrun", {}).get("ok"):
        guarded("svc_tests_valdry", svc_tests, f"{REPO}/dryW", "valdry")
    stage_freeze_candidate()

# ----------------------------------------------------------------------------------------------- b: frozen TEST-F evaluation
@stage("verify_frozen")
def stage_verify_frozen():
    """configs/final.json (committed before this kernel was built) must match the mounted test-stage checkpoints byte for byte."""
    fz = jload(f"{REPO}/configs/final.json"); bad = []
    for name, h in fz["test_stage_ckpt_sha256"].items():
        p = f"{REPO}/runs/{name}"
        if not os.path.exists(p) or sha256(p) != h:
            bad.append(name)
    if bad:
        raise RuntimeError(f"frozen checkpoints missing or changed: {bad}")
    chk = """
import sys, torch, json
fz = json.load(open(sys.argv[1])); pre = json.load(open(sys.argv[2]))
for n in fz['test_stage_ckpt_sha256']:
    ck = torch.load(sys.argv[3] + '/' + n, map_location='cpu', weights_only=False); a = ck['args']; fk = str(ck['freq']); s = fz['selection'][fk]
    assert (a['hid'], a['emb'], a['J'], a['dmeans'], a['steps']) == (s['hid'], s['emb'], s['J'], s['dmeans'], pre['steps']), (n, a)
    assert a['train_end'] == pre['cutoffs']['test'], (n, a['train_end'])
    assert ck['params'] == s['params'], (n, ck['params'], s['params'])
print('ok', len(fz['test_stage_ckpt_sha256']))
"""
    r = subprocess.run([PY, "-c", chk, f"{REPO}/configs/final.json", f"{REPO}/configs/prereg.json", f"{REPO}/runs"], capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        raise RuntimeError("checkpoint metadata does not match the frozen selection: " + (r.stderr or r.stdout)[-600:])
    return dict(checked=len(fz["test_stage_ckpt_sha256"]), meta=r.stdout.strip())

@stage("assemble_test")
def stage_assemble_test():
    fz = jload(f"{REPO}/configs/final.json")
    W = f"{REPO}/finalW"; info = assemble_weights("test", W, fz["spread"])
    shutil.copytree(W, f"{OUT}/service_weights", dirs_exist_ok=True)
    return {k: dict(params=v["params"], params_per_net=v["params_per_net"], fp32_bytes=v["fp32_bytes"], int8_bytes=v["int8_bytes"], spread=v["spread"], fp32_sha256=v["fp32_sha256"], int8_sha256=v["int8_sha256"]) for k, v in info.items()}

@stage("dryrun_codepaths")
def stage_dryrun_codepaths():
    """Every evaluation path of the TEST-F list on five VAL-F rounds (the models saw VAL-F: code check only, numbers are not used)."""
    fz = jload(f"{REPO}/configs/final.json"); W = f"{REPO}/finalW"
    jobs = variants("test", W, fz["spread"], ("2026-01-01", "2026-01-05"), False, nohol_prefix="test")
    rcs = run_chains([(n, [c]) for n, c in jobs], 4)
    bad = {k: v for k, v in rcs.items() if v != 0}
    if bad:
        raise RuntimeError(f"dry run failed, TEST-F not started: {bad}")
    out = {}
    for n, _ in jobs:
        r = jload(f"{OUT}/valdry/{n}.json")["summary"]
        out[n] = dict(mase=round(r["mase"], 5), missing=r["n_missing"], routing=r.get("routing"))
    for fk, d in (("15", 2), ("60", 5)):
        sv = out[f"d{d}_service_int8"]
        if (sv.get("routing") or {}).get("fallback", 0) != 0 or (sv.get("routing") or {}).get("specialist", 0) != 50:
            raise RuntimeError(f"service routing not all-specialist on the dry run: {sv}")
    return out

@stage("test_f")
def stage_test_f():
    fz = jload(f"{REPO}/configs/final.json"); W = f"{REPO}/finalW"
    ledger = f"{OUT}/TEST-LOG-entries.md"
    if os.path.exists(ledger) and "variant:" in open(ledger).read():
        raise RuntimeError("TEST-F entries already exist in this kernel's output: refusing to evaluate twice")
    jobs = variants("test", W, fz["spread"], TESTF, True, nohol_prefix="test")
    log("TEST-F:", len(jobs), "evaluations (each logged once)")
    rcs = run_chains([(n, [c]) for n, c in jobs], 4)
    collect_faith()
    res = {n: jload(f"{OUT}/test/{n}.json")["summary"] for n, _ in jobs if rcs.get(n) == 0}
    return dict(rcs=rcs, summaries={n: dict(mase=round(r["mase"], 5), sql=round(r["sql"], 5), cover80=round(r["cover80"], 4), rounds=r["rounds"], missing=r["n_missing"], routing=r.get("routing")) for n, r in res.items()})

@stage("test_evidence")
def stage_test_evidence():
    asm = STATUS["stages"].get("assemble_test", {}).get("result") or {}
    pm = {fk: {"gridload ens3 fp32": asm[fk]["params"] / 1e6, "gridload ens3 service int8": asm[fk]["params"] / 1e6, "gridload single s0": asm[fk]["params_per_net"] / 1e6,
               "gridload no-holiday s0": asm[fk]["params_per_net"] / 1e6} for fk in asm}
    names = {"gridload ens3 fp32": "ens3_torch_fp32", "gridload ens3 service int8": "service_int8", "gridload single s0": "single_s0", "gridload no-holiday s0": "nohol_s0"}
    jobs = []
    for window in (TESTF, ("2026-02-08", "2026-03-14"), ("2026-03-15", "2026-03-31")):
        jobs += evidence("test", "test_f", window, True, names, {}, pm, src_window=TESTF)
    rcs = run_chains([(n, [c]) for n, c in jobs], 3)
    for f in glob.glob(f"{REPO}/reports/test_f_*.md"):
        shutil.copy(f, OUT)
    return rcs

@stage("bench_cpu")
def stage_bench():
    W = f"{REPO}/finalW"; res = {}
    for fk in ("15", "60"):
        res[fk] = run([PY, "-m", "tsb.bench", "--onnx", f"{W}/net{fk}.onnx", f"{W}/net{fk}_int8.onnx", "--freq", fk, "-n", "300"], f"bench_{fk}", None, f"{REPO}/src", True)
        shutil.copy(f"{OUT}/logs/bench_{fk}.log", f"{OUT}/bench_kaggle_cpu_{fk}.json")
    return res

def plan_b():
    stage_env()
    sv = threading.Thread(target=lambda: guarded("svc_prepare", svc_prepare)); sv.start()
    restore_base()
    guarded("restore_field", restore_field)
    guarded("restore_ckpts", restore_ckpts)
    stage_verify_frozen()
    if not STATUS["stages"].get("verify_frozen", {}).get("ok"):
        log("frozen verification failed: stopping before any TEST-F evaluation"); return
    stage_assemble_test()
    stage_dryrun_codepaths()
    if not STATUS["stages"].get("dryrun_codepaths", {}).get("ok"):
        log("dry run failed: stopping before any TEST-F evaluation"); return
    stage_test_f()
    stage_test_evidence()
    sv.join()
    guarded("svc_tests_final", svc_tests, f"{REPO}/finalW", "final")
    stage_bench()

# ----------------------------------------------------------------------------------------------- plans
def plan_a1():
    stage_env()
    stage_fetch()
    stage_tables()
    stage_repro_eval()
    stage_smoke()
    if STATUS["stages"].get("smoke", {}).get("ok"):
        guarded("svc_tests_smoke", svc_tests, f"{REPO}/smokeW", "smoke")
    stage_export_cache()

def main():
    plan = sys.argv[1] if len(sys.argv) > 1 else "a1"
    STATUS["plan"] = plan
    try:
        shutil.copy(f"{REPO}/BUILD.json", f"{OUT}/BUILD.json")
    except Exception:
        pass
    {"a1": plan_a1, "a2": plan_a2, "b": plan_b}[plan]()
    finish()

if __name__ == "__main__":
    main()
