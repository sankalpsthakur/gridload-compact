"""Latency / size benchmark: 1 thread, batch 1, median of N after warm-up.  Reports model-only and end-to-end (features+model) per series."""
import argparse, json, os, platform, statistics, subprocess, time
import numpy as np
from tsb import feats, infer, cal_np, ROOT

def sysinfo():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if not chip:
            raise RuntimeError
    except Exception:
        try:
            chip = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        except Exception:
            chip = platform.processor()
    return dict(chip=chip, platform=platform.platform(), python=platform.python_version(), cpus=os.cpu_count())

def pct(xs, cpu=None):
    xs = sorted(xs); n = len(xs)
    d = dict(p50_ms=1e3 * xs[n // 2], p90_ms=1e3 * xs[int(0.9 * n)], p99_ms=1e3 * xs[min(n - 1, int(0.99 * n))], n=n)
    if cpu is not None:
        cs = sorted(cpu); d["cpu_p50_ms"] = 1e3 * cs[n // 2]
    return d

def make_ort(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def fn(ctx, lag, step, base):
        return sess.run(["q"], {"ctx": ctx, "lag": lag, "step": step, "base": base})[0]
    return fn

def run(onnx_path, freq, n=200, warm=20):
    from tsb.export import spec_kwargs_for
    sp = feats.Spec(freq, **spec_kwargs_for(onnx_path))
    z = np.load(f"{ROOT}/data/cache/hol_table.npz"); C = cal_np.Calendar(int(z["day0"]), z["table"])
    rng = np.random.default_rng(0)
    # synthetic but realistic context: daily + weekly seasonality + noise, platform clock grid
    L = sp.L; t = np.arange(L)
    per_day = sp.P
    base = 10000 + 3000 * np.sin(2 * np.pi * t / per_day) + 800 * np.sin(2 * np.pi * t / (7 * per_day))
    vals = base + rng.normal(0, 60, L)
    last_ts = int(np.datetime64("2026-03-10T07:45:00").astype("datetime64[s]").astype(np.int64)) if freq == 15 else int(np.datetime64("2026-03-10T08:00:00").astype("datetime64[s]").astype(np.int64))
    ts = last_ts + (t - (L - 1)) * sp.step_s
    fn = make_ort(onnx_path)
    # model-only
    f = feats.build(sp, np.concatenate([vals.astype(np.float32), np.full(sp.H, np.nan, np.float32)])[None],
                    {k: v[None] for k, v in C.features(last_ts + (np.arange(L + sp.H) - (L - 1)) * sp.step_s).items()})
    for _ in range(warm): fn(f["ctx"], f["lag"], f["step"], f["base"])
    tm, tmc = [], []
    for _ in range(n):
        a = time.perf_counter(); c = time.process_time(); fn(f["ctx"], f["lag"], f["step"], f["base"]); tm.append(time.perf_counter() - a); tmc.append(time.process_time() - c)
    te, tec = [], []
    for _ in range(warm): infer.forecast_series(fn, C, sp, ts, vals, sp.H)
    for _ in range(n):
        a = time.perf_counter(); c = time.process_time(); infer.forecast_series(fn, C, sp, ts, vals, sp.H); te.append(time.perf_counter() - a); tec.append(time.process_time() - c)
    return dict(onnx=os.path.basename(onnx_path), freq=freq, bytes=os.path.getsize(onnx_path), model_only=pct(tm, tmc), end_to_end_per_series=pct(te, tec), loadavg=os.getloadavg())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True); ap.add_argument("--freq", type=int, required=True)
    ap.add_argument("-n", type=int, default=200)
    a = ap.parse_args()
    out = dict(system=sysinfo(), results=[run(p, a.freq, a.n) for p in a.onnx])
    print(json.dumps(out, indent=1))
