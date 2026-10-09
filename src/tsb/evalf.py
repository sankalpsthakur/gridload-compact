"""Evaluate checkpoints / ONNX files / the service code path on faithful archive rounds (served contexts, first-published truth).
python -m tsb.evalf --def 2 --ckpt runs/a.pt [runs/b.pt ...] --days 2026-01-01 2026-02-07 [--spread S] [--final --note VARIANT]
python -m tsb.evalf --def 2 --onnx weights/net15_int8.onnx --freq 15 --days ...      (spec kwargs come from net15_info.json)
python -m tsb.evalf --def 2 --service weights/ --precision int8 --days ...
--final marks a TEST-F evaluation: an entry is appended to TEST-LOG.md (and to $TSB_TESTLOG_OUT when set, immediately)."""
import argparse, datetime as dt, json, os, sys
import numpy as np, pandas as pd
from tsb import feats, infer, faith, cal_np, metrics, ROOT

NO_HOL = {}
SPEC_KW = {}

def load_models(paths):
    import torch
    from tsb import net
    torch.set_num_threads(1)
    ms = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = net.QNet(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        ms.append((m, ck["freq"]))
        NO_HOL[p] = bool(ck.get("args", {}).get("no_hol", False))
        SPEC_KW[p] = ck.get("spec_kwargs", {})
    assert len({f for _, f in ms}) == 1
    assert len({NO_HOL[p] for p in paths}) == 1, "do not mix holiday and no-holiday members"
    assert len({json.dumps(SPEC_KW[p], sort_keys=True) for p in paths}) == 1, "members must share the feature spec"
    return [m for m, _ in ms], ms[0][1]

def make_predict_np(models):
    import torch
    def fn(ctx, lag, step, base):
        with torch.no_grad():
            qs = [m(torch.from_numpy(ctx), torch.from_numpy(lag), torch.from_numpy(step), torch.from_numpy(base)).numpy() for m in models]
        return np.mean(qs, 0)
    return fn

def make_predict_np_ort(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def fn(ctx, lag, step, base):
        return sess.run(["q"], {"ctx": ctx, "lag": lag, "step": step, "base": base})[0]
    return fn

def calendar(no_hol=False):
    z = np.load(f"{ROOT}/data/cache/hol_table.npz")
    return cal_np.Calendar(int(z["day0"]), np.zeros_like(z["table"]) if no_hol else z["table"])

def run(ckpts, defn, d0, d1, predict_np=None, spec=None, quant_scale=None, no_hol=None, rounds=None):
    if predict_np is None:
        models, freq = load_models(ckpts); predict_np = make_predict_np(models); spec = feats.Spec(freq, **SPEC_KW[ckpts[0]])
    if no_hol is None:
        no_hol = any(NO_HOL.get(c, False) for c in ckpts) if ckpts else False
    C = calendar(no_hol=no_hol)
    if rounds is None:
        rounds = faith.load_rounds(defn, d0, d1)
    def ff(ts, vals):
        res = infer.forecast_series(predict_np, C, spec, ts, vals, spec.H)
        if res is None or quant_scale is None:
            return res
        fut, q = res
        med = q[:, 4:5]
        return fut, med + (q - med) * quant_scale
    df = faith.score_model(rounds, ff, spec.H, spec.step_s)
    return df, rounds

def run_service(weights_dir, defn, d0, d1, precision, rounds=None):
    """Score the exact service code (numpy features + onnxruntime + routing gates) on the faithful rounds."""
    sys.path.insert(0, f"{ROOT}/submission/model-services/gridload-compact")
    os.environ["GRIDLOAD_WEIGHTS"] = weights_dir
    from app import model as svc
    m = svc.GridLoadCompact(precision=precision)
    freq = "15min" if defn == 2 else "h"; step = 900 if defn == 2 else 3600; H = 96 if defn == 2 else 72
    if rounds is None:
        rounds = faith.load_rounds(defn, d0, d1)
    paths = {}
    def ff(ts, vals):
        items = [{"ts": pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z"), "value": (None if not np.isfinite(v) else float(v))} for t, v in zip(ts, vals)]
        last, q = m.predict_series(items, H, freq)
        paths[m.last_path] = paths.get(m.last_path, 0) + 1
        return last + np.arange(1, H + 1, dtype=np.int64) * step, q
    df = faith.score_model(rounds, ff, H, step)
    df.attrs["paths"] = paths
    print("routing:", paths)
    return df

def summarize(df, label):
    g = df.groupby("day")[["mase", "sql", "cover80"]].mean()
    return dict(label=label, rounds=len(g), mase=float(g.mase.mean()), sql=float(g.sql.mean()), cover80=float(g.cover80.mean()), n_missing=int(df.mase.isna().sum()))

def _sha(path):
    from tsb.export import sha256
    return sha256(path)

def write_entry(text):
    with open(f"{ROOT}/TEST-LOG.md", "a") as f:
        f.write(text)
    extra = os.environ.get("TSB_TESTLOG_OUT")
    if extra:
        os.makedirs(os.path.dirname(extra) or ".", exist_ok=True)
        with open(extra, "a") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="defn", type=int, required=True)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--onnx", default=None, help="evaluate an exported ONNX file instead of checkpoints (needs --freq)")
    ap.add_argument("--freq", type=int, default=None)
    ap.add_argument("--no-hol", action="store_true")
    ap.add_argument("--service", default=None, help="evaluate the shipped service code path (weights dir) instead of the research harness: needs --precision")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--spread", type=float, default=None, help="post-hoc decile spread scale (torch checkpoints only; baked into ONNX at export)")
    ap.add_argument("--days", nargs=2, required=True)
    ap.add_argument("--final", action="store_true", help="TEST-F run: appends an entry to TEST-LOG.md")
    ap.add_argument("--note", default="", help="variant name for the log entry")
    ap.add_argument("--tag", default=None, help="file tag (default derived from the artifact name)")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    artifacts = []
    if a.final:      # a TEST-F evaluation counts from the moment it starts, also when it crashes before writing its result
        write_entry(f"\n### {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC  def {a.defn}  {a.days[0]}..{a.days[1]}  variant: {a.note or 'unnamed'}  STARTED"
                    f"  (kernel {os.environ.get('TSB_KERNEL', 'local')}, code {os.environ.get('TSB_COMMIT', 'unknown')}); the result entry follows when the run completes\n")
    if a.service:
        df = run_service(a.service, a.defn, a.days[0], a.days[1], a.precision)
        label = f"service:{os.path.basename(a.service.rstrip('/'))}:{a.precision}"; tag = f"service_{a.precision}"
        artifacts = [os.path.join(a.service, f) for f in sorted(os.listdir(a.service)) if f.endswith(".onnx") and (("_int8" in f) == (a.precision == "int8"))]
    elif a.onnx:
        from tsb.export import spec_kwargs_for
        spec = feats.Spec(a.freq, **spec_kwargs_for(a.onnx))
        df, _ = run(None, a.defn, a.days[0], a.days[1], predict_np=make_predict_np_ort(a.onnx), spec=spec, no_hol=a.no_hol)
        label = os.path.basename(a.onnx); tag = label.replace(".onnx", ""); artifacts = [a.onnx]
    else:
        df, _ = run(a.ckpt, a.defn, a.days[0], a.days[1], quant_scale=a.spread)
        label = ",".join(os.path.basename(c) for c in a.ckpt); tag = os.path.basename(a.ckpt[0]).replace(".pt", "") + (f"+{len(a.ckpt)-1}" if len(a.ckpt) > 1 else "")
        artifacts = list(a.ckpt)
    if a.tag:
        tag = a.tag
    s = summarize(df, label)
    if "paths" in df.attrs:
        s["routing"] = df.attrs["paths"]
    print(json.dumps(s))
    out = f"{ROOT}/runs/faith_{'final' if a.final else 'val'}_def{a.defn}_{tag}_{a.days[0]}_{a.days[1]}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        json.dump(dict(summary=s, parquet=out, def_=a.defn, days=a.days, spread=a.spread, note=a.note), open(a.out_json, "w"), indent=1)
    if a.final:
        shas = "; ".join(f"{os.path.basename(p)}={_sha(p)[:16]}" for p in artifacts)
        write_entry(f"\n### {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC  def {a.defn}  {a.days[0]}..{a.days[1]}  variant: {a.note or tag}\n"
                    f"- code: {os.environ.get('TSB_COMMIT', 'unknown')}  kernel: {os.environ.get('TSB_KERNEL', 'local')}\n"
                    f"- artifacts (sha256 prefix): {shas}\n- spread: {a.spread}\n- result: {json.dumps(s)}\n- per-round file: {os.path.basename(out)}\n")
