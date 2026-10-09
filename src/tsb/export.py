"""Export a checkpoint (or an average of several) to ONNX fp32 + dynamic int8; report sizes.
python -m tsb.export --ckpt runs/a.pt [runs/b.pt ...] --out weights/net15"""
import argparse, os, json
import numpy as np, torch
from tsb import net, feats

class Wrapper(torch.nn.Module):
    """Averages the quantile curves of several nets into one graph so the service has a single input/output signature."""
    def __init__(self, models, spread=1.0):
        super().__init__(); self.ms = torch.nn.ModuleList(models); self.spread = float(spread)
    def forward(self, ctx, lag, step, base):
        qs = [m(ctx, lag, step, base) for m in self.ms]
        q = torch.stack(qs, 0).mean(0) if len(qs) > 1 else qs[0]
        if self.spread != 1.0:                      # global calibration of the decile spread around the median
            med = q[..., 4:5]
            q = med + (q - med) * self.spread
        return q

def load(ckpts):
    ms = []; freq = None; kw = None
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = net.QNet(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval(); ms.append(m); freq = ck["freq"]
        k = ck.get("spec_kwargs", {})
        assert kw is None or kw == k, "ensemble members must share the feature spec"
        kw = k
    return ms, freq, kw

def _export_onnx(w, args, path, opset):
    kw = dict(input_names=["ctx", "lag", "step", "base"], output_names=["q"],
              dynamic_axes={k: {0: "batch"} for k in ["ctx", "lag", "step", "base", "q"]}, opset_version=opset)
    try:
        torch.onnx.export(w, args, path, dynamo=False, **kw)          # TorchScript exporter: no extra dependencies
    except TypeError:                                                 # torch without the dynamo kwarg
        torch.onnx.export(w, args, path, **kw)

def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def info_path_for(onnx_path):
    """weights/net15.onnx and weights/net15_int8.onnx -> weights/net15_info.json"""
    return os.path.splitext(onnx_path)[0].replace("_int8", "") + "_info.json"

def spec_kwargs_for(onnx_path):
    p = info_path_for(onnx_path)
    return json.load(open(p))["spec_kwargs"] if os.path.exists(p) else {}

def export(ckpts, out_prefix, opset=17, spread=1.0):
    ms, freq, kw = load(ckpts)
    sp = feats.Spec(freq, **kw)
    w = Wrapper(ms, spread).eval()
    B = 2
    args = (torch.zeros(B, sp.n_ctx), torch.zeros(B, sp.H, sp.K, sp.n_lag_f), torch.zeros(B, sp.H, sp.n_step_f), torch.zeros(B, sp.H))
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    fp32 = out_prefix + ".onnx"
    _export_onnx(w, args, fp32, opset)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = out_prefix + "_int8.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    info = dict(freq=freq, spec_kwargs=kw, n_nets=len(ms), spread=spread, params=int(sum(net.n_params(m) for m in ms)), params_per_net=int(net.n_params(ms[0])),
                fp32_bytes=os.path.getsize(fp32), int8_bytes=os.path.getsize(int8), state_dict_bytes=sum(p.numel() * 4 for m in ms for p in m.parameters()),
                members=[dict(file=os.path.basename(c), sha256=sha256(c)) for c in ckpts], fp32_sha256=sha256(fp32), int8_sha256=sha256(int8))
    json.dump(info, open(out_prefix + "_info.json", "w"), indent=1)
    return info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--spread", type=float, default=1.0)
    a = ap.parse_args()
    print(json.dumps(export(a.ckpt, a.out, spread=a.spread)))
