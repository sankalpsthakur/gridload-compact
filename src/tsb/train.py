"""Train the compact quantile net on final-data history of the 10 SMARD net-load series (platform clock).
Usage: python -m tsb.train --freq 15 --train-end 2025-06-30T23:45 --val 2025-07-01 2025-12-31 --steps 4000 --seed 0 --out runs/x.pt"""
import argparse, json, os, time, math
import numpy as np, pandas as pd, torch
from tsb import sim, feats, net, metrics, ROOT

def valid_origins(bk, end_idx, start_idx):
    """boolean [S, N]: origin ok for training (finite origin value, >=90% finite weekly window and targets, all inside range)."""
    sp = bk.spec; X = bk.X; S, N = X.shape
    fin = np.isfinite(X).astype(np.float32)
    cs = np.concatenate([np.zeros((S, 1), np.float32), np.cumsum(fin, 1)], 1)
    def cnt(a, b):      # finite count in [a, b) for each origin vector index
        return cs[:, b] - cs[:, a]
    o = np.arange(N)
    ok = np.zeros((S, N), bool)
    lo = max(start_idx, sp.L - 1); hi = min(end_idx - sp.H, N - sp.H - 1)
    oo = np.arange(lo, hi + 1)
    wk = cs[:, oo + 1] - cs[:, oo + 1 - sp.W7]
    tg = cs[:, oo + sp.H + 1] - cs[:, oo + 1]
    ok[:, oo] = (fin[:, oo] > 0) & (wk >= 0.9 * sp.W7) & (tg >= 0.9 * sp.H)
    return ok

def make_sampler(bk, ok, tau_years):
    """per-series pools of valid origins and (optional) recency weights exp(-age/tau)."""
    S, N = ok.shape
    pools = [np.flatnonzero(ok[s]) for s in range(S)]
    cdfs = None
    if tau_years and tau_years > 0:
        tau_steps = tau_years * 365.25 * 86400 / bk.step
        cdfs = []
        for p in pools:
            w = np.exp((p - p.max()) / tau_steps)
            c = np.cumsum(w); cdfs.append(c / c[-1])
    return pools, cdfs

def sample_batch(bk, ok, rng, B, phase_boost, sampler=None):
    S, N = ok.shape
    pools, cdfs = sampler if sampler is not None else make_sampler(bk, ok, 0)
    def draw(s):
        if cdfs is None:
            return pools[s][rng.integers(0, len(pools[s]))]
        return pools[s][min(np.searchsorted(cdfs[s], rng.random()), len(pools[s]) - 1)]
    ss = rng.integers(0, S, B)
    oo = np.array([draw(s) for s in ss], dtype=np.int64)
    if phase_boost:
        hours = (bk.P[oo] % 86400) / 3600.0
        bad = np.flatnonzero((rng.random(B) < 0.5) & ~((hours >= 4.0) & (hours <= 10.0)))
        for i in bad:
            for _ in range(20):
                o = draw(ss[i])
                h = (bk.P[o] % 86400) / 3600.0
                if 4.0 <= h <= 10.0:
                    oo[i] = o; break
    return ss, oo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=15)
    ap.add_argument("--train-start", default="2018-02-01T00:00")
    ap.add_argument("--train-end", default="2025-06-30T23:45")
    ap.add_argument("--val", nargs=2, default=["2025-07-01", "2025-12-31"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--nsteps", type=int, default=32, help="target steps sampled per sample")
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--wfloor", type=float, default=0.03)
    ap.add_argument("--no-hol", action="store_true", help="ablation: zero all holiday/bridge features")
    ap.add_argument("--recency-tau", type=float, default=0.0, help="years; >0 samples recent origins more often (exp(-age/tau))")
    ap.add_argument("--J", type=int, default=48)
    ap.add_argument("--dmeans", type=int, default=0)
    ap.add_argument("--out", default="runs/net.pt")
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    bk = sim.Bank(a.freq, no_hol=a.no_hol, spec_kwargs=dict(J=a.J, dmeans=a.dmeans)); sp = bk.spec
    end_idx = bk.idx_of(a.train_end); start_idx = bk.idx_of(a.train_start)
    ok = valid_origins(bk, end_idx, start_idx)
    sampler = make_sampler(bk, ok, a.recency_tau)
    print(f"freq {a.freq}: valid origins {ok.sum()} (per series {ok.sum(1).tolist()})", flush=True)
    model = net.QNet(sp.n_ctx, sp.K, sp.n_lag_f, sp.n_step_f, hid=a.hid, emb=a.emb)
    print("params", net.n_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1, anneal_strategy="cos")
    val_origins = {ph: sim.round_origins(bk, a.val[0], a.val[1], ph) for ph in (["07:45", "05:30"] if a.freq == 15 else ["08:00", "06:00"])}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    def evaluate():
        res = {}
        model.eval()
        for ph, org in val_origins.items():
            df = sim.score_rounds(bk, org, lambda ss, oo: sim.predict(model, bk, ss, oo)[0])
            res[ph] = (float(df.mase.mean()), float(df.sql.mean()), float(df.cover80.mean()))
        model.train()
        return res
    t0 = time.time(); run = 0.0; last_val = None
    for step in range(1, a.steps + 1):
        ss, oo = sample_batch(bk, ok, rng, a.bs, phase_boost=True, sampler=sampler)
        f = bk.batch(ss, oo, with_targets=True)
        keep = f["ok"] & np.isfinite(f["mae_naive"]) & (f["mae_naive"] > 0)
        if keep.sum() < 8:
            continue
        sel = rng.integers(0, sp.H, (a.bs, a.nsteps))
        bi = np.arange(a.bs)[:, None]
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        lag = t(f["lag"][bi, sel]); step_f = t(f["step"][bi, sel]); base = t(f["base"][bi, sel])
        y = t(f["y"][bi, sel]); ym = t(f["ymask"][bi, sel])
        w = torch.from_numpy((keep / np.maximum(f["mae_naive"], a.wfloor)).astype(np.float32))
        q = model(t(f["ctx"]), lag, step_f, base)
        loss = net.pinball(q, y, ym, w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run:.4f} {time.time()-t0:.0f}s", flush=True)
        if a.eval_every < 10**6 and (step % a.eval_every == 0 or step == a.steps):
            last_val = evaluate()
            print("  VAL", json.dumps({k: [round(x, 4) for x in v] for k, v in last_val.items()}), flush=True)
    secs = time.time() - t0
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, args=vars(a), freq=a.freq, spec_kwargs=dict(J=a.J, dmeans=a.dmeans),
                    params=net.n_params(model), val_s=last_val, train_seconds=secs), a.out)
    if last_val is not None:     # VAL-S sidecar: {phase: [mase, sql, cover80]} of the last evaluation (step == steps)
        json.dump(dict(freq=a.freq, params=net.n_params(model), hid=a.hid, emb=a.emb, J=a.J, dmeans=a.dmeans, steps=a.steps, seed=a.seed,
                       train_end=a.train_end, val_window=a.val, val_s=last_val, train_seconds=secs), open(a.out + ".val.json", "w"))
    print("saved", a.out, f"{secs:.0f}s", flush=True)

if __name__ == "__main__":
    main()
