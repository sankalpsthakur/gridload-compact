"""One pass over the archive: per-round contexts/truth arrays + per-(round,model,series) official-style scores
for the whole field (32 models), against BOTH the archive (first-published) truth and the current final raw data.
Memory-light: streams one round at a time."""
import sys, time, json
import numpy as np, pandas as pd
from tsb import ROOT, archive, data, metrics

QC = [f"q_{l}" for l in metrics.LEVELS]
H = {2: 96, 5: 72}

def run(defn):
    rd = archive.round_dirs(defn)
    raw = {reg: data.quarter_series(reg) for reg in data.REGIONS}
    if defn == 5:
        raw = {reg: data.hourly_from_quarter(s) for reg, s in raw.items()}
    uids = list(archive.UID2REG)
    ctx_arr, tr_arr, ctx_t0, tr_t0, days, rids = [], [], [], [], [], []
    rows = []
    t0 = time.time()
    for i, (reg_start, sub) in enumerate(sorted(rd.items())):
        ctx, rid = archive.context(defn, sub); tr = archive.truth(defn, sub); fc = archive.forecasts(defn, sub)
        days.append(reg_start); rids.append(rid)
        yf_all = {uid: raw[archive.UID2REG[uid]].reindex(tr[uid].index) for uid in uids}   # once per (round, series)
        for m, per in fc.items():
            for uid, f in per.items():
                reg = archive.UID2REG[uid]
                yp = tr[uid]; yf = yf_all[uid]
                j = f.join(yp.rename("yp")).join(yf.rename("yf"))
                ok = j[["value", "yp"]].notna().all(axis=1)
                j = j[ok]
                if len(j) == 0:
                    continue
                have_q = all(c in j.columns for c in QC) and j[QC].notna().all().all()
                q = j[QC].to_numpy().T if have_q else None
                naive = ctx[uid].iloc[-1]
                a = metrics.arena_series_scores(j.yp.to_numpy(), j.value.to_numpy(), q, naive)
                jf = j[j.yf.notna()]
                b = metrics.arena_series_scores(jf.yf.to_numpy(), jf.value.to_numpy(), q[:, j.yf.notna().to_numpy()] if q is not None else None, naive) if len(jf) else dict(mase=np.nan, sql=np.nan)
                rows.append((reg_start, rid, m, reg, a["mase"], a["sql"], a["mae"], a["mae_naive"], have_q, len(j), b["mase"], b["sql"]))
        if i % 10 == 0:
            print(f"def {defn} round {i}/{len(rd)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["day", "round_id", "model", "series", "mase", "sql", "mae", "mae_naive", "has_q", "n", "mase_vs_final", "sql_vs_final"])
    df.to_parquet(f"{ROOT}/data/cache/field_def{defn}.parquet", index=False)
    np.savez_compressed(f"{ROOT}/data/cache/rounds_def{defn}.npz", days=np.array(days), round_ids=np.array(rids), uids=np.array(uids))
    print("done", defn, df.shape, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for d in [int(x) for x in sys.argv[1].split(",")]:
        run(d)
