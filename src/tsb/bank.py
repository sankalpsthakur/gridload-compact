"""Regular platform-clock grids of the 10 SMARD net-load series (float32, NaN = missing) + calendar arrays."""
import numpy as np, pandas as pd
from tsb import ROOT, data, cal

def build(freq_minutes):
    assert freq_minutes in (15, 60)
    series = {}
    for reg in data.REGIONS:
        q = data.quarter_series(reg)
        series[reg] = q if freq_minutes == 15 else data.hourly_from_quarter(q)
    start = min(s.index.min() for s in series.values()).ceil("1h")
    end = max(s.index.max() for s in series.values())
    idx = pd.date_range(start, end, freq=f"{freq_minutes}min", tz="UTC")
    X = np.full((len(data.REGIONS), len(idx)), np.nan, dtype=np.float32)
    for i, reg in enumerate(data.REGIONS):
        X[i] = series[reg].reindex(idx).to_numpy(np.float32)
    c = cal.calendar_arrays(idx)
    out = dict(X=X, t0=np.int64(idx[0].timestamp()), step=np.int64(freq_minutes * 60), regions=np.array(data.REGIONS),
               local_dow=c["local_dow"], local_hour=c["local_hour"], hol=c["hol"], doy=c["doy"], shift=c["shift"])
    np.savez_compressed(f"{ROOT}/data/cache/bank{freq_minutes}.npz", **out)
    return out

def load(freq_minutes):
    z = np.load(f"{ROOT}/data/cache/bank{freq_minutes}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}

if __name__ == "__main__":
    import sys
    for f in (15, 60):
        o = build(f)
        print(f, o["X"].shape, pd.to_datetime(int(o["t0"]), unit="s"), "nan frac", float(np.isnan(o["X"]).mean()))
