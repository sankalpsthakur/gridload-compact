"""Fetch raw SMARD (Bundesnetzagentur, CC BY 4.0) grid-load history for the 10 series used by
the TS-Arena 'SMARD Net Load' challenges. Public API, no key. Output: data/smard/<region>.parquet
with columns ts_ms (int64 UTC epoch ms, as published) and value (MWh per quarter hour)."""
import json, os, sys, time, concurrent.futures as cf
import pandas as pd, requests
from tsb import ROOT
from tsb.util import disk_ok, free_gb

REGIONS = ["AT", "DE-LU", "DE", "LU", "50Hertz", "APG", "Amprion", "Creos", "TenneT", "TransnetBW"]
FILTER, RES = "410", "quarterhour"
BASE = "https://www.smard.de/app/chart_data"
OUT = f"{ROOT}/data/smard"
sess = requests.Session()
sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3))

def index(region):
    r = sess.get(f"{BASE}/{FILTER}/{region}/index_{RES}.json", timeout=30)
    r.raise_for_status()
    return r.json()["timestamps"]

def block(region, ts):
    url = f"{BASE}/{FILTER}/{region}/{FILTER}_{region}_{RES}_{ts}.json"
    for a in range(5):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                return r.json().get("series", [])
            if r.status_code == 404:
                return []
        except Exception:
            pass
        time.sleep(1 + a)
    raise RuntimeError(f"failed {url}")

def fetch_region(region, since_ms):
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/{region}.parquet"
    have = pd.read_parquet(dst) if os.path.exists(dst) else None
    stamps = [t for t in index(region) if t >= since_ms]
    if have is not None and len(have):
        # always refetch the last 3 blocks (data is still being finalised) and anything missing
        last = have.ts_ms.max()
        stamps = [t for t in stamps if t >= last - 3 * 7 * 86400 * 1000]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for series in ex.map(lambda t: block(region, t), stamps):
            rows.extend((int(p[0]), p[1]) for p in series)
    df = pd.DataFrame(rows, columns=["ts_ms", "value"]).dropna()
    if have is not None:
        df = pd.concat([have, df])
    df = df.drop_duplicates("ts_ms", keep="last").sort_values("ts_ms").reset_index(drop=True)
    df.to_parquet(dst, index=False)
    return region, len(df), pd.to_datetime(df.ts_ms.min(), unit="ms"), pd.to_datetime(df.ts_ms.max(), unit="ms")

def _one(rg, since):
    if not disk_ok(34.0):
        print('STOP: free disk below 34 GB', free_gb(), flush=True); return None
    if os.path.exists(f'{OUT}/{rg}.parquet') and len(sys.argv) > 2 and sys.argv[2] == 'skip-done':
        return None
    t0 = time.time()
    res = fetch_region(rg, since)
    print(*res, f"{time.time()-t0:.0f}s", flush=True)
    return res

if __name__ == "__main__":
    since = int(pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2018-01-01", tz="UTC").timestamp() * 1000)
    workers = int(os.environ.get("TSB_REGION_WORKERS", "1"))     # regions fetched concurrently (each uses 4 block workers)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda rg: _one(rg, since), REGIONS))
