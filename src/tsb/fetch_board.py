"""Fetch public TS-Arena dashboard data (no key needed through the public site proxy):
round lists per definition, per-round leaderboards (official per-series MASE/SQL of every model)
and per-round series metadata (per-series context end).  Read-only, sequential, cached on disk."""
import json, os, sys, time, requests
from tsb import ROOT
from tsb.util import disk_ok, free_gb
OUT = f"{ROOT}/data/board"
S = requests.Session()
BASE = "https://ts-arena.live/api/v1"

def get(path, timeout=120, tries=4):
    for a in range(tries):
        try:
            r = S.get(f"{BASE}/{path}", timeout=timeout)
            if r.status_code == 200:
                return r.json()
            print("HTTP", r.status_code, path, flush=True)
        except Exception as e:
            print("ERR", type(e).__name__, path, flush=True)
        time.sleep(2 + 3 * a)
    return None

def rounds(defn):
    dst = f"{OUT}/rounds_def{defn}.json"
    if os.path.exists(dst):
        return json.load(open(dst))
    items, page = [], 1
    while True:
        d = get(f"definitions/{defn}/rounds?status=completed&page={page}&page_size=100")
        if d is None:
            raise SystemExit("rounds listing failed")
        items += d["items"]
        print(f"def {defn} page {page}/{d['pagination']['total_pages']} -> {len(items)}", flush=True)
        if not d["pagination"]["has_next"]:
            break
        page += 1
    json.dump(items, open(dst, "w"))
    return items

def round_blobs(rid):
    for kind in ("leaderboard", "series"):
        dst = f"{OUT}/r{rid}_{kind}.json"
        if os.path.exists(dst) and os.path.getsize(dst) > 10:
            continue
        d = get(f"rounds/{rid}/{kind}")
        if d is not None:
            json.dump(d, open(dst, "w"))
        time.sleep(0.2)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    defs = [int(x) for x in sys.argv[1].split(",")]
    since = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    for dfn in defs:
        its = rounds(dfn)
        sel = [r for r in its if r["registration_start"][:10] >= since]
        print(f"def {dfn}: {len(its)} rounds listed, fetching {len(sel)} since {since}", flush=True)
        for i, r in enumerate(sorted(sel, key=lambda r: r["registration_start"])):
            if not disk_ok(34.0):
                raise SystemExit(f'STOP: free disk below 34 GB: {free_gb():.1f}')
            round_blobs(r["id"])
            if i % 20 == 0:
                print(f"  def {dfn} {i}/{len(sel)}", flush=True)
    print("DONE", flush=True)
