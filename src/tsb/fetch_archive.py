"""Threaded downloader for selected partitions of the public TS-Arena-Archive (CC BY 4.0), pinned to one revision.
The file list comes from the HF API at that revision (cached in data/raw/hf_archive_meta.json).
usage: python -m tsb.fetch_archive smard_load_challenge_24h_15min,smard_load_challenge_72h_1h [--only context_data,ground_truth]"""
import json, os, re, sys, threading, time, concurrent.futures as cf, requests
from tsb import ROOT
OUT = f"{ROOT}/data/archive"
REV = os.environ.get("TSB_HF_REV", "72c6cb0fe11a7c0349b063c74129ff73a87542cc")   # DAG-UPB/TS-Arena-Archive, lastModified 2026-06-18
META = f"{ROOT}/data/raw/hf_archive_meta.json"

def ensure_meta():
    if os.path.exists(META):
        return json.load(open(META))
    os.makedirs(os.path.dirname(META), exist_ok=True)
    for url in (f"https://huggingface.co/api/datasets/DAG-UPB/TS-Arena-Archive/revision/{REV}?blobs=true",
                "https://huggingface.co/api/datasets/DAG-UPB/TS-Arena-Archive?blobs=true"):
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200:
                    m = r.json()
                    if m.get("sha") != REV:
                        print("WARNING: archive sha", m.get("sha"), "!= pinned", REV, flush=True)
                    json.dump(m, open(META, "w"))
                    return m
            except Exception as e:
                print("meta retry", type(e).__name__, flush=True)
            time.sleep(2 + 2 * attempt)
    raise SystemExit("cannot fetch the archive file list")

class Limiter:
    """Client-side pacing: the Hub allows about 3000 resolver requests per 5 minutes for anonymous clients (ratelimit header)."""
    def __init__(self, rate):
        self.interval = 1.0 / rate; self.lock = threading.Lock(); self.next = time.time()
    def wait(self):
        with self.lock:
            now = time.time(); t = max(self.next, now); self.next = t + self.interval
        if t > now:
            time.sleep(t - now)

def main():
    schedules = sys.argv[1].split(",")
    only = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
    meta = ensure_meta()
    files = [(s["rfilename"], s.get("size") or 0) for s in meta["siblings"]
             if any(f"schedule_id={sc}/" in s["rfilename"] for sc in schedules)
             and (only is None or s["rfilename"].split("/")[0] in only)]
    print(len(files), "files", sum(z for _, z in files) / 1e6, "MB; revision", meta.get("sha"), flush=True)
    base = f"https://huggingface.co/datasets/DAG-UPB/TS-Arena-Archive/resolve/{meta.get('sha') or REV}/"
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=2)
    sess.mount("https://", adapter)
    lim = Limiter(float(os.environ.get("TSB_FETCH_RATE", "8.5")))

    def get(item):
        path, size = item
        dst = os.path.join(OUT, path)
        if os.path.exists(dst) and (size == 0 or os.path.getsize(dst) == size):
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(12):
            lim.wait()
            try:
                r = sess.get(base + path, timeout=60)
                if r.status_code == 200:
                    tmp = dst + ".part"
                    with open(tmp, "wb") as f:
                        f.write(r.content)
                    os.replace(tmp, dst)
                    return len(r.content)
                if r.status_code == 429:
                    m = re.search(r"t=(\d+)", r.headers.get("ratelimit", ""))
                    wait = float(r.headers.get("retry-after") or (float(m.group(1)) + 1 if m else 5))
                    time.sleep(min(wait, 120)); continue
                time.sleep(1 + attempt)
            except Exception:
                time.sleep(1 + attempt)
        return -1

    t0 = time.time(); done = 0; nbytes = 0; failed = []
    with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("TSB_FETCH_WORKERS", "16"))) as ex:
        for item, n in zip(files, ex.map(get, files)):
            done += 1
            if n < 0: failed.append(item)
            else: nbytes += n
            if done % 500 == 0:
                print(f"{done}/{len(files)} {nbytes/1e6:.0f}MB {time.time()-t0:.0f}s failed={len(failed)}", flush=True)
    if failed:                                          # repair pass, gently
        print("repair pass for", len(failed), "files", flush=True)
        again = []
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            for item, n in zip(failed, ex.map(get, failed)):
                if n < 0: again.append(item)
        failed = again
    print(f"DONE {done} files, {nbytes/1e6:.0f} MB new, {time.time()-t0:.0f}s, failed={len(failed)}", flush=True)
    for path, _ in failed[:10]:
        print("FAILED", path, flush=True)
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
