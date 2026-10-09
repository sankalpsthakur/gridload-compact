#!/usr/bin/env python3
"""Local coordination helper (no compute): wait for a free program Kaggle CPU slot, push one kernel, poll its status, fetch the small outputs,
release the slot.  One background command per kernel.
usage: python scripts/kwait.py <slug> [--poll 300] [--pattern REGEX]
Slots: ../locks/kaggle-cpu-1..4 (mkdir is atomic); the lock is taken just before the push and removed when the kernel ends (or the push fails)."""
import argparse, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKS = os.path.join(os.path.dirname(ROOT), "locks")

def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))

def take_slot(owner_text):
    while True:
        for n in (1, 2, 3, 4):
            d = os.path.join(LOCKS, f"kaggle-cpu-{n}")
            try:
                os.mkdir(d)
            except FileExistsError:
                continue
            open(os.path.join(d, "owner"), "w").write(owner_text + "\n")
            return d
        print(utc(), "all four slots busy, waiting", flush=True)
        time.sleep(20)

def kaggle(*args, check=False):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} -> {r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug"); ap.add_argument("--poll", type=int, default=300); ap.add_argument("--pattern", default=r"^(out|ckpt)/.*")
    ap.add_argument("--expect-min", type=int, default=60, help="expected run time for the lock note")
    a = ap.parse_args()
    ref = f"sankalpsthakur/{a.slug}"
    res = os.path.join(ROOT, "results", a.slug); os.makedirs(res, exist_ok=True)
    lock = take_slot(f"timeseries {ref} {utc()} queued, expected run about {a.expect_min} min")
    print(utc(), "took", lock, flush=True)
    try:
        r = kaggle("kernels", "push", "-p", os.path.join(ROOT, "kaggle", a.slug))
        print(utc(), "push rc", r.returncode, r.stdout.strip(), r.stderr.strip()[-500:], flush=True)
        if r.returncode != 0:
            return 2
        t0 = time.time()
        open(os.path.join(lock, "owner"), "w").write(f"timeseries {ref} {utc(t0)} running, expected end about {utc(t0 + 60 * a.expect_min)}\n")
        live = subprocess.Popen(["kaggle", "kernels", "logs", "-f", ref], stdout=open(os.path.join(res, "live.log"), "w"), stderr=subprocess.STDOUT)
        last = ""
        while True:
            time.sleep(a.poll)
            s = kaggle("kernels", "status", ref)
            line = (s.stdout + s.stderr).strip().replace("\n", " ")
            if line != last:
                print(utc(), line, flush=True); last = line
            low = line.lower()
            if any(k in low for k in ("complete", "error", "cancel", "failed")):
                break
        live.terminate()
        time.sleep(5)
        # size guard: the owner's rule is no fetch over 50 MB; list first (names and sizes), refuse if the pattern would match more
        try:
            lst = kaggle("kernels", "files", ref, "--format", "json", "--page-size", "200")
            open(os.path.join(res, "files.json"), "w").write(lst.stdout)
            import json as _json
            items = _json.loads(lst.stdout)
            rows = items if isinstance(items, list) else items.get("files", [])
            tot = 0
            for it in rows:
                nm = it.get("name") or it.get("fileName") or ""
                if re.search(a.pattern, nm):
                    tot += int(it.get("size") or it.get("totalBytes") or 0)
            print(utc(), "files listed:", len(rows), "matched bytes:", tot, flush=True)
            if tot > 45 * 1024 * 1024:
                print(utc(), "REFUSING to fetch: matched output exceeds 45 MB; fetch by hand with a tighter pattern", flush=True)
                return 3
        except Exception as e:
            print(utc(), "size guard could not list files:", type(e).__name__, e, flush=True)
        o = kaggle("kernels", "output", ref, "-p", res, "-o", "--file-pattern", a.pattern)
        print(utc(), "output rc", o.returncode, (o.stdout + o.stderr).strip()[-800:], flush=True)
        # full file listing for the record (names and sizes only)
        f = kaggle("kernels", "files", ref, "--page-size", "200")
        open(os.path.join(res, "files.txt"), "w").write(f.stdout + f.stderr)
        return 0
    finally:
        try:
            os.remove(os.path.join(lock, "owner")); os.rmdir(lock)
            print(utc(), "released", lock, flush=True)
        except Exception as e:
            print("lock release problem", e, flush=True)

if __name__ == "__main__":
    sys.exit(main())
