"""Train the frozen recipe: python -m tsb.run_final configs/final.json --stage val|test [--parallel 2]
stage val  -> training cutoff 2025-12-31 23:45 (models used for VAL-F and the spread calibration)
stage test -> training cutoff 2026-02-06 23:45 (models evaluated once on TEST-F and exported)"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from tsb import ROOT

CUTOFF = {"val": "2025-12-31T23:45", "test": "2026-02-06T23:45"}

def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=f"{ROOT}/src")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--stage", required=True, choices=["val", "test"])
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    jobs = []
    for fkey, c in cfg["nets"].items():
        for seed in range(c["seeds"]):
            tag = f"{cfg['name']}_{a.stage}_{fkey}_s{seed}"
            out = f"{ROOT}/runs/{tag}.pt"
            if os.path.exists(out):
                continue
            cmd = [sys.executable, "-m", "tsb.train", "--freq", str(c["freq"]), "--steps", str(c["steps"]), "--hid", str(c["hid"]), "--emb", str(c["emb"]),
                   "--J", str(c["J"]), "--dmeans", str(c["dmeans"]), "--seed", str(seed), "--train-end", CUTOFF[a.stage],
                   "--eval-every", "1000000", "--threads", "1", "--out", out] + (["--no-hol"] if c.get("no_hol") else [])
            jobs.append((cmd, f"{ROOT}/logs/{tag}.log"))
    print(len(jobs), "training jobs", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for rc in ex.map(lambda j: run(*j), jobs):
            print("job done rc", rc, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
