#!/usr/bin/env python3
"""Assemble the ts-arena-models PR as a patch (nothing is pushed or published).
usage: python scripts/package_submission.py --weights results/bench-timeseries-b/out/service_weights [--params-json out/service_weights/net15_info.json ...]
Steps: reset the local clone ref/ts-arena-models to cbbb347, copy the service (code + weights), add compose/gridload-compact.yml, the include line,
the config.json entry and the README line, then `git add -N` + `git diff --binary` -> submission/ts-arena-models.patch.
Local git/file work only (no model code runs)."""
import argparse, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAM = f"{ROOT}/ref/ts-arena-models"
BASE = "cbbb347"

def git(*a, check=True):
    r = subprocess.run(["git", "-C", TAM, *a], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(a)}: {r.stderr[-400:]}")
    return r.stdout

COMPOSE = """services:
  gridload-compact:
    image: gridload-compact
    build:
      context: ../model-services/gridload-compact
      dockerfile: Dockerfile
    profiles: ["gridload-compact", "all-models"]
    labels:
      - managed.by=controller
    networks:
      - internal
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--params", type=int, default=0, help="total parameters (both frequencies, all ensemble members), informational")
    ap.add_argument("--org", default=None)
    a = ap.parse_args()
    git("reset", "-q", "--hard", BASE); git("clean", "-fdq")
    dst = f"{TAM}/model-services/gridload-compact"
    shutil.copytree(f"{ROOT}/submission/model-services/gridload-compact", dst, ignore=shutil.ignore_patterns("weights", "__pycache__", ".pytest_cache"))
    os.makedirs(f"{dst}/weights", exist_ok=True)
    for f in os.listdir(a.weights):
        shutil.copy(f"{a.weights}/{f}", f"{dst}/weights/{f}")
    if os.path.exists(f"{ROOT}/submission/model-services/gridload-compact/README.md"):
        shutil.copy(f"{ROOT}/submission/model-services/gridload-compact/README.md", f"{dst}/README.md")
    open(f"{TAM}/compose/gridload-compact.yml", "w").write(COMPOSE)
    dc = open(f"{TAM}/docker-compose.yml").read()
    dc = dc.replace("  - compose/t0.yml\n", "  - compose/t0.yml\n  - compose/gridload-compact.yml\n", 1)
    open(f"{TAM}/docker-compose.yml", "w").write(dc)
    cfgp = f"{TAM}/challenge-uploads/src/config.json"
    raw = open(cfgp).read()
    cfg = json.loads(raw)
    cfg["gridload-compact"] = {"name": "sankalpsthakur/gridload-compact", "model_type": "Specialist", "model_family": "gridload-compact", "model_size": 0, "hosting": "UPB",
                               "architecture": "MLP quantile nets (3-seed ensemble per frequency)",
                               "pretraining_data": "SMARD net-load history 2018 to 2026-02 (CC BY 4.0), public data only; no third-party model weights",
                               "publishing_date": "2026-10-07", "parameters": {"additionalProp1": {}},      # same placeholder as the other entries; exact count in the PR text
                               "organization_id": None}
    indent = 4 if raw.startswith("{\n    ") else 2
    open(cfgp, "w").write(json.dumps(cfg, indent=indent, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    rd = open(f"{TAM}/README.md").read()
    rd = rd.replace("`t0`, `visionts`.", "`t0`, `gridload-compact`, `visionts`.", 1)
    open(f"{TAM}/README.md", "w").write(rd)
    git("add", "-N", ".")
    patch = git("diff", "--binary", BASE)
    os.makedirs(f"{ROOT}/submission", exist_ok=True)
    open(f"{ROOT}/submission/ts-arena-models.patch", "w").write(patch)
    print(git("diff", "--stat", BASE))

if __name__ == "__main__":
    main()
