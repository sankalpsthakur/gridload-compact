#!/usr/bin/env bash
# Reproduce data, models, evaluation and the submission patch from a clean checkout.  NOTHING runs on the Mac: training, evaluation, export and service
# tests run in private Kaggle CPU kernels (owner rule: no model compute on the 16 GB Mac).  This script only builds the kernels and pushes them.
# Needs: python 3, the kaggle CLI logged in (kernels are private, named bench-timeseries-*), git.  The program's CPU slots are taken with scripts/kwait.py.
#
#   a1  environment pins (numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, holidays 0.106, onnxruntime 1.30.0), SMARD API + TS-Arena-Archive (pinned revision)
#       fetched inside the kernel, banks, holiday table, v0 reproducibility gate (only when the Mac dev checkpoint is embedded: --with-ref), smoke, service tests.
#   a2  field cache (forecast partitions, paced to the Hub rate limit, about 11 min), sweep, the pre-registered selection (configs/prereg.json), val-stage and
#       test-stage 3-seed ensembles, spread calibration on VAL-F, VAL-F run of every evaluation path, freeze_candidate.json.
#   --  freeze: python scripts/freeze.py results/bench-timeseries-a2/out/freeze_candidate.json ; commit configs/final.json ; add the freeze note to TEST-LOG.md ; commit.
#   b   verifies the frozen checkpoint hashes, exports, dry-runs, TEST-F once per pre-registered variant, evidence tables, service tests + harness, CPU latency.
#
# Training is not bit-reproducible across hardware (float32 reduction order): a re-run of a2 gives the same selection rule and seeds, close but not identical
# checkpoints.  To re-verify the TEST-F numbers of the frozen submission, run b against the a2 output that configs/final.json hashes refer to.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/make_kernel.py a1                       # add --with-ref only if assets/ref (Mac v0 checkpoint) exists
python3 scripts/kwait.py bench-timeseries-a1 --poll 120 --expect-min 25

python3 scripts/make_kernel.py a2 --sources sankalpsthakur/bench-timeseries-a1          # no --with-ref: Kaggle rejects scripts of about 4 MB (HTTP 400), 0.84 MB passes
python3 scripts/kwait.py bench-timeseries-a2 --poll 300 --expect-min 80

python3 scripts/freeze.py results/bench-timeseries-a2/out/freeze_candidate.json
# ... review, commit configs/final.json, write the freeze note in TEST-LOG.md, commit ...

python3 scripts/make_kernel.py b --sources sankalpsthakur/bench-timeseries-a1,sankalpsthakur/bench-timeseries-a2
python3 scripts/kwait.py bench-timeseries-b --poll 300 --expect-min 40

# package (local git/file work only): the ts-arena-models patch from the b kernel's service weights
python3 scripts/package_submission.py --weights results/bench-timeseries-b/out/service_weights --params "$(python3 -c "import json;print(sum(json.load(open('results/bench-timeseries-b/out/service_weights/net%s_info.json'%k))['params'] for k in (15,60)))")"
