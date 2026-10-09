# gridload-compact

A compact probabilistic load forecaster for the two SMARD net-load challenges of TS-Arena (24 h at 15 min, 72 h at 1 h).
Two small MLP quantile nets, 160,278 parameters in total, 144 kB and 150 kB as int8 ONNX, about 0.4 to 1.2 ms per series on one CPU thread.
Self-evaluated on the TS-Arena Archive; not an official ranking.

## What it is

- **Two MLP quantile nets**, one per challenge (15 min x 96 steps, 1 h x 72 steps).  Each net is a context encoder plus a shared per-step head (hidden width 64) that emits nine monotone deciles
  per step; the point forecast is the median.  Each frequency is a **3-seed ensemble** whose quantile curves are averaged inside one ONNX graph.
  Parameters: 25,817 per net at 15 min and 27,609 hourly; 77,451 and 82,827 per ensemble; **160,278 in total**.
- A post-hoc scalar on the decile spread (1.05 at 15 min, 1.20 hourly) was chosen on validation data only.
- **ONNX**: int8 (dynamic quantization) 144,175 B and 149,554 B; fp32 344,362 B and 365,869 B.
- **Service** (`submission/model-services/gridload-compact`): numpy feature pipeline, ONNX Runtime on one thread, no torch, FastAPI `POST /predict`.  A clock-signature gate compares the daily profile of the
  context with a fixed reference profile; anything that is not a SMARD net-load series on the platform clock goes to a generic seasonal fallback, for which no claim is made.
- Input: the last 1,000 points of the series.  Public holiday calendars (DE federal and state, AT) are baked into `weights/hol_table.npz` and derived from timestamps only (python `holidays`, MIT).
- Training data: SMARD net-load history (revised values) from 2018-01 up to the cutoff 2026-02-06 23:45.  No third-party model weights.

## Results (self-evaluated)

Self-evaluated on the TS-Arena Archive, window TEST-F (52 rounds, 2026-02-08 .. 2026-03-31; training cutoff 2026-02-06 23:45; configuration frozen and each variant evaluated once, pre-registered).
Headline variant = the service code path with the int8 weights, scored with the platform's arena MASE and SQL against the 32 reference models of the Archive's field:

- **Definition 2 (24 h at 15 min): lower mean MASE than every reference model on the full window.**  MASE 0.217 [0.173, 0.273] against 0.251 [0.184, 0.327] for the best reference model of the window, visiontspp-large (307 M parameters,
  about 4,000 times larger); per-round difference -0.034 [-0.059, -0.009]; SQL 0.173 against 0.201, difference -0.028 [-0.046, -0.011].  **Caveat that belongs to the claim:** the interval excludes 0 over
  the full window, but over the first 35 rounds (02-08 .. 03-14, which the live board mostly reproduces; 03-06 .. 03-08 are partly re-scored there) it does not (-0.012 [-0.026, +0.006]); the difference is carried by the 17 archive-only late-March rounds
  (-0.070 [-0.111, -0.021] against that window's best reference, flowstate).
- **Definition 5 (72 h at 1 h): a tie, not a win.**  MASE 0.185 [0.161, 0.214] against 0.187 [0.157, 0.226] for chronos-2 (120 M parameters, about 1,400 times larger); difference -0.002 [-0.015, +0.007];
  SQL 0.148 against 0.150, difference -0.002 [-0.014, +0.006].  The model matches the best reference at a fraction of its size; no win is claimed.

Scoring is the platform's arena MASE (MAE of the forecast divided by the MAE of the last-context-value persistence forecast over the same horizon) and SQL (scaled quantile loss, same scale),
reimplemented in `src/tsb/metrics.py` from the MIT backend of TS-Arena.  This is a self-obtained evaluation on the TS-Arena Archive and **not an official ranking**; no first-place or state-of-the-art claim is made.

### By window

| def | window | metric | rounds | best reference (params M) | its mean [CI] | service int8 mean [CI] | diff vs best ref [CI] | win-rate |
|---|---|---|---|---|---|---|---|---|
| 2 | 2026-02-08 .. 2026-03-31 | MASE | 52 | visiontspp-large (307) | 0.251 [0.184, 0.327] | 0.217 [0.173, 0.273] | -0.034 [-0.059, -0.009] | 62% |
| 2 | 2026-02-08 .. 2026-03-31 | SQL | 52 | visiontspp-large (307) | 0.201 [0.148, 0.262] | 0.173 [0.135, 0.220] | -0.028 [-0.046, -0.011] | 58% |
| 2 | 2026-02-08 .. 2026-03-14 | MASE | 35 | visiontspp-large (307) | 0.184 [0.164, 0.201] | 0.172 [0.161, 0.180] | -0.012 [-0.026, +0.006] | 49% |
| 2 | 2026-02-08 .. 2026-03-14 | SQL | 35 | visiontspp-large (307) | 0.149 [0.132, 0.164] | 0.135 [0.127, 0.143] | -0.014 [-0.026, +0.000] | 46% |
| 2 | 2026-03-15 .. 2026-03-31 | MASE | 17 | flowstate (9) | 0.381 [0.274, 0.472] | 0.311 [0.237, 0.360] | -0.070 [-0.111, -0.021] | 59% |
| 2 | 2026-03-15 .. 2026-03-31 | SQL | 17 | moirai-2-small (14) | 0.299 [0.258, 0.315] | 0.252 [0.186, 0.294] | -0.048 [-0.072, -0.021] | 71% |
| 5 | 2026-02-08 .. 2026-03-31 | MASE | 52 | chronos-2 (120) | 0.187 [0.157, 0.226] | 0.185 [0.161, 0.214] | -0.002 [-0.015, +0.007] | 48% |
| 5 | 2026-02-08 .. 2026-03-31 | SQL | 52 | chronos-2 (120) | 0.150 [0.126, 0.183] | 0.148 [0.129, 0.172] | -0.002 [-0.014, +0.006] | 42% |
| 5 | 2026-02-08 .. 2026-03-14 | MASE | 35 | chronos-2 (120) | 0.163 [0.144, 0.181] | 0.168 [0.155, 0.181] | +0.005 [-0.001, +0.010] | 46% |
| 5 | 2026-02-08 .. 2026-03-14 | SQL | 35 | chronos-2 (120) | 0.130 [0.114, 0.145] | 0.134 [0.124, 0.144] | +0.004 [-0.001, +0.010] | 37% |
| 5 | 2026-03-15 .. 2026-03-31 | MASE | 17 | moirai-2-small (14) | 0.229 [0.179, 0.280] | 0.219 [0.168, 0.265] | -0.009 [-0.038, +0.006] | 59% |
| 5 | 2026-03-15 .. 2026-03-31 | SQL | 17 | moirai-2-small (14) | 0.185 [0.143, 0.228] | 0.177 [0.135, 0.213] | -0.008 [-0.034, +0.007] | 53% |

"diff" is mean(ours - best reference) per round; a win needs an interval that excludes 0.  Definition 2: win over the full window, win on 03-15 .. 03-31, no win on 02-08 .. 03-14 (the SQL interval
reaches +0.000).  Definition 5: no win in any window.  The Elo values of the evidence files (platform procedure, K=4) are not quoted because four variants of ours are inserted into the field at once.

### All 16 pre-registered TEST-F evaluations (each run once)

#### definition 2 (24 h at 15 min)

| variant | MASE | SQL | cover 80 % | rounds | missing | routing |
|---|---|---|---|---|---|---|
| service code path, int8 (headline) | 0.2173 | 0.1729 | 0.770 | 52 | 9 | {'specialist': 513, 'fallback': 7} |
| ensemble, torch fp32 | 0.2154 | 0.1706 | 0.767 | 52 | 9 |  |
| ensemble, onnxruntime fp32 | 0.2154 | 0.1706 | 0.767 | 52 | 9 |  |
| ensemble, onnxruntime int8 | 0.2137 | 0.1695 | 0.767 | 52 | 9 |  |
| single net, seed 0 | 0.2149 | 0.1703 | 0.758 | 52 | 9 |  |
| single net, seed 1 | 0.2172 | 0.1723 | 0.764 | 52 | 9 |  |
| single net, seed 2 | 0.2162 | 0.1713 | 0.767 | 52 | 9 |  |
| single net, seed 0, no holiday features | 0.2191 | 0.1746 | 0.771 | 52 | 9 |  |

Seeds (single nets): MASE 0.2161 +- 0.0012 (sd over 3 seeds), SQL 0.1713 +- 0.0010

#### definition 5 (72 h at 1 h)

| variant | MASE | SQL | cover 80 % | rounds | missing | routing |
|---|---|---|---|---|---|---|
| service code path, int8 (headline) | 0.1848 | 0.1480 | 0.856 | 52 | 0 | {'specialist': 513, 'fallback': 7} |
| ensemble, torch fp32 | 0.1834 | 0.1468 | 0.858 | 52 | 0 |  |
| ensemble, onnxruntime fp32 | 0.1834 | 0.1468 | 0.858 | 52 | 0 |  |
| ensemble, onnxruntime int8 | 0.1832 | 0.1468 | 0.858 | 52 | 0 |  |
| single net, seed 0 | 0.1843 | 0.1475 | 0.861 | 52 | 0 |  |
| single net, seed 1 | 0.1843 | 0.1477 | 0.861 | 52 | 0 |  |
| single net, seed 2 | 0.1834 | 0.1466 | 0.846 | 52 | 0 |  |
| single net, seed 0, no holiday features | 0.1904 | 0.1541 | 0.858 | 52 | 0 |  |

Seeds (single nets): MASE 0.1840 +- 0.0005 (sd over 3 seeds), SQL 0.1473 +- 0.0006

- Routing of the headline: 7 of 520 (round, series) pairs per definition went to the seasonal fallback (Amprion on 2026-03-25; LU and Creos on 2026-03-29 .. 03-31); all other scorable pairs equal the
  onnxruntime int8 ensemble to 2e-7 in MASE.  The fallback costs +0.0036 (definition 2) and +0.0015 (definition 5) MASE against the onnxruntime int8 ensemble.
- Ensemble, torch against onnxruntime fp32: max |difference| 1.5e-7 / 2.8e-7 MASE over 520 pairs per definition.  int8 against fp32 ensemble (onnxruntime, paired per round): -0.0017 [-0.0029, -0.0005]
  MASE for definition 2, -0.0002 [-0.0008, +0.0005] for definition 5: quantization costs nothing measurable.
- Holiday features (no-holiday minus seed 0, paired): definition 2 +0.0042 [-0.0015, +0.0105]; definition 5 +0.0061 [+0.0020, +0.0115] (helps at 1 h; at 15 min inside the noise).
- Definition 2 has 9 unscorable pairs for every model (round 6363, 2026-03-25, has truth for Amprion only), so "missing 9" is not a model failure; 93 further pairs of definition 2 and 76 of definition 5 are scored
  on a truncated horizon because the archive truth is incomplete in late March, for all models alike.

### Limits that belong to the claim

- Specialist: only the ten SMARD net-load series on the platform's current timestamp convention; everything else gets a generic seasonal fallback and no claim is made for it.
- Specialist versus zero-shot generalists: a small model trained on the same SMARD series (cutoff before the window) is compared with foundation models that saw none of it.  The fair reading is "a specialist this
  small is competitive or better on its own series", not "better architecture".
- The definition-2 win is concentrated in the 17 archive-only late-March rounds (stale contexts, partial truth, DST change); the headline path sent 7 of 520 pairs per definition to its seasonal fallback there
  (the gate reasons are not logged).
- The intervals are the platform's ISO-week block bootstrap: 9 blocks in the full window, 6 and 4 in the two sub-windows, so they are crude.  An i.i.d. round bootstrap and a moving-block (7-round) bootstrap give the
  same conclusions, except that the moving-block interval of definition 5 on 02-08..03-14 just excludes 0 in chronos-2's favour (lower bound +0.0004) (TEST-LOG post-run note 9).  On definition 2 our MASE is below
  visiontspp-large's on every series but 50Hertz (equal).
- The calendar (DE federal and state holidays, AT) is baked into the weights and derived from timestamps only; the no-holiday ablation is reported above.
- The archive and the live board differ for 2026-03-06..08 (partly) and 2026-03-15..31 (`notes/revisions.md`, item 7), so TEST-F is also given for 02-08..03-14 and 03-15..03-31.  Scoring reproduces the official
  per-series MASE exactly for 87.8 % of the definition-2 pairs (`INTEGRITY.md`).
- The Elo values in the evidence files (`results/bench-timeseries-b/out/test_f_*.md`) are not quoted: four of our variants are inserted into the field at once.
- Training is not bit-reproducible across hardware; the SHA-256 of the checkpoints behind the numbers are in `configs/final.json`.

## Protocol

- **Pre-registration.**  Selection rule, ensemble, spread grid, the eight TEST-F variants per challenge and the headline variant (the service code path, int8) were written down before any sweep, validation-stage or
  test-stage number of the Kaggle phase existed: `TEST-LOG.md` (Frozen protocol, Addenda 1 and 2) and the machine-readable `configs/prereg.json`.
- **Cutoffs.**  Training targets end strictly before the evaluation window: sweep models 2025-06-30 23:45, validation-stage models 2025-12-31 23:45, test-stage models 2026-02-06 23:45; the first TEST-F round
  registers on 2026-02-08.  The spread scalar uses validation rounds only (VAL-F, 2026-01-01 .. 02-07).
- **Freeze.**  `configs/final.json` (selection, spread, seeds, SHA-256 of the eight test-stage checkpoints) was committed on 2026-10-08 20:35 UTC, before the TEST-F kernel was built; the kernel verifies the hashes
  first and stops otherwise.
- **One run per variant.**  8 variants x 2 challenges = 16 TEST-F evaluations, each logged at start and at completion in `TEST-LOG.md` (copied unedited from the kernel output
  `results/bench-timeseries-b/out/TEST-LOG-entries.md`).  The TEST-F kernel was pushed once.
- **Re-run.**  The frozen kernel was pushed a second time, unchanged, as `bench-timeseries-b-verify` (logged in `TEST-LOG.md` as a further evaluation, no selection): all six TEST-F evidence tables
  (`results/bench-timeseries-b-verify/out/test_f_*.md`) are byte-identical to the original run's.
- **Faithfulness.**  The service code path scores like the research harness (torch against ONNX Runtime fp32 differ by at most 2.8e-7 in MASE over 520 pairs per challenge); details and the full checklist in
  `INTEGRITY.md`.
- No TEST-F truth, context or forecast of any model enters training, calibration or selection.  The reference models' TEST-F scores were printed during an earlier analysis of the bar
  to beat (not of our model); see "No leakage" in `INTEGRITY.md`.

## Size and latency

| | 15 min (definition 2) | hourly (definition 5) |
|---|---|---|
| parameters per net / ensemble of 3 | 25,817 / 77,451 | 27,609 / 82,827 |
| ONNX ensemble, fp32 / int8 | 344,362 B / 144,175 B | 365,869 B / 149,554 B |
| spread (post-hoc scalar on the deciles, chosen on VAL-F) | 1.05 | 1.20 |
| latency, int8, model only, p50 (Kaggle CPU, AMD EPYC 7B12, 1 thread, batch 1, n=300) | 0.27 ms | 0.23 ms |
| latency, int8, end to end per series (numpy features + model), p50 | 1.16 ms | 1.23 ms |
| latency, fp32, model only / end to end, p50 | 0.28 ms / 1.18 ms | 0.24 ms / 1.27 ms |
| latency, Apple M4 (macOS), 1 thread, batch 1, int8, model only / end to end, p50 | 0.20 ms / 0.44 ms | 0.16 ms / 0.44 ms |
| latency, Apple M4 (macOS), fp32, model only / end to end, p50 | 0.14 ms / 0.39 ms | 0.12 ms / 0.39 ms |

Sources: `results/bench-timeseries-b/out/bench_kaggle_cpu_{15,60}.json` (Kaggle CPU) and `results/bench-timeseries-b/m4_latency_{15,60}.json` (Apple M4, 2026-10-08; median of 300 calls after 20 warm-ups,
numpy features included in the end-to-end figure, measured with a 1-minute load average of 3.7 to 3.9).  On the M4 the int8 graphs are slightly slower than fp32: dynamic quantization is for size, not for speed.

## Run the service

```bash
cd submission/model-services/gridload-compact
pip install -r requirements.txt pytest httpx        # fastapi, uvicorn, pydantic, numpy, onnxruntime (pinned) plus the test tools
python -m pytest tests -q                           # 41 service tests
uvicorn app.main:app --port 8000
```

`POST /predict` takes `{"history": [{"ts": "2026-03-10T07:45:00.000Z", "value": 12345.0}, ...], "horizon": 96, "freq": "15min"}` (`"freq": "h"` with `"horizon": 72` for the hourly challenge; a list of such lists is a
batch) and returns `{"prediction": [{"ts": ..., "value": ..., "probabilistic_values": {"q_0.1": ..., ..., "q_0.9": ...}}, ...]}`; `GET /health`.  `GRIDLOAD_PRECISION=int8` (default) or `fp32` selects the graphs,
`GRIDLOAD_WEIGHTS` points to another weights directory.  The tests use the contract checker of the TS-Arena models repository (`submission/model-services/tests/quantile_contract.py`, a verbatim copy, see `NOTICE`);
`run_harness.py` next to it posts synthetic histories to a running service:

```bash
python submission/model-services/tests/run_harness.py --url http://localhost:8000/predict --horizon 96 --freq 15min --mode batch
```

The service README (`submission/model-services/gridload-compact/README.md`) and the Dockerfile are the files of the TS-Arena pull request, unchanged.

## Reproduce

Training, evaluation, export and service tests ran in private Kaggle CPU kernels (`scripts/reproduce.sh`; the kernels themselves are not published).  The chain, as run:

```bash
python3 scripts/make_kernel.py a1 [--with-ref]  &&  python3 scripts/kwait.py bench-timeseries-a1     # env pins, data fetch, tables, reproducibility gate, smoke tests
python3 scripts/make_kernel.py a2 --sources sankalpsthakur/bench-timeseries-a1  &&  python3 scripts/kwait.py bench-timeseries-a2    # sweep, pre-registered selection, stage 2, calibration
python3 scripts/freeze.py results/bench-timeseries-a2/out/freeze_candidate.json                          # then commit configs/final.json and the freeze note
python3 scripts/make_kernel.py b --sources sankalpsthakur/bench-timeseries-a1,sankalpsthakur/bench-timeseries-a2  &&  python3 scripts/kwait.py bench-timeseries-b
python3 scripts/package_submission.py --weights results/bench-timeseries-b/out/service_weights
```

- **a1** pins the environment (Python 3.13, numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, torch 2.11.0+cpu, onnxruntime 1.30.0, holidays 0.106), fetches SMARD (public API) and the TS-Arena Archive (Hugging Face,
  pinned revision `72c6cb0fe11a7c0349b063c74129ff73a87542cc`), and runs the **reproducibility gate**: the v0 development checkpoint, re-evaluated on VAL-F (definition 2) inside the kernel, must reproduce the
  per-(round, series) MASE and SQL of the local run.  It did, for all 380 pairs (max |difference| 1.2e-7 in MASE and 9.4e-8 in SQL; mean MASE 0.189901 against 0.1899):
  `results/bench-timeseries-a1/out/repro_v0_eval.json`, `faith_val_def2_v0_15_s0_kernel.parquet`, `dev_val_f_def2_kernel.md`.  Part 2 of the gate (the rebuilt field cache has the pre-registered row counts
  28,448 / 28,736 and equals the local copy) runs inside a2 (`results/bench-timeseries-a2/out/field_manifest_copy.json`).
- **a2** builds the field cache, runs the sweep and the pre-registered selection (`configs/prereg.json`), trains the validation-stage and test-stage ensembles, calibrates the spread on VAL-F and writes
  `freeze_candidate.json`.  **b** verifies the frozen checkpoint hashes, exports the ONNX graphs, evaluates TEST-F once per pre-registered variant, builds the evidence tables, runs the service tests and the
  harness, and measures CPU latency.  Summaries: `python3 scripts/summarize.py results/bench-timeseries-a2/out`, `python3 scripts/tables.py results/bench-timeseries-b/out test`.
- Training is not bit-reproducible across hardware: a re-run of a2 gives the same rule and seeds but close, not identical, checkpoints.  To re-verify the TEST-F numbers of the shipped weights, run b against the a2
  output that `configs/final.json` hashes refer to.

What a clone of this repository cannot do without changes:

- The kernel ids use the author's Kaggle namespace (`sankalpsthakur/...` in `scripts/make_kernel.py`, `scripts/kwait.py`, `scripts/reproduce.sh` and the `--sources` arguments above); replace it with your own in all of them.
- `scripts/kwait.py` takes one of four CPU-slot directories `../locks/kaggle-cpu-{1..4}` (create `../locks` first) and needs the `kaggle` CLI logged in.
- `--with-ref` embeds the local v0 development checkpoint from `assets/ref/`, which is not part of this repository; part 1 of the gate cannot be re-run from here, its recorded result is above.
- `scripts/package_submission.py` rebuilds the TS-Arena patch in a local clone of DAG-UPB/ts-arena-models at `ref/ts-arena-models` (base `cbbb347`).
- The modules of `src/tsb` import as `tsb` with `PYTHONPATH=src`.
- `TEST-LOG.md` and `INTEGRITY.md` are unedited copies.  They mention files that are not part of this repository (`PLAN.md`, `bin/guard.sh`) and commit ids (`b9f6339`, `ad57898`, `d4c7c23`, ...) of the private workspace
  repository this snapshot was taken from (its head was `400d9c5`); those ids do not exist in this repository's history.

## Repository layout

| path | what |
|---|---|
| `src/tsb/` | research code: data fetch, features, nets, training, export, scoring (`metrics.py`), evaluation (`evalf.py`), evidence tables |
| `scripts/` | kernel builder and driver (`make_kernel.py`, `kdriver.py`), waiter (`kwait.py`), freeze, packaging, summaries, `reproduce.sh` |
| `configs/` | `prereg.json` (pre-registration), `final.json` (frozen configuration with checkpoint hashes) |
| `assets/` | `clock_ref.npz` (reference daily profile of the service gate), `models.json` (public TS-Arena model metadata, used to label reference models) |
| `TEST-LOG.md`, `INTEGRITY.md` | protocol, the ledger of the 16 TEST-F evaluations, post-run notes, integrity checklist |
| `notes/revisions.md` | platform data conventions found on the way (clock, revisions, contexts, archive against live board) |
| `reports/` | development report on the validation window |
| `results/bench-timeseries-{a1,a2,b,b-verify}/out/` | tables, json summaries, per-round score files (`faith/*.parquet`), ledgers; no logs, checkpoints or caches |
| `submission/model-services/gridload-compact/` | the service exactly as in the TS-Arena pull request, with the final ONNX weights, info files, holiday table and clock reference |
| `submission/model-services/tests/` | contract checker and harness of the TS-Arena models repository (MIT), copied verbatim |
| `HF-MODEL-CARD.md` | model card for the Hugging Face repository |
| `LICENSE`, `NOTICE`, `MANIFEST.txt` | Apache-2.0 text; data and third-party attributions; `sha256  path` of every other file, sorted |

## TS-Arena submission

Offered to the TS-Arena Reference Track as a pull request to ts-arena-models: https://github.com/DAG-UPB/ts-arena-models/pull/5.  Whether it is listed there, and under which model type, is the maintainers' decision.

## Hugging Face model repository

Upload these files to the model repository: `net15.onnx`, `net15_int8.onnx`, `net60.onnx`, `net60_int8.onnx`, `net15_info.json`, `net60_info.json`, `hol_table.npz`, `clock_ref.npz` (all from
`submission/model-services/gridload-compact/weights/`), and `HF-MODEL-CARD.md` as `README.md` (Hugging Face reads the model card from `README.md`).

## Licences and attribution

- Code and model weights in this repository: Apache-2.0 (`LICENSE`), except the third-party files named in `NOTICE`.  The `gridload-compact` service directory is also contributed to the MIT-licensed ts-arena-models repository in the pull request above, where it
  is offered under that repository's MIT licence (this is what its README says).
- Training data: SMARD, Bundesnetzagentur | SMARD.de, CC BY 4.0 (https://www.smard.de/).  The weights are derived from it; changes were made.
- Evaluation data: TS-Arena-Archive (DAG-UPB/TS-Arena-Archive), CC BY 4.0.  Nothing from it, and no forecast of any other model, enters training, calibration or the released weights.
- Baked holiday table: generated with the python `holidays` package (MIT).
- Third-party files and full attributions: `NOTICE`.

## Status

Self-evaluated, not peer-reviewed.  Sankalp Thakur, October 2026.

## Licence note

Everything in this repository is Apache-2.0 (see `LICENSE`) except: the directory `submission/model-services/gridload-compact/` (service code and weights), which is also offered under the MIT licence exactly as contributed to [DAG-UPB/ts-arena-models](https://github.com/DAG-UPB/ts-arena-models/pull/5) (its own README's "this repository" refers to that MIT repository); the vendored test helpers under `submission/model-services/tests/` (MIT, ts-arena-models); and the data attributions in `NOTICE` (SMARD CC BY 4.0, TS-Arena-Archive CC BY 4.0, `holidays` MIT).

