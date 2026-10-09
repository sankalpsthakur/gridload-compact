---
license: apache-2.0
library_name: onnx
tags:
- time-series-forecasting
- onnx
- edge
- probabilistic-forecasting
- electricity-load
---

# gridload-compact

A compact probabilistic load forecaster for the two SMARD net-load challenges of TS-Arena (24 h at 15 min, 72 h at 1 h): two small MLP quantile nets, 160,278 parameters in total, 144 kB and 150 kB as int8 ONNX,
about 0.4 to 1.2 ms per series on one CPU thread.  Self-evaluated on the TS-Arena Archive; not an official ranking.  Code, protocol, ledger of the evaluations and reproduction scripts:
https://github.com/sankalpsthakur/gridload-compact

## Model description

- **Two MLP quantile nets**, one per challenge (15 min x 96 steps, 1 h x 72 steps).  Each net is a context encoder plus a shared per-step head (hidden width 64) that emits nine monotone deciles per step
  (0.1 to 0.9); the point forecast is the median.
- **3-seed ensembles** per frequency; the quantile curves are averaged inside one ONNX graph.  Parameters: 25,817 per net at 15 min and 27,609 hourly; 77,451 and 82,827 per ensemble; **160,278 in total**.
- A post-hoc scalar on the decile spread (1.05 at 15 min, 1.20 hourly) was chosen on validation data only.
- **ONNX**, fp32 (344,362 B and 365,869 B) and int8 by dynamic quantization (144,175 B and 149,554 B).  The service default is int8.
- **Inference**: a numpy feature pipeline (no torch) and ONNX Runtime on one thread.  A clock-signature gate compares the daily profile of the context with a fixed reference profile (`clock_ref.npz`); anything that is not
  a SMARD net-load series on the platform clock goes to a generic seasonal fallback with residual deciles, for which no claim is made.
- Public holiday calendars (DE federal and state, AT) are baked into `hol_table.npz` and derived from timestamps only.

## Intended use and limits

- **Intended use**: probabilistic net-load forecasts for the ten SMARD net-load series of the two TS-Arena challenges, on the platform's timestamp convention (UTC stamps that run 1 to 2 h early; the hourly series
  is the mean of the four quarter-hours), from a context of 1,000 points; CPU-only and edge-sized deployments; benchmarking and research.
- **Not covered**: other series, frequencies or horizons are served by the seasonal fallback and no claim is made for them.  Its held-out evaluation covers only archive rounds of 2026-02-08 .. 2026-03-31; it has no live
  or operational track record and is not validated for grid operation or trading.
- The comparison with foundation models is a specialist trained on the same SMARD series (cutoff before the window) against zero-shot generalists that saw none of it: the fair reading is "a specialist this small is
  competitive or better on its own series", not "better architecture".
- Training is not bit-reproducible across hardware; the SHA-256 of the checkpoints behind the numbers are in `net15_info.json`, `net60_info.json` and `configs/final.json` of the GitHub repository.

## Training data and cutoff

- SMARD net load (Bundesnetzagentur | SMARD.de, CC BY 4.0), revised history from 2018-01 fetched from the public API.  The shipped weights are the test-stage ensembles, trained on targets up to
  **2026-02-06 23:45**; the architecture selection used models cut at 2025-06-30 23:45 and the spread scalar used models cut at 2025-12-31 23:45.  The first evaluation round registers on 2026-02-08.
- Public holiday table generated with the python `holidays` package (MIT), version 0.106.
- No third-party model weights, and nothing from the TS-Arena Archive, were used for training or calibration.

## Evaluation

Self-evaluated on the TS-Arena Archive, window TEST-F (52 rounds, 2026-02-08 .. 2026-03-31; training cutoff 2026-02-06 23:45; configuration frozen and each variant evaluated once, pre-registered).
Headline variant = the service code path with the int8 weights, scored with the platform's arena MASE and SQL against the 32 reference models of the Archive's field:

- **Definition 2 (24 h at 15 min): lower mean MASE than every reference model on the full window.**  MASE 0.217 [0.173, 0.273] against 0.251 [0.184, 0.327] for the best reference model of the window, visiontspp-large (307 M parameters,
  about 4,000 times larger); per-round difference -0.034 [-0.059, -0.009]; SQL 0.173 against 0.201, difference -0.028 [-0.046, -0.011].  **Caveat that belongs to the claim:** the interval excludes 0 over
  the full window, but over the first 35 rounds (02-08 .. 03-14, which the live board mostly reproduces; 03-06 .. 03-08 are partly re-scored there) it does not (-0.012 [-0.026, +0.006]); the difference is carried by the 17 archive-only late-March rounds
  (-0.070 [-0.111, -0.021] against that window's best reference, flowstate).
- **Definition 5 (72 h at 1 h): a tie, not a win.**  MASE 0.185 [0.161, 0.214] against 0.187 [0.157, 0.226] for chronos-2 (120 M parameters, about 1,400 times larger); difference -0.002 [-0.015, +0.007];
  SQL 0.148 against 0.150, difference -0.002 [-0.014, +0.006].  The model matches the best reference at a fraction of its size; no win is claimed.

Scoring is the platform's arena MASE (MAE of the forecast divided by the MAE of the last-context-value persistence forecast over the same horizon) and SQL (scaled quantile loss, same scale), reimplemented from the
MIT backend of TS-Arena.  This is a **self-obtained evaluation on the TS-Arena Archive and not an official ranking**.  Protocol: selection rule, ensemble, spread grid and the eight evaluated variants were
pre-registered before any sweep, validation-stage or test-stage number existed; the configuration was frozen (hashes of the test-stage checkpoints) before the TEST-F evaluation; each variant was evaluated once
(16 evaluations), and an unchanged re-run of the frozen kernel gives byte-identical evidence tables.  The definition-2 difference is carried by 17 archive-only late-March rounds; the intervals are the platform's
ISO-week block bootstrap (9 blocks in the full window) and are crude.  Full tables, sub-windows, the ledger of all evaluations and the integrity checklist are in the GitHub repository (`TEST-LOG.md`,
`INTEGRITY.md`, `results/`).

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

Sources: `results/bench-timeseries-b/out/bench_kaggle_cpu_{15,60}.json` and `results/bench-timeseries-b/m4_latency_{15,60}.json` in the GitHub repository (median of 300 calls after warm-up, numpy features
included in the end-to-end figure).

## How to run the service

The service (FastAPI, `POST /predict`, `GET /health`) is in the GitHub repository under `submission/model-services/gridload-compact`:

```bash
git clone https://github.com/sankalpsthakur/gridload-compact
cd gridload-compact/submission/model-services/gridload-compact
pip install -r requirements.txt            # fastapi, uvicorn, pydantic, numpy, onnxruntime (pinned)
uvicorn app.main:app --port 8000           # reads ./weights, the same files as this repository
```

To use the files of this Hugging Face repository instead (run both from the service directory):

```python
from huggingface_hub import snapshot_download
snapshot_download("sankalpsthakur/gridload-compact", local_dir="hf-weights")
```

```bash
GRIDLOAD_WEIGHTS=hf-weights uvicorn app.main:app --port 8000      # GRIDLOAD_PRECISION=fp32 selects the fp32 graphs (default int8)
```

Request: `{"history": [{"ts": "2026-03-10T07:45:00.000Z", "value": 12345.0}, ...], "horizon": 96, "freq": "15min"}` (`"freq": "h"` with `"horizon": 72` for the hourly challenge; a list of such lists is a batch).
Response: `{"prediction": [{"ts": ..., "value": ..., "probabilistic_values": {"q_0.1": ..., ..., "q_0.9": ...}}, ...]}`.

## Files

| file | what |
|---|---|
| `net15.onnx`, `net15_int8.onnx` | 15 min x 96 steps, 3-seed ensemble, fp32 (344,362 B) and int8 (144,175 B) |
| `net60.onnx`, `net60_int8.onnx` | 1 h x 72 steps, 3-seed ensemble, fp32 (365,869 B) and int8 (149,554 B) |
| `net15_info.json`, `net60_info.json` | feature specification, spread, parameter counts, SHA-256 of the member checkpoints and of the ONNX files |
| `hol_table.npz` | baked public holiday calendar table (DE federal and state, AT) |
| `clock_ref.npz` | reference daily profile of the clock gate |
| `README.md` | this card |

## Licences

- Code and model weights: Apache-2.0.
- Training data: SMARD, Bundesnetzagentur | SMARD.de, CC BY 4.0 (https://www.smard.de/); the weights are derived from it and changes were made.
- Evaluation data: TS-Arena-Archive (DAG-UPB/TS-Arena-Archive), CC BY 4.0; used for evaluation only, not contained in the weights.
- Holiday table: generated with the python `holidays` package (MIT).

Self-evaluated, not peer-reviewed.  Sankalp Thakur, October 2026.
