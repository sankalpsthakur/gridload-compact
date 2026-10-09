# Integrity checklist (timeseries track)

Status as of the last commit; each line states what was checked and where the evidence is.  Compute note: from 2026-10-07 10:00 UTC every training, evaluation,
export and service test ran in private Kaggle CPU kernels `bench-timeseries-*` (the Mac did coordination only).

## Evaluation faithfulness
- [x] Scoring reimplemented from the MIT backend (arena MASE = MAE / persistence-MAE, SQL with the fev quantile loss and the same scale, per-round mean over series, K=4 bootstrapped Elo).  `src/tsb/metrics.py`.
- [x] Official numbers reproduced before any training: all 280 (model, series) MASE values of round 887 match to 0.0; 20 138 of 22 984 pairs of definition 2 Q1 match to 1e-9 (every round 2026-01-01..03-05 and 03-09..03-14).  The remainder (03-06..08 partly, 03-15..03-31) differ because the live board re-ran or re-scored those rounds after the archive dump (`notes/revisions.md` 7).  Windows are scored from the archive for every model, ours included.
- [x] Platform conventions verified on data, not assumed: timestamp clock (1 h / 2 h early), hourly = mean of four quarter-hours, 1000-point contexts, per-series anchors, first-published truth (`notes/revisions.md`).
- [x] Service code path scores like the research harness.  Smoke nets (a1): torch vs onnxruntime fp32 differ by 3e-9 / 4e-9 (def 2 / def 5), int8 onnxruntime vs the service code by 7e-9 / 2e-8, routing 30 of 30 specialist.  Real final recipe on VAL-F (a2, 380 pairs per definition): torch vs onnxruntime fp32 max |diff| 2.3e-7 / 1.5e-7 MASE, onnxruntime int8 vs service int8 1.0e-7 / 1.1e-7, routing 380 of 380 specialist.  On TEST-F (b, 520 pairs per definition): torch vs onnxruntime fp32 1.5e-7 / 2.8e-7 MASE; service int8 vs onnxruntime int8 identical to 2e-7 on every scorable pair except 7 per definition that the service gates sent to the seasonal fallback (Amprion 2026-03-25; LU and Creos 2026-03-29 .. 03-31): intended behaviour, part of the headline and reported with its routing counts.  `results/bench-timeseries-a2/out/faith/`, `results/bench-timeseries-b/out/faith/`, TEST-LOG post-run notes 4-5.
- [x] Kernel reproducibility gate (a1): the Mac v0 dev checkpoint, re-evaluated on VAL-F def 2 in a Kaggle kernel (Python 3.13.15, numpy 2.5.3, pandas 3.0.6, torch 2.11.0+cpu), reproduces all 380 (round, series) MASE/SQL values with max |difference| 1.2e-7 / 9.4e-8 (mean MASE 0.189901 vs 0.1899).  `results/bench-timeseries-a1/`.

## No leakage
- [x] Frozen window plan written before any faithful-window number was looked at (`TEST-LOG.md`, commit 45679b5); selection rule, ensemble size, spread grid and the TEST-F variant list pre-registered before the Kaggle phase produced any number (Addendum 1, commit 54a9a2d; headline variant in Addendum 2; Addendum 3 was written after the a1 numbers but before any sweep, validation-stage or test-stage number).
- [x] Training data = revised SMARD history from 2018-01 only; sweep models cut at 2025-06-30 23:45, val-stage models at 2025-12-31 23:45, test-stage models at 2026-02-06 23:45 (first TEST-F round registers 2026-02-08; the frozen text allows 2026-02-07 07:45Z, the code uses the earlier cutoff).  The archive's own rule (cutoff strictly before the evaluation period) holds.
- [x] TEST-F truth, contexts and forecasts of any model are not used in training, calibration or selection.  The field's TEST-F scores were printed during the earlier analysis of the bar (not of our model).
- [x] Every TEST-F evaluation is listed in `TEST-LOG.md`: 16 STARTED lines and 16 result entries (8 variants x 2 definitions, kernel `bench-timeseries-b` v1 pushed once, code d4c7c23), copied unedited from the kernel's `TEST-LOG-entries.md`; no TEST-F evaluation was repeated or discarded.  The b kernel verified the 8 test-stage checkpoint hashes and their training arguments against `configs/final.json` (commit ad57898, frozen 2026-10-08 20:35 UTC) before the first evaluation; between the a2 build commit b9f6339 and the b build commit nothing under `src/`, `scripts/*.py`, `submission/` or `assets/` changed (TEST-LOG Addendum 4 item 7).
- [x] Calibration (decile spread factor) uses VAL-F only, with the val-stage 3-seed ensemble (cutoff 2025-12-31 23:45): one scalar per frequency from the pre-registered grid, argmin of mean VAL-F SQL, ties to the smaller scale (1.05 at 15 min, 1.20 hourly; `results/bench-timeseries-a2/out/spread_def*.json`), applied unchanged to the test-stage ensemble and its members.
- [x] Moirai weights (CC BY-NC) were not used for training or distillation; no third-party model weights were used at all.

## Licences
- SMARD (Bundesnetzagentur) CC BY 4.0: training data, attribution "Bundesnetzagentur | SMARD.de".  CC BY 4.0 allows released weights derived from it (also commercially) with that attribution and a note that changes were made; no share-alike, no non-commercial clause.
- TS-Arena-Archive CC BY 4.0: evaluation data only (contexts, truth, forecasts of other models); nothing from it, and no forecast of any other model, enters training, calibration or the released weights.  Results are self-obtained evaluations on the Archive, not official rankings (the Archive's rule).
- `holidays` python package (MIT) generated the baked holiday table (DE federal + 16 state shares, AT), pinned to 0.106; its SHA256 equals the Mac table.
- Our weights/code: to be offered under the MIT licence of ts-arena-models (owner decides; nothing was published).

## Limits stated in the claim
- Specialist for the SMARD net-load series on the platform's current timestamp convention; other families are served by a seasonal fallback and no claim is made for them.
- Holiday calendars are public deterministic knowledge baked into the model (derived from timestamps only, nothing fetched at run time); an ablation without them is reported.
- Elo on a 52-round window with a 32-model field is not the live board Elo (which uses every round since 2025).
- Late-March archive rounds (03-15..03-31) are archive-only: the live board re-ran them (`notes/revisions.md` 7); both sub-windows are reported.

## Added 2026-10-08 (kernels a2 and b)
- [x] Field cache: the cache rebuilt inside a2 (28448 / 28736 rows) equals the provisional a1 cache on every shared row and the four cache files are byte-identical (SHA256) to the Mac copies; b used the a2 export.
- [x] Retrains in the kernel agree with the Mac runs: the informational v0 retrains reproduce the Mac VAL-S numbers to 4 digits (`results/bench-timeseries-a2/out/info_v0.json`).
- [x] Selection followed the pre-registered rule (smallest qualifying width that passes the VAL-F gate: hid 64 at both frequencies; table in TEST-LOG Addendum 4 item 3) and was frozen before the TEST-F kernel was built.
- [x] Service tests (41), repository contract tests (36) and the live harness (single and batch, 15 min x 96, 1 h x 24, 1 h x 72) passed on the dry-run weights (a2) and on the final weights (b).  The `wrapt` "Error in sitecustomize" line in the service logs
  is the Kaggle image's own sitecustomize failing inside the pip-less venv; it does not touch our code.
- [x] The first fetch of a2's output had stopped after 15 of 141 files; it was fetched again and checked name by name against the kernel's file list (same check done for b: 133 of 133).
- [x] Like-for-like scoring against the field on TEST-F: for every pair shared with the field the scored-point count `n` equals that of all 32 field models (0 mismatches, both sub-windows), round ids agree, the field's naive-forecast has MASE 1 to 0.0 on every pair; 13 pairs that only we score (9 unscorable pairs of round 6363, LU and Creos on 2026-03-27) do not change the headline when removed (TEST-LOG post-run note 8).
- [x] Local work on the Mac in this phase: git, file edits, the kaggle CLI and reading small per-round result files (pandas, one 2000-draw block bootstrap over 9 weekly blocks), all through `bin/guard.sh`.  No model was loaded or run on the Mac.
- [x] M4 latency (1 thread, batch 1) measured once the 1-minute load average was under 4 (3.7-3.9): `results/bench-timeseries-b/m4_latency_{15,60}.json`; Kaggle CPU latency in `results/bench-timeseries-b/out/bench_kaggle_cpu_*.json`.  Container latency (`POST /predict`) is a lead step.
- [x] Docker image build and container harness: done by the lead on GitHub Actions in the fork (run 37845299662, 2026-10-08): image builds, `/health` 200, harness single+batch PASS for 15 min x 96, 1 h x 72, 1 h x 24, compose build OK; single-mode harness wall time 0.05 s.

## Added 2026-10-09 15:18 UTC (lead)
- The statements "nothing was published" above are superseded: the service was submitted as https://github.com/DAG-UPB/ts-arena-models/pull/5 on 2026-10-08, and this evaluation kit is public at https://github.com/sankalpsthakur/gridload-compact with the model card and weights at https://huggingface.co/sankalpsthakur/gridload-compact. Licences: the kit is Apache-2.0; the service directory is also MIT as contributed upstream.
- Lead re-run of kernel b (`bench-timeseries-b-verify`): the six TEST-F evidence tables are byte-identical to the original run.

