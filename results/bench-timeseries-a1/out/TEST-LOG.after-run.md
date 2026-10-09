# TEST-LOG

Every evaluation on the final (TEST-F) window is logged here, including failed or discarded ones.
Nothing in this file may be used for selection after the entry date.

## Frozen protocol (written 2026-10-07 04:00 UTC, before any faithful-window number was looked at)

Why the windows look like this: the platform scores against FIRST-PUBLISHED (preliminary) SMARD values and serves a
partly preliminary context (last ~10 h).  The current SMARD API returns REVISED values, which differ from the
platform truth by 0.6-5.7 % (std) for DE / 50Hertz / AT (see `notes/revisions.md`).  The only public data that
carries the platform's own truth, served contexts and every model's real forecasts is the TS-Arena-Archive
(Q1 2026, 90 daily rounds per definition).  So the faithful windows are inside Q1 2026.

| Window | Rounds (registration day, UTC) | Data | Use |
|---|---|---|---|
| VAL-S | 2025-07-01 .. 2025-12-31, origins 07:45Z/05:30Z (def 2), 08:00Z/06:00Z (def 5) | simulated on revised data | architecture / hyper-parameter selection, many rounds |
| VAL-F | 2026-01-01 .. 2026-02-07 (38 rounds per def) | archive: served context, first-published truth, real field | select among finalists, quantile calibration |
| TEST-F | 2026-02-08 .. 2026-03-31 (52 rounds per def) | same | final evaluation, one run per frozen variant |

Training cutoffs (targets strictly before): VAL-S runs 2025-06-30 23:45Z; VAL-F runs 2025-12-31 23:45Z;
TEST-F run 2026-02-07 07:45Z (first TEST-F origin is 2026-02-08).  Revised SMARD history from 2018-01 is the
training data (CC BY 4.0).  No TEST-F target and no TEST-F truth enters training, calibration or selection.

Scoring = TS-Arena arena MASE (MAE / MAE of the last-context-value persistence over the horizon) and SQL, per
(round, series), averaged over series per round; Elo = the platform's own procedure (K=4, 500 bootstraps) with our
model inserted into the field of real archive forecasts (all 28-32 models).  `src/tsb/metrics.py` reproduces the
official per-series MASE of all 280 pairs of round 887 exactly (max abs diff 0.0).

Frozen variants to be scored once on TEST-F: fp32 (torch), fp32 (onnxruntime), int8 (onnxruntime dynamic quantization).

## Addendum 1 (2026-10-07 10:20 UTC): selection rule and variant list, fixed before the Kaggle phase

Written before any sweep, validation-stage or test-stage number of the Kaggle phase exists.  Machine-readable copy:
`configs/prereg.json` (the scripts read it).  Nothing in the frozen protocol above is changed; this only makes it operational.

1. Compute.  All training and evaluation after this point runs in private Kaggle CPU kernels (`bench-timeseries-*`); the Mac does
   coordination only.  Kernels carry `src/tsb`, fetch SMARD (API) and the TS-Arena-Archive (HF, pinned revision 72c6cb0f) themselves.
   Before a kernel is used for anything new it must reproduce the Mac result of the v0 dev checkpoint on VAL-F (def 2):
   per-(round, series) MASE/SQL within 1e-4 and the field-cache row counts 28448 / 28736; otherwise stop and report.
2. Cutoffs.  Sweep models: targets before 2025-06-30 23:45 (VAL-S is 2025-07..12).  Val-stage models: 2025-12-31 23:45.
   Test-stage models: 2026-02-06 23:45 (this is what `run_final.py` and INTEGRITY.md use; it is earlier than the 2026-02-07 07:45Z
   the frozen text allows, so it is strictly more conservative; no TEST-F origin is before 2026-02-08).
3. Selection (per frequency, 15 min and hourly).  Candidates: E2 features (J=96; per-day means 10 at 15 min, 14 hourly), hidden
   width 64 / 128 / 256 (emb 16 / 32 / 64), 4000 steps, seed 0.  A candidate qualifies when its VAL-S mean MASE is within 4 % of the
   best candidate in EACH phase separately.  Take the qualifying candidate with the fewest parameters.  Gate: its VAL-F mean MASE
   (faithful archive scoring) must be below the best reference model's mean over the same rounds in the real field; CIs are reported
   and do not gate.  If it fails the gate, step up to the next larger qualifying candidate; if none passes, take the lowest VAL-F
   MASE and record `gate_failed`.  The two v0 retrains (J=48, no per-day means, hid 256) are informational, not candidates.
4. Final recipe.  Ensemble of 3 seeds (0, 1, 2) of the selected candidate, quantile curves averaged in one ONNX graph; the single
   members are reported too.  Spread: one global scalar per frequency on the grid 0.90..1.50 (step 0.05), chosen by minimum mean
   VAL-F SQL of the 3-seed ensemble trained to 2025-12-31 23:45 (ties: smaller scale), applied unchanged to the test-stage
   ensemble and members.  Ablation: no-holiday features, seed 0, compared with the seed-0 member.
5. TEST-F variants.  Each of these is ONE TEST-F evaluation per definition (2 and 5), 16 in total, and gets one entry below:
   ens3 torch fp32; ens3 onnxruntime fp32; ens3 onnxruntime int8; service code path (int8); single member seed 0, seed 1, seed 2
   (torch fp32); no-holiday ablation seed 0.  The sub-window tables (2026-02-08..03-14 official-reproducible, 2026-03-15..03-31
   archive-only) are views of the same evaluation, not new ones.  Per-evaluation entries are written to the kernel output
   immediately; any re-run that touches TEST-F is a further evaluation and is logged as one, with the reason.
6. Freeze.  Before the TEST-F kernel is pushed: `configs/final.json` (selection, spread, seeds), SHA256 of the test-stage
   checkpoints, and the code commit are committed here.  The TEST-F kernel verifies the hashes before it evaluates.
7. Claim wording stays as in PLAN.md; the Archive's own rules apply: results are self-obtained evaluation on the TS-Arena Archive,
   not official platform rankings, with a training cutoff strictly before the evaluation period.

## Entries

(none yet)
