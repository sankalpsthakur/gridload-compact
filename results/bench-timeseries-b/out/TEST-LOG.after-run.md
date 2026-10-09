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

## Addendum 2 (2026-10-07 10:50 UTC): headline variant, gate order, a1 findings; still before any sweep, validation-stage or test-stage number

1. Headline variant.  For each definition the headline TEST-F result is the **service code path, int8** (what the Docker image runs by default).  The other seven
   variants of Addendum 1 are reported next to it; none is chosen on TEST-F.  The service result is always printed with its routing counts (specialist vs
   fallback), because the service has routing gates that the torch and onnxruntime paths do not, and the DST change of 2026-03-29 and the late-March contexts
   that end 1-2 days before registration (notes/revisions.md items 7 and 8) can make the paths diverge.
2. Reproducibility gate.  Part 1 (the Mac v0 dev checkpoint re-evaluated on VAL-F def 2 in a kernel) passed in kernel a1: 380 of 380 (round, series) pairs, max
   |difference| 1.2e-7 in MASE and 9.4e-8 in SQL, mean MASE 0.189901 (Mac 0.1899), no missing values.  Part 2 (field cache: row counts 28448 / 28736, per-row
   equality with the Mac cache, regenerated dev report) now runs inside the a2 kernel beside the sweep, and the selection stage is gated on it.  If the kernel's
   rebuild of the field cache fails or differs from the Mac cache, the Mac cache (same pinned archive revision, verified earlier against the official per-series
   scores) is used and the kernel status says `field_source=mac_fallback`.
3. a1 findings.  The Hub rate-limits anonymous resolver requests (about 3000 per 5 minutes); the first a1 attempt lost 168 forecast files to it and the fetch is
   now paced.  The service venv needs `--without-pip` on the Kaggle image.  Neither affects any model or number.

## Addendum 3 (2026-10-07, written at 15:50 UTC after the a1 results; still before any sweep, validation-stage or test-stage number)

1. a1 outcome.  v0 gate passed (Addendum 2).  The field cache built in a1: definition 2 has 28448 rows and is identical to the Mac cache on every row and every archive-derived
   column (max difference 0.0); definition 5 has 27056 of 28736 rows (168 forecast files were lost to the Hub rate limit, all in rounds 2026-02-10..15, i.e. TEST-F days) and is
   identical to the Mac cache on all 27056 shared rows.  Verified offline on the Mac from the 2.3 MB of a1 output.
2. Change of mechanism.  Kaggle rejected a 3.8 MB kernel script (HTTP 400 on SaveKernel; 0.84 MB passes), so the Mac field cache cannot ship inside the kernel.  a2 therefore uses
   a1's cache as the provisional field cache for the VAL-F gate (valid: the VAL-F rounds are complete and verified equal to the Mac copy) and rebuilds the complete cache beside the
   sweep for b's evidence tables; the rebuild is compared with the provisional cache inside the kernel and with the Mac copy offline.  The selection rule is unchanged.
3. Dev report.  `reports/dev_val_f_def2.md` was replaced by the kernel's regeneration: point estimates and Elo are identical to the earlier Mac file; the CI endpoints of the
   per-round means differ because that file came from an older bootstrap implementation (a recomputation with the current code on the Mac gives the kernel's value, 0.1458 / 0.2477).
4. Clock note.  The session clock jumped 5 h between 10:48 and 15:48 UTC (suspension); a2 was pushed at 15:48:07 UTC, after the 14:30 time box, by a waiter started before the jump was noticed.

## Addendum 4 (2026-10-08, written 20:40 UTC): a2 read-out and FREEZE.  No TEST-F evaluation of any model of ours exists at this point

1. a2 outcome.  Kernel `sankalpsthakur/bench-timeseries-a2` (code commit b9f6339, started 2026-10-07 15:48 UTC, 3391 s): all 15 stages ok.  The first fetch of its output had stopped after 15 of the 141
   `out/` files (cause not established; the live-log follower of the same waiter had hit a DNS failure and a read timeout).  The output was fetched again on 2026-10-08 at 20:27 UTC; the local copy was
   compared name by name with the kernel's file listing (nothing missing) and the 30 checkpoints were re-hashed against `ckpt_manifest.json` (all equal).
2. Reproducibility.  The field cache rebuilt inside a2 (28448 / 28736 rows, the pre-registered counts) equals the provisional a1 cache on every shared row (maximum difference 0.0 in MASE, SQL, MAE, naive
   MAE and n; definition 5 gains the 1680 rows from 2026-02-10 that a1 had lost to the Hub rate limit), and the four cache files (`field_def2/5.parquet`, `rounds_def2/5.npz`) are byte-identical (SHA256) to the
   Mac copies in `data/cache/`.  The two informational v0 retrains in the kernel reproduce the Mac runs: VAL-S MASE 0.1910 / 0.2099 (15 min, phases 07:45 / 05:30) against the Mac's 0.1909 / 0.2098, and
   0.2035 / 0.1998 (hourly, phases 08:00 / 06:00) against the Mac's 0.2035 / 0.1998; VAL-F MASE of the 15-min retrain 0.18990 (Mac dev report 0.1899).
3. Selection by the rule of Addendum 1 (`results/bench-timeseries-a2/out/selection.json`).  Every width qualifies at both frequencies (largest gap to the best on a VAL-S phase 2.52 %; the tolerance is 4 %) and
   every one passes the VAL-F gate (best reference in the field over the same rounds: flowstate, MASE 0.2275 at 15 min; chronos-2, MASE 0.2427 hourly), so the smallest candidate is selected at both frequencies:
   hid 64, 25817 parameters per net (15 min) and 27609 (hourly), gate_failed=false.

   | freq | hid | params per net | VAL-S gap to best (phase 1 / phase 2) | VAL-F MASE | VAL-F SQL |
   |---|---|---|---|---|---|
   | 15 min | **64** | 25817 | +2.52 % / +1.57 % | 0.1863 | 0.1474 |
   | 15 min | 128 | 63913 | +0.74 % / +0.37 % | 0.1800 | 0.1420 |
   | 15 min | 256 | 176969 | 0 / 0 | 0.1837 | 0.1449 |
   | 1 h | **64** | 27609 | +1.5 % / +1.4 % | 0.2028 | 0.1584 |
   | 1 h | 128 | 67497 | +0.7 % / +0.6 % | 0.2033 | 0.1587 |
   | 1 h | 256 | 184137 | 0 / 0 | 0.2058 | 0.1607 |

4. Spread.  One scalar per frequency, the argmin of the VAL-F SQL of the val-stage 3-seed ensemble over the grid: 1.05 at 15 min (SQL 0.144582; 1.00 and 1.10 give 0.144692 and 0.144595, a near-tie) and 1.20
   hourly (0.157864; 1.15 and 1.25 give 0.157909 and 0.157932).
5. Dry run of every evaluation path on VAL-F with the final recipe at the validation stage (real VAL-F numbers: the val-stage models end 2025-12-31 23:45).  Ensemble fp32, MASE / SQL: 0.1834 / 0.1446
   (definition 2) and 0.2031 / 0.1579 (definition 5).  Service code path int8: 0.1862 / 0.1465 and 0.2042 / 0.1587, routing 380 of 380 (round, series) pairs to the specialist and none to the fallback on both
   definitions.  Service tests on the dry-run weights: 41 service tests, 36 contract tests, and the live harness on single and batch requests of 15 min x 96, 1 h x 24 and 1 h x 72 all passed.  The `wrapt`
   sitecustomize message in the service logs is the Kaggle image's own sitecustomize failing inside the pip-less venv: noise.
6. FREEZE, 2026-10-08 20:35 UTC.  `configs/final.json` (commit ad57898) is the frozen configuration.  Nothing about TEST-F may be chosen after this entry.  Freeze note:

   - frozen 2026-10-08 20:35 UTC from the a2 kernel (bench-timeseries-a2, code commit b9f63390f0); `configs/final.json` is the frozen configuration.
   - def 2, 15 min x 96: E2 features J=96 dmeans=10, hid 64 emb 16, 25817 parameters per net, 3 seeds [0, 1, 2], steps 4000; spread 1.05; gate_failed=False (smallest qualifying candidate that passes the VAL-F gate).
   - def 5, 1 h x 72: E2 features J=96 dmeans=14, hid 64 emb 16, 27609 parameters per net, 3 seeds [0, 1, 2], steps 4000; spread 1.2; gate_failed=False (smallest qualifying candidate that passes the VAL-F gate).
   - test-stage checkpoints (cutoff 2026-02-06T23:45) SHA256 prefixes: test_15_nohol_s0.pt=cd6d0fd28b91, test_15_s0.pt=d0b03c3ef63f, test_15_s1.pt=1edff5f7cadf, test_15_s2.pt=4e67a872b20a, test_60_nohol_s0.pt=89bd5531bdda, test_60_s0.pt=f184a96c54e0, test_60_s1.pt=77fd6aa75421, test_60_s2.pt=3cabd3e212c6
   - headline variant per definition: service code path, int8 (Addendum 2); the b kernel verifies the hashes above before any TEST-F evaluation and stops otherwise.

7. Code state for b.  Between the a2 build commit b9f6339 and the commit b is built from, nothing under `src/`, `scripts/*.py`, `submission/` or `assets/` changes; the differences are `configs/final.json`
   (added), a comment in `scripts/reproduce.sh`, `.gitignore` (result folders are tracked) and this file.  b mounts the a1 and a2 outputs, restores the data cache from a1 and the checkpoints and the complete
   field cache from a2, checks the SHA256 of the 8 test-stage checkpoints against `configs/final.json` and their training arguments against the selection, and stops before any TEST-F evaluation if anything differs.
8. Rules from here.  Each variant is evaluated once per definition: 8 variants x 2 definitions = 16 TEST-F evaluations (Addendum 1 item 5).  The kernel's own once-only check does not carry across pushes: if b
   fails after its `test_f` stage has started, any re-push is a further TEST-F evaluation and is logged as such, with the reason.  A failure before `test_f` (frozen-hash check, weights assembly, five-round dry
   run) is not a TEST-F evaluation, but the attempt is logged here.  Results carry intervals; a win over the best reference model of the TEST-F table is claimed only where the interval of the per-round
   difference excludes 0.

## Entries

(none yet)

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 service code path int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15.onnx=81e7e6b9f9c850d6
- spread: None
- result: {"label": "net15.onnx", "rounds": 52, "mase": 0.21538568160136615, "sql": 0.1705881498404233, "cover80": 0.7667741683740044, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_ort_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5
- spread: None
- result: {"label": "net15_int8.onnx", "rounds": 52, "mase": 0.2136769221459295, "sql": 0.1695305560099699, "cover80": 0.7665386474862937, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_ort_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 1 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s0.pt=d0b03c3ef63ffabd; test_15_s1.pt=1edff5f7cadfda47; test_15_s2.pt=4e67a872b20ac422
- spread: 1.05
- result: {"label": "test_15_s0.pt,test_15_s1.pt,test_15_s2.pt", "rounds": 52, "mase": 0.2153856822962949, "sql": 0.1705881493092261, "cover80": 0.7667741683740044, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_torch_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 2 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s0.pt=d0b03c3ef63ffabd
- spread: 1.05
- result: {"label": "test_15_s0.pt", "rounds": 52, "mase": 0.21489883192613685, "sql": 0.17030263588969735, "cover80": 0.7575240414088081, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 1 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s1.pt=1edff5f7cadfda47
- spread: 1.05
- result: {"label": "test_15_s1.pt", "rounds": 52, "mase": 0.21723457616807007, "sql": 0.17225651377372564, "cover80": 0.7638345230658785, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s1_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 no-holiday ablation seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 2 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s2.pt=4e67a872b20ac422
- spread: 1.05
- result: {"label": "test_15_s2.pt", "rounds": 52, "mase": 0.21624105543034122, "sql": 0.1713112243541424, "cover80": 0.7672213115766902, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s2_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 service code path int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5; net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "service:finalW:int8", "rounds": 52, "mase": 0.21730961966070916, "sql": 0.1729146064293282, "cover80": 0.7696489440176428, "n_missing": 9, "routing": {"specialist": 513, "fallback": 7}}
- per-round file: faith_final_def2_test_d2_service_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 2  2026-02-08..2026-03-31  variant: def2 no-holiday ablation seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_nohol_s0.pt=cd6d0fd28b910b35
- spread: 1.05
- result: {"label": "test_15_nohol_s0.pt", "rounds": 52, "mase": 0.2190762635240415, "sql": 0.17455040139015682, "cover80": 0.7714243431736326, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_nohol_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s0.pt=f184a96c54e0ce44; test_60_s1.pt=77fd6aa7542120e4; test_60_s2.pt=3cabd3e212c6b549
- spread: 1.2
- result: {"label": "test_60_s0.pt,test_60_s1.pt,test_60_s2.pt", "rounds": 52, "mase": 0.18341776724696707, "sql": 0.14677138379844218, "cover80": 0.8575820643978889, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_torch_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 service code path int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net60.onnx=6d610b1c66c6509e
- spread: None
- result: {"label": "net60.onnx", "rounds": 52, "mase": 0.18341776781311142, "sql": 0.1467713855292288, "cover80": 0.8575820643978889, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_ort_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 1 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "net60_int8.onnx", "rounds": 52, "mase": 0.183232555787468, "sql": 0.14681973075949795, "cover80": 0.8577164102612967, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_ort_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 2 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s0.pt=f184a96c54e0ce44
- spread: 1.2
- result: {"label": "test_60_s0.pt", "rounds": 52, "mase": 0.18427172604388348, "sql": 0.14751533639377545, "cover80": 0.8611559499378039, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 1 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s1.pt=77fd6aa7542120e4
- spread: 1.2
- result: {"label": "test_60_s1.pt", "rounds": 52, "mase": 0.18430093636974906, "sql": 0.14773921359075992, "cover80": 0.8608823460737539, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s1_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 no-holiday ablation seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 2 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s2.pt=3cabd3e212c6b549
- spread: 1.2
- result: {"label": "test_60_s2.pt", "rounds": 52, "mase": 0.1833676209694743, "sql": 0.14656283296946987, "cover80": 0.8461777300346143, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s2_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 no-holiday ablation seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_nohol_s0.pt=89bd5531bdda0c0c
- spread: 1.2
- result: {"label": "test_60_nohol_s0.pt", "rounds": 52, "mase": 0.1904035151503274, "sql": 0.15408170023699513, "cover80": 0.8576233454546626, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_nohol_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 20:41 UTC  def 5  2026-02-08..2026-03-31  variant: def5 service code path int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5; net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "service:finalW:int8", "rounds": 52, "mase": 0.1847649701804282, "sql": 0.14801052613656857, "cover80": 0.8561927784205007, "n_missing": 0, "routing": {"specialist": 513, "fallback": 7}}
- per-round file: faith_final_def5_test_d5_service_int8_2026-02-08_2026-03-31.parquet
