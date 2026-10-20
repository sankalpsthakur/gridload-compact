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

Kernel `sankalpsthakur/bench-timeseries-b` v1 (code commit d4c7c23, pushed once at 2026-10-08 20:39:38 UTC, ran 123 s on Kaggle CPU; frozen configuration `configs/final.json`, commit ad57898).  Its stages
`verify_frozen` (SHA256 of the 8 test-stage checkpoints and their training arguments against the freeze), `assemble_test`, `dryrun_codepaths` (every path on five VAL-F rounds; service routing 50 of 50
specialist), `test_f`, `test_evidence`, `svc_tests_final` and `bench_cpu` all ended ok (`results/bench-timeseries-b/out/status_b.json`).

The 16 evaluations below (8 variants x 2 definitions, each run once) are the only TEST-F evaluations of any model of ours.  The ledger holds 16 STARTED lines and 16 result entries; it is copied unedited from
`results/bench-timeseries-b/out/TEST-LOG-entries.md` (times are UTC; the evaluations ran four at a time).  The result entries carry the SHA256 prefixes of the artifacts, the spread and the summary
(mean MASE and SQL over rounds of the per-round mean over series, 80 % coverage, missing pairs, routing counts).

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

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 20:40 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 1 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

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

### Post-run notes (written 2026-10-08 after the ledger above; they select nothing)

1. No second TEST-F evaluation exists: kernel b was pushed once and every variant ran once.  The numbers below and the tables in `results/bench-timeseries-b/out/test_f_def*_*.md` are views of the same 16 per-round files
   (the two sub-windows of Addendum 1 item 5 are views, not new evaluations), and so are the paired comparisons computed offline from those files.
2. Headline (Addendum 2: service code path, int8), full window 2026-02-08..03-31, 52 rounds, MASE / SQL, 95 % block-bootstrap interval over ISO weeks, difference to the best reference model of the same window
   (named in the evidence file; the best reference differs by definition):
   definition 2 (24 h at 15 min): 0.217 / 0.173 against visiontspp-large 0.251 / 0.201; difference -0.034 [-0.059, -0.009] / -0.028 [-0.046, -0.011].
   definition 5 (72 h at 1 h): 0.185 / 0.148 against chronos-2 0.187 / 0.150; difference -0.002 [-0.015, +0.007] / -0.002 [-0.014, +0.006]: indistinguishable on this window.
   Sub-windows, MASE difference to the best reference of that window: definition 2: 02-08..03-14 (35 rounds, official-reproducible) -0.012 [-0.026, +0.006] against visiontspp-large; 03-15..03-31 (17 rounds,
   archive-only) -0.070 [-0.111, -0.021] against flowstate.  Definition 5: 02-08..03-14 +0.005 [-0.001, +0.010] against chronos-2; 03-15..03-31 -0.009 [-0.038, +0.006] against moirai-2-small.
   So the definition-2 win over the full window is carried by the last 17 rounds; in the first 35 the interval contains 0.
3. Data facts that shape the reading.  Round 6363 of definition 2 (registered 2026-03-25) has ground truth for Amprion only: 9 of the 10 series of that round are unscorable for every model, so
   "missing 9" is the same for all variants and its per-round mean is a single series (the 32 models of the field have a score for Amprion only).  Because the archive truth is incomplete in late March
   (and in 2026-03-07), a further 93 pairs of definition 2 and 76 of definition 5 are scored on fewer than the full horizon, for every model alike.
4. Routing of the headline path.  The service sent 7 of 520 (round, series) pairs of each definition to its seasonal fallback: Amprion on 2026-03-25 (round 6363) and LU and Creos (the same series) on
   2026-03-29, 03-30 and 03-31 (the specialist's input gates rejected these contexts; the gate reason is not logged, and the last three days are the DST-change days with contexts that end before the
   registration day, notes/revisions.md 8).  Every other scorable pair is identical to the onnxruntime int8 ensemble to 2e-7 in MASE.  Cost of the fallback against the onnxruntime int8 ensemble:
   +0.0036 MASE (definition 2), +0.0015 (definition 5); paired difference service int8 minus ensemble fp32 per round: +0.0019
   [-0.0065, +0.0117] and +0.0013 [-0.0006, +0.0043].
5. Path equivalence on TEST-F (520 pairs per definition): torch fp32 against onnxruntime fp32, max |difference| 1.5e-7 / 2.8e-7 in MASE and 1.2e-7 / 1.2e-7 in SQL (definition 2 / 5).
6. Ablation and seeds.  No-holiday minus seed-0 single net (paired, per round): definition 2 MASE +0.0042 [-0.0015, +0.0105]; definition 5 +0.0061 [+0.0020, +0.0115].  Single nets over seeds 0/1/2, MASE
   mean +- sd: 0.2161 +- 0.0012 (definition 2), 0.1840 +- 0.0005 (definition 5).
7. Elo in the evidence files is computed with four of our variants inserted into the field at once; use the MASE / SQL means and the paired intervals for claims.
8. Like-for-like check (offline, from the per-round files and the field cache; written after the ledger).  On every TEST-F pair that our service-int8 file and the field share (509 of 520 pairs in
   definition 2, 518 of 520 in definition 5) the number of scored points `n` equals that of each of the 32 field models (0 mismatches, in both sub-windows), the round ids agree, and the field's
   naive-forecast has MASE 1 to 0.0 on every pair (the persistence scale is the same for everyone).  For 11 pairs of definition 2 (the 9 unscorable ones plus LU and Creos on 2026-03-27) and 2 of
   definition 5 (LU and Creos on 2026-03-27) no field model has a row while our file scores them; restricting ours to the pairs the field has changes the headline to MASE 0.2166 (definition 2) and
   0.1846 (definition 5) and the paired differences to -0.0349 [-0.0601, -0.0086] and -0.0021 [-0.0154, +0.0071]; nothing in the reading changes (sub-windows: definition 2 03-15..03-31 -0.0718
   [-0.1161, -0.0207], 02-08..03-14 unchanged).
9. Sensitivity of the intervals (offline, same per-round files; the platform's ISO-week block bootstrap has 9 blocks in the full window and only 6 and 4 in the sub-windows, so it is crude).  MASE, service int8
   minus the window's best reference, per round: platform block bootstrap / i.i.d. round bootstrap / moving-block bootstrap with 7-round blocks (4000 draws each):
   definition 2 full window -0.0342 [-0.0588, -0.0086] / [-0.0578, -0.0135] / [-0.0618, -0.0118], 32 of 52 rounds won; 02-08..03-14 -0.0122 [-0.0256, +0.0055] / [-0.0371, +0.0079] / [-0.0335, +0.0047], 17 of 35;
   03-15..03-31 -0.0697 [-0.1111, -0.0207] / [-0.1284, -0.0212] / [-0.1358, -0.0167], 10 of 17.  Definition 5 full window -0.0019 [-0.0149, +0.0071] / [-0.0122, +0.0069] / [-0.0092, +0.0093], 25 of 52;
   02-08..03-14 +0.0053 [-0.0006, +0.0103] / [-0.0010, +0.0115] / [+0.0004, +0.0115], 16 of 35; 03-15..03-31 -0.0094 [-0.0380, +0.0058] / [-0.0303, +0.0118] / [-0.0345, +0.0182], 10 of 17.  The three procedures
   agree on every conclusion except the moving-block interval of definition 5 on 02-08..03-14, which just excludes 0 on the side of chronos-2 (+0.0004).  Per series (definition 2, full window, pairs shared with the field)
   our MASE is below visiontspp-large's on every series except 50Hertz (equal, -0.001): -0.048 (AT, APG) to -0.023 (LU, Creos); definition 5 against chronos-2: better on TenneT (-0.029), worse on 50Hertz (+0.018),
   within 0.006 elsewhere.

## Lead re-run, 2026-10-08 21:20 UTC
Additional TEST-F evaluation, logged per protocol: the frozen kernel b was pushed again unchanged as `sankalpsthakur/bench-timeseries-b-verify` v1 (same a1/a2 sources, same checkpoints hashed in configs/final.json) by the lead as a clean-kernel verification. All 16 variants were re-evaluated once; the six TEST-F evidence tables (`results/bench-timeseries-b-verify/out/test_f_*.md`) are byte-identical to the original run's. No selection, no change to any number.

## TEST-L1: shadow check on completed live TS-Arena rounds (pre-registration, written 2026-10-19 19:14 UTC; committed before any in-window leaderboard row, forecast or truth value was read)

Question.  If gridload-compact v1 (frozen, unchanged) had been submitted to the Open Track on every round of the last 8 weeks, where would it stand on the platform's own Arena Score, on definitions 2 and 5?  Purpose: decide whether to ask the owner for an Open Track entry (account, API key as a GitHub secret, a daily job; none of this is approved or done).  This is a shadow check on public data: no account, no key, no submission, nothing published.  It is not an official ranking.

1. Model under test.  gridload-compact v1 exactly as in the public kit (github.com/sankalpsthakur/gridload-compact, commit 37b3a42, `submission/model-services/gridload-compact/`): service code `app/*.py`, int8 ONNX nets, `net15_info.json` / `net60_info.json` (spread 1.05 / 1.20; spec J=96, dmeans=10 / 14), `hol_table.npz`, `clock_ref.npz`; SHA-256 of every file in `kernels/shadow/V1-MANIFEST.txt`; they equal the weights that produced the TEST-F numbers (kernel b `service_weights`).  Training cutoff 2026-02-06 23:45; no retraining, recalibration, spread change or re-selection.  The kernel asserts the SHA-256 of the files (and the int8 hashes in the info files) at start and stops if anything differs.  Path = `GridLoadCompact(precision="int8").predict(batch, horizon, freq)`, which is what `POST /predict` calls; batch = the 10 series of one task group; definition 2: freq "15min", horizon 96; definition 5: freq "h", horizon 72.  The forecast of a series is the nine deciles `q_0.1..q_0.9` (point forecast = the median), timestamps continuing from the last context timestamp.

2. Data.  Public dashboard API of the platform, `https://ts-arena.live/api/v1/` (confirmed: the API-portal host `api.ts-arena.live` answers 401 "Invalid API Key" on every data route and is open only at `/` and `/health`; its `openapi.json` is public but describes only the key-gated portal).  Nothing is requested with a key or an account.  Window W: the completed rounds of definitions 2 (SMARD net load, 24 h at 15 min) and 5 (72 h at 1 h) with registration start 2026-08-24 .. 2026-10-18 UTC, as frozen in `kernels/shadow/rounds_frozen.json` (SHA-256 161cb1e9...; listing fetched 2026-10-19 18:49 UTC): 56 rounds of definition 2 (ids 12048 .. 12928, ids-hash cfa3972f) and 54 of definition 5 (12050 .. 12898, ids-hash 0e316ef4); 10 series each.  A task is one (round, series).  Per task the kernel reads: the series metadata (`/rounds/{id}/series`: context start and end), one `/rounds/{id}/series/{sid}/data` call covering the context and the horizon (context = the points from `context_start_time` to `context_end_time`; truth = the `H` points `context_end + step .. context_end + H*step`; these are the platform's scoring windows), the stored forecasts of every model (`/rounds/{id}/series/{sid}/forecasts`) and the stored leaderboard rows (`/rounds/{id}/leaderboard`).  Eligible task: truth present and finite at all `H` points, a non-empty context, and our forecast produced; any other task is dropped for every model and counted by reason.  If more than 5 % of tasks of a definition are lost to fetch or run errors, the run is incomplete and the verdict is NOT EVALUABLE.

3. Scoring (the platform's current recipe, not the track's old `src/tsb/metrics.py`, whose denominator the platform replaced on 2026-10-15).  The kernel imports the platform's own `forecast_metrics.compute_score_fields` (MIT, vendored unmodified in `kernels/shadow/vendor/`, SHA-256 e39c065d..., from ts-arena-backend main at b8babf7, last change 2026-10-15) and rescores EVERY model, ours included, from its stored forecasts (ours: its own forecasts) against the truth read from the `/data` endpoint: `mae`, `scale` = mean |x_t - x_(t-m)| over the context with the lag by timestamp (m = 96 at 15 min, 24 hourly), `mase` = mae / scale, `sql_score` = mean over the nine deciles of the fev quantile loss over the same scale (missing deciles completed and levels sorted as the platform does), `n_points`, and the persistence quantities (`naive_value` = the last finite context value, `naive_mae`).  Stored scores are used only for the reproduction check (4); stored and self-computed scores are never mixed in an Arena Score.  A naive anchor row (the internal naive of the platform: error = `naive_mae` for the point ranking, `naive_mae/scale` for the probabilistic one, `n_points` of the task's fullest row) joins every task.  An old-style "arena MASE" (mae / naive_mae, the metric of our public texts) is also computed per row for continuity (7).

4. Reproduction gate R0, evaluated before anything involving our model is read.  For each comparator c in {visiontspp-large (model id 131), SCCH/PCFM-FT (4240), farang/fx2-medium-thinking (2328), chronos-2 (107)}: over eligible tasks where c has a stored leaderboard row with `is_final` true, the share of rows with |recomputed - stored| <= 1e-3 * stored for both `mase` and `sql_score` (and the same `has_quantiles`).  PASS(c) iff at least 140 such rows and a share >= 0.95.  R0 holds iff PASS(131) and PASS(4240) (the two gate comparators); the other two are flagged and, if they fail, left out of the report.  If R0 fails the verdict is NOT EVALUABLE (Protocol rule 1: stop and report): no gap involving our model is computed or quoted, the failure is described, and a different recipe would need its own pre-registration (TEST-L2) written before our scores are looked at.  The share of matching rows is also reported for all models.

5. Arena Score, gap and intervals.  The platform's `arena_score.py` (MIT, vendored unmodified, SHA-256 e667b6b6..., commit b8babf7) is the engine: rows as in the platform's `arena_score_service.summarize_definition` (point ranking: `mae` of every model; probabilistic ranking: `sql_score` of rows with `has_quantiles` only, as `arena_scores_repository` keeps them; the naive anchor row in both), `day` = UTC day number of the round's registration start, `pair_sums` (outcome err_B / (err_A + err_B), weight 1/(k-1), pairs with unequal `n_points` or exactly one zero error skipped) and the Bradley-Terry fit `fit`; score = 100 * (theta - theta_naive).  Per definition and per metric (point, probabilistic) one joint fit over all models of the window and ours (a fit that does not converge makes that metric NOT EVALUABLE).  The Arena-gap of model A over model B is score_A - score_B (about +1 per 1 % lower error); a positive gap favours A.  Interval of a gap: 95 % percentile interval of the paired gap over refits of the whole fit under a circular block bootstrap of the days of the window (the platform's resampling with the block length set to 7 days, 1000 refits, seed 0; sensitivity: 14 days with 500 refits and the platform's 28 days with 200 refits, which gives only 2 blocks in 8 weeks).  The replica of the scout (`scripts/arena_replica/bt_replica2.py`) is used only as a cross-check of the engine on the Archive tables.

6. Comparators and decision rule (fixed now).  Gate comparators: visiontspp-large and SCCH/PCFM-FT.  Reported comparators: farang/fx2-medium-thinking and chronos-2 (read with the same +5 / -5 convention but not gating) and the naive anchor.  Only rounds both models played enter a gap through the joint fit; a gate comparator needs at least 140 common eligible tasks in the definition with ours, otherwise its condition is not evaluable.
   GO iff all of
   - R0 holds;
   - R1: definition 2, probabilistic ranking, context variant V1: Arena-gap(ours over visiontspp-large) >= +5.0, point estimate of the joint fit;
   - R2: definition 2, probabilistic ranking, V1: Arena-gap(ours over SCCH/PCFM-FT) >= -5.0, point estimate;
   - R3: fallback share <= 5 % of eligible definition-2 tasks (a task is a fallback when the service routed the series to its generic seasonal fallback, `path == "fallback"`; shares are also reported for definition 5, not gated).
   Otherwise NO-GO; if R0, R1's or R2's comparator, or the data volume is not evaluable the verdict is NOT EVALUABLE, which also means: do not ask the owner for the entry on this evidence.  Definition 5 is reported without a gate.  The thresholds were proposed in the task brief before any in-window number existed and are not tuned to one.  They are point-estimate thresholds: at about 560 tasks per definition and 8 weekly blocks the 95 % interval of a gap is expected to be several points wide, so a GO is a decision to ask the owner, not evidence of superiority.
   Claim rule (independent of GO): ours is "ahead of" a model only where the lower end of the 7-day-block interval of the gap is above 0, "behind" where the upper end is below 0, otherwise "not distinguishable"; every claim reads "self-evaluated shadow check on completed live rounds, 2026-08-24..2026-10-18, contexts reconstructed (not the served ones), not an official ranking, as of <date>".

7. Reported only, whatever they show (one table each, no selection): (a) definition 2 point ranking, definition 5 point and probabilistic rankings, and the farang/fx2 and chronos-2 rows of everything, with the three interval variants; the place of ours and the top 10 of each fit; (b) the same gaps for context variant V2 (below) and with the stale rounds left out (a round is stale when, for any series, registration start - 2 h - `context_end_time` > 6 h; the clock offset is 2 h because the whole window is CEST); if V2 or the stale-free subset moves R1 or R2 across its threshold, the verdict text says "fragile"; (c) the per-task paired log-ratio 100*mean ln(err_B/err_A) with ISO-week block intervals, the share of tasks won, and the old-style arena MASE (per-round mean over series, then over rounds) with its per-round difference and ISO-week block interval, so the live window can be set beside the wording of our public texts; (d) routing counts, latency per series (1 thread, Kaggle CPU), and the context diagnostic (below).

8. Context variants and disclosure.  V1 (primary, gates): the context is what `/data` returns for `context_start_time .. context_end_time`, i.e. CURRENT stored values.  The field's models received the served context, whose last hours were first-published values and could differ.  So V1 is cleaner than the field's live contexts, to our advantage; stale contexts (the platform's context ends hours before the registration) are reproduced exactly because the context end comes from the series metadata.  V2 (secondary): V1 with the last P points (P = 96 at 15 min, 24 hourly) replaced by the stored forecast of the platform's `Statistical/Seasonal-Naive` model, which repeats the last season of the served context; used only for tasks where that model has the forecast and the identity check passes (at least 90 % of its points equal V1's context values lagged by P within 1e-6 relative; else V2 = V1 for that task and the count is reported).  Diagnostic: per task, the share of the last P points where served and current values differ, and the mean absolute difference over the context scale.  Other limits: truth is the current stored value, the platform scored at evaluation time (R0 measures the effect); the whole window is CEST, so the DST switch of 2026-10-25 is not tested; registration timing (definition 2: 08:00-08:15 UTC; definition 5: 09:00-10:00 UTC) and the key-gated upload path are not tested; the field changes inside the window (14 to 56 models per round at definition 2); the Open Track entry would not be in the Reference Track; the platform's own board uses the full history, not these 8 weeks.

9. Read before this entry (disclosure).  From the public site: the definitions and model lists, the round listings of definitions 2 and 5 (ids, dates, status, model counts; page 1, 100 rounds each), and the series metadata (context ends) of the newest 60 rounds of definition 2, used for the staleness check (stale, i.e. > 6 h: ids 12144 (08-30, 12.5-13.3 h), 12176 (09-01, 12.3-12.8 h), 12192 (09-02, 19.3-19.8 h), 12288 (09-08, 8.8-10.8 h), 12304 (09-09, 25.5-26.3 h), 12480 (09-20, three series up to 14 h); the other 54 rounds of that listing have every series within 5.75 h); one in-window `/data` call (round 12928, series 109, 2026-10-17 05:00Z .. 10-18 05:15Z) compared with SMARD's own API for the clock check (platform timestamp = SMARD UTC - 2 h, 98 of 98 points equal) and never with a forecast or score.  Out of window and for the recipe check only: round 11493 (definition 2, 2026-07-20): series, models, the whole leaderboard, one series' forecasts, context and truth; the recomputed MASE of visiontspp-large, chronos-2, seasonal-naive and flowstate on series 109 matched the stored `mase` to 6 digits (that is why the recipe in 3 is the pre-registered one).  Platform source (`arena_score.py`, `forecast_metrics.py`, `arena_score_service.py`, `arena_scores_repository.py`, `mase_window.py`) and the 2026-10-15 Arena Score announcement.  Archive-era numbers only for the replica (TEST-F, `scripts/arena_replica/`).  No in-window leaderboard row, stored forecast, truth value (other than the 98 points above) or score of any model has been read, and none of our forecasts for the window exists.

10. Procedure.  Private Kaggle CPU kernel `sankalpsthakur/bench-timeseries-shadow` (internet on), built from the committed code; stages: asserts (hashes), smoke on out-of-window rounds 11493 (definition 2) and 11494 (definition 5) (fetch, recompute, compare with stored scores, run the service) - the kernel stops before the window if the smoke fails -, fetch and score the window, analysis.  Pushed once, under a Kaggle CPU slot lock.  A fix and re-push before any window score exists is logged as an attempt; a re-push after window scores exist is a further TEST-L1 evaluation and is logged as such with the reason; no change of definition after this entry is made silently (deviations are listed in the post-run notes).  No model, round or series is dropped after the fact except by the eligibility rules in 2 and the stale-free subset in 7(b).  The smoke-stage forecasts on the out-of-window rounds are not part of TEST-L1.  Everything not listed above is exploratory and labelled so.  The verdict is computed locally from the fetched tables by `scripts/shadow_verdict.py` (through `bin/guard.sh`), which reads R0 first.

### TEST-L1 Amendment 1 (written 2026-10-19 20:03 UTC; before the kernel was pushed; since the pre-registration only out-of-window rounds have been read, no in-window leaderboard row, stored forecast, truth value or score)

Why.  Plumbing tests of the kernel code on out-of-window rounds (definition 2: ids 11477, 11493, 11509, 11523; definition 5: 11478, 11494, 11510, 11524; 80 tasks; a stand-in service replaced our nets, nothing was loaded locally) showed that the recipe of items 3-4 would have failed for a reason that has nothing to do with the model:
- With the scale taken from the CURRENT `/data` context, the recomputed MASE and SQL of every model differ from the stored ones by one common factor per task (spread across models <= 5e-16, factors 0.988 .. 1.010); the scale equals the stored one (to 1e-9) in only 12 of 40 (definition 2) and 7 of 40 (definition 5) tasks, and differs by more than 1e-3 in 40 % and 23 % of the 30 plumbing-test tasks of each definition.  Forecasts and truth are reproduced exactly (the MAE ratio is the same for all models).  Under the pre-registered R0 the run would have been NOT EVALUABLE.
- The stored scale comes from the context the platform SERVED, whose last hours are first-published values that SMARD has since revised.  Replacing the last P points (P = 96 at 15 min, 24 hourly) of the current context by the stored forecast of the platform's `Statistical/Seasonal-Naive` model (it repeats the last season of the history it received, by position, so its first P forecast values are the last P served points) reproduces the stored scale in 80 of 80 tasks to 2e-16, and then every stored `mase` and `sql_score` of every field model on the 60 plumbing-test tasks (1130 and 1080 rows) within 1e-3 (visiontspp-large: relative difference 0.0).  Served and current values differ on 0 .. 83 % of those P points per series (mean 29 % and 33 % over the plumbing-test tasks of definitions 2 and 5; mean absolute difference up to 14 % of the context scale).

Changes (everything else in the pre-registration stands):
- A1, V2 (item 8): the served context = the current context with its last P points replaced, by position, by the first P forecast values of `Statistical/Seasonal-Naive`; applied when the current context has exactly 1000 points, that forecast starts one step after the context end and has P finite values, and the replaced values stay within 5 % of the level of the current ones.  This replaces the "at least 90 % identical" test, which would have rejected exactly the revised cases V2 is for.  Otherwise V2 = V1 for the task (counted).
- A2, scale (item 3): per task, `scale` and the naive value (the last finite context value) are those of the V2 context when V2 is built, else of V1; every row of the task (field, ours, naive anchor) is scored with them.  The Arena Score gaps are invariant to this per-task constant; the levels and R0 are not.  The old-style arena MASE (mae / naive_mae) uses the same naive value.
- A3, gate variant (items 6, 7b): R1, R2 and R3 are evaluated on V2.  Reason: V2 is the context the field received (checked by the exact scale match), so it is the "same contexts" of the brief, whereas V1 (current data) is cleaner than any live context and would overstate a live entry.  V1 stays in the report as the optimistic bound; the "fragile" flag compares V2 with the stale-free subsets, and a separate line says whether V1 alone would change the verdict.  The fallback share of R3 is the V2 one (V1 also reported).
- A4, R0 (item 4) keeps its form (|recomputed - stored| <= 1e-3 * stored for `mase` and `sql_score`, same `has_quantiles`, share >= 0.95, at least 140 rows, for the two gate comparators) with "recomputed" using the A2 scale.  Added as diagnostics, not gates: the share reproduced for `rmse`, and the relative difference between the V1 and the served scale per task.
- A5, smoke (item 10): the smoke stage also requires a field match share >= 0.9 with the A2 scale (local stand-in run: 1.0 on both smoke rounds).  Tables: `scores_*` gain `rmse` / `stored_rmse`; `tasks_*` gain `scale_v1`, `scale_v2` and lose `sn_identity`.

Code and checks before the push.  `kernels/shadow/{shadow_lib,shadow_analysis,shadow_kernel,build_kernel}.py` (commit recorded in the STARTED entry).  Checks run locally (all through `bin/guard.sh`, none loads a model): (a) the plumbing test above; (b) `scripts/arena_replica/check_vs_platform.py`: the platform's vendored engine on the Archive TEST-F tables gives the same Bradley-Terry gaps as the scout's replica (definition 5, 35 rounds, point: -4.1 against chronos-2; 52 rounds -0.3; SQL -5.3 / -1.0) and the whole TEST-L1 analysis code on those tables reproduces the replica's paired log-ratios (-4.47 [-7.43, -1.05], SQL -5.48 [-9.81, -1.85]) and the public old-style arena MASE differences (+0.0053 [-0.0006, +0.0103] on 35 rounds, -0.0021 [-0.0154, +0.0071] on 52).

### TEST-L1 Amendment 2 (written 2026-10-19 20:21 UTC; before the kernel was pushed; since Amendment 1 only out-of-window and non-window data were read: rounds of definitions 3 and 6 listed below; no in-window row of definitions 2 or 5)

Why.  (1) Coupling.  Amendment 1 kept R0 as "the two gate comparators are reproduced with at least 140 stored rows each".  If SCCH/PCFM-FT joined late or does not play definition 5, that clause alone would delete the whole report, including the visiontspp-large gap, for a reason that is about a comparator's sample size and not about the pipeline.  (2) Age.  A test of the field-only rescoring (stand-in service, V2 scale) on 12 rounds of the non-window SMARD generation definitions 3 (24 h at 15 min; ids 12922, 12906, 12842, 12762, 12618, 12330) and 6 (72 h at 1 h; ids 12893, 12877, 12845, 12765, 12621, 12333), aged 2 to 39 days, showed that the stored scores of recent rounds are not a fixed reference: the unconditional share of stored `mase` / `sql_score` reproduced within 1e-3 was 0.49 .. 0.94, because inputs changed after the platform scored (the scale-free `rmse` is reproduced in only 55 .. 86 % of the rows of rounds up to 14 days old, 25 .. 56 % at 1 h at the 1e-6 level, and in 100 % of the oldest rounds: truth revised after the evaluation, or fewer points scored then) and because the served context differs from V2 beyond the last season in some tasks (scale factor f != 1 in 8 .. 62 % of tasks).  Both act on all models of a task alike and cannot move an Arena Score gap.  With a screen on identical inputs (rmse within 1e-6) and ONE factor per task, the structure of `mase`, `sql_score` and `has_quantiles` was reproduced in 100.0 % of the informative rows of definition 3 at all six ages (75 .. 100 % of the rows informative) and in 97.1 .. 100 % at definition 6 (25 .. 56 % informative for rounds up to 21 days old, 100 % for the oldest).

Changes (everything else, including Amendment 1, stands; B1 replaces the R0 text of item 4 and of A4):
- B1, R0 per definition, on the R0 comparator visiontspp-large (id 131) only.  Rows: its stored `is_final` rows of eligible tasks.  A row is informative when the recomputed `rmse` equals the stored one within 1e-6 relative (same forecasts, same truth, same points as when the platform scored).  Per task, f = the median over the informative rows of ALL models of mase_recomputed / mase_stored (the ratio of the stored context scale to ours).  An informative row passes when mase_recomputed / mase_stored and sql_recomputed / sql_stored both equal f within 1e-6 relative (or both are undefined) and `has_quantiles` agrees.  R0 holds iff at least 100 informative rows exist, at least 95 % of them pass, and V2 was built for at least 90 % of the eligible tasks.  Rows that are not informative are reported by round age, not scored as failures; f is a diagnostic (the share of tasks with f = 1 is the share whose served context V2 reproduces exactly, which the report states) and does not enter any gap.
- B2, other comparators (SCCH/PCFM-FT, farang/fx2-medium-thinking, chronos-2): the same test with at least 20 informative rows.  A comparator that fails it, or has too few rows, is dropped from the report; for SCCH/PCFM-FT that makes R2 not evaluable and the verdict NOT EVALUABLE.
- B3, evaluability of a gap (separate from R0, and my addition to the brief's "rounds they played"): a gate comparator needs at least 140 common eligible tasks with ours in definition 2 (14 rounds) for its gap to count; below that the gap is reported and the verdict is NOT EVALUABLE.  The floor stops a gate from being decided on a handful of tasks.
- B4, a definition whose R0 fails is not evaluated (no gap involving ours is computed for it); a failure at definition 5 does not touch the definition-2 verdict.

### TEST-L1 entries

### 2026-10-19 20:23 UTC  TEST-L1  definitions 2 and 5  window 2026-08-24..2026-10-18  variant: gridload-compact v1, int8 service path (gate: V2 served context; bound: V1 current context)  STARTED  (kernel sankalpsthakur/bench-timeseries-shadow v1; code commit 631e6a6ecaa1e1a450967b3150f21f87f4c5961b; kernels/shadow/script.py sha256 7ddc034dd55e458729e313179ee0b7f89f689e4c70f5287fe9d0c1c6cee68a2b, kernel-metadata.json sha256 4f99e52637f9d28670db73be1d0457ad93f1167b438443d060455c3f763cac29; pre-registration ffef896, Amendment 1 61eea8b, Amendment 2 ee6fa56; Kaggle CPU slot locks/kaggle-cpu-2).  First and only push so far; no in-window score of any model exists; the result entry follows when the run completes.  Local checks before the push (stand-in service, out-of-window and non-window rounds only): plumbing run, packaging run of the generated script from an extracted tree, analysis code on the Archive TEST-F tables.

### 2026-10-19 20:48 UTC  TEST-L1  definitions 2 and 5  result  (kernel sankalpsthakur/bench-timeseries-shadow v1: COMPLETE 2026-10-19 20:44:43 UTC, 1,078.6 s in the kernel, 2,465 API calls, 0 retries, 0 API errors; 9 files, 2.5 MB, fetched once and checked against the kernel manifest: none missing, none different; smoke on the out-of-window rounds passed; slot kaggle-cpu-2 released)
R0 FAILED in both definitions, on the V2-coverage clause alone.  V2 was built for 293 of 488 eligible tasks of definition 2 (0.600) and 184 of 416 of definition 5 (0.442), under the required 0.90.  The reproduction itself is sound: visiontspp-large has 340 informative rows of 468 stored final rows in definition 2 (structure share 1.0000) and 276 of 416 in definition 5 (0.9928).  Pre-registered consequence (item 4): verdict NOT EVALUABLE; no gap involving our model is computed or quoted.  None was: the kernel's analysis returned before the Arena fits, and `scripts/shadow_verdict.py` stopped after R0 (`results/shadow/verdict.json` holds the R0 figures only).  No score, rank or gap of our model has been read.
Read afterwards, field rows and task flags only (`scratchpad`-style diagnostics, not stored): (i) 195 / 232 eligible tasks had no V2: 182 / 229 failed the 5 %-of-level sanity check, 13 / 3 failed a structural test; the Seasonal-Naive forecast differs from the current context on 89-100 % of the last-season points (mean absolute difference 0.26 context scales where built, 0.9-1.1 where rejected), which was not the case in the July rounds of Amendment 1.  (ii) V2 does not reproduce the stored context scale in the window: tasks with a defined factor f, exact scale (|f-1| <= 1e-9) in 16 % (definition 2) and 7.5 % (definition 5) of the tasks where V2 was built, against 35 % and 20 % where it was not; median |f-1| 0.7 % / 0.3 % against 0.03 % / 0.01 %.  So the stored Seasonal-Naive forecast is not the last season of the served context for most window tasks, and V2 is NOT validated in the window; the 80-of-80 match of Amendment 1 came from rounds 2026-07-19 .. 07-22.  (iii) Truth drift: the rmse of visiontspp-large is reproduced in 36 % of its rows of rounds up to 13 days old, 73 % at 14-27 days and 100 % beyond 28 days.  (iv) SCCH/PCFM-FT has 50 stored final rows in definition 2 (5 rounds; 19 informative) and 30 in definition 5 (14), so the 140-task floor of B3 cannot be met; farang/fx2-medium-thinking has 214 / 208 rows, chronos-2 488 / 416.

## TEST-L2: the same tables read under the original context variant (pre-registration, written 2026-10-19 20:48 UTC; before any statistic involving our model was computed or viewed)

Why.  TEST-L1 ended NOT EVALUABLE on a clause (V2 coverage) whose premise, that the Seasonal-Naive forecast is the served last season, holds for the July rounds and not for the window (facts (i)-(ii) above).  The pre-registration (item 4) allows a different recipe only as a new pre-registration written before our scores are looked at; this is it.  It uses the SAME fetched tables (no new fetch, no new run; our forecasts are those of kernel v1 computed from the current-data context, the variant the brief and the original item 8 call V1).  Everything not changed here stands as in the pre-registration and Amendments 1-2.
- L2-1, gate variant V1 (the brief's literal rule): R1, R2, R3 of item 6 are evaluated on the V1 rows (`ours_v1`).  V2 is dropped from the gate and from the report (not validated).  V1 is CLEANER than any live context: the field received the served context, which differs from the current data (the stored context scale differs from ours by more than 1e-3 in a large share of the tasks, reported in the result).  So V1 is an upper bound for a live entry: NO-GO on V1 is conclusive for asking the owner; GO on V1 only means "not excluded by this check", and a live entry would still need a trial on served contexts.
- L2-2, R0 (per definition): visiontspp-large (id 131) has at least 100 informative rows and a structure share >= 0.95, as in Amendment 2 B1 without the V2-coverage clause (met in both definitions, see the result above).  Other comparators: reproduced iff at least 10 informative rows (L1: 20) and share >= 0.95; one that fails is dropped from the report.
- L2-3, scale: scoring scales are the per-task `scale` column of the tables (a constant per task shared by all rows, V2's where it was built, else V1's); Arena gaps do not depend on it.
- L2-4, two readings of the verdict, both reported, none chosen afterwards.  SCCH/PCFM-FT has played 5 rounds of definition 2.  Strict: B3 as written (at least 140 common tasks), which yields NOT EVALUABLE if the floor is not met.  Literal brief ("rounds they played"): R2 is evaluated on the common tasks if there are at least 30 (3 rounds), with its interval reported, and the verdict (GO / NO-GO) follows R1-R3 on V1.  The recommendation states the strict reading first.
- L2-5, statistics, intervals and claim rule as in items 5-7 (joint fit of the platform's engine; 7-day circular blocks with 1000 refits, 14-day with 500, the platform's 28-day with 200, seed 0; paired log ratio; old-style arena MASE), computed locally by `scripts/shadow_l2.py` through `bin/guard.sh` in stages (the 60 s limit), committed before it is run.

### 2026-10-19 20:55 UTC  TEST-L2  definitions 2 and 5  result  (`scripts/shadow_l2.py`, commit b44f45f, on the TEST-L1 tables as fetched; details `results/shadow/l2/verdict_l2.md`, `l2_points.json`, `ci/`)
R0 (L2-2) holds: visiontspp-large 340 informative rows, structure share 1.0000 (definition 2); 276, 0.9928 (definition 5); SCCH/PCFM-FT 19 and 14 informative rows, reproduced.  Definition 2, probabilistic ranking, V1 (current-data contexts), 488 eligible tasks of 560, ours 145.8 points, place 14 of 46 fitted models:
- R1, gap over visiontspp-large +1.7 [-4.2, +8.5] (7-day blocks; 14-day [-3.6, +9.1]; 28-day [-0.6, +4.6]), 468 common tasks in 49 rounds: below +5.0, not met.  Paired per-task log ratio +2.78 [-3.27, +10.61]; old-style arena MASE ours 0.3148 against 0.3199, -0.0051 [-0.0311, +0.0204], 63 % of rounds won.
- R2, gap over SCCH/PCFM-FT -17.4 [-29.4, -8.7], 50 common tasks in 5 rounds: below -5.0, not met (the whole interval is below -5).
- R3, fallback share 0.1578 (77 of 488 tasks): above 0.05, not met.  All 77 fall in the rounds of 2026-09-03 .. 09-23 (0 of 250 tasks from 09-24 on); the current series of the platform have gaps of up to 63 h inside the context windows of those rounds (platform-side data gap in early September), which the service's input gates send to the seasonal fallback.
- Stale-free subset (6 stale rounds out): R1 +1.6, R2 -17.9: no flip.  Point ranking (MAE): over visiontspp-large +4.9 [-1.3, +13.1], over SCCH/PCFM-FT -14.5 [-25.7, -5.5], over farang/fx2-medium-thinking -21.2 [-41.1, -4.9], over chronos-2 -8.2 [-21.8, +2.8]; place 12 of 61.  Platform-style score 145.8 [131.3, 160.3], rank range 1-17.
Readings (L2-4): strict (B3, at least 140 common tasks per gate comparator): NOT EVALUABLE (SCCH/PCFM-FT has 50); literal brief (rounds they played, at least 30 tasks): NO-GO (R1, R2 and R3 all unmet).  V1 is the optimistic context: over the 370 (definition 2) and 281 (definition 5) tasks with a defined factor, the V1 context scale equals the stored (served) scale exactly in 42 % and 27 %, differs by more than 1e-3 in 29 % and 17 %, by more than 1 % in 8.6 % and 0 % (p90 0.73 % and 0.18 %, max 3.1 % and 0.9 %).  A NO-GO on V1 is therefore conclusive for asking the owner now.
Definition 5 (reported, no gate), V1, 416 eligible tasks, SQL: over visiontspp-large +17.8 [+5.3, +28.7]; over chronos-2 -22.1 [-51.2, -4.4] (MAE -18.3 [-45.2, -2.1]; old-style arena MASE +0.0660 [+0.0054, +0.1778] against chronos-2); over SCCH/PCFM-FT -3.9 on 30 tasks (3 rounds); place 14 of 45; fallback share 0.1659 (69 of 416, same early-September rounds).
Exploratory (not pre-registered, changes no verdict; `results/shadow/l2/explore_l2.md`): from 2026-09-24 on (no fallbacks) definition 2 SQL: +0.7 [-4.9, +9.4] over visiontspp-large, -12.1 [-21.7, -3.9] over SCCH/PCFM-FT, place 9 of 46; definition 5 SQL: -3.3 [-7.9, +1.1] against chronos-2, +24.8 over visiontspp-large, place 3 of 45.  On the specialist-served tasks of all rounds, definition 2 SQL over visiontspp-large +6.8 [+0.4, +17.9], over SCCH/PCFM-FT -8.9 [-19.0, +0.6].

### TEST-L1 / TEST-L2 post-run notes (written 2026-10-19 20:55 UTC; they select nothing)
1. One kernel run exists (v1, pushed once 2026-10-19 20:23:54 UTC); no further evaluation of our model on the window was run.  The L1 verdict (gate V2, Amendments 1-2) is NOT EVALUABLE; the L2 readings are above.
2. Order of events, for the record: pre-registration ffef896 -> Amendment 1 61eea8b and Amendment 2 ee6fa56 (both before the push, from out-of-window and non-window rounds only) -> push 20:23:54 -> fetch -> R0 failed on the V2-coverage clause (8348778 records it and pre-registers L2 before any statistic of our model was computed) -> `shadow_l2.py` committed (b44f45f) -> run.  Facts about the field and the task flags were read between the R0 failure and the L2 registration; no score, rank or gap of our model was.
3. Code state.  The kernel ran build commit 631e6a6.  Later commits do not change what the kernel computed: `shadow_analysis.analyze_definition` got an optional `platform=` flag (default as before), plus `scripts/shadow_verdict.py` (L1), `scripts/shadow_l2.py` and `scripts/shadow_l2_explore.py` (local analyses); the `platform` stage of `shadow_l2.py` (platform-style score and rank range, item 7a) was added after the points stage had run.
4. Unverified: whether the served context (what a live entry receives) changes the L2 numbers beyond what V1's scale deviations suggest; V2 could not be validated in the window (L1 result, facts (i)-(ii)); the DST switch of 2026-10-25, registration timing and the key-gated upload path are untested; farang/fx2-medium-thinking has no deciles in the probabilistic fit (point ranking only); the late-March archive rounds do not exist on the live board.
5. Rule history, in one place.  The gate was V1 in the pre-registration (the brief), V2 in Amendment 1 (before the push), and V1 again in TEST-L2 (after V2 failed in the window).  The two readings of the verdict (strict / literal) and the lowering of the other-comparator floor from 20 to 10 informative rows were set after TEST-L1 had failed and after the field was known (SCCH/PCFM-FT had 19 informative rows in definition 2, so at 20 it would have been dropped and the literal reading would read NOT EVALUABLE, not NO-GO).  None of these changes touched a number of our model, and none can change GO versus no-GO: R1 (+1.7 against +5) and R3 (fallback 0.158 against 0.05) fail under every version, so no reading can give GO whatever SCCH/PCFM-FT does.
6. Wording of "reproduced".  The stored scores of visiontspp-large are reproduced up to one scale factor per task (1e-6) on the rows whose inputs are unchanged: 73 % of its rows in definition 2 (66 % in definition 5); the others differ because the truth was revised after the platform scored.  Absolute MASE and SQL values match only where the context scale matches (with V1 contexts in 42 % of the tasks of definition 2 and 27 % of definition 5).  "Upper bound" for V1 is an argument (its context is consistent with the revised truth, the field's served tails were first-published), not a measurement.  The naive anchor and the old-style arena MASE denominators use the last value of the V2 context in the tasks where V2 was built (about 60 % / 44 %; unvalidated in the window); the Arena Score gaps do not depend on it.
