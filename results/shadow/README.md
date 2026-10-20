# results/shadow: shadow check of gridload-compact v1 on completed live TS-Arena rounds (TEST-L1 / TEST-L2)

Self-evaluated shadow check on public data, not an official ranking.  Window: the completed live rounds of definitions 2 and 5 registered 2026-08-24 to 2026-10-18; read from the platform's public dashboard API on
2026-10-19 (kernel run 20:24 to 20:42 UTC); verdict `l2/verdict_l2.md` written 2026-10-19 (UTC).  Headline, definition 2, probabilistic ranking: 14th of 46 fitted models; gap over visiontspp-large +1.7 [-4.2, +8.5]
(the pre-registered rule needed >= +5.0); fallback share 15.8 % (at most 5 % allowed); the rule failed, so no live entry was made.  The contexts are rebuilt from current data (variant V1), which is probably optimistic against
the contexts the field received.

Pre-registration, amendments, ledger and post-run notes: `../../TEST-LOG.md` (sections "TEST-L1", "TEST-L1 Amendment 1/2", "TEST-L1 entries", "TEST-L2").  Sources: `../../kernels/shadow/` and `../../scripts/shadow_*.py`.
Data and licences: `../../NOTICE`, items 7 and 8.

| path | what |
|---|---|
| `out/` | files fetched from the private Kaggle kernel `sankalpsthakur/bench-timeseries-shadow` v1: `manifest.json` (SHA-256 of every file, of the kernel sources and of the service files), the per-task tables of our model `tasks_def{2,5}.csv.gz`, `model_names.json`, `smoke.json`, `log.txt`; `analysis.json` holds R0 only (the kernel stopped before the Arena fits).  **Not included:** `scores_def{2,5}.csv.gz` (the platform's stored per-(round, series, model) scores of about 70 field models, 2.4 MB); they come from the platform's dashboard API and we did not find a licence statement for them (see NOTICE item 7).  Their SHA-256 are in `manifest.json`; the files are available on request |
| `kernel_log.txt` | the Kaggle log of the run |
| `verdict.json` | TEST-L1 verdict from `scripts/shadow_verdict.py`: R0 figures; NOT EVALUABLE (V2 coverage), nothing about our model |
| `l2/verdict_l2.md`, `verdict_l2.json`, `l2_points.json`, `platform_style.json`, `ci/` | TEST-L2: V1 gate, both readings of the verdict, gaps with 7 / 14 / 28-day-block intervals, paired statistics, old-style arena MASE |
| `l2/explore_l2.md`, `explore_l2.json` | exploratory cuts (not pre-registered): rounds from 2026-09-24, specialist-served tasks |
| `archive_crosscheck/` | Archive TEST-F cross-checks: a replica of the platform's Arena Score re-run, the platform's own engine (vendored in `kernels/shadow/vendor/`) on the same tables, the two metrics side by side (the replica scripts need the Archive field cache, which is not part of this repository) |

Re-running: `scripts/shadow_verdict.py` and `scripts/shadow_l2.py` read `out/scores_def{2,5}.csv.gz`, which are not included, so the published `l2/` files can be inspected and the hashes of their inputs checked, but not recomputed from this
repository alone.  A fresh fetch with the kernel sources in `kernels/shadow/` would differ in details because the platform re-scores recent rounds (TEST-LOG, TEST-L1 Amendment 2).  `kernels/shadow/shadow_analysis.py` is the version
after commit 6bb7b73 of the private working repository: it adds an optional `platform=` flag to the version that ran in the kernel (SHA-256 `d9312a94...` in `out/manifest.json`); the default behaviour is unchanged.  The other kernel
sources equal the hashes in `out/manifest.json`.  The generated kernel script (`script.py`, about 260 kB with the service files embedded) is not committed, as for the other kernels of this repository; the service files it embedded are
the ones in `submission/model-services/gridload-compact/` (hashes in `kernels/shadow/V1-MANIFEST.txt`).
