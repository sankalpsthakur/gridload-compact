# TEST-L2 verdict: gridload-compact v1 against the live field (current-data contexts, V1)

Window 2026-08-24 .. 2026-10-18 (completed live rounds), kernel build `631e6a6eca`; self-evaluated shadow check on public data, contexts reconstructed from the current data (consistent with the revised truth used for scoring, unlike the first-published tails of the field's served contexts: probably optimistic for a live entry, by an amount not measured); not an official ranking.

**Strict reading (B3, at least 140 common tasks per gate comparator): NOT EVALUABLE (SCCH/PCFM-FT has 50 common tasks with ours in definition 2, under the 140 of B3)**  
**Literal-brief reading (SCCH/PCFM-FT on the rounds it played, at least 30 tasks): NO-GO**

**No reading can give GO: R1, R3 fail on their own, whatever SCCH/PCFM-FT does.**

| condition | value | threshold | met |
|---|---|---|---|
| R1 definition 2, SQL gap vs visiontspp-large | +1.7 [-4.2, +8.5] (468 common tasks) | >= +5.0 (point) | False |
| R2 definition 2, SQL gap vs SCCH/PCFM-FT | -17.4 [-29.4, -8.7] (50 common tasks) | >= -5.0 (point) | False |
| R3 definition 2 fallback share (V1) | 0.1578 | <= 0.05 | False |
| stale-free subset flips R1 or R2 | {'nonstale': {'R1_gap': 1.5798140648778087, 'R2_gap': -17.92975690853072, 'R1': False, 'R2': False}} | - | fragile = False |

## Definition 2

Tasks 560 (56 rounds), eligible 488, dropped {'truth_incomplete': 72}; stale rounds (> 6 h): ['12144', '12176', '12192', '12288', '12304', '12480'].  Fallback share (V1) 0.1578; latency 4.81 ms per series (p50, Kaggle CPU, 1 thread).  R0: visiontspp-large 340 informative rows, structure share 1.0000; the stored scores of recent rounds drift (rmse reproduced in 73 % of its rows overall).  V1 context scale versus the stored (served) scale over 370 tasks with a defined factor: exact in 42 %, off by more than 1e-3 in 29 % and by more than 1 % in 9 % (median 0.00 %, p90 0.73 %): V1 is probably optimistic against the served context (not measured).

**probabilistic ranking (SQL), context V1, all rounds**: ours 145.8 points (place 14 of 46 fitted models), 488 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +1.7 | [-4.2, +8.5] | [-3.6, +9.1] | [-0.6, +4.6] | 468 | 49 |
| SCCH/PCFM-FT | -17.4 | [-29.4, -8.7] | [-27.7, -10.0] | [-28.9, -9.5] | 50 | 5 |
| farang/fx2-medium-thinking | n/a (not in the fit) | | | | 0 | |
| chronos-2 | -13.3 | [-30.6, +0.9] | [-35.4, +1.6] | [-32.3, +1.4] | 488 | 51 |

top 10 of the fit: SCCH/PCFM-FT 163.2; timestral-123m 161.2; chronos-2 159.1; timestral-209m 155.7; tabby-pretrain 155.3; tabby 155.1; timestral-28m 154.0; t0-gamma 151.8; timestral-5m 151.4; moirai-2-small 147.3

platform-style score [interval of the 28-day bootstrap with the weekly floor], rank range: gridload-compact v1 (int8) 145.8 [131.3, 160.3] ranks 1-17; visiontspp-large 144.1 [129.6, 158.6] ranks 1-17; SCCH/PCFM-FT 163.2 [127.9, 198.6] ranks n/a; chronos-2 159.1 [144.6, 173.6] ranks 1-16

**probabilistic ranking (SQL), context V1, stale rounds left out**: ours 146.0 points (place 14 of 46 fitted models), 478 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +1.6 | [-4.3, +8.5] | n/a | n/a | 458 | 47 |
| SCCH/PCFM-FT | -17.9 | [-30.1, -9.4] | n/a | n/a | 50 | 5 |
| farang/fx2-medium-thinking | n/a (not in the fit) | | | | 0 | |
| chronos-2 | -13.6 | [-31.2, +0.7] | n/a | n/a | 478 | 49 |

**point ranking (MAE), context V1, all rounds**: ours 128.7 points (place 12 of 61 fitted models), 488 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +4.9 | [-1.3, +13.1] | [-1.6, +14.3] | [+0.1, +11.8] | 468 | 49 |
| SCCH/PCFM-FT | -14.5 | [-25.7, -5.5] | [-24.0, -7.3] | [-21.1, -6.2] | 50 | 5 |
| farang/fx2-medium-thinking | -21.2 | [-41.1, -4.9] | [-39.3, -5.8] | [-30.8, -9.4] | 214 | 22 |
| chronos-2 | -8.2 | [-21.8, +2.8] | [-25.4, +3.4] | [-22.3, +2.3] | 488 | 51 |

top 10 of the fit: farang/fx2-medium-thinking 149.9; SCCH/PCFM-FT 143.3; timestral-123m 139.9; chronos-2 136.9; t0-gamma 135.5; timestral-28m 133.9; timestral-209m 133.7; tabby 133.6; tabby-pretrain 133.4; greydow 133.1

platform-style score [interval of the 28-day bootstrap with the weekly floor], rank range: gridload-compact v1 (int8) 128.7 [114.1, 143.4] ranks 1-21; visiontspp-large 123.8 [109.2, 138.5] ranks 1-23; SCCH/PCFM-FT 143.3 [108.8, 177.8] ranks n/a; farang/fx2-medium-thinking 149.9 [131.7, 168.1] ranks 1-16; chronos-2 136.9 [122.3, 152.9] ranks 1-20

**point ranking (MAE), context V1, stale rounds left out**: ours 129.1 points (place 12 of 61 fitted models), 478 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +5.0 | [-1.3, +13.3] | n/a | n/a | 458 | 47 |
| SCCH/PCFM-FT | -14.9 | [-26.1, -5.9] | n/a | n/a | 50 | 5 |
| farang/fx2-medium-thinking | -21.8 | [-42.0, -5.0] | n/a | n/a | 207 | 21 |
| chronos-2 | -8.3 | [-22.1, +2.8] | n/a | n/a | 478 | 49 |

paired log ratio of per-task errors, 100 * mean ln(err_comparator / err_ours), ISO-week block interval, tasks won:

| comparator | metric | mean | 95 % interval | tasks | share won |
|---|---|---|---|---|---|
| chronos-2 | point | -9.43 | [-25.30, +1.86] | 488 | 0.47 |
| chronos-2 | prob | -14.12 | [-33.91, -0.27] | 488 | 0.45 |
| visiontspp-large | point | +4.77 | [-3.19, +14.89] | 468 | 0.57 |
| visiontspp-large | prob | +2.78 | [-3.27, +10.61] | 468 | 0.54 |
| farang/fx2-medium-thinking | point | -26.48 | [-52.53, -11.38] | 214 | 0.32 |
| SCCH/PCFM-FT | point | -13.38 | [-13.38, -13.38] | 50 | 0.26 |
| SCCH/PCFM-FT | prob | -11.69 | [-11.69, -11.69] | 50 | 0.24 |

old-style arena MASE of our public texts (mae / flat-naive mae; mean over series, then rounds), ours minus comparator, ISO-week bootstrap:

| comparator | ours | comparator | difference | 95 % interval | rounds | rounds won |
|---|---|---|---|---|---|---|
| chronos-2 | 0.3147 | 0.2697 | +0.0450 | [-0.0082, +0.1168] | 51 | 0.43 |
| visiontspp-large | 0.3148 | 0.3199 | -0.0051 | [-0.0311, +0.0204] | 49 | 0.63 |
| farang/fx2-medium-thinking | 0.3283 | 0.2338 | +0.0945 | [+0.0250, +0.2209] | 22 | 0.27 |
| SCCH/PCFM-FT | 0.2181 | 0.1964 | +0.0217 | [+0.0217, +0.0217] | 5 | 0.00 |

## Definition 5

Tasks 540 (54 rounds), eligible 416, dropped {'truth_incomplete': 124}; stale rounds (> 6 h): ['12146', '12178', '12194', '12210', '12290', '12306', '12482'].  Fallback share (V1) 0.1659; latency 4.84 ms per series (p50, Kaggle CPU, 1 thread).  R0: visiontspp-large 276 informative rows, structure share 0.9928; the stored scores of recent rounds drift (rmse reproduced in 66 % of its rows overall).  V1 context scale versus the stored (served) scale over 281 tasks with a defined factor: exact in 27 %, off by more than 1e-3 in 17 % and by more than 1 % in 0 % (median 0.01 %, p90 0.18 %): V1 is probably optimistic against the served context (not measured).

**probabilistic ranking (SQL), context V1, all rounds**: ours 162.5 points (place 14 of 45 fitted models), 416 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +17.8 | [+5.3, +28.7] | [+2.7, +30.7] | [+2.5, +26.4] | 416 | 45 |
| SCCH/PCFM-FT | -3.9 | [-20.3, +6.1] | [-22.0, +6.8] | [-23.8, +7.5] | 30 | 3 |
| farang/fx2-medium-thinking | n/a (not in the fit) | | | | 0 | |
| chronos-2 | -22.1 | [-51.2, -4.4] | [-54.3, -2.6] | [-52.7, -3.2] | 416 | 45 |

top 10 of the fit: chronos-2 184.6; moirai-2-small 182.5; timestral-123m 170.4; timestral-28m 168.8; moirai-base-model 168.6; toto-2.0-2.5b 167.0; tabby 167.0; SCCH/PCFM-FT 166.3; moirai-large 165.4; toto-2.0-313m 165.2

platform-style score [interval of the 28-day bootstrap with the weekly floor], rank range: gridload-compact v1 (int8) 162.5 [141.8, 182.7] ranks 1-23; visiontspp-large 144.7 [124.4, 164.9] ranks 1-24; SCCH/PCFM-FT 166.4 [123.1, 209.6] ranks n/a; chronos-2 184.6 [164.4, 204.9] ranks 1-15

**probabilistic ranking (SQL), context V1, stale rounds left out**: ours 164.4 points (place 14 of 45 fitted models), 399 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +17.9 | [+4.7, +29.5] | n/a | n/a | 399 | 43 |
| SCCH/PCFM-FT | -3.3 | [-20.0, +6.7] | n/a | n/a | 30 | 3 |
| farang/fx2-medium-thinking | n/a (not in the fit) | | | | 0 | |
| chronos-2 | -21.8 | [-51.4, -3.9] | n/a | n/a | 399 | 43 |

**point ranking (MAE), context V1, all rounds**: ours 142.6 points (place 12 of 57 fitted models), 416 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +17.6 | [+6.1, +29.0] | [+4.5, +30.6] | [+5.2, +25.7] | 416 | 45 |
| SCCH/PCFM-FT | -0.0 | [-14.7, +9.9] | [-16.1, +10.3] | [-18.1, +10.3] | 30 | 3 |
| farang/fx2-medium-thinking | -22.3 | [-52.3, +0.4] | [-52.0, +2.3] | [-50.0, +2.7] | 208 | 22 |
| chronos-2 | -18.3 | [-45.2, -2.1] | [-48.1, -0.4] | [-46.1, -1.6] | 416 | 45 |

top 10 of the fit: farang/fx2-medium-thinking 165.0; chronos-2 161.0; moirai-2-small 160.9; toto-2.0-2.5b 146.4; moirai-base-model 146.0; timestral-123m 145.9; timestral-28m 145.0; toto-2.0-313m 144.6; tabby 143.7; moirai-large 143.1

platform-style score [interval of the 28-day bootstrap with the weekly floor], rank range: gridload-compact v1 (int8) 142.6 [122.8, 162.5] ranks 1-27; visiontspp-large 125.1 [105.2, 144.9] ranks 1-30; SCCH/PCFM-FT 142.7 [100.2, 185.2] ranks n/a; farang/fx2-medium-thinking 165.0 [140.6, 189.4] ranks 1-17; chronos-2 161.0 [141.1, 180.8] ranks 1-17

**point ranking (MAE), context V1, stale rounds left out**: ours 144.3 points (place 10 of 57 fitted models), 399 tasks.

| comparator | gap (points) | 95 % interval, 7-day blocks | 14-day | 28-day (platform) | common tasks | common rounds |
|---|---|---|---|---|---|---|
| visiontspp-large | +17.5 | [+5.3, +29.5] | n/a | n/a | 399 | 43 |
| SCCH/PCFM-FT | +0.1 | [-15.0, +10.1] | n/a | n/a | 30 | 3 |
| farang/fx2-medium-thinking | -22.2 | [-52.2, +1.3] | n/a | n/a | 201 | 21 |
| chronos-2 | -18.3 | [-45.8, -1.6] | n/a | n/a | 399 | 43 |

paired log ratio of per-task errors, 100 * mean ln(err_comparator / err_ours), ISO-week block interval, tasks won:

| comparator | metric | mean | 95 % interval | tasks | share won |
|---|---|---|---|---|---|
| chronos-2 | point | -20.40 | [-50.89, -2.95] | 416 | 0.44 |
| chronos-2 | prob | -24.31 | [-57.24, -5.18] | 416 | 0.39 |
| visiontspp-large | point | +18.87 | [+7.34, +30.72] | 416 | 0.70 |
| visiontspp-large | prob | +18.91 | [+5.93, +29.47] | 416 | 0.72 |
| farang/fx2-medium-thinking | point | -27.09 | [-68.83, +0.32] | 208 | 0.38 |
| SCCH/PCFM-FT | point | +2.29 | [+2.29, +2.29] | 30 | 0.50 |
| SCCH/PCFM-FT | prob | +2.05 | [+2.05, +2.05] | 30 | 0.57 |

old-style arena MASE of our public texts (mae / flat-naive mae; mean over series, then rounds), ours minus comparator, ISO-week bootstrap:

| comparator | ours | comparator | difference | 95 % interval | rounds | rounds won |
|---|---|---|---|---|---|---|
| chronos-2 | 0.2729 | 0.2068 | +0.0660 | [+0.0054, +0.1778] | 45 | 0.47 |
| visiontspp-large | 0.2729 | 0.3138 | -0.0410 | [-0.0773, -0.0077] | 45 | 0.76 |
| farang/fx2-medium-thinking | 0.3080 | 0.2008 | +0.1072 | [+0.0029, +0.3001] | 22 | 0.41 |
| SCCH/PCFM-FT | 0.1557 | 0.1528 | +0.0028 | [+0.0028, +0.0028] | 3 | 0.67 |

