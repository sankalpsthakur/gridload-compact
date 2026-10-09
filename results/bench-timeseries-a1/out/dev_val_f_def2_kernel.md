# VAL-F (dev check): SMARD Net Load definition 2, rounds 2026-01-01 .. 2026-02-07


## MASE (arena definition; lower is better), 38 rounds, 95% block-bootstrap CI over ISO weeks

Best reference model in this window: **flowstate**.  Diff = mean(model - flowstate) per round with CI; win-rate = share of rounds with lower score than flowstate.

| rank | model | params (M) | mean [95% CI] | diff vs best ref [95% CI] | win-rate vs best ref | Elo [95% CI] |
|---|---|---|---|---|---|---|
| 1 | **gridload-v0** | 0.150 | 0.190 [0.146, 0.248] | -0.038 [-0.069, -0.012] | 68% | 1428 [1357, 1485] |
| 2 | flowstate | 9 | 0.228 [0.161, 0.316] | - | - | 1344 [1272, 1398] |
| 3 | visiontspp-large | 307 | 0.234 [0.177, 0.321] | +0.006 [-0.011, +0.023] | 58% | 1360 [1303, 1413] |
| 4 | chronos-bolt-base | 205 | 0.245 [0.172, 0.337] | +0.018 [+0.002, +0.033] | 39% | 1286 [1218, 1347] |
| 5 | visiontspp-base | 86 | 0.249 [0.184, 0.342] | +0.021 [+0.005, +0.033] | 34% | 1288 [1224, 1352] |
| 6 | chronos-bolt-small | 48 | 0.250 [0.170, 0.344] | +0.022 [-0.001, +0.047] | 39% | 1279 [1200, 1344] |
| 7 | chronos-bolt-mini | 21 | 0.251 [0.180, 0.339] | +0.023 [+0.008, +0.040] | 34% | 1243 [1168, 1312] |
| 8 | timesfm-2.5-200m | 200 | 0.252 [0.176, 0.339] | +0.024 [+0.006, +0.047] | 32% | 1278 [1209, 1339] |
| 9 | chronos-bolt-tiny | 9 | 0.266 [0.175, 0.369] | +0.038 [+0.009, +0.076] | 29% | 1216 [1140, 1281] |
| 10 | chronos-2 | 120 | 0.266 [0.186, 0.367] | +0.039 [+0.020, +0.060] | 42% | 1233 [1163, 1313] |

Field for this metric: 32 models + ours; Elo = platform procedure (K=4, base 1000, 500 bootstraps, per-round mean over series).

## SQL (arena definition; lower is better), 38 rounds, 95% block-bootstrap CI over ISO weeks

Best reference model in this window: **flowstate**.  Diff = mean(model - flowstate) per round with CI; win-rate = share of rounds with lower score than flowstate.

| rank | model | params (M) | mean [95% CI] | diff vs best ref [95% CI] | win-rate vs best ref | Elo [95% CI] |
|---|---|---|---|---|---|---|
| 1 | **gridload-v0** | 0.150 | 0.150 [0.115, 0.196] | -0.031 [-0.057, -0.010] | 71% | 1311 [1255, 1359] |
| 2 | flowstate | 9 | 0.182 [0.127, 0.254] | - | - | 1193 [1137, 1243] |
| 3 | visiontspp-large | 307 | 0.185 [0.137, 0.255] | +0.003 [-0.009, +0.015] | 66% | 1225 [1176, 1269] |
| 4 | visiontspp-base | 86 | 0.198 [0.144, 0.271] | +0.017 [+0.005, +0.024] | 34% | 1121 [1062, 1172] |
| 5 | chronos-bolt-base | 205 | 0.199 [0.137, 0.277] | +0.018 [+0.003, +0.033] | 42% | 1119 [1056, 1174] |
| 6 | chronos-bolt-small | 48 | 0.201 [0.136, 0.279] | +0.020 [-0.000, +0.041] | 39% | 1130 [1063, 1184] |
| 7 | chronos-bolt-mini | 21 | 0.201 [0.140, 0.278] | +0.020 [+0.005, +0.035] | 37% | 1112 [1052, 1166] |
| 8 | timesfm-2.5-200m | 200 | 0.202 [0.139, 0.275] | +0.020 [+0.006, +0.039] | 37% | 1111 [1063, 1159] |
| 9 | chronos-2 | 120 | 0.211 [0.148, 0.293] | +0.030 [+0.017, +0.045] | 37% | 1087 [1035, 1146] |
| 10 | chronos-bolt-tiny | 9 | 0.215 [0.137, 0.304] | +0.034 [+0.008, +0.068] | 32% | 1072 [1014, 1120] |

Field for this metric: 19 models + ours; Elo = platform procedure (K=4, base 1000, 500 bootstraps, per-round mean over series).
