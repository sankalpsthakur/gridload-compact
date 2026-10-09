
### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 service code path int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15.onnx=81e7e6b9f9c850d6
- spread: None
- result: {"label": "net15.onnx", "rounds": 52, "mase": 0.21538568160136615, "sql": 0.1705881498404233, "cover80": 0.7667741683740044, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_ort_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 onnxruntime int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5
- spread: None
- result: {"label": "net15_int8.onnx", "rounds": 52, "mase": 0.2136769221459295, "sql": 0.1695305560099699, "cover80": 0.7665386474862937, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_ort_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 ens3 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s0.pt=d0b03c3ef63ffabd; test_15_s1.pt=1edff5f7cadfda47; test_15_s2.pt=4e67a872b20ac422
- spread: 1.05
- result: {"label": "test_15_s0.pt,test_15_s1.pt,test_15_s2.pt", "rounds": 52, "mase": 0.2153856791435324, "sql": 0.17058814824707932, "cover80": 0.7667741683740044, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_ens3_torch_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 1 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 2 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s0.pt=d0b03c3ef63ffabd
- spread: 1.05
- result: {"label": "test_15_s0.pt", "rounds": 52, "mase": 0.21489883236945978, "sql": 0.17030263760806846, "cover80": 0.757504009357526, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 1 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s1.pt=1edff5f7cadfda47
- spread: 1.05
- result: {"label": "test_15_s1.pt", "rounds": 52, "mase": 0.21723457703303475, "sql": 0.1722565122794671, "cover80": 0.7638345230658785, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s1_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 single member seed 2 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_s2.pt=4e67a872b20ac422
- spread: 1.05
- result: {"label": "test_15_s2.pt", "rounds": 52, "mase": 0.2162410578770824, "sql": 0.17131122402257187, "cover80": 0.7672213115766902, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_single_s2_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:15 UTC  def 2  2026-02-08..2026-03-31  variant: def2 no-holiday ablation seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:15 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 2  2026-02-08..2026-03-31  variant: def2 service code path int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5; net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "service:finalW:int8", "rounds": 52, "mase": 0.21730961966070916, "sql": 0.1729146064293282, "cover80": 0.7696489440176428, "n_missing": 9, "routing": {"specialist": 513, "fallback": 7}}
- per-round file: faith_final_def2_test_d2_service_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 2  2026-02-08..2026-03-31  variant: def2 no-holiday ablation seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_15_nohol_s0.pt=cd6d0fd28b910b35
- spread: 1.05
- result: {"label": "test_15_nohol_s0.pt", "rounds": 52, "mase": 0.21907626189942997, "sql": 0.17455040172396522, "cover80": 0.7714243431736326, "n_missing": 9}
- per-round file: faith_final_def2_test_d2_nohol_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net60.onnx=6d610b1c66c6509e
- spread: None
- result: {"label": "net60.onnx", "rounds": 52, "mase": 0.18341776781311142, "sql": 0.1467713855292288, "cover80": 0.8575820643978889, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_ort_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s0.pt=f184a96c54e0ce44; test_60_s1.pt=77fd6aa7542120e4; test_60_s2.pt=3cabd3e212c6b549
- spread: 1.2
- result: {"label": "test_60_s0.pt,test_60_s1.pt,test_60_s2.pt", "rounds": 52, "mase": 0.18341776706544197, "sql": 0.1467713836690754, "cover80": 0.8575820643978889, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_torch_fp32_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 service code path int8  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 1 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 ens3 onnxruntime int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "net60_int8.onnx", "rounds": 52, "mase": 0.183232555787468, "sql": 0.14681973075949795, "cover80": 0.8577164102612967, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_ens3_ort_int8_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 2 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s0.pt=f184a96c54e0ce44
- spread: 1.2
- result: {"label": "test_60_s0.pt", "rounds": 52, "mase": 0.18427172674414433, "sql": 0.1475153369774498, "cover80": 0.8611559499378039, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 1 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s1.pt=77fd6aa7542120e4
- spread: 1.2
- result: {"label": "test_60_s1.pt", "rounds": 52, "mase": 0.18430093571005365, "sql": 0.14773921417048783, "cover80": 0.8608823460737539, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s1_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 no-holiday ablation seed 0 torch fp32  STARTED  (kernel bench-timeseries-b, code d4c7c23b264f668fa2ef05a9121b918fce3f631a); the result entry follows when the run completes

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 single member seed 2 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_s2.pt=3cabd3e212c6b549
- spread: 1.2
- result: {"label": "test_60_s2.pt", "rounds": 52, "mase": 0.1833676193985934, "sql": 0.14656283326009711, "cover80": 0.8461777300346143, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_single_s2_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 no-holiday ablation seed 0 torch fp32
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): test_60_nohol_s0.pt=89bd5531bdda0c0c
- spread: 1.2
- result: {"label": "test_60_nohol_s0.pt", "rounds": 52, "mase": 0.1904035165632751, "sql": 0.15408170000481944, "cover80": 0.8576233454546626, "n_missing": 0}
- per-round file: faith_final_def5_test_d5_nohol_s0_2026-02-08_2026-03-31.parquet

### 2026-10-08 21:16 UTC  def 5  2026-02-08..2026-03-31  variant: def5 service code path int8
- code: d4c7c23b264f668fa2ef05a9121b918fce3f631a  kernel: bench-timeseries-b
- artifacts (sha256 prefix): net15_int8.onnx=1e1dfaca195e8cb5; net60_int8.onnx=549c86f26bc0c8fb
- spread: None
- result: {"label": "service:finalW:int8", "rounds": 52, "mase": 0.1847649701804282, "sql": 0.14801052613656857, "cover80": 0.8561927784205007, "n_missing": 0, "routing": {"specialist": 513, "fallback": 7}}
- per-round file: faith_final_def5_test_d5_service_int8_2026-02-08_2026-03-31.parquet
