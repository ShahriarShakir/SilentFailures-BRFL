# Where each claim in the paper is supported

Section numbers refer to the revised manuscript. "Verify" gives the command or the file to open.

| section | claim | artifact | verify |
|---|---|---|---|
| 4.1 | data-poisoning attacks are configured and never applied | `1_audited_source/experiments/attacks/byzantine_attacks.py` | `grep -rn "\.poison_data(" 1_audited_source --include=*.py` is empty |
| 4.1 | 63 of 90 campaign cells are unattacked | `1_audited_source/original_logs/` | 7 of 10 attacks act only through `poison_data` |
| 4.2 | the classification head is skipped and re-initialised every round | `1_audited_source/fl/client_yolo10s.py` | `set_parameters`; log lines "kept local copies of 12 layer(s)" |
| 4.3 | 297 buffers never leave the client | `1_audited_source/fl/client_yolo10s.py` | `get_parameters` uses `named_parameters()`; `5_derived_results/contract_state.json` counts 322 / 198 / 99 |
| 4.4 | the screen receives a hard-coded constant | `1_audited_source/fl/client_yolo10s.py` | fallback `{'mu': 0.29, 'sigma': 0.19}`; `5_derived_results/b1_poison_verification.json` |
| 4.5 | the screen retained every client in 2174 invocations | `1_audited_source/original_logs/` | `6_regenerate/computed_values.py` computes `ScreenInvocations` and `ScreenDistinctOutcomes` from the logs |
| 4.5, 4.6 | minimum score 0.516; attacked clients score higher than honest | `5_derived_results/g1_photometric_separation.json` | open the file |
| 4.7 | reference profile taken from a different dataset | `1_audited_source/fl/utils/quality_scoring.py` and config | the constants (0.2942, 0.1888) |
| 4.8 | all 2100 trainer configurations carry seed=0 | `1_audited_source/original_logs/robust_baselines.log` | `grep -c engine/trainer` gives 2100; `grep engine/trainer \| grep -oE 'seed=[0-9]+' \| sort -u` gives only `seed=0`. Note: the shipped client source is from June 2026 and already passes `seed=` to the training call; the logs are from the May campaign, before that change. Section 5 of the paper shows the added argument does not reach the data loader in any case. |
| 5.1 | 65.91 AP points lost with buffers withheld; reported metric moves +0.26 | `4_run_records/T3_full_s{1..5}.json`, `T3_params_s{1..5}.json` | `computed_values.py` -> `GlobalDiffAP`, `LocalDiffAP` |
| 5.2 | within-run correlation negative under the failing protocol | same ten records | `computed_values.py` -> `CorrFail*`, `CorrOk*` |
| 5.2 | corpus of 112 runs, 19 negative, all with a collapsed aggregate | `4_run_records/` per `run_manifest.txt` | `computed_values.py` -> `NRuns`, `NNegative`, `NCollapsed` |
| 5.3 | 2x2 factorial: buffers cost 0.6464, head 0.0360 | `4_run_records/F_head-*_bn-*.json` | `computed_values.py` -> `Fac*`, `Eff*` |
| 5.4 | batch-norm re-estimation recovers 98.4% of a dead checkpoint; head recovers none | `5_derived_results/c3_healthy.json`, `c3_damaged.json` | `computed_values.py` -> `Cthree*` |
| 5.4 | epoch sweep | `4_run_records/E{1,3,5}_{full,params}_s1.json` | Appendix E table |
| 5.5 | brightness flooding: global -1.63, reported -6.23 | `4_run_records/G2_fedavg_*`, `S_fedavg_*` | `computed_values.py` -> `Bf*` |
| 6 | eight rules under four attacks | `4_run_records/M_*.json` | `make_figures.py` -> `fig_attacks.pdf`; Appendix D table |
| 6 | the repairs, verified | `2_corrected_harness/` | `client_fix.py`, `model_fix.py`, `data_poison.py`, `runner.py` |
| 7.1 | dose-response, 1.60 AP over the sweep | `4_run_records/C5_bf_i*.json` | `computed_values.py` -> `Dose*` |
| 7.2 | backdoor ASR 95.53% naive, 52.13% specific | `4_run_records/A4_*.json`, `5_derived_results/a4_asr.json` | `computed_values.py` -> `Asr*` |
| 7.3 | local-only control | `5_derived_results/c4_localonly.json` | `computed_values.py` -> `Cfour*` |
| 8 | the five tests as code | `3_tests/falsification_suite.py` | `python 3_tests/falsification_suite.py` reproduces T2 (minimum score 0.516, reject region unreachable) and T4 (3 of 26 arms bit-identical: krum, median, trimmed mean) from `5_derived_results/` and `1_audited_source/original_results/` |
| 9 | injection: 6 of 9 caught, 1 false alarm | `3_tests/fault_injection/`, `5_derived_results/c6_injection.json` | `c6_values.py` |
| 11 | selection regret | `4_run_records/M_*.json` | `computed_values.py` -> `Reg*` |
| App. F | the five contracts | `2_corrected_harness/contracts.py`, `5_derived_results/contract_*.json` | the split-integrity check needs the dataset; its output and the metric-identity, hash and buffer-policy results are in the three `contract_*.json` files |
| App. F | state hashes, buffer policy, timeline in a live run | `4_run_records/CONTRACT_demo.json` | fields `agg_state_hash`, `buffer_policy`, `timeline` |

Section 10 (the two independent codebases) is supported by the public repositories named there,
at the commits given, and is not part of this archive.
