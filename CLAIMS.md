# Where each claim in the paper is supported

Section numbers refer to the paper. "Verify" gives the command or the file to open.

| section | claim | artifact | verify |
|---|---|---|---|
| 4.1 | data-poisoning attacks are configured and never applied | `1_audited_source/experiments/attacks/byzantine_attacks.py` | `grep -rn "\.poison_data(" 1_audited_source --include=*.py` is empty |
| 4.1 | 63 of 90 campaign cells are unattacked | `1_audited_source/original_logs/` | 7 of 10 attacks act only through `poison_data` |
| 4.2 | the classification head is skipped and re-initialised every round | `1_audited_source/fl/client_yolo10s.py` | `set_parameters`; log lines "kept local copies of 12 layer(s)" |
| 4.3 | 297 buffers never leave the client | `1_audited_source/fl/client_yolo10s.py` | `get_parameters` uses `named_parameters()`; `5_derived_results/contract_state.json` counts 322 / 198 / 99 |
| 4.4 | with the archived source, the screen receives a hard-coded constant | `1_audited_source/fl/client_yolo10s.py` | fallback `{'mu': 0.29, 'sigma': 0.19}`; `5_derived_results/b1_poison_verification.json`. The logged runs used an earlier routine that returned random draws (`1_audited_source/PROVENANCE.md`) |
| 4.5 | the screen retained every client in 2174 invocations | `1_audited_source/original_logs/` | `6_regenerate/computed_values.py` computes `ScreenInvocations` and `ScreenDistinctOutcomes` from the logs |
| 4.5, 4.6 | minimum score 0.516 over 84 observed points; attacked clients score higher than honest | `5_derived_results/g1_photometric_separation.json` | `computed_values.py` -> `ScreenMinDeployed`, `ScreenNSupport`, `ScreenSigma*` |
| 4.7 | reference profile taken from a different dataset | `1_audited_source/fl/utils/quality_scoring.py`, `1_audited_source/data/bn_statistics/iara_config.json` | the deployed constants (0.2942, 0.1888) equal the `nightowls` entry of the profile file; the `ecp` entry, for the dataset trained on, is (0.2157, 0.1483); `computed_values.py` -> `ScreenMinCorrect` scores the observed inputs under it |
| 4.8 | in the experiment behind the three identical arms all 2100 trainer configurations carry seed=0; of the 26 multi-seed arms, 16 give the same first-round metric under both seeds and 10 differ by 0.013 | `1_audited_source/original_logs/robust_baselines.log`, `1_audited_source/original_results/` | `grep engine/trainer robust_baselines.log \| grep -oE 'seed=[0-9]+' \| sort \| uniq -c` gives `2100 seed=0`; `computed_values.py` -> `ArchArms`, `ArchIdentArms`, `ArchRoundOneSame`, `ArchRoundOneDiff`. The seed argument was added on 6 May, partway through the campaign (`1_audited_source/PROVENANCE.md`). |
| 4.9 | the averaging step of the screened strategy and its FedAvg, FedProx and FedBN baselines returns one tensor; 110 logged runs and 12,166 warning lines; reported metric between 0.42 and 0.56 in the 80 result files of these strategies | `1_audited_source/fl/server_strategies.py`, `fl/server_strategies_fedbn.py`, `fl/server_strategies_fedprox.py`, `original_logs/`, `original_results/` | `_weighted_average` allocates from `weights_results[:1]`; `grep -c "Not enough parameters. Expected 2, got 1"` on the logs; `computed_values.py` -> `OneTensorRuns`, `OneTensorLines`, `OneTensorFiles`, `OneTensorRepMin`, `OneTensorRepMax` |
| 5.1 | 65.91 AP points lost with buffers withheld; reported metric moves +0.26 | `4_run_records/T3_full_s{1..5}.json`, `T3_params_s{1..5}.json` | `computed_values.py` -> `GlobalDiffAP`, `LocalDiffAP` |
| App. A | pooled across-seed SD 0.0028 from the six arms run at three seeds | `4_run_records/` | `computed_values.py` -> `PooledSD`; used for the ratios in Sections 7.1 and D and Table 11 |
| 5.1 | three-seed check: aggregated-state hashes agree in 47 of 48 rounds while the client return order differs in 48 of 48 | `5_derived_results/seed_check.json` | `computed_values.py` -> `SeedHash*`, `SeedOrder*` |
| 5.2 | within-run correlation negative under the failing protocol | same ten records | `computed_values.py` -> `CorrFail*`, `CorrOk*` |
| 5.2 | corpus of 112 runs, 109 eligible, 19 negative: 10 of the 11 runs whose aggregation collapsed, and 9 under Min-Max or IPM | `4_run_records/` per `run_manifest.txt` | `computed_values.py` -> `NRuns`, `NNegative`, `NCollapsed`, `NNegAmongCollapsed`, `CollapsedNotNeg*`, and `NegRunRows` (the rows of Table 13) |
| 5.3 | 2x2 factorial: buffers cost 0.6464, head 0.0360 | `4_run_records/F_head-*_bn-*.json` | `computed_values.py` -> `Fac*`, `Eff*` |
| 5.4 | batch-norm re-estimation recovers 98.4% of a dead checkpoint; head recovers none | `5_derived_results/c3_healthy.json`, `c3_damaged.json` | `computed_values.py` -> `Cthree*` |
| 5.4 | epoch sweep | `4_run_records/E{1,3,5}_{full,params}_s1.json` | Appendix E table |
| 5.5 | brightness flooding: global -1.63, reported -6.23 | `4_run_records/G2_fedavg_*`, `S_fedavg_*` | `computed_values.py` -> `Bf*` |
| 6 | eight rules under four attacks | `4_run_records/M_*.json` | `make_figures.py` -> `fig_attacks.pdf`; Appendix D table |
| 6 | the repairs, verified | `2_corrected_harness/` | `client_fix.py`, `model_fix.py`, `data_poison.py`, `runner.py` |
| 7.1 | dose-response, 1.60 AP over the sweep | `4_run_records/C5_bf_i*.json` | `computed_values.py` -> `Dose*` |
| 7.2 | backdoor ASR 95.53% naive, 52.13% specific | `4_run_records/A4_*.json`, `5_derived_results/a4_asr.json` | `computed_values.py` -> `Asr*` |
| 7.3 | local-only control | `5_derived_results/c4_localonly.json` | `computed_values.py` -> `Cfour*` |
| 8 | the five tests as code | `3_tests/falsification_suite.py` | `python 3_tests/falsification_suite.py` reproduces T2 (minimum score 0.516 over the observed inputs, so no observed input is rejected) and T4 (3 of 26 arms give the same series under both seeds, compared to nine decimals: krum, median, trimmed mean) from `5_derived_results/` and `1_audited_source/original_results/` |
| 9 | injection: 6 of 9 caught, 1 false alarm; T2 fires on 4 faults and T5, run as specified (perturb and compare), on 3; a textbook 3-MAD screen is reported as a probe | `3_tests/fault_injection/`, `5_derived_results/c6_injection.json` | `python 3_tests/fault_injection/drive.py` re-runs the study and rewrites the JSON; `c6_values.py` reads it |
| 11 | selection regret | `4_run_records/M_*.json` | `computed_values.py` -> `Reg*` |
| App. F | the five contracts | `2_corrected_harness/contracts.py`, `5_derived_results/contract_*.json` | the split-integrity check needs the dataset; its output and the metric-identity, hash and buffer-policy results are in the three `contract_*.json` files |
| App. F | state hashes, buffer policy, timeline in a live run | `4_run_records/CONTRACT_demo.json` | fields `agg_state_hash`, `buffer_policy`, `timeline` |

Section 10 (the two independent codebases) is supported by the public repositories named there,
at the commits given, and is not part of this archive.
