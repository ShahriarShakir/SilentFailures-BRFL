# Code and data for "When the Metric Cannot See the Failure"

This archive has everything needed to check the paper: the code that was audited, the
corrected code, the tests, the run records, and the scripts that turn those records into the
numbers and figures in the paper. `CLAIMS.md` maps each claim in the paper to the file that
supports it. `1_audited_source/PROVENANCE.md` documents the origin, version limitations, anonymization changes,
and contents of the audited snapshot.

## Layout

```
1_audited_source/     the pipeline the paper audits, as it was at audit time, its 38 campaign logs, its 111 result files,
                      and the reference-profile file its screen reads (data/bn_statistics/iara_config.json)
                      (two logs are shipped gzip-compressed; see below)
2_corrected_harness/  the repaired pipeline (Section 6)
3_tests/              the five tests (Section 8), the contracts (Appendix F), the fault-injection harness (Section 9)
4_run_records/        112 run records, one JSON per training run
5_derived_results/    results computed from the run records and checkpoints
6_regenerate/         scripts that produce every number and figure in the paper
```

## Check the paper's numbers

Two logs are shipped compressed; decompress them first, then regenerate:

```
gzip -dk 1_audited_source/original_logs/ablation_layers.log.gz
gzip -dk 1_audited_source/original_logs/P0_master.log.gz
cd 6_regenerate
python computed_values.py    # writes computed_values.tex
python c6_values.py          # writes c6_values.tex
python make_figures.py       # writes the figures
cd ..
python 3_tests/falsification_suite.py   # reproduces the T2 and T4 findings
python 3_tests/fault_injection/drive.py # re-runs the fault-injection study of Section 9 (uses a GPU if present)
```

The two `.tex` files are byte-identical to the ones the PDF was built from. The regeneration
scripts read `4_run_records/`, `5_derived_results/`, and, for the screen-invocation count,
`1_audited_source/original_logs/`; the score under the correct reference profile (Section 8) reads
`1_audited_source/data/bn_statistics/iara_config.json`. They need Python 3.10 with numpy, scipy and matplotlib, and
no GPU.

## Find the nine faults

Each fault in Section 4 names the routine it lives in. In `1_audited_source/`:

| fault | where to look |
|---|---|
| 4.1 attacks never applied | `grep -rn "\.poison_data(" --include=*.py` returns nothing; the method is defined in `experiments/attacks/byzantine_attacks.py` and never called |
| 4.2 head never aggregated | `fl/client_yolo10s.py`, `set_parameters`: shape-mismatched tensors are skipped and logged as "kept local copies" |
| 4.3 buffers never exchanged | `fl/client_yolo10s.py`, `get_parameters`: iterates `named_parameters()`, not `state_dict()` |
| 4.4 screen input is a constant | `fl/client_yolo10s.py`, `_extract_bn_stats`: the fallback `{'mu': 0.29, 'sigma': 0.19}`. The logged runs used an earlier version of this routine that returned random draws; see `1_audited_source/PROVENANCE.md` |
| 4.5 to 4.7 screen faults | `fl/utils/quality_scoring.py` and `fl/server_strategies.py`; the observed scores are in `5_derived_results/g1_photometric_separation.json`; the per-dataset reference profiles are in `data/bn_statistics/iara_config.json` |
| 4.8 seeds inert | `original_logs/robust_baselines.log`: 2100 trainer-configuration lines, every one `seed=0`. The 26 multi-seed arms are in `original_results/`; `6_regenerate/computed_values.py` compares their two seeds (macros `Arch*`). A seed argument was added to the training call on 6 May, partway through the campaign: logs written before it record `seed=0`, later ones the configured seed (see `1_audited_source/PROVENANCE.md`). |
| 4.9 averaging returns one tensor | `fl/server_strategies.py`, `_weighted_average` of `FedAvgStrategy`, `FedBNStrategy` and `PHOTOSCREENStrategy` (`FedProxStrategy` in `fl/server_strategies_fedprox.py` inherits it; `fl/server_strategies_fedbn.py` has a second copy): the output list is allocated from `weights_results[:1]` and `weights[0]`, so one tensor is returned. Clients then print `WARNING: Not enough parameters. Expected 2, got 1` (`fl/client_yolo10s.py`, `set_parameters`): `grep -c "Not enough parameters" original_logs/*.log`. `6_regenerate/computed_values.py` counts the warning lines, the runs and the result files (macros `OneTensor*`). |

## Two compressed logs

`original_logs/ablation_layers.log` (247 MB) and `original_logs/P0_master.log` (444 MB) are
shipped as `.gz` files because of the host's per-file size limit. Decompress them in place
before running anything that scans `original_logs/*.log`:

```
cd 1_audited_source/original_logs
gzip -dk ablation_layers.log.gz P0_master.log.gz
```

SHA-256 of the decompressed files:
`fe293aaff2c99b8b0d26b69bb129231f88d1a156c7ff303bd20dfa3a12b889dd  ablation_layers.log`
`952f88bc367a1b7bce5db6bdcc4446d6aadab12ba3dc3a79bd96a1ffb902126c  P0_master.log`

`computed_values.py` counts screen invocations across every log, so the count of 2174 in the
paper depends on these two being decompressed first.

## Run records

Each file in `4_run_records/` is one training run: the configuration, the malicious client set,
per-round global mAP50 of the aggregated model on the held-out split, per-round client-local
metrics, which clients were accepted and rejected, and client update norms.
`6_regenerate/run_manifest.txt` lists the 112 tags that make up the corpus.

## Re-running training

Training needs Flower 1.24, Ultralytics 8, PyTorch 2.6, Ray, and the EuroCity Persons night
subset, which has its own licence and is not included. `2_corrected_harness/runner.py` is the
entry point; `runner.py --help` lists the flags. `client_fix.py` seeds the detector library's
data-loader generator from the run seed (the block that patches `InfiniteDataLoader`, added on
9 September 2026). The runs in `4_run_records/` were made before that change, under the library's
constant loader seed (Section 5.1); remove that block to reproduce their data order.

## What was changed for review

In `1_audited_source/`, identifying paths and strings were replaced, the execution host's IP
address was replaced with `ANON_IP`, and the method's strategy and screen names were changed
to `PhotoScreen` and `QualityGate`; the identifiers of its scoring module keep their original names.
Nothing else in the audited source was touched;
its comments are the original author's. In `data/bn_statistics/iara_config.json` the method name in
the description string was replaced; its values are unchanged. In the other directories the comments
were edited for clarity. No code logic, constant, or data value was changed anywhere. All Python
files parse.

## Version history

- 12 September 2026: first release.
- 7 October 2026: the fault-injection harness runs T5 as specified in Section 8 (apply the attack's
  own transformation to each honest client's input and compare the admit score before and after) and
  T1 against the undefended baseline, and adds a textbook 3-MAD screen as a probe;
  `5_derived_results/c6_injection.json` is its output. Added `5_derived_results/seed_check.json`, the
  three-seed check of Section 5.1, and the reference-profile file of Section 4.7. The regeneration
  scripts also compute the comparison of the archive's multi-seed arms (Section 4.8), the pooled
  across-seed standard deviation, and the counts of Section 4.9. The provenance note lists both
  differences between the logged code and the June source; this README and `CLAIMS.md` cover the
  averaging fault of Section 4.9. `3_tests/falsification_suite.py` states that its T4 comparison is
  made to nine decimals, and a comment in `2_corrected_harness/client_fix.py` dates the loader-seed
  block. `computed_values.tex` and `c6_values.tex` remain byte-identical to the ones the PDF was built from.
