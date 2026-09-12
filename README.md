# Code and data for "When the Metric Cannot See the Failure"

This archive has everything needed to check the paper: the code that was audited, the
corrected code, the tests, the run records, and the scripts that turn those records into the
numbers and figures in the paper. `CLAIMS.md` maps each claim in the paper to the file that
supports it.

## Layout

```
1_audited_source/     the pipeline the paper audits, as it was at audit time, its 38 campaign logs, and its 111 result files
                      (two logs are shipped gzip-compressed; see below)
2_corrected_harness/  the repaired pipeline (Section 6)
3_tests/              the five tests (Section 8), the contracts (Appendix F), the fault-injection harness (Section 9)
4_run_records/        112 run records, one JSON per training run
5_derived_results/    results computed from the run records and checkpoints
6_regenerate/         scripts that produce every number and figure in the paper
```

## Check the paper's numbers

```
cd 6_regenerate
python computed_values.py    # writes computed_values.tex
python c6_values.py          # writes c6_values.tex
python make_figures.py       # writes the figures
cd ..
python 3_tests/falsification_suite.py   # reproduces the T2 and T4 findings
```

The two `.tex` files are byte-identical to the ones the PDF was built from. The scripts read
`4_run_records/` and `5_derived_results/` only. They need Python 3.10 with numpy, scipy and
matplotlib, and no GPU.

## Find the eight faults

Each fault in Section 4 names the routine it lives in. In `1_audited_source/`:

| fault | where to look |
|---|---|
| 4.1 attacks never applied | `grep -rn "\.poison_data(" --include=*.py` returns nothing; the method is defined in `experiments/attacks/byzantine_attacks.py` and never called |
| 4.2 head never aggregated | `fl/client_yolo10s.py`, `set_parameters`: shape-mismatched tensors are skipped and logged as "kept local copies" |
| 4.3 buffers never exchanged | `fl/client_yolo10s.py`, `get_parameters`: iterates `named_parameters()`, not `state_dict()` |
| 4.4 screen input is a constant | `fl/client_yolo10s.py`, `_extract_bn_stats`: the fallback `{'mu': 0.29, 'sigma': 0.19}` |
| 4.5 to 4.7 screen faults | `fl/utils/quality_scoring.py` and `fl/server_strategies.py`; the observed scores are in `5_derived_results/g1_photometric_separation.json` |
| 4.8 seeds inert | `original_logs/robust_baselines.log`: 2100 trainer-configuration lines, every one `seed=0`. The client source shipped here is from June 2026 and already passes a seed to the training call; the May logs predate that change, and Section 5 shows the argument is inert regardless. |

## Two compressed logs

`original_logs/ablation_layers.log` (247 MB) and `original_logs/P0_master.log` (444 MB) are
shipped as `.gz` files because of the host's per-file size limit. Decompress them in place
before running anything that scans `original_logs/*.log`:

```
cd 1_audited_source/original_logs
gzip -dk ablation_layers.log.gz P0_master.log.gz
```

SHA-256 of the decompressed files:
`53e1e2f0adf48053605caf589169eefa84ed92830a49dab8accf4c7399b99493  ablation_layers.log`
`8bf075d62870f4fc71d8facdf63b4524b96132846b9f2bbb95f97cde0967b9b7  P0_master.log`

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
entry point; `runner.py --help` lists the flags.

## What was changed for review

In `1_audited_source/`, identifying paths and strings were replaced and the method's identifiers
were renamed to `PhotoScreen` and `QualityGate`. Nothing else in the audited source was touched;
its comments are the original author's. In the other directories the comments were edited for
clarity. No code logic, constant, or data value was changed anywhere. All Python files parse.
