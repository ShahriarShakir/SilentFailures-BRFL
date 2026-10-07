# Provenance of the audited source

## What `1_audited_source/` is

The single federated learning pipeline the paper examines, as it stood when the audit was
performed. It supports inspection of the nine faults reported in Section 4. It is separate from
`2_corrected_harness/`, the repaired implementation used for the controlled experiments.

## Version

The project was not under version control when the audit was performed. The artifact address in
Section 3 of the paper identifies the snapshot. The archive contains no upstream history,
tag, repository address, or licence file from which an earlier revision can be recovered.

The client source is the June 2026 state. The campaign logs in `original_logs/` record
executions from 3 to 11 May 2026. The code that produced them differs from the June source in two
respects.

1. The seed argument. It was added to the training call on 6 May 2026, partway through the
   campaign. The 11 logs written before that change record `seed=0` in every trainer configuration
   (2100 of them in `robust_baselines.log`); the 17 written after it record the configured seeds; the other 10 contain no trainer configuration.
   Section 4.8 of the paper uses both groups, and Section 5 shows that the argument, where present,
   does not reach the data loader.
2. The photometric routine `_extract_bn_stats` in `fl/client_yolo10s.py`. The June source reads the
   client's images and, on this project's `.txt` image lists, falls back to the constant of Section
   4.4. The code that produced the logs returned random draws in its place,
   `np.random.normal(0.29, 0.05)` and `np.random.normal(0.19, 0.03)`. Neither version gives the
   screen a measurement of the client's images.

The authors can provide the Action Editor with contemporaneous workspace records privately if
additional verification of the snapshot is required.

## Contents

- 30 Python source files
- 38 campaign logs in `original_logs/`; two are shipped gzip-compressed (see README)
- 111 result files in `original_results/`
- 1 configuration file, `data/bn_statistics/iara_config.json`: the per-dataset reference profiles the
  screen reads (Section 4.7 of the paper). The method name in its description string was replaced;
  its values are unchanged.

`CLAIMS.md` maps each claim in the paper to a file and a verification command.

## Preparation for review

Identifying paths and strings were replaced, the execution host's IP address was replaced with
`ANON_IP`, one author docstring was removed, and the method's strategy and screen names were
changed to `PhotoScreen` and `QualityGate`; the identifiers of its scoring module keep their original
names. Executable logic,
hyperparameters, constants, and recorded values were not changed. All Python files parse.

## Oversized logs

| file | raw bytes | SHA-256 of the decompressed sanitised file |
|---|---|---|
| `original_logs/ablation_layers.log` | 247,344,113 | `fe293aaff2c99b8b0d26b69bb129231f88d1a156c7ff303bd20dfa3a12b889dd` |
| `original_logs/P0_master.log` | 444,304,716 | `952f88bc367a1b7bce5db6bdcc4446d6aadab12ba3dc3a79bd96a1ffb902126c` |

Shipped as `.gz` (27.8 MB and 49.8 MB). Decompress before running scripts that scan
`original_logs/*.log`; 1604 of the 2174 screen invocations counted in Section 4.5 are in these
two files.

## Data and software

The EuroCity Persons night subset is under its own licence and is not redistributed. Re-running
training needs Python 3.10, Flower 1.24, Ultralytics 8, PyTorch 2.6, and Ray. The regeneration
scripts need NumPy, SciPy, and Matplotlib, and no GPU.

## Licence

This archive is provided for peer review of the paper. No licence is granted beyond access for
that purpose; a licence will be attached on publication.
