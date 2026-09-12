# Provenance of the audited source

## What `1_audited_source/` is

The single federated learning pipeline the paper examines, as it stood when the audit was
performed. It supports inspection of the eight faults reported in Section 4. It is separate from
`2_corrected_harness/`, the repaired implementation used for the controlled experiments.

## Version

The project was not under version control when the audit was performed. The archive hash in
Section 3 of the paper identifies the snapshot. The archive contains no upstream history,
tag, repository address, or licence file from which an earlier revision can be recovered.

The client source is the June 2026 state. The campaign logs in `original_logs/` record
executions from 3 to 11 May 2026. The two differ in one respect that matters for Section 4.8:
the June source passes a seed argument to the training call, and the May logs show `seed=0` in
every one of the 2100 recorded trainer configurations, because that argument did not exist when
the campaign ran. Section 5 of the paper reports that the argument, once present, does not reach
the data loader either.

The authors will supply the repository location privately to the Action Editor on request, and
will add it to the public record when anonymity no longer applies.

## Contents

- 30 Python source files
- 38 campaign logs in `original_logs/`; two are shipped gzip-compressed (see README)
- 111 result files in `original_results/`

`CLAIMS.md` maps each claim in the paper to a file and a verification command.

## Preparation for review

Identifying paths and strings were replaced, one author docstring was removed, and the method's
identifiers were renamed to `PhotoScreen` and `QualityGate` throughout. Executable logic,
hyperparameters, constants, and recorded values were not changed. All Python files parse.

## Oversized logs

| file | raw bytes | SHA-256 of the raw file |
|---|---|---|
| `original_logs/ablation_layers.log` | 247,344,449 | `53e1e2f0adf48053605caf589169eefa84ed92830a49dab8accf4c7399b99493` |
| `original_logs/P0_master.log` | 444,305,248 | `8bf075d62870f4fc71d8facdf63b4524b96132846b9f2bbb95f97cde0967b9b7` |

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
