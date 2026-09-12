# Provenance of the audited source

## Scope

`1_audited_source/` is the anonymized, read-only snapshot of the single Byzantine-robust federated-learning pipeline examined in the paper *When the Metric Cannot See the Failure*. It is retained to support inspection of the eight implementation and evaluation faults reported in Section 4. It is distinct from `2_corrected_harness/`, which contains the repaired implementation used for the controlled validation experiments.

## Version identification

The anonymized snapshot first entered the review artifact in Git commit `6b078871ec7072170fd4a179710c1832ced73a82` (12 September 2026). That commit identifies the exact code and result snapshot distributed for review; it is an artifact commit, not an upstream development commit.

The supplied archive contains no upstream `.git` history, release tag, repository URL, or license file from which an earlier development revision can be independently recovered. The code snapshot is described in the artifact as the June 2026 state of the audited client, while the retained campaign logs record executions from 3–11 May 2026. This distinction matters for the seed-liveness finding: the June client snapshot passes a `seed` argument, whereas the May logs show `seed=0` in every recorded trainer configuration. The paper tests whether that later argument reaches the data-loader path.

During double-blind review, any identifying upstream repository location is withheld from the public artifact. The authors can provide the original repository identity and any available development revision privately to the Action Editor, and will add the de-anonymized source location to the public record when review anonymity no longer applies.

## Contents of the supplied snapshot

The complete source archive contains:

- 30 Python source files under `1_audited_source/`;
- 38 campaign log files under `1_audited_source/original_logs/`;
- 111 JSON result files under `1_audited_source/original_results/`.

`CLAIMS.md` maps each manuscript claim to the relevant source file, run record, derived result, and verification command. `6_regenerate/` contains the scripts that derive the reported values and figures from the retained records.

## Preparation for double-blind review

The review snapshot was prepared by replacing identifying absolute paths with `/ANON`, removing one author-identifying docstring, and removing comments that named a venue or review cycle. Method identifiers were replaced consistently with `PhotoScreen` and `QualityGate`. These transformations were limited to identifiers and prose: executable logic, hyperparameters, constants, and recorded values were not intentionally changed. Comments outside `1_audited_source/` were edited for clarity. All included Python files pass syntax validation.

## Oversized log files

Two raw logs in the complete archive exceed GitHub's 100 MB per-file limit and therefore cannot be stored as ordinary GitHub blobs:

| File | Raw size (bytes) | SHA-256 |
|---|---:|---|
| `original_logs/ablation_layers.log` | 247,344,449 | `53e1e2f0adf48053605caf589169eefa84ed92830a49dab8accf4c7399b99493` |
| `original_logs/P0_master.log` | 444,305,248 | `1c5df738853d94237d6cbb0fa364949a7b3b20a64428675ef7fe6ce3bfe6370f` |

For the hosted review artifact, these files should be provided as gzip-compressed files (`ablation_layers.log.gz` and `P0_master.log.gz`) together with instructions to decompress them before running scripts that scan `original_logs/*.log`. The hashes above refer to the decompressed raw files.

## External data and software

The EuroCity Persons night subset is not redistributed because it is governed by its own license. Re-running training requires Python 3.10, Flower 1.24, Ultralytics 8, PyTorch 2.6, and Ray. The numerical regeneration scripts require NumPy, SciPy, and Matplotlib and do not require a GPU.

No license file is present in the supplied archive. Consequently, this provenance record does not assign a license to the audited source or grant permissions beyond access for peer-review verification.
