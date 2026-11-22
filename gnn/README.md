# GNN and reference-selection experiments

This is the shared baseline implementation for **Crys_METH**. It contains the
reference selection, graph, training, evaluation, and analysis code migrated from
Crys_JEPA. No sibling checkout or `REFERENCE_BASE_PATH` setting is needed.

The root `evaluation.ipynb` remains the entry point for the frozen candidate-ranking
experiment. This package additionally provides the historical baseline workflows
and the implementation behind `extra/`'s appendix experiments.

## Layout

```text
gnn/
  run.py                 Common experiment launcher
  README.md              Contents, commands and data requirements
  reference_baselines/   Shared implementation (stable Python import name)
data/eval/               Frozen inputs, downloaded models and caches
outputs/gnn/             Generated runs, graphs, scores, figures and reports
  summaries/             Curated results from the four appendix experiments
```

## Contents

The directories below are inside `gnn/reference_baselines/`.

| Directory | Contents |
| --- | --- |
| `core/` | Exact retrieval, angular distances, file integrity, settings, fixed-hull thermodynamics |
| `datasets/` | Pinned release downloads, hull labels/decompositions, reference pools and training datasets |
| `baselines/` | Chemical and nearest-neighbor reference sets, energy/angle rules, formula balance, random controls, conductance, spectral and mean ablations |
| `graphs/` | Euclidean/cosine graphs, co-occurrence, diffusion, eigenmodes, reference graphs, SOAP/phase connectivity and relational edge features |
| `models/` | Reference-utility GNNs, edge-attribute and relational-edge variants; legacy candidate models |
| `training/` | Base and edge-feature GNN training, prediction, and historical candidate-model training |
| `evaluation/` | Ranking metrics, duplicates, held-out comparisons, budget studies, GNN evaluations, MatterSim audits and recoverability probes |
| `structure/` | SOAP descriptors, geometry comparisons, validity, novelty, case studies and the local structure-energy adapter |
| `embedding_structure/` | Communities, space-group concentration, energy capture, confound controls, and probes |
| `reference_set_comparison/` | Independent JEPA reference export and chemical/kNN/union screening, MEGNet/eSEN hull pipelines, report analysis, UMAP and `inspect_train.ipynb` |
| `checks/` | Retrieval, encoder, training, outcome and interpretation checks against prepared inputs |
| `reporting/` | Tables, figures and reports generated from completed experiments |
| `configs/` | Fixed TOML experiment protocols |
| `requirements/` | Workflow-specific dependency lists |
| `tests/` | Synthetic scientific checks, pipeline integration tests and migration regressions |
| `paths.py`, `prepare_inputs.py` | Repository-relative locations and non-destructive staging of existing evaluation inputs |

The appendix folders retain explanatory READMEs, configuration snapshots, and
short compatibility entry points. Their 13 result files now live under
[`outputs/gnn/summaries/`](../outputs/gnn/summaries/). The algorithms live here once.
Cluster-specific shell/Slurm scripts and redundant historical README collections
were not imported. The inspection notebook is retained without saved outputs.

## Run from Crys_METH

Use the environment described in the repository README. One launcher handles all
experiments and adds the repository and `gnn/` to Python's import path:

```bash
python gnn/run.py --list
python gnn/run.py prepare
python gnn/run.py reference train --help
python gnn/run.py edge-attribute train --help
python gnn/run.py relational-edge train --help
python gnn/run.py connectivity train --help
python gnn/run.py space-group communities --help
python gnn/run.py proxy --help
```

| Experiment | Actions |
| --- | --- |
| `reference` | `graphs`, `train`, `prepare`, `evaluate` |
| `edge-attribute` | `train`, `prepare`, `evaluate` |
| `relational-edge` | `features`, `train`, `prepare`, `evaluate`, `probe` |
| `connectivity` | `graphs`, `train`, `prepare`, `evaluate` |
| `space-group` | `communities`, `capture`, `map` |
| `proxy` | Pipeline stages such as `prepare`, `embed`, `score`, `select`, `predict`, `hull`, `report` |
| `embeddings` | JEPA reference export; accepts `--limit`, `--device`, etc. |
| `prepare` | Stage the existing frozen input bundle |

Options after the action are forwarded to that experiment. For example:

```bash
python gnn/run.py connectivity train --graph soap --model gcn
python gnn/run.py proxy report --model esen
python gnn/run.py graphs.build --help
```

Connectivity training automatically uses its own configuration and output
folder; `--smoke` selects a separate smoke folder. Each training command runs
only the requested configuration/topology, not an entire graph sweep. Any other
package module remains accessible by its dotted name through the same launcher.
The temporary root `baselines.py` launcher has been replaced by `gnn/run.py`.

The launcher also works from another working directory when invoked by absolute
path. Explicit relative CLI arguments still refer to your current directory;
configuration file paths are resolved relative to the repository.

For direct module commands or interactive imports, from the repository root:

```bash
export PYTHONPATH="$PWD/gnn:$PWD/utils/eval:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m reference_baselines.training.train_reference_gnn --help
python -m unittest discover -s gnn/reference_baselines/tests -v
```

Existing appendix commands such as `python extra/edge_attribute_gnn/train.py
--help` work without an `.env` file. They forward to this package. Their local
`config.toml` files are historical snapshots; executable defaults use `configs/`
here. The connectivity experiment uses the base reference-GNN trainer with its
connectivity configuration.

## Inputs and outputs

Source code stays under `gnn/reference_baselines`. Inputs and model caches stay
under `data/eval`; generated experiment results go to `outputs/gnn`:

| Location | Purpose |
| --- | --- |
| `data/eval/inputs/references/` | Frozen reference vectors and metadata |
| `data/eval/inputs/candidates/` | Frozen candidate vectors, metadata and decompositions |
| `data/eval/inputs/splits/` | Frozen train/validation/reserved partitions and protocol |
| `data/eval/inputs/embeddings/` | Flat input view created by `prepare_inputs` for older baseline commands |
| `data/eval/graphs/`, `data/eval/checkpoints/` | Existing frozen notebook graph and checkpoint bundles |
| `outputs/gnn/` | Baseline artifacts, reference-GNN checkpoints and new experiment results |
| `outputs/gnn/reports/` | Newly generated baseline reports and figures |
| `data/eval/inputs/release/` | Historical baseline release downloads (archive member layout retained) |
| `data/eval/reference_set_comparison/` | Downloaded proxy models, scalers and caches |
| `outputs/gnn/reference_set_comparison/` | Reference exports, proxy run outputs, reports and UMAP figures |
| `outputs/gnn/summaries/` | Curated appendix tables and recoverability summary |

`prepare_inputs` creates relative file links for embedding inputs and independent
copies of decompositions, splits and physical/cosine graphs in the baseline results
tree. This prevents baseline rebuilds from changing the notebook inputs. `--copy`
uses copies for embedding inputs too, for systems without symlinks. It checks existing destinations and refuses conflicting files. Re-run
it after downloading inputs in a fresh checkout. It does not fabricate missing
data, compute labels, change splits, or run training.

All 135 source reference-GNN checkpoints were retained locally beneath
`outputs/gnn/{connectivity,edge_attribute,relational_edge}_reference_gnn/runs/`.
Of these, 114 were absent from Crys_METH and 21 also exist in the frozen notebook
bundle. All are independent copies so retraining cannot overwrite that bundle.
These files are not committed to Git. Keep or
transfer `data/eval` and `outputs/gnn` separately when reproducing the experiments elsewhere.

The old workflow requires additional intermediate artifacts beyond the frozen
notebook bundle: reference pools, compositions, cosine neighbors, labels, SOAP
features, and phase edges, depending on the experiment. Their builders are
included; raw release files, full hull databases, exact scalers, and missing
intermediates must still be supplied or generated. Copying code does not recreate
these historical datasets. The unique checkpoints alone are insufficient to
rerun every ablation. Existing result summaries live under `outputs/gnn/summaries/`.
Regenerated run artifacts are Git-ignored; curated summaries are retained in Git.
The existing frozen notebook run under `data/eval/results/heldout_600` remains
separate from these GNN experiment outputs.

## Workflow guide

- **Fixed baselines:** `graphs.build`, `graphs.analysis`, `checks.retrieval`,
  `baselines.reference_sets`, `baselines.energy`, and the other rule/ablation
  modules. These consume the aligned frozen vector/metadata view; downstream
  commands require the preceding graph/pool/label outputs.
- **Reference-utility GNNs:** `datasets.decomposition`,
  `graphs.build_reference_graph`, `training.train_reference_gnn`, and
  `evaluation.reference_gnn`; the edge-attribute and relational variants have
  correspondingly named modules. `graphs.connectivity_adjacency` builds the
  SOAP/phase topology ablations. Preserve the fixed splits and protocol.
- **Embedding analysis:** run `embedding_structure.*` modules with `--help`;
  commands accepting `--data` take `data/eval/inputs/references`. This includes
  community maps, energy and space-group capture, and confound controls.
- **Independent proxy comparison:** inspect
  `reference_set_comparison/experiment.yml`, export reference embeddings with
  `reference_set_comparison.generate_embeddings`, then run pipeline stages
  `prepare -> embed -> score -> select`. For each proxy, use `fetch-hull`,
  `phases`, `download-model`, `smoke-model`, `predict`, and `hull` as applicable;
  finish with `report` (optionally `--model megnet` or `--model esen`). The pipeline
  checks prerequisites and receipts. Reference and candidate exports must share
  the same checkpoint, scaler and preprocessing. A reconstructed scaler is an
  approximation, not proof of reproducing the original embedding space.
- **Historical audits:** the `checks/` and reporting modules expect specific
  completed experiment outputs; they are not generic smoke tests. Use `tests/`
  for data-independent verification.

The migrated encoder code imports `utils.jepa.JEPA` and uses `configs/jepa.yml`.
Proxy environments have conflicting dependency requirements: use their separate
`requirements-jepa.txt`, `requirements-megnet.txt`, and `requirements-esen.txt`
files rather than combining them. SOAP requires `dscribe` from
`requirements/structure.txt`; it loads only when constructing descriptors.
MatterSim workflows require the corresponding inference dependencies. Historical
pinned requirement files describe those workflows, not a single tested lock for
the root environment.

## Scientific and provenance boundaries

Reference-utility GNN targets are reference participation in candidate
fixed-hull decompositions. Candidate stability labels must not enter their
training or checkpoint selection. Older `training.train`/`training.predict` and
candidate-model code are retained for completeness, but those candidate-stability
runs are not evidence for the reference-selection claim.

MatterSim, MEGNet and eSEN results are proxy assessments, not DFT confirmation.
Previously inspected partitions cannot become unseen test sets again. Frozen
receipts remain enforced. The notebook graph checker normalizes only the migrated
decomposition input-path edit when computing its historical algorithm digest;
algorithm edits still invalidate the receipt. Newly exported proxy embeddings
record the local encoder hash and must not silently reuse stale source receipts.

## Migration verification

All 130 Python files from the source package are present, plus the additional
baseline tests and the local structure adapter. The relocated suite ran 131 tests:
128 passed and three data-dependent checks were skipped (SOAP graph, phase graph,
and a local MatterSim checkpoint were unavailable). The implementation modules imported, named experiment CLI help commands passed
from outside the repository, and the existing notebook graph and training
receipts verified. Full training, downloads and production
MEGNet/eSEN/MatterSim inference were not run as part of the migration.
