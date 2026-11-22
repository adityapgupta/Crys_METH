# Crys_METH

Crys_METH is a materials-discovery evaluation repository built around JEPA
crystal embeddings. It contains:

- JEPA embedding visualisation and nearest-neighbour analysis;
- chemical-reference and embedding-kNN graph analysis;
- a frozen candidate-ranking and stability evaluation;
- reference-selection baselines and reference-utility GNNs; and
- four appendix experiments that test graph wiring, edge attributes, relational
  features, and space-group concentration.

The repository includes source code, notebooks, configuration snapshots, curated
result tables, and some generated assets. Large datasets, downloaded models,
checkpoints, and run directories are intentionally excluded from Git; reproduce
or transfer those separately when needed.

## Repository map

| Path | Purpose |
| --- | --- |
| [`visualise.py`](visualise.py) | Embed MP-20 and create a crystal-system t-SNE view |
| [`neighbours.py`](neighbours.py) | Query cosine-nearest JEPA neighbours for a material |
| [`knn.ipynb`](knn.ipynb) | Inspect embeddings, build/sweep kNN graphs, and analyse reference sets |
| [`evaluation.ipynb`](evaluation.ipynb) | Main frozen candidate-ranking and stability workflow |
| [`utils/`](utils/) | JEPA extraction plus reusable evaluation and reporting utilities |
| [`gnn/`](gnn/) | Shared baseline, graph, GNN, proxy, reporting, and test implementation |
| [`extra/`](extra/) | Compatibility entry points and documentation for appendix experiments |
| [`data/`](data/) | Dataset inputs, frozen evaluation inputs, graphs, and result tables |
| [`logs/`](logs/) | JEPA and experiment logs/checkpoints when supplied locally |
| [`outputs/`](outputs/) | Visualisations, evaluation products, and GNN results |
| [`docs/`](docs/) | Pipeline diagrams, presentation material, and report assets |

## Environment

The commands below describe the environment used by the repository. They assume
Conda and a CUDA 12.4 PyTorch wheel are available. Adjust the PyTorch install
for a CPU-only machine if required.

```bash
conda create -n crys_meth python=3.12.0
conda activate crys_meth
conda install -c conda-forge mattersim==1.2.0
conda install -c conda-forge pymatgen=2025.6.14 ase=3.25.0 matminer=0.9.3
pip install --no-cache-dir --force-reinstall \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric p_tqdm lmdb easydict einops atomate2 \
  umap-learn plotly scikit-learn matplotlib pandas numpy tqdm pyyaml \
  jupyterlab ipykernel
pip install setuptools==80.9.0 lmdb==1.6.2 scipy==1.16.2 smact==3.2.0
```

The GNN package has workflow-specific requirement files under
[`gnn/reference_baselines/requirements/`](gnn/reference_baselines/requirements/)
and the proxy comparison has separate JEPA, MEGNet, and eSEN requirements under
[`gnn/reference_baselines/reference_set_comparison/`](gnn/reference_baselines/reference_set_comparison/).
Do not combine those conflicting environments into one lockfile.

## Data and experiment restoration

The ordinary JEPA workflows use:

- MP-20 splits in `data/base/mp_20/`;
- optional Alex-MP-20 data in `data/base/alex_mp_20/`;
- JEPA source data in `data/jepa/`; and
- the checkpoint and scaler under `logs/jepa/` and `data/jepa/`.

If experiment archives are supplied separately, restore them relative to the
repository root:

```text
exp_logs.zip      -> logs/
exp_finetune.zip  -> data/finetune/
```

The frozen evaluation/GNN workflows use the staged bundle under `data/eval/`.
Its manifest records file hashes. `prepare_inputs` only stages or links existing
files; it does not download data, invent labels, alter splits, or train models.

```bash
python gnn/run.py prepare
```

Keep the frozen notebook inputs under `data/eval/` separate from ordinary JEPA
data under `data/` and model logs under `logs/`.

## JEPA visualisation

[`visualise.py`](visualise.py) embeds the selected MP-20 splits with the JEPA
checkpoint, derives crystal-system/formula/atom-count metadata, and writes a
Plotly t-SNE projection:

```bash
python visualise.py
```

Default inputs are `data/base/mp_20/{train,val,test}.csv.gz`,
`logs/jepa/saved_model/model_1972.pt`, `data/jepa/mean_std_scaler.pt`, and
`configs/jepa.yml`. Default outputs are:

```text
outputs/visualization/embeddings.npy
outputs/visualization/metadata.csv
outputs/visualization/tsne_coordinates.csv
outputs/visualization/tsne_crystal_system.html
```

If `embeddings.npy` and `metadata.csv` already exist, they are validated and
reused. Useful options include:

```bash
python visualise.py --metric cosine --perplexity 30 --seed 42 --device auto
python visualise.py --limit-per-split 1000 --out-dir outputs/visualization-small
python visualise.py --splits train val --device cpu
```

Supported devices are `auto`, `cpu`, and `cuda`; the t-SNE metric is `cosine` or
`euclidean`. The embedding rows and metadata rows must remain aligned.

## JEPA neighbours

Query the cached visualisation embeddings by material ID:

```bash
python neighbours.py --material-id mp-149 --k 10
python neighbours.py --material-id mp-149 --k 10 --plot
```

The script normalises embeddings, performs chunked cosine-distance search,
excludes all occurrences of the query, and returns distinct material IDs. It
writes `logs/jepa/<material_id>_neighbors.csv` by default, plus a local PCA
plot when `--plot` is supplied. It also writes the original chemical-system
reference lookup to a companion CSV. Use `--occurrence N` when an ID occurs
multiple times.

To use another NumPy cache:

```bash
python neighbours.py --material-id mp-149 --input path/embeddings.npy \
  --metadata path/metadata.csv --k 20
```

For the legacy PyTorch bundle:

```bash
python neighbours.py --extract
python neighbours.py --extract --dataset both \
  --extract-output logs/jepa/jepa_embeddings.pt
python neighbours.py --material-id mp-149 \
  --input logs/jepa/jepa_embeddings.pt --reference-dataset alex_mp_20
```

`--dataset` accepts `mp_20`, `alex_mp_20`, or `both`. Extraction uses the same
checkpoint, scaler, config, batch size, and device options as the visualisation
workflow.

## kNN notebook

[`knn.ipynb`](knn.ipynb) is the exploratory, reusable analysis notebook. It:

1. validates the embedding matrix and aligned metadata;
2. inspects chemical-system reference sets;
3. constructs a symmetric Gaussian-weighted Euclidean kNN graph;
4. caches exact sorted neighbours (up to the configured top-100);
5. sweeps `k` for directed and symmetric neighbourhoods;
6. compares reference precision/recall, connectivity, and absolute energy gaps;
7. plots the k trade-offs; and
8. demonstrates chemical-only and augmented embedding reference selection for
   newly generated crystals.

The notebook stores the neighbour cache and sweep tables alongside the selected
data directory. The cache is keyed to the ordered embedding contents, shape,
dtype, and algorithm version, so changed or reordered embeddings invalidate it.
Start Jupyter from the repository root:

```bash
conda activate crys_meth
jupyter lab
```

The notebook's kNN graph is an analysis tool; it is distinct from the physical
co-occurrence and cosine graphs prepared for the frozen GNN experiments.

## Frozen evaluation

[`evaluation.ipynb`](evaluation.ipynb) runs the frozen candidate-ranking and
stability evaluation. Each experiment supplies its selected final candidates
and predictions to the notebook, which checks the frozen inputs, splits,
graphs, checkpoints, and transfer receipts before continuing through phase
prediction and hull analysis.

The notebook then compares the final outputs from the different experiment
families using stability-focused metrics: candidate predictions, duplicate
checks, hull coverage, discovery fractions, and aggregate selection results.
CHGNet and MEGNet are used as proxy evaluators for the returned structures;
their comparisons measure relative stability and discovery performance, not
DFT-confirmed stability. Shared evaluation code is under
[`utils/eval/evaluation/`](utils/eval/evaluation/), with reports in
`data/eval/report/`.

## GNN and reference-selection experiments

The shared implementation lives in
[`gnn/reference_baselines/`](gnn/reference_baselines/). It covers:

- chemical, nearest-neighbour, energy, angular, formula-balance, random,
  conductance, spectral, and mean-ablation baselines;
- cosine, physical/co-occurrence, connectivity, diffusion, spectral, and
  relational-edge graph construction;
- reference-utility GNNs and edge-attribute/relational-edge variants;
- fixed-hull decomposition datasets and reference pools;
- held-out, duplicate, budget, non-parametric, recoverability, and proxy
  evaluations;
- embedding structure, community, energy-capture, space-group, and confound
  analyses; and
- reporting, integrity checks, and scientific regression tests.

Use the common launcher from the repository root:

```bash
python gnn/run.py --list
python gnn/run.py reference graphs --help
python gnn/run.py reference train --help
python gnn/run.py reference evaluate --help
python gnn/run.py edge-attribute train --help
python gnn/run.py relational-edge train --help
python gnn/run.py connectivity train --help
python gnn/run.py space-group communities --help
python gnn/run.py proxy --help
python gnn/run.py embeddings --help
```

Named workflows are:

| Launcher name | Main actions |
| --- | --- |
| `reference` | physical/reference graph construction, training, preparation, evaluation |
| `edge-attribute` | explicit edge-feature GNN training and evaluation |
| `relational-edge` | independent edge features, training, evaluation, recoverability probe |
| `connectivity` | SOAP/phase connectivity graphs, training, evaluation |
| `space-group` | communities, space-group capture, and community maps |
| `proxy` | prepare, embed, score, select, proxy inference, hulls, and reporting |
| `embeddings` | export JEPA reference embeddings |
| `prepare` | stage existing frozen inputs |

For direct package tests:

```bash
export PYTHONPATH="$PWD/gnn:$PWD/utils/eval:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s gnn/reference_baselines/tests -v
```

Generated checkpoints and run artifacts are normally written under
`outputs/gnn/`; curated, portable result tables are retained under
[`outputs/gnn/summaries/`](outputs/gnn/summaries/). Existing frozen notebook
results under `data/eval/results/heldout_600/` are separate from newly generated
GNN runs.

### Scientific boundaries

Reference-utility GNNs predict reference participation in fixed-hull candidate
decompositions. Candidate stability labels must not be used for their training
or checkpoint selection. Older candidate-model training modules remain for
historical completeness and are not evidence for the reference-selection claim.
MatterSim, MEGNet, and eSEN results are proxy assessments, not DFT confirmation.
Preserve the frozen splits, receipts, and provenance when extending an analysis.

## Extra appendix experiments

The folders under [`extra/`](extra/) retain compatibility commands, historical
configuration snapshots, and focused documentation. The executable
implementation is shared with `gnn/reference_baselines/`; no external checkout,
`.env`, or `REFERENCE_BASE_PATH` is required.

| Experiment | Question and conclusion | Entry point |
| --- | --- | --- |
| **Connectivity GNN** | Does embedding-kNN (`cosine`), SOAP geometry, or convex-hull tie-line (`phase`) wiring beat the physical co-occurrence graph or no-graph readout? **No wiring did.** | [`extra/connectivity_gnn/`](extra/connectivity_gnn/) |
| **Edge-attribute GNN** | Do embedding cosine, same-space-group, and electronegativity-profile edge features improve scalar co-occurrence weights? **No reliable gain.** | [`extra/edge_attribute_gnn/`](extra/edge_attribute_gnn/) |
| **Relational-edge GNN** | Do embedding-independent SOAP, co-occurrence-PMI, and hull tie-line features help? **No reliable gain.** SOAP was largely unrecoverable from embeddings (`R² ≈ 0.05`); hull compatibility was partially recoverable (`AUC ≈ 0.71`). | [`extra/relational_edge_gnn/`](extra/relational_edge_gnn/) |
| **Space-group concentration** | Do space groups concentrate in a few embedding communities? **61 of 73 eligible groups** passed BH-adjusted `q ≤ 0.05` under a size-matched permutation null. | [`extra/space_group_concentration/`](extra/space_group_concentration/) |

Examples:

```bash
python gnn/run.py prepare
python extra/edge_attribute_gnn/train.py --help
python extra/relational_edge_gnn/train.py --help
python extra/connectivity_gnn/evaluate.py --help
```

The small headline tables are in
[`outputs/gnn/summaries/`](outputs/gnn/summaries/), grouped as
`connectivity_gnn`, `edge_attribute_gnn`, `relational_edge_gnn`, and
`space_group_concentration`. Some full experiment inputs, release files,
intermediates, and checkpoints are not tracked and must be restored before
rerunning a historical result.

