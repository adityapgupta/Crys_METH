# Crys_METH
## Python environment setup with Conda
```
conda create -n crys_meth python=3.12.0
conda activate crys_meth
conda install -c conda-forge mattersim==1.2.0
conda install -c conda-forge pymatgen=2025.6.14 ase=3.25.0 matminer=0.9.3
pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric p_tqdm lmdb easydict einops atomate2 umap-learn plotly scikit-learn matplotlib pandas numpy tqdm pyyaml jupyterlab ipykernel
pip install setuptools==80.9.0 lmdb==1.6.2 scipy==1.16.2 smact==3.2.0
```

## JEPA visualisation
From the repository root, generate or reuse the MP-20 embedding and t-SNE cache:
```bash
python visualise.py
```
This writes `outputs/visualization/embeddings.npy`,
`outputs/visualization/metadata.csv`, `tsne_coordinates.csv`, and
`tsne_crystal_system.html`. If `embeddings.npy` and `metadata.csv` already exist,
the script reuses them instead of regenerating the embeddings.

Useful options include `--metric cosine|euclidean`, `--perplexity 30`,
`--seed 42`, `--device auto|cpu|cuda`, and `--out-dir <directory>`. To regenerate
only a subset of MP-20, use `--limit-per-split <count>`.

## JEPA neighbor queries
By default, `neighbours.py` reads:

- `outputs/visualization/embeddings.npy`
- `outputs/visualization/metadata.csv`

Query a material ID with:
```bash
python neighbours.py --material-id mp-149 --k 10
```
This saves a neighbor CSV under `logs/jepa/` by default and optionally creates a
PCA plot when `--plot` is included:
```bash
python neighbours.py --material-id mp-149 --k 10 --plot
```
Use `--occurrence 1` when a material ID has multiple rows. For a custom cache,
pass `--input <embeddings.npy> --metadata <metadata.csv>`.

To create a legacy `.pt` bundle instead, run:
```bash
python neighbours.py --extract
```
This extracts MP-20 and writes `logs/jepa/jepa_embeddings.pt` by default. Use
`--extract-output <bundle.pt>` to change it. Include Alex-MP-20 with
`--dataset both`, or extract only that dataset with `--dataset alex_mp_20`. Query
a `.pt` bundle with `--input <bundle.pt>`.
For an Alex-MP-20 reference lookup, add `--reference-dataset alex_mp_20`.

## Download and Reproduce
Place and unzip the archives as follows, relative to the `Crys_METH` repository root:

- Unzip `exp_logs.zip` in the repository root, then place or rename its extracted `exp_logs` directory as `logs/`.
- Unzip `exp_finetune.zip` inside `data/`, then place or rename its extracted `exp_finetune` directory as `data/finetune/`.

## Evaluation notebook

The repository-root [`evaluation.ipynb`](evaluation.ipynb) is the main entry point
for the frozen candidate-ranking and stability evaluation. Its reusable Python
utilities live in [`utils/eval/evaluation/`](utils/eval/evaluation/) and are imported
by the notebook; cluster scripts and report sources remain alongside those utilities.
From the repository root, launch Jupyter Lab with the `crys_meth` environment:

```bash
conda activate crys_meth
jupyter lab
```

Open [`evaluation.ipynb`](evaluation.ipynb) and select the `crys_meth` kernel.
The notebook finds the repository root from the current working directory, so
start Jupyter in the `Crys_METH` directory. Complete each section in order:

1. Verify the frozen inputs and available saved results.
2. Download/verify the graph and prepared-structure bundles under `data/eval/`.
3. Generate candidate scores and freeze the selections.
4. Stop at the SoC handoff when the notebook directs it.
5. Upload/download the configured SoC phase and candidate-prediction outputs.
6. Set `RETURNED_READY = True` in the notebook only after the returned bundle is
   downloaded and verified, then resume the hull and stability analysis.

The evaluation's isolated frozen inputs, graphs, GNN checkpoints, and prepared run
bundle are staged under `data/eval/` and excluded from Git. Notebook helper
code and the small vendored baseline source set are in `utils/eval/`.
`data/eval/manifest.json` records the staged files and
SHA-256 hashes. These are separate from the ordinary JEPA data under `data/` and
model logs under `logs/`.

## Appendix experiments

The focused appendix analyses and reference-GNN ablations live in
[`extra/`](extra/), separate from the reusable evaluation utilities in
`utils/eval/`. Each experiment directory contains its own code, configuration,
documentation, and committed result summaries; shared reference-baseline code
is resolved through `REFERENCE_BASE_PATH`. See [`extra/README.md`](extra/README.md)
for the experiment index and setup instructions.
