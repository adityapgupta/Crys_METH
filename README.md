# Crys_METH
## Python environment setup with Conda
```
conda create -n crys_meth python=3.12.0
conda activate crys_meth
conda install -c conda-forge mattersim==1.2.0
conda install -c conda-forge pymatgen=2025.6.14 ase=3.25.0 matminer=0.9.3
pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric p_tqdm lmdb easydict einops atomate2 umap-learn plotly scikit-learn matplotlib pandas numpy tqdm pyyaml
pip install setuptools==80.9.0 lmdb==1.6.2 scipy==1.16.2 smact==3.2.0
```

## JEPA Embeddings and Neighbor Queries
From the repository root, create the embedding bundle once:
```bash
python neighbours.py --extract
```
This extracts MP-20 by default and writes `logs/jepa/jepa_embeddings.pt`. To include Alex-MP-20 as well, use `python neighbours.py --extract --dataset both`; to extract only Alex-MP-20, use `--dataset alex_mp_20`.

Then query a material ID using the saved bundle:
```bash
python neighbours.py --material-id mp-149 --k 10
```
For an Alex-MP-20 reference lookup, add `--reference-dataset alex_mp_20`.

## Download and Reproduce
Place and unzip the archives as follows, relative to the `Crys_METH` repository root:

- Unzip `exp_logs.zip` in the repository root, then place or rename its extracted `exp_logs` directory as `logs/`.
- Unzip `exp_finetune.zip` inside `data/`, then place or rename its extracted `exp_finetune` directory as `data/finetune/`.
