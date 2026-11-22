"""Reusable JEPA embedding extraction for MP-20 and Alex-MP-20."""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from utils.inference import encode_prepared, load_encoder, prepare_structure


DATASET_SPLITS = {
    "mp_20": ("train", "val", "test"),
    "alex_mp_20": ("train",),
}


def _split_path(data_root, dataset, split):
    directory = Path(data_root) / dataset
    for suffix in (".csv.gz", ".csv"):
        path = directory / f"{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {split}.csv[.gz] for {dataset} in {directory}")


class JEPAEmbeddingExtractor:
    """Load a JEPA model once and encode structures in batches."""

    def __init__(self, checkpoint, scaler, config, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.means, self.stds, self.config = load_encoder(
            checkpoint, scaler, config, self.device
        )

    def embed_structures(self, structures, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not structures:
            return np.empty((0, self.config.model.hidden_dim), dtype=np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(structures), batch_size):
                prepared = [
                    prepare_structure(structure)
                    for structure in structures[start:start + batch_size]
                ]
                batches.append(encode_prepared(
                    prepared, self.model, self.means, self.stds, self.device
                ))
        return np.concatenate(batches, axis=0)


def extract_embedding_bundle(
    output="logs/jepa/jepa_embeddings.pt",
    dataset="mp_20",
    data_root="data/base",
    checkpoint="logs/jepa/saved_model/model_1972.pt",
    scaler="data/jepa/mean_std_scaler.pt",
    config="configs/jepa.yml",
    batch_size=32,
    device="auto",
    splits=None,
):
    """Embed selected dataset split(s), save the query-ready bundle, and return it."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    datasets = ("mp_20", "alex_mp_20") if dataset == "both" else (dataset,)
    invalid = set(datasets) - DATASET_SPLITS.keys()
    if invalid:
        raise ValueError(f"Unsupported dataset selection: {sorted(invalid)}")

    frames = []
    source_paths = []
    for name in datasets:
        selected_splits = splits or DATASET_SPLITS[name]
        if isinstance(selected_splits, str):
            selected_splits = (selected_splits,)
        for split in selected_splits:
            path = _split_path(data_root, name, split)
            frame = pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else None,
                                keep_default_na=False)
            if not {"material_id", "cif"}.issubset(frame.columns) or frame.empty:
                raise ValueError(f"{path} must contain material_id, cif, and at least one row")
            frame["material_id"] = frame["material_id"].astype(str)
            frame["dataset"] = name
            frame["split"] = split
            frames.append(frame)
            source_paths.append(path)

    table = pd.concat(frames, ignore_index=True)
    if table["material_id"].eq("").any():
        raise ValueError("Material IDs must be nonempty")
    formulas = []
    extractor = JEPAEmbeddingExtractor(checkpoint, scaler, config, device)
    embedding_batches = []
    for start in range(0, len(table), batch_size):
        rows = table.iloc[start:start + batch_size]
        structures = []
        for row in rows.itertuples():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    structure = Structure.from_str(row.cif, fmt="cif")
                structures.append(structure)
                formulas.append(structure.composition.reduced_formula)
            except Exception as exc:
                raise ValueError(
                    f"Invalid CIF for {row.material_id} ({row.dataset}/{row.split})"
                ) from exc
        embedding_batches.append(
            extractor.embed_structures(structures, batch_size=batch_size)
        )
        print(f"Embedded {min(start + len(rows), len(table))}/{len(table)} crystals", flush=True)
    embeddings = np.concatenate(embedding_batches, axis=0)
    formation_column = "formation_energy_per_atom"
    energies = (
        pd.to_numeric(table[formation_column], errors="coerce").to_numpy(dtype=np.float32)
        if formation_column in table.columns
        else np.full(len(table), np.nan, dtype=np.float32)
    )
    bundle = {
        "embedding": torch.from_numpy(embeddings),
        "material_id": table["material_id"].tolist(),
        "ef_per_atom": torch.from_numpy(energies),
        "formula": formulas,
        "dataset": table["dataset"].tolist(),
        "split": table["split"].tolist(),
        "embedding_type": "unaugmented_backbone_cls",
        "checkpoint": str(Path(checkpoint).resolve()),
        "source_files": [str(path) for path in source_paths],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(
        f"Saved {len(table)} {dataset} embeddings with shape {tuple(embeddings.shape)} "
        f"to {output_path}"
    )
    return bundle
