"""Shared JEPA checkpoint loading and crystal embedding helpers."""

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from components.jepa import JEPA


def load_jepa_model(config, checkpoint_path, device, matrix_scaler=None):
    model = JEPA(config, matrix_scaler=matrix_scaler).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def extract_loader_embeddings(model, loader, material_ids):
    from tqdm import tqdm

    embeddings, energies = [], []
    with torch.inference_mode():
        for data in tqdm(loader, desc="Extracting JEPA embeddings"):
            embedding, ef_per_atom = model.get_emb(data)
            embeddings.append(embedding)
            energies.append(ef_per_atom)
    embeddings = torch.cat(embeddings, dim=0)
    energies = torch.cat(energies, dim=0)
    if embeddings.shape[0] != len(material_ids):
        raise RuntimeError("Embedding count does not match the material ID count")
    return embeddings, energies


def prepare_structure(structure):
    """Apply the JEPA training preprocessing to one pymatgen structure."""
    structure = structure.get_primitive_structure().get_reduced_structure()
    if not structure.is_ordered or not 1 <= len(structure) <= 499:
        raise ValueError("Need an ordered structure with 1..499 sites")
    if any(not 1 <= number < 100 for number in structure.atomic_numbers):
        raise ValueError("Encoder supports atomic numbers 1..99")

    coordinates = torch.tensor(structure.frac_coords, dtype=torch.float32)
    numbers = torch.tensor(structure.atomic_numbers, dtype=torch.float32)
    key = (1000 * numbers + 100 * coordinates[:, 0]
           + 10 * coordinates[:, 1] + coordinates[:, 2])
    order = torch.argsort(key)

    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    left, singular, right = torch.linalg.svd(lattice)
    rotation = left @ right
    polar = rotation @ (right.T @ torch.diag(singular) @ right) @ rotation.T
    polar[torch.abs(polar) < 1e-5] = 0.0
    indices = torch.triu_indices(3, 3)
    features = polar[indices[0], indices[1]] / len(structure) ** (1 / 3)
    if not torch.isfinite(features).all() or not torch.isfinite(coordinates).all():
        raise ValueError("Nonfinite structure features")
    return structure, coordinates[order], numbers[order].long(), features


def encode_prepared(prepared, model, means, stds, device):
    import torch.nn.functional as functional

    counts = torch.tensor([len(item[0]) for item in prepared], device=device)
    matrices = torch.stack([item[3] for item in prepared]).to(device)
    matrices = (matrices - means) / stds
    batch = torch.repeat_interleave(torch.arange(len(prepared), device=device), counts)
    coordinates = torch.cat([item[1] for item in prepared]).to(device)
    numbers = torch.cat([item[2] for item in prepared]).to(device)
    features = torch.cat([
        coordinates, functional.one_hot(numbers, 100).float(), matrices[batch],
    ], dim=-1)
    return model.encode(features, batch).detach().cpu().numpy()


def embed_structures(structures, model, means, stds, device, batch_size=32):
    """Embed pymatgen structures in input order using the shared JEPA encoder."""
    import numpy as np

    results = []
    with torch.inference_mode():
        for start in range(0, len(structures), batch_size):
            prepared = [prepare_structure(s) for s in structures[start:start + batch_size]]
            results.append(encode_prepared(prepared, model, means, stds, device))
    return np.concatenate(results, axis=0)


def load_encoder(checkpoint_path, scaler_path, config_path, device):
    device = torch.device(device)
    packed = torch.load(scaler_path, map_location="cpu", weights_only=True)
    if isinstance(packed, (tuple, list)):
        packed = torch.stack(list(packed))
    if not isinstance(packed, torch.Tensor) or packed.shape != (2, 6):
        raise ValueError("Expected scaler tensor with shape (2, 6)")
    if not torch.isfinite(packed).all() or not (packed[1] > 0).all():
        raise ValueError("Scaler must be finite with positive standard deviations")

    config_data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = SimpleNamespace(model=SimpleNamespace(**config_data["model"]), device=device)
    model, _ = load_jepa_model(config, checkpoint_path, device)
    means, stds = packed.float().to(device)
    return model, means, stds, config
