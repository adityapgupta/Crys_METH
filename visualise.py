"""Embed MP20 crystals and plot a t-SNE projection by crystal system."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_inputs(directory, splits, limit=None):
    import numpy as np
    import pandas as pd

    frames = []
    for split in splits:
        path = next(
            (
                directory / f"{split}{extension}"
                for extension in (".csv.gz", ".csv")
                if (directory / f"{split}{extension}").is_file()
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                f"Missing {split}.csv or {split}.csv.gz in {directory}"
            )
        frame = pd.read_csv(path, keep_default_na=False, nrows=limit)
        if not {"material_id", "cif"}.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{path} must contain material_id and cif and at least one row")
        frame["source_row"] = np.arange(len(frame))
        frame["split"] = split
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    if table.material_id.eq("").any() or table.material_id.duplicated().any():
        raise ValueError("Material IDs must be nonempty and unique across selected splits")
    table["embedding_row"] = np.arange(len(table))
    return table


def embed(args):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from utils.extract import JEPAEmbeddingExtractor

    table = load_inputs(args.data_dir, args.splits, args.limit_per_split)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    extractor = JEPAEmbeddingExtractor(
        args.checkpoint, args.scaler, args.config, device
    )
    embeddings = np.empty(
        (len(table), extractor.config.model.hidden_dim), dtype=np.float32
    )
    labels = []

    with torch.inference_mode():
        for start in range(0, len(table), args.batch_size):
            stop = min(start + args.batch_size, len(table))
            structures = []
            for row in table.iloc[start:stop].itertuples():
                try:
                    structure = Structure.from_str(row.cif, fmt="cif")
                    structures.append(structure)
                    labels.append(
                        {
                            "crystal_system": SpacegroupAnalyzer(
                                structure, symprec=args.symprec
                            ).get_crystal_system(),
                            "formula": structure.composition.reduced_formula,
                            "num_atoms": len(structure),
                        }
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Failed {row.material_id} ({row.split}, row {row.source_row})"
                    ) from exc

            values = extractor.embed_structures(
                structures, batch_size=args.batch_size
            )
            if values.shape != (stop - start, embeddings.shape[1]) or not np.isfinite(
                values
            ).all():
                raise ValueError(f"Invalid embeddings at rows {start}:{stop}")
            embeddings[start:stop] = values
            print(f"Embedded {stop}/{len(table)} crystals", flush=True)

    metadata = table.drop(columns="cif").copy()
    for key in labels[0]:
        metadata[key] = [label[key] for label in labels]
    np.save(args.out_dir / "embeddings.npy", embeddings, allow_pickle=False)
    metadata.to_csv(args.out_dir / "metadata.csv", index=False)
    return embeddings, metadata


def load_cached(args):
    import numpy as np
    import pandas as pd

    embeddings_path = args.out_dir / "embeddings.npy"
    metadata_path = args.out_dir / "metadata.csv"
    if not embeddings_path.is_file() and not metadata_path.is_file():
        return None
    if not embeddings_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Cached output is incomplete; embeddings.npy and metadata.csv are both required"
        )

    embeddings = np.load(embeddings_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError("Cached embedding and metadata shapes do not match")
    if "crystal_system" not in metadata or "material_id" not in metadata:
        raise ValueError("Cached metadata must contain material_id and crystal_system")
    return embeddings, metadata


def plot_tsne(values, metadata, args):
    import numpy as np
    import plotly.express as px
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize

    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("t-SNE needs at least four finite embedding rows")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Zero embedding cannot be normalized")

    coordinates = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, len(values) - 1),
        metric=args.metric,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(normalize(values) if args.metric == "cosine" else values)

    frame = metadata.copy()
    frame["x"], frame["y"] = coordinates[:, 0], coordinates[:, 1]
    frame.to_csv(args.out_dir / "tsne_coordinates.csv", index=False)

    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="crystal_system",
        render_mode="webgl",
        hover_name="material_id",
        hover_data=["formula", "split"],
        title=f"MP20 JEPA — t-SNE — crystal system (n={len(frame):,})",
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.65})
    fig.update_layout(template="plotly_white", legend={"itemsizing": "constant"})
    fig.write_html(args.out_dir / "tsne_crystal_system.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/base/mp_20")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "logs/jepa/saved_model/model_1972.pt",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=ROOT / "data/jepa/mean_std_scaler.pt",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jepa.yml")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/mp20_visualization"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--symprec", type=float, default=0.1)
    args = parser.parse_args()

    if (
        args.batch_size < 1
        or args.perplexity <= 0
        or args.symprec <= 0
        or not 0 <= args.seed < 2**32
        or (args.limit_per_split is not None and args.limit_per_split < 1)
    ):
        parser.error("Invalid batch size, perplexity, symprec, limit, or seed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cached = load_cached(args)
    if cached is None:
        values, metadata = embed(args)
    else:
        values, metadata = cached
        print(f"Loaded cached embeddings from {args.out_dir}", flush=True)
    plot_tsne(values, metadata, args)
    print(f"Saved t-SNE projection to {args.out_dir}")


if __name__ == "__main__":
    main()
