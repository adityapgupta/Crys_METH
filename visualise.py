"""Embed every MP20 crystal and plot unsupervised UMAP/t-SNE projections.

Requires the repository's JEPA environment plus umap-learn and plotly.
Example (uses the downloaded model_1972 checkpoint and its lattice scaler)::

    python visualise.py --method both

Inputs default to data/base/mp_20/{train,val,test}.csv[.gz]. No downloading
or scaler reconstruction is performed. One point represents one crystal,
not one chemical element. Labels never enter the embedding or projection.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREPROCESSING = "primitive-reduced-type-sorted-float32-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prototypes():
    """Ideal references; anonymous matching permits chemical substitutions."""
    from pymatgen.core import Lattice, Structure

    cubic = Lattice.cubic(5.6)
    specs = [
        ("rock salt", 225, cubic, ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]]),
        ("cesium chloride", 221, cubic, ["Cs", "Cl"], [[0, 0, 0], [.5, .5, .5]]),
        ("zinc blende", 216, cubic, ["Zn", "S"], [[0, 0, 0], [.25, .25, .25]]),
        ("fluorite / antifluorite", 225, cubic, ["Ca", "F"], [[0, 0, 0], [.25, .25, .25]]),
        ("cubic perovskite", 221, cubic, ["Ca", "Ti", "O"],
         [[0, 0, 0], [.5, .5, .5], [.5, .5, 0]]),
        ("diamond", 227, cubic, ["C"], [[.125, .125, .125]]),
        ("fcc", 225, cubic, ["Cu"], [[0, 0, 0]]),
        ("bcc", 229, cubic, ["Fe"], [[0, 0, 0]]),
        ("hcp", 194, Lattice.hexagonal(3, 3 * (8 / 3) ** .5),
         ["Mg"], [[1 / 3, 2 / 3, .25]]),
    ]
    return [(name, Structure.from_spacegroup(group, lattice, species, coords))
            for name, group, lattice, species, coords in specs]


def structure_labels(structure, references, matcher, symprec):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    # Filter by anonymous stoichiometry, but require actual geometry matching.
    formula = structure.composition.anonymized_formula
    matches = [name for name, ref in references
               if ref.composition.anonymized_formula == formula
               and matcher.fit_anonymous(ref, structure)]
    return {
        "derived_spacegroup": analyzer.get_space_group_number(),
        "crystal_system": analyzer.get_crystal_system(),
        "prototype": matches[0] if len(matches) == 1 else
                     ("ambiguous" if matches else "unmatched"),
        "prototype_matches": "; ".join(matches),
    }


def load_inputs(directory, splits, limit=None):
    import numpy as np
    import pandas as pd

    frames, sources = [], []
    for split in splits:
        path = next((directory / f"{split}{ext}" for ext in (".csv.gz", ".csv")
                     if (directory / f"{split}{ext}").is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing {split}.csv or {split}.csv.gz in {directory}")
        frame = pd.read_csv(path, keep_default_na=False, nrows=limit)
        if not {"material_id", "cif"}.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{path} must contain material_id and cif and at least one row")
        frame["source_row"] = np.arange(len(frame))
        frame["split"] = split
        frames.append(frame)
        sources.append(path)
    table = pd.concat(frames, ignore_index=True)
    if table.material_id.eq("").any() or table.material_id.duplicated().any():
        raise ValueError("Material IDs must be nonempty and unique across selected splits")
    table["embedding_row"] = np.arange(len(table))
    return table, sources


def embed(args):
    from importlib.metadata import version
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from utils.extract import JEPAEmbeddingExtractor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    table, sources = load_inputs(args.data_dir, args.splits, args.limit_per_split)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    extractor = JEPAEmbeddingExtractor(args.checkpoint, args.scaler, args.config, device)
    references = prototypes()
    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol,
                               primitive_cell=True, scale=True, attempt_supercell=False)
    embeddings = np.empty((len(table), extractor.config.model.hidden_dim), dtype=np.float32)
    labels = []
    with torch.inference_mode():
        for start in range(0, len(table), args.batch_size):
            stop = min(start + args.batch_size, len(table))
            structures = []
            for row in table.iloc[start:stop].itertuples():
                try:
                    structure = Structure.from_str(row.cif, fmt="cif")
                    structures.append(structure)
                    label = structure_labels(structure, references, matcher, args.symprec)
                    label.update(formula=structure.composition.reduced_formula,
                                 num_atoms=len(structure))
                    labels.append(label)
                except Exception as exc:
                    raise ValueError(f"Failed {row.material_id} ({row.split}, row {row.source_row}); "
                                     "no crystals silently skipped") from exc
            values = extractor.embed_structures(structures, batch_size=args.batch_size)
            if values.shape != (stop - start, embeddings.shape[1]) or not np.isfinite(values).all():
                raise ValueError(f"Invalid embeddings at rows {start}:{stop}")
            if start == 0:
                np.testing.assert_allclose(
                    values[:1], extractor.embed_structures(structures[:1], batch_size=1),
                                           rtol=1e-4, atol=1e-5)
            embeddings[start:stop] = values
            print(f"Embedded and labeled {stop}/{len(table)} crystals", flush=True)
    metadata = table.drop(columns="cif").copy()
    for key in labels[0]:
        metadata[key] = [label[key] for label in labels]
    np.save(args.out_dir / "embeddings.npy", embeddings, allow_pickle=False)
    metadata.to_csv(args.out_dir / "metadata.csv", index=False)
    manifest = {
        "sources": {str(p.resolve()): sha256(p) for p in sources},
        "checkpoint_sha256": sha256(args.checkpoint), "scaler_sha256": sha256(args.scaler),
        "config_sha256": sha256(args.config), "preprocessing": PREPROCESSING,
        "rows": len(table), "shape": list(embeddings.shape),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {name: version(name) for name in ("torch", "pymatgen", "numpy", "pandas")},
        "script_sha256": sha256(Path(__file__)),
        "label_note": "Prototype labels are derived geometry matches to a limited ideal library, "
                      "not MP20 ground truth. Unmatched does not mean invalid.",
        "embeddings_sha256": sha256(args.out_dir / "embeddings.npy"),
        "metadata_sha256": sha256(args.out_dir / "metadata.csv"),
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return embeddings, metadata


def project(values, method, args):
    import numpy as np
    from sklearn.preprocessing import normalize

    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Projection needs at least four finite embedding rows")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Zero embedding cannot be normalized")
    # Cosine geometry follows JEPA's contrastive objective. No supervised labels.
    x = normalize(values) if args.metric == "cosine" else values
    if method == "umap":
        from umap import UMAP
        return UMAP(n_components=2, n_neighbors=min(args.neighbors, len(x) - 1),
                    min_dist=args.min_dist, metric=args.metric, random_state=args.seed,
                    init="random").fit_transform(x)
    from sklearn.manifold import TSNE
    # Euclidean distance on unit vectors is monotonic in cosine distance;
    # use exact requested cosine metric here, rather than silently changing it.
    return TSNE(n_components=2, perplexity=min(args.perplexity, len(x) - 1),
                metric=args.metric, init="pca", learning_rate="auto",
                random_state=args.seed).fit_transform(x)


def plot(values, metadata, args):
    from importlib.metadata import version
    import plotly.express as px

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    for method in methods:
        xy = project(values, method, args)
        frame = metadata.copy()
        frame["x"], frame["y"] = xy[:, 0], xy[:, 1]
        frame.to_csv(args.out_dir / f"{method}_coordinates.csv", index=False)
        labels = ["prototype", "derived_spacegroup", "crystal_system", "split"]
        if args.include_spacegroup_number:
            labels.append("spacegroup.number")
        for label in labels:
            if label not in frame:
                continue
            frame[label] = frame[label].astype(str)
            fig = px.scatter(frame, x="x", y="y", color=label, render_mode="webgl",
                             hover_name="material_id",
                             hover_data=["formula", "split", "prototype_matches", "derived_spacegroup"],
                             title=f"MP20 JEPA — {method.upper()} — {label} (n={len(frame):,})")
            fig.update_traces(marker={"size": 4, "opacity": .65})
            fig.update_layout(template="plotly_white", legend={"itemsizing": "constant"})
            fig.write_html(args.out_dir / f"{method}_{label}.html", include_plotlyjs=True)
    metadata.prototype.value_counts().rename_axis("prototype").to_csv(args.out_dir / "prototype_counts.csv")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    packages = ["scikit-learn", "plotly"] + (["umap-learn"] if "umap" in methods else [])
    settings["versions"] = {name: version(name) for name in packages}
    (args.out_dir / "projection_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/base/mp_20")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "logs/jepa/saved_model/model_1972.pt")
    parser.add_argument("--scaler", type=Path, default=ROOT / "data/jepa/mean_std_scaler.pt",
                        help="Exact training lattice scaler, tensor of shape (2, 6)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jepa.yml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mp20_visualization")
    parser.add_argument("--plot-only", action="store_true", help="Reuse verified embeddings and metadata in out-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-split", type=int, help="Optional smoke test; default uses EVERY row")
    parser.add_argument("--method", choices=["umap", "tsne", "both"], default="umap")
    parser.add_argument("--include-spacegroup-number", "--spacegroup-number",
                        dest="include_spacegroup_number", action="store_true",
                        help="Also plot using the original MP-20 spacegroup.number field")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=.1)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--symprec", type=float, default=.1)
    parser.add_argument("--ltol", type=float, default=.2)
    parser.add_argument("--stol", type=float, default=.3)
    parser.add_argument("--angle-tol", type=float, default=5)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.neighbors < 2 or args.perplexity <= 0
            or not 0 <= args.min_dist <= 1 or args.symprec <= 0
            or min(args.ltol, args.stol, args.angle_tol) <= 0
            or not 0 <= args.seed < 2**32
            or (args.limit_per_split is not None and args.limit_per_split < 1)):
        parser.error("Invalid batch size, projection settings, limit, seed or matching tolerance")
    if not args.plot_only and (args.checkpoint is None or args.scaler is None):
        parser.error("--checkpoint and --scaler are required unless --plot-only is used")
    if args.plot_only:
        import numpy as np
        import pandas as pd
        manifest = json.loads((args.out_dir / "provenance.json").read_text(encoding="utf-8"))
        for name in ("embeddings", "metadata"):
            path = args.out_dir / (name + (".npy" if name == "embeddings" else ".csv"))
            if sha256(path) != manifest[name + "_sha256"]:
                raise ValueError(f"Modified cached {name}")
        values = np.load(args.out_dir / "embeddings.npy", allow_pickle=False)
        metadata = pd.read_csv(args.out_dir / "metadata.csv", keep_default_na=False)
        if values.shape != tuple(manifest["shape"]) or len(values) != len(metadata):
            raise ValueError("Embedding/metadata shape mismatch")
        np.testing.assert_array_equal(metadata.embedding_row, np.arange(len(values)))
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        values, metadata = embed(args)
    plot(values, metadata, args)
    print(f"Saved embeddings, labels and interactive plots to {args.out_dir}")


if __name__ == "__main__":
    main()
