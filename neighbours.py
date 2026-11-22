import argparse
import csv
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from itertools import combinations
from pymatgen.core import Structure


def load_material_metadata(material_ids):
    requested_ids = set(material_ids)
    labels = {}
    formulas = {}
    sources = [
        ("mp_20", "data/base/mp_20/train.csv.gz", "e_above_hull", "pretty_formula"),
        ("mp_20", "data/base/mp_20/val.csv.gz", "e_above_hull", "pretty_formula"),
        ("mp_20", "data/base/mp_20/test.csv.gz", "e_above_hull", "pretty_formula"),
        (
            "alex_mp_20",
            "data/base/alex_mp_20/train.csv.gz",
            "energy_above_hull",
            "reduced_formula",
        ),
    ]
    for dataset, path, column, formula_column in sources:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=["material_id", column, formula_column],
        )
        for material_id, value, formula in zip(
            frame["material_id"], frame[column], frame[formula_column]
        ):
            material_id = str(material_id)
            if material_id in requested_ids:
                formulas.setdefault(material_id, str(formula))
                labels.setdefault(material_id, []).append(
                    {"dataset": dataset, "hull_energy": float(value)}
                )
    missing_ids = requested_ids - formulas.keys()
    for path in ["data/jepa/mp.csv.gz", "data/jepa/mptrj.csv.gz"]:
        if not missing_ids or not os.path.exists(path):
            continue
        for frame in pd.read_csv(
            path,
            compression="gzip",
            usecols=["material_id", "cif"],
            chunksize=50000,
        ):
            frame["material_id"] = frame["material_id"].astype(str)
            for _, row in frame[frame["material_id"].isin(missing_ids)].iterrows():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    structure = Structure.from_str(row["cif"], fmt="cif")
                formulas[row["material_id"]] = structure.composition.reduced_formula
            missing_ids -= set(frame["material_id"]) & missing_ids
            if not missing_ids:
                break
    return labels, formulas


def expand_into_subsystems(chemical_system):
    elements = chemical_system.split("-")
    return {
        "-".join(sorted(subsystem))
        for size in range(1, len(elements) + 1)
        for subsystem in combinations(elements, size)
    }


def load_reference_neighbors(material_id, dataset):
    reference_path = f"data/base/{dataset}/train.csv.gz"
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

    reference = pd.read_csv(reference_path, compression="gzip")
    query_rows = pd.DataFrame()
    for split in ["train", "val", "test"]:
        query_path = f"data/base/{dataset}/{split}.csv.gz"
        if not os.path.exists(query_path):
            continue
        query_source = pd.read_csv(query_path, compression="gzip")
        query_rows = query_source[
            query_source["material_id"].astype(str) == material_id
        ]
        if not query_rows.empty:
            break
    if query_rows.empty:
        raise ValueError(
            f"{material_id!r} was not found in the available {dataset} splits."
        )

    query_structure = Structure.from_str(query_rows.iloc[0]["cif"], fmt="cif")
    query_system = query_structure.composition.chemical_system
    subsystems = expand_into_subsystems(query_system)
    neighbors = reference[reference["chemical_system"].isin(subsystems)].copy()
    neighbors = neighbors[
        neighbors["material_id"].astype(str) != material_id
    ].copy()
    neighbors.insert(0, "matched_subsystem", neighbors["chemical_system"])
    neighbors.insert(0, "query_chemical_system", query_system)
    return neighbors, query_system, sorted(subsystems)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find nearest JEPA embedding neighbors for a material ID."
    )
    parser.add_argument("--material-id", default=None)
    parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Zero-based occurrence when a material ID appears multiple times.",
    )
    parser.add_argument("--input", default="logs/jepa/jepa_embeddings.pt")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract embeddings only, save the bundle, and exit.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mp_20", "alex_mp_20", "both"],
        default="mp_20",
        help="Dataset to embed when --extract is used (default: mp_20).",
    )
    parser.add_argument("--checkpoint", default="logs/jepa/saved_model/model_1972.pt")
    parser.add_argument("--scaler", default="data/jepa/mean_std_scaler.pt")
    parser.add_argument("--config", default="configs/jepa.yml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--reference-dataset",
        choices=["mp_20", "alex_mp_20"],
        default="mp_20",
        help="Training set used for the original chemical-system reference lookup.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=262144,
        help="Number of embeddings scored per search chunk.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Neighbor CSV path; defaults to logs/jepa/<material_id>_neighbors.csv.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PCA plot containing the query and its nearest neighbors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.extract:
        from utils.extract import extract_embedding_bundle

        extract_embedding_bundle(
            output=args.input,
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            scaler=args.scaler,
            config=args.config,
            batch_size=args.batch_size,
            device=args.device,
        )
        return
    if not args.material_id:
        raise ValueError("--material-id is required unless --extract is used")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Embedding bundle not found: {args.input}. Run `python query.py --extract` first."
        )
    bundle = torch.load(args.input, map_location="cpu", weights_only=False)
    material_ids = [str(value) for value in bundle["material_id"]]
    matches = [index for index, value in enumerate(material_ids) if value == args.material_id]
    if not matches:
        raise ValueError(f"{args.material_id!r} was not found in {args.input}")
    if args.occurrence < 0 or args.occurrence >= len(matches):
        raise ValueError(
            f"{args.material_id!r} occurs {len(matches)} times; "
            f"--occurrence must be between 0 and {len(matches) - 1}."
        )
    if len(matches) > 1:
        print(
            f"Warning: {args.material_id!r} occurs {len(matches)} times; "
            f"using occurrence {args.occurrence}."
        )
    query_index = matches[args.occurrence]
    if args.k < 1 or args.k >= len(material_ids):
        raise ValueError("--k must be between 1 and the number of materials minus 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    embeddings = bundle["embedding"].float()
    embeddings /= embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    query = embeddings[query_index]
    candidate_k = min(len(embeddings), max(args.k * 32, args.k + len(matches)))
    best_distances = torch.full((candidate_k,), float("inf"))
    best_indices = torch.full((candidate_k,), -1, dtype=torch.long)
    for start in range(0, len(embeddings), args.chunk_size):
        end = min(start + args.chunk_size, len(embeddings))
        distances = 1.0 - embeddings[start:end] @ query
        for excluded_index in matches:
            if start <= excluded_index < end:
                distances[excluded_index - start] = float("inf")
        chunk_k = min(candidate_k, distances.numel())
        chunk_distances, local_indices = torch.topk(
            distances, chunk_k, largest=False
        )
        chunk_indices = local_indices + start
        candidates_distances = torch.cat((best_distances, chunk_distances))
        candidates_indices = torch.cat((best_indices, chunk_indices))
        selected = torch.topk(
            candidates_distances, candidate_k, largest=False
        ).indices
        best_distances = candidates_distances[selected]
        best_indices = candidates_indices[selected]

    nearest = torch.stack((best_distances, best_indices), dim=1)
    nearest = nearest[torch.argsort(nearest[:, 0])]
    selected_ids = set()
    distinct_nearest = []
    for distance, index in nearest:
        index = int(index)
        if index < 0 or material_ids[index] in selected_ids:
            continue
        selected_ids.add(material_ids[index])
        distinct_nearest.append((distance, index))
        if len(distinct_nearest) == args.k:
            break
    if len(distinct_nearest) < args.k:
        raise RuntimeError(
            f"Only found {len(distinct_nearest)} distinct neighbors. "
            "Increase the candidate pool multiplier in the script."
        )
    nearest = distinct_nearest
    labels, formulas = load_material_metadata(
        [args.material_id] + [material_ids[index] for _, index in nearest]
    )
    query_energy = float(bundle["ef_per_atom"][query_index])
    reference_neighbors, query_system, subsystems = load_reference_neighbors(
        args.material_id, args.reference_dataset
    )

    rows = []
    for rank, (distance, index) in enumerate(nearest, start=1):
        rows.append(
            {
                "rank": rank,
                "query_material_id": args.material_id,
                "neighbor_material_id": material_ids[index],
                "neighbor_formula": formulas.get(material_ids[index], ""),
                "cosine_distance": float(distance),
                "query_ef_per_atom": query_energy,
                "neighbor_ef_per_atom": float(bundle["ef_per_atom"][index]),
                "neighbor_stability": labels.get(material_ids[index], []),
            }
        )

    output = args.output or f"logs/jepa/{args.material_id}_neighbors.csv"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Query: {args.material_id}")
    print(f"Query formula: {formulas.get(args.material_id, 'formula unavailable')}")
    print(f"Query energy per atom: {query_energy}")
    print(f"Saved {len(rows)} neighbors to {output}")
    reference_output = (
        os.path.splitext(output)[0] + f"_{args.reference_dataset}_references.csv"
    )
    reference_neighbors.to_csv(reference_output, index=False)
    print(
        f"Reference chemical system: {query_system}; "
        f"matched {len(reference_neighbors)} crystals across {len(subsystems)} systems"
    )
    print(f"Saved original screening references to {reference_output}")
    for row in rows:
        print(
            f"{row['rank']:>3}: {row['neighbor_material_id']} "
            f"({row['neighbor_formula'] or 'formula unavailable'}) "
            f"distance={row['cosine_distance']:.6f}"
        )

    if args.plot:
        indices = [query_index] + [int(index) for _, index in nearest]
        points = embeddings[indices]
        _, _, v = torch.pca_lowrank(points, q=2, center=True)
        centered = points - points.mean(dim=0, keepdim=True)
        points_2d = centered @ v[:, :2]
        plot_path = os.path.splitext(output)[0] + ".png"
        plt.figure(figsize=(8, 7))
        plt.scatter(points_2d[1:, 0], points_2d[1:, 1], s=45, label="Neighbors")
        plt.scatter(
            points_2d[0, 0],
            points_2d[0, 1],
            s=100,
            marker="*",
            color="red",
            label=args.material_id,
        )
        for row, point in zip(rows, points_2d[1:]):
            plt.annotate(row["neighbor_material_id"], (point[0], point[1]), fontsize=7)
        plt.xlabel("Local PCA 1")
        plt.ylabel("Local PCA 2")
        plt.title(f"JEPA neighbors of {args.material_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)
        plt.close()
        print(f"Saved neighbor plot to {plot_path}")


if __name__ == "__main__":
    main()
