# Space-group concentration

Do individual space groups concentrate in a small number of embedding
communities? Each space group's top-five-community share is compared against a
size-matched label-permutation null.

## Files (run in order)

1. `build_graph.py` — weighted kNN graph over the frozen embeddings.
2. `community_lists_and_map.py` — Louvain communities → `crystals_by_community.csv`.
3. `spacegroup_capture.py` — concentration vs. the permutation null → `spacegroup_capture.csv`.

`common.py` holds the shared argument parsing, provenance and I/O helpers.

## Results ([`outputs/gnn/summaries/space_group_concentration/`](../../outputs/gnn/summaries/space_group_concentration/))

- `spacegroup_capture.csv` — per-group concentration, enrichment, permutation p, BH q.
- `community_summary.csv` — per-community size and dominant space group / chemistry.
- `capture_vs_homogeneity.csv` — concentration vs. label homogeneity (exploratory).

Headline: 61 of 73 eligible groups have BH-adjusted q ≤ 0.05.
