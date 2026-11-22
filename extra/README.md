# Appendix experiments

Cleaned, self-documenting copies of the code and headline results for four
experiments, collected for the report appendix:

| Folder | Experiment | Result |
|---|---|---|
| `space_group_concentration/` | Do individual space groups concentrate in a few embedding communities? | 61/73 eligible groups concentrate (BH q ≤ 0.05) under a size-matched permutation null |
| `edge_attribute_gnn/` | Do embedding-similarity / same-space-group / electronegativity edge features help the reference GNN? | Null: no gain over the scalar co-occurrence weight |
| `relational_edge_gnn/` | Do edge features built *independently* of the embeddings (SOAP geometry, co-occurrence PMI, hull tie-lines) help? | Null: no reliable gain; features verified largely non-redundant |
| `connectivity_gnn/` | Does any graph *wiring* (embedding-kNN, SOAP geometry, hull tie-line) beat the no-graph readout? | Null: no wiring beats the no-graph readout or the co-occurrence graph |

Each folder holds that experiment's own code plus its headline result CSVs under
`results/`.

## The shared base (`REFERENCE_BASE_PATH`)

These experiments are **not duplicated here** — they build on a shared base
pipeline (the base dataset/graph, the reference GNN, the trainer, the evaluator,
and all precomputed inputs) that lives in the `reference_baselines` package,
maintained separately. The code here imports it as `reference_baselines.*`.

To make those imports resolve, point `REFERENCE_BASE_PATH` at the directory that
*contains* the `reference_baselines` package:

1. `cp .env.example .env`
2. edit `.env` and set `REFERENCE_BASE_PATH` to your checkout.

`env.py` reads `.env` and puts that directory on `sys.path`. Every module here
loads it before importing anything from `reference_baselines`. Operational paths
(inputs, configs, where fresh runs write) therefore come from the base tree; the
`config.toml` and `results/*.csv` kept in each folder are the committed snapshots
for the appendix.

## What is intentionally not here

- Cluster submit / sync scripts (`*.slurm`, `*.sh`) — the base repo keeps those.
- The shared base modules themselves — referenced via `REFERENCE_BASE_PATH`.
- Intermediate run artifacts (smoke runs, superseded pre-fix runs) — only the
  final reserved-evaluation CSVs are kept.
