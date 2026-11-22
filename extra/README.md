# Appendix experiments

Cleaned, self-documenting copies of the code and headline results for four
experiments, collected for the report appendix:

| Folder | Experiment | Result |
|---|---|---|
| `space_group_concentration/` | Do individual space groups concentrate in a few embedding communities? | 61/73 eligible groups concentrate (BH q ≤ 0.05) under a size-matched permutation null |
| `edge_attribute_gnn/` | Do embedding-similarity / same-space-group / electronegativity edge features help the reference GNN? | Null: no gain over the scalar co-occurrence weight |
| `relational_edge_gnn/` | Do edge features built *independently* of the embeddings (SOAP geometry, co-occurrence PMI, hull tie-lines) help? | Null: no reliable gain; features verified largely non-redundant |
| `connectivity_gnn/` | Does any graph *wiring* (embedding-kNN, SOAP geometry, hull tie-line) beat the no-graph readout? | Null: no wiring beats the no-graph readout or the co-occurrence graph |

Each folder holds compatibility entry points and documentation. Headline result
tables are under [`outputs/gnn/summaries/`](../outputs/gnn/summaries/), grouped by
the same experiment names.

## Shared implementation

All implementations now live in
[`../gnn/reference_baselines/`](../gnn/README.md).
The Python files here are compatibility entry points, so existing commands still
work. `env.py` locates the bundled package automatically; no `.env` file or
`REFERENCE_BASE_PATH` setting is used.

From the repository root, stage the existing evaluation inputs once:

```bash
python gnn/run.py prepare
python extra/edge_attribute_gnn/train.py --help
python extra/relational_edge_gnn/train.py --help
```

Experiment `config.toml` files here and result summaries under `outputs/gnn/summaries/`
are historical snapshots.
Executable configurations live in the shared package's `configs/` folder. New
outputs and checkpoints live under `outputs/gnn/`. Some experiments require
additional raw inputs and intermediate artifacts; see the shared package README.
