# Relational edge-feature reference GNN

Follow-up to the edge-attribute experiment: do edge features built to be
*independent* of the node embeddings — SOAP geometry, co-occurrence PMI, hull
tie-line compatibility — help on the fixed physical topology? **Null result.**

## Files

- `edge_features.py` — precompute the three relational edge-feature sources (fit-side only).
- `reference_graph.py` — assemble the standardised edge-feature table on the physical graph.
- `model.py` — learned edge encoder reusing the edge-attribute gate/attention layers; variant masks.
- `train.py` — trains each variant × model × seed to the frozen checkpoint rule.
- `evaluate.py` — scores frozen checkpoints on the reserved partition.
- `recoverability_probe.py` — tests how far each edge feature is recoverable from the embeddings.
- `config.toml`, `test.py`.

`edge_features.py` needs `results/reference_hull_energies.csv` and the SOAP
fingerprints in the base tree (see the module docstring).

## Results ([`outputs/gnn/summaries/relational_edge_gnn/`](../../outputs/gnn/summaries/relational_edge_gnn/))

`metrics.csv`, `scores.csv`, `controlled_comparisons.csv` (reserved partition) and
`recoverability_probe.json`. Headline: no variant reliably beats the scalar
baseline; SOAP geometry is essentially unrecoverable from the embeddings (R²≈0.05),
hull compatibility is partially recoverable (AUC≈0.71).
