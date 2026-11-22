# Edge-attribute reference GNN

On the fixed physical (hull co-occurrence) topology, do explicit edge features —
embedding cosine, same-space-group, electronegativity-profile distance — improve
reference weighting over the scalar co-occurrence weight? **Null result.**

## Files

- `reference_graph.py` — wraps the base `PoolDataset` with the three edge features.
- `model.py` — edge-gated GCN/GAT/SAGE layers, the no-message `readout` control, `build_model`.
- `train.py` — trains each variant × model × seed to the frozen checkpoint rule.
- `evaluate.py` — scores frozen checkpoints on the reserved partition.
- `config.toml` — models, variants, seeds, budgets.
- `test.py` — synthetic unit checks.

Training target is reference decomposition-membership (not candidate stability);
MatterSim labels are opened only at reserved evaluation.

## Results (`results/`)

`metrics.csv`, `scores.csv`, `controlled_comparisons.csv` from the reserved
partition. Headline: `rich − scalar`, `rich − shuffled`, `rich − readout`
confidence intervals all include zero.
