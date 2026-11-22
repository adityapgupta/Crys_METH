# Connectivity reference GNN

The edge-feature experiments pointed at the graph *wiring*, not the edge
information, as the thing still worth varying. This experiment varies wiring
directly — embedding-kNN (`cosine`), SOAP geometry (`soap`), convex-hull tie-line
(`phase`) — against the physical co-occurrence graph and the no-graph readout.
**Null result:** no wiring beats the no-graph readout or the co-occurrence graph.

## Files

- `connectivity_adjacency.py` — build the `soap` kNN and `phase` tie-line adjacencies.
- `evaluate.py` — score frozen checkpoints on the reserved partition; GNN-minus-readout contrasts.
- `config.toml`, `test.py`.

### No new model or training code

This experiment reuses the base `models/reference_gnn.py` and
`training/train_reference_gnn.py` unchanged. The `soap` and `phase` graphs are
registered in the base `datasets/reference_graph.py` `GRAPH_NAMES`, and
`physical`/`cosine` come from the base `graphs/build_reference_graph.py`. Those
one-line registrations live in the shared base (under `gnn/reference_baselines`), not
here, so this folder only holds the new adjacency builder, evaluator, config and
tests.

## Results ([`outputs/gnn/summaries/connectivity_gnn/`](../../outputs/gnn/summaries/connectivity_gnn/))

`metrics.csv`, `controlled_comparisons.csv`, `checkpoints.csv` from the reserved
partition.
