# Saved experiment summaries

These are the existing appendix results relocated from `extra/*/results/`.
No new training or scientific evaluation was performed during the move.

| Experiment | Saved results |
| --- | --- |
| [Connectivity GNN](connectivity_gnn/) | Metrics, controlled comparisons and checkpoint inventory |
| [Edge-attribute GNN](edge_attribute_gnn/) | Metrics, controlled comparisons and scores |
| [Relational-edge GNN](relational_edge_gnn/) | Metrics, controlled comparisons, scores and recoverability probe |
| [Space-group concentration](space_group_concentration/) | Capture, community summary and capture versus homogeneity |

The 13 original result files are kept in Git. New runs, intermediate graphs,
checkpoints and reports go into the corresponding folders under `outputs/gnn/`
and are Git-ignored. See [the launcher and experiment guide](../../../gnn/README.md).
