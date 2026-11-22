"""Common launcher for Crys_METH's GNN and reference-selection experiments.

Examples:
  python gnn/run.py --list
  python gnn/run.py reference train --help
  python gnn/run.py connectivity train --graph soap --model gcn
  python gnn/run.py proxy report --model esen
  python gnn/run.py graphs.build --help
"""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
GNN_DIR = ROOT / 'gnn'

EXPERIMENTS = {
    'reference': {
        'train': ('training.train_reference_gnn', []),
        'prepare': ('evaluation.reference_gnn', ['--prepare']),
        'evaluate': ('evaluation.reference_gnn', ['--evaluate']),
        'graphs': ('graphs.build_reference_graph', []),
    },
    'edge-attribute': {
        'train': ('training.train_edge_attribute_reference_gnn', []),
        'prepare': ('evaluation.edge_attribute_reference_gnn', ['--prepare']),
        'evaluate': ('evaluation.edge_attribute_reference_gnn', ['--evaluate']),
    },
    'relational-edge': {
        'train': ('training.train_relational_edge_reference_gnn', []),
        'prepare': ('evaluation.relational_edge_reference_gnn', ['--prepare']),
        'evaluate': ('evaluation.relational_edge_reference_gnn', ['--evaluate']),
        'features': ('graphs.relational_edge_features', []),
        'probe': ('evaluation.embedding_recoverability_probe', []),
    },
    'connectivity': {
        'train': ('training.train_reference_gnn', []),
        'prepare': ('evaluation.connectivity_reference_gnn', ['--prepare']),
        'evaluate': ('evaluation.connectivity_reference_gnn', ['--evaluate']),
        'graphs': ('graphs.connectivity_adjacency', []),
    },
    'space-group': {
        'communities': ('embedding_structure.spectral_community', []),
        'capture': ('embedding_structure.spacegroup_capture', []),
        'map': ('embedding_structure.community_lists_and_map', []),
    },
}


def has_option(arguments, option):
    return any(arg == option or arg.startswith(option + '=') for arg in arguments)


def resolve(experiment, arguments):
    """Return a module and forwarded arguments without importing model libraries."""
    arguments = list(arguments)
    if experiment in EXPERIMENTS:
        if not arguments or arguments[0] in ('-h', '--help'):
            raise ValueError(f'{experiment} actions: ' + ', '.join(EXPERIMENTS[experiment]))
        action = arguments.pop(0)
        if action not in EXPERIMENTS[experiment]:
            raise ValueError(f'Unknown {experiment} action {action!r}; choose ' + ', '.join(EXPERIMENTS[experiment]))
        module, defaults = EXPERIMENTS[experiment][action]
        arguments = list(defaults) + arguments
        if experiment == 'connectivity' and action == 'train':
            if not has_option(arguments, '--config'):
                arguments += ['--config', str(GNN_DIR / 'reference_baselines/configs/connectivity_reference_gnn.toml')]
            if not has_option(arguments, '--run-dir'):
                folder = 'smoke' if '--smoke' in arguments else 'runs'
                arguments += ['--run-dir', str(ROOT / 'outputs/gnn/connectivity_reference_gnn' / folder)]
        if experiment == 'space-group' and action in ('communities', 'map') and not has_option(arguments, '--data'):
            arguments += ['--data', str(ROOT / 'data/eval/inputs/references')]
    elif experiment == 'proxy':
        module = 'reference_set_comparison.pipeline'
    elif experiment == 'embeddings':
        module = 'reference_set_comparison.generate_embeddings'
    elif experiment == 'prepare':
        module = 'prepare_inputs'
    else:
        module = experiment.removeprefix('reference_baselines.')
        if not module or any(not part.isidentifier() for part in module.split('.')):
            raise ValueError('Use a listed experiment or a dotted baseline module name')
    return 'reference_baselines.' + module, arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='List named experiments and actions')
    parser.add_argument('experiment', nargs='?', help='Experiment name or dotted module name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list:
        for name, actions in EXPERIMENTS.items():
            print(f'{name:16} ' + ', '.join(actions))
        print('proxy            prepare, embed, score, select, fetch-hull, phases, download-model, smoke-model, predict, hull, report, status')
        print('embeddings       JEPA reference export (--help for options)')
        print('prepare          Stage existing inputs for baseline workflows')
        print('Other modules    e.g. graphs.build, baselines.energy, embedding_structure.energy_capture')
        return
    if not args.experiment:
        parser.error('Specify an experiment, or use --list')
    if args.experiment in EXPERIMENTS and (not args.arguments or args.arguments == ['--help']):
        print(f'{args.experiment} actions: ' + ', '.join(EXPERIMENTS[args.experiment]))
        return
    try:
        module, arguments = resolve(args.experiment, args.arguments)
    except ValueError as exc:
        parser.error(str(exc))
    for path in (ROOT, GNN_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
