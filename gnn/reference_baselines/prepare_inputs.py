"""Expose staged Crys_METH inputs at the paths used by the baseline workflows.

Creates relative file symlinks (or verified copies with --copy); never replaces
existing data. Run after downloading the evaluation input and graph bundles.
"""
import argparse
import os
from pathlib import Path
import shutil

from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, DATA_DIR, EMBEDDINGS_DIR, RESULTS_DIR, DOCS_DIR


def input_links():
    for name in ('embeddings.npy', 'metadata.csv'):
        yield DATA_DIR / 'references' / name, EMBEDDINGS_DIR / name
    for name in ('candidate_embeddings.npy', 'candidate_metadata.csv'):
        yield DATA_DIR / 'candidates' / name, EMBEDDINGS_DIR / name
    yield DATA_DIR / 'candidates/candidate_decompositions.csv', RESULTS_DIR / 'candidate_decompositions.csv'
    for name in ('train_rows.npy', 'validation_rows.npy', 'reserved_rows.npy',
                 'split_protocol.json', 'split_manifest.csv'):
        yield DATA_DIR / 'splits' / name, RESULTS_DIR / 'gnn_comparison' / name
    for name in ('physical.npz', 'cosine.npz'):
        yield ROOT / 'data/eval/graphs' / name, RESULTS_DIR / 'reference_gnn/graphs' / name


def prepare(copy=False):
    missing = []
    for source, destination in input_links():
        if not source.is_file():
            missing.append(source)
            continue
        independent = copy or destination.is_relative_to(RESULTS_DIR)
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or sha256(source) != sha256(destination):
                raise FileExistsError(f'Conflicting baseline input: {destination}')
            if independent and destination.is_symlink():
                destination.unlink()
                shutil.copy2(source, destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if independent:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(os.path.relpath(source, destination.parent))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy', action='store_true', help='Copy instead of linking (for systems without symlinks)')
    args = parser.parse_args()
    missing = prepare(args.copy)
    if missing:
        parser.exit(1, 'Prepared available inputs. Missing staged files:\n' + '\n'.join(map(str, missing)) + '\n')
    print(f'Baseline inputs ready: {EMBEDDINGS_DIR}')


if __name__ == '__main__':
    main()
