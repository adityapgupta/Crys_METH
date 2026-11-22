"""Rebuild the original reference graphs from saved vectors and FIT decompositions.

The array-level functions have no repository-path assumptions. Both the notebook
and CLI call build_graphs; inference calls load_graph. Historical physical-edge
counting, material-ID mapping, and exact neighbour retrieval are reused directly.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from evaluation.common import input_files, read_json, seal, sha, verify


def cosine_from_neighbours(indices, distances, reference_count):
    """Original symmetric-max cosine graph, clipped similarities and unit loops."""
    from scipy import sparse
    indices, distances = np.asarray(indices), np.asarray(distances, dtype=np.float64)
    if indices.shape != distances.shape or indices.ndim != 2 or len(indices) != reference_count:
        raise ValueError('Neighbour table shape does not match reference bank')
    if indices.dtype.kind not in 'iu' or np.any((indices < 0) | (indices >= reference_count)):
        raise ValueError('Neighbour IDs outside reference bank')
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite neighbour distances')
    source = np.repeat(np.arange(reference_count), indices.shape[1])
    similarity = np.clip(1.0 - distances.reshape(-1), 1e-6, 1.0)
    graph = sparse.csr_matrix((similarity, (source, indices.reshape(-1))),
                              shape=(reference_count, reference_count))
    graph = graph.maximum(graph.T)
    graph = graph + sparse.eye(reference_count, format='csr', dtype=np.float64)
    graph.sum_duplicates()
    return graph.tocsr()


def build_cosine_graph(reference_embeddings, k=10, batch_size=512):
    """Use the historical float64 unit-vector/squared-distance retrieval exactly.

    Distances are squared Euclidean distances between normalized vectors divided
    by two, as in graphs/cosine.py. Reference queries exclude their own row.
    """
    from reference_baselines.core.angular import unit_vectors
    from reference_baselines.core.retrieval import exact_knn
    if not 0 < k < len(reference_embeddings):
        raise ValueError('Graph k must be smaller than the reference bank')
    vectors = unit_vectors(reference_embeddings)
    # The original retrieval exported 50 neighbours before the graph took ten.
    count = min(max(50, k), len(vectors)-1)
    ids, squared = exact_knn(vectors, vectors, count, batch_size=batch_size,
                             exclude_self=True, label='reference graph rebuild')
    ids, distance = ids[:, :k], np.clip(squared[:, :k]/2, 0, 2)
    return cosine_from_neighbours(ids, distance, len(vectors)), ids, distance


def build_physical_graph(fit_rows, reference_metadata, decompositions, excluded_rows=()):
    """Count co-occurrence only in original fit candidates, never held-out rows."""
    from reference_baselines.datasets.decomposition import build_targets, target_stats
    from reference_baselines.graphs.cooccurrence import cooccurrence_graph
    fit = np.asarray(fit_rows)
    excluded = np.asarray(excluded_rows)
    if fit.ndim != 1 or fit.dtype.kind not in 'iu' or len(np.unique(fit)) != len(fit):
        raise ValueError('Invalid fit row indices')
    if np.intersect1d(fit, excluded).size:
        raise ValueError('Held-out rows cannot enter physical graph construction')
    np.testing.assert_array_equal(decompositions.candidate_row, np.arange(len(decompositions)))
    if np.any((fit < 0) | (fit >= len(decompositions))):
        raise ValueError('Fit rows are not aligned with decomposition table')
    ids = reference_metadata.material_id.astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate reference material IDs')
    positives, unmapped, counts = build_targets(fit, table=decompositions,
                                               index={name:i for i,name in enumerate(ids)})
    graph, pair_counts = cooccurrence_graph(positives, len(ids))
    return graph, {'cooccurrence_pair_count': len(pair_counts),
                   'target_stats': target_stats(fit, positives, unmapped, counts)}


def _source_hashes():
    from reference_baselines.datasets import decomposition
    from reference_baselines.graphs import cooccurrence
    from reference_baselines.core import angular, retrieval
    # Keep the original graph receipt valid after relocating this helper. These
    # package-qualified imports are the only differences from the frozen source;
    # hashing the normalized text preserves the algorithm fingerprint without
    # consulting the legacy evaluation/ directory.
    source = Path(__file__).read_text(encoding='utf-8').replace('import hashlib\n', '')
    source = source.replace(
        'from evaluation.common import',
        'from evaluation.common import',
    ).replace(
        "'evaluation.inference.graphs'",
        "'evaluation.inference.graphs'",
    ).replace(
        'from evaluation.inputs import',
        'from evaluation.inputs import',
    )
    start = source.index('def _source_hashes():')
    end = source.index('\n\ndef graph_inputs', start)
    source = source[:start] + '''def _source_hashes():
    from reference_baselines.datasets import decomposition
    from reference_baselines.graphs import cooccurrence
    from reference_baselines.core import angular, retrieval
    return {module.__name__: sha(Path(module.__file__))
            for module in (decomposition, cooccurrence, angular, retrieval)} | {
                'evaluation.inference.graphs': sha(Path(__file__))}
''' + source[end:]
    graph_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()
    return {module.__name__: sha(Path(module.__file__))
            for module in (decomposition, cooccurrence, angular, retrieval)} | {
                'evaluation.inference.graphs': graph_hash}


def graph_inputs(c):
    files = input_files(c)
    names = ('embeddings.npy', 'metadata.csv', 'candidate_decompositions.csv',
             'train_rows.npy', 'validation_rows.npy', 'reserved_rows.npy',
             'split_protocol.json', 'split_manifest.csv')
    return {name: sha(files[name]) for name in names}


def verify_graphs(c):
    record = verify(c['graphs'], 'graph_build.json')
    if record['input_sha256'] != graph_inputs(c):
        raise ValueError('Graph source data changed; keep the frozen inputs or choose new graph/output folders')
    if record['source_sha256'] != _source_hashes():
        raise ValueError('Graph implementation changed; review and rebuild into a new graph folder')
    return record


def build_graphs(c):
    """Build once, record all inputs, then verify and reuse on later invocations."""
    from scipy import sparse
    from reference_baselines.graphs.cooccurrence import graph_stats
    import torch
    directory = c['graphs']
    if (directory/'graph_build.json').is_file():
        return verify_graphs(c)
    if (c['output']/'selection_complete.json').is_file():
        raise ValueError('Selections already frozen; do not reconstruct their graphs in place')
    if any((directory/f'{name}.npz').exists() for name in ('physical','cosine')):
        raise FileExistsError('Graph files without a build receipt exist; preserve them and use a new graph folder')
    # Reuse the full split audit; missing graphs are reported but do not prevent construction.
    from evaluation.inputs import audit
    audit(c)
    files = input_files(c)
    references = np.load(files['embeddings.npy'], mmap_mode='r', allow_pickle=False)
    metadata = pd.read_csv(files['metadata.csv'], keep_default_na=False)
    decompositions = pd.read_csv(files['candidate_decompositions.csv'],
                                usecols=['candidate_row','decomposition'], keep_default_na=False)
    candidates = pd.read_csv(files['candidate_metadata.csv'], keep_default_na=False)
    if len(decompositions) != len(candidates):
        raise ValueError('Candidate decomposition table is not aligned with the original candidate universe')
    fit = np.load(files['train_rows.npy'], allow_pickle=False)
    validation = np.load(files['validation_rows.npy'], allow_pickle=False)
    reserved = np.load(files['reserved_rows.npy'], allow_pickle=False)
    physical, stats = build_physical_graph(fit, metadata, decompositions, np.r_[validation,reserved])
    torch.set_num_threads(c.get('graph_threads', 8))
    cosine, indices, distances = build_cosine_graph(references, k=10,
                            batch_size=c.get('graph_retrieval_batch_size', 512))
    directory.mkdir(parents=True, exist_ok=True)
    for name,graph in (('physical',physical),('cosine',cosine)):
        sparse.save_npz(directory/f'{name}.npz', graph)
    np.save(directory/'reference_cosine_indices.npy', indices, allow_pickle=False)
    np.save(directory/'reference_cosine_distances.npy', distances, allow_pickle=False)
    products = [directory/name for name in ('physical.npz','cosine.npz',
                 'reference_cosine_indices.npy','reference_cosine_distances.npy')]
    seal(directory, 'graph_build.json', products,
         input_sha256=graph_inputs(c), source_sha256=_source_hashes(),
         mode='reconstructed_from_supplied_original_inputs',
         historical_graph_identity='Original graph checksum unavailable; historical input/code-based reconstruction, not certified byte identity.',
         reference_count=len(metadata), fit_rows=len(fit), validation_rows_excluded=len(validation),
         reserved_rows_excluded=len(reserved), candidate_stability_labels_read=False,
         physical=graph_stats(physical,'Fit-only decomposition co-occurrence'),
         cosine=graph_stats(cosine,'Historical symmetric cosine k=10'), **stats)
    return read_json(directory/'graph_build.json')


def load_graph(c, name):
    """Load the same persisted graph for any downstream inference call."""
    from scipy import sparse
    if name not in ('physical','cosine'):
        raise ValueError('Choose physical or cosine')
    if (c['graphs']/'graph_build.json').exists():
        verify_graphs(c)
    graph = sparse.load_npz(c['graphs']/f'{name}.npz').tocsr()
    count = len(pd.read_csv(input_files(c)['metadata.csv'], usecols=['material_id']))
    if graph.shape != (count,count) or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError('Invalid reference graph')
    return graph
