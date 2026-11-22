"""Portable paths, immutable run manifests, and verified transfer receipts."""
from pathlib import Path
import hashlib
import json
import sqlite3
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[3]
PATHS = ('inputs', 'graphs', 'checkpoints', 'training_protocols', 'release_root',
         'hull_database', 'megnet_directory', 'output', 'cache')

INPUT_LAYOUT = {
    'candidate_embeddings.npy': 'candidates/candidate_embeddings.npy',
    'candidate_metadata.csv': 'candidates/candidate_metadata.csv',
    'candidate_decompositions.csv': 'candidates/candidate_decompositions.csv',
    'embeddings.npy': 'references/embeddings.npy',
    'metadata.csv': 'references/metadata.csv',
    'split_manifest.csv': 'splits/split_manifest.csv',
    'split_protocol.json': 'splits/split_protocol.json',
    'train_rows.npy': 'splits/train_rows.npy',
    'validation_rows.npy': 'splits/validation_rows.npy',
    'reserved_rows.npy': 'splits/reserved_rows.npy',
}


def input_files(c):
    """Named data roles shared by graph construction, inference, and the notebook."""
    return {name: c['inputs']/relative for name, relative in INPUT_LAYOUT.items()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path=None):
    c = read_json(path or ROOT / 'data/eval/config.json')
    for key in PATHS:
        p = Path(c[key]).expanduser()
        c[key] = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    if c['batch_size'] < 1 or not c['fractions'] or any(not 0 < x <= 1 for x in c['fractions']):
        raise ValueError('Invalid selection settings')
    if c['neighbours'] != sorted(set(c['neighbours'])) or min(c['neighbours']) < 1:
        raise ValueError('Neighbour counts must be sorted positive unique integers')
    return c


def seal(directory, name, files, **extra):
    directory = Path(directory)
    write_json(directory / name, {'files': {str(Path(f).relative_to(directory)): sha(f)
               for f in files}, **extra})


def verify(directory, name):
    directory = Path(directory).resolve()
    record = read_json(directory / name)
    for rel, expected in record['files'].items():
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Changed or missing artifact: {rel}')
    return record


@contextmanager
def database(path):
    # DELETE journaling is compatible with shared home filesystems; do not enable WAL.
    db = sqlite3.connect(path, timeout=120)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()
