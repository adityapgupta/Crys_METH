"""Validate the supplied frozen split and recover structures by original identity."""
import io
from itertools import combinations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .common import sha, read_json, write_json, seal, database, digest, input_files


def checkpoints(c):
    specs = [(kind, graph, seed) for graph in ('physical', 'cosine')
             for kind in ('gcn', 'gat', 'sage') for seed in c['gnn_seeds']]
    specs += [('no_graph', 'physical', seed) for seed in c['gnn_seeds']]
    return [(kind, graph, seed, c['checkpoints'] / f'{kind}_{graph}_seed{seed}')
            for kind, graph, seed in specs]


def audit(c):
    files = input_files(c)
    protocol = read_json(files['split_protocol.json'])
    q = np.load(files['candidate_embeddings.npy'], mmap_mode='r', allow_pickle=False)
    r = np.load(files['embeddings.npy'], mmap_mode='r', allow_pickle=False)
    cm = pd.read_csv(files['candidate_metadata.csv'], keep_default_na=False)
    rm = pd.read_csv(files['metadata.csv'], keep_default_na=False)
    manifest = pd.read_csv(files['split_manifest.csv'], keep_default_na=False)
    splits = {s: np.load(files[f'{s}_rows.npy'], allow_pickle=False)
              for s in ('train', 'validation', 'reserved')}
    if q.shape != (len(cm), 512) or r.shape != (len(rm), 512) or len(rm) != 27136:
        raise ValueError('Embedding/metadata dimensions do not match the trained 27,136-node bank')
    for a in (q, r):
        if not all(np.isfinite(a[i:i+8192]).all() for i in range(0, len(a), 8192)):
            raise ValueError('Nonfinite embeddings')
    if not rm.material_id.is_unique or cm.duplicated(['eval_folder', 'idx']).any():
        raise ValueError('Duplicate reference or candidate identities')
    np.testing.assert_array_equal(manifest.candidate_row, np.arange(len(cm)))
    for col in ('eval_folder', 'idx', 'chemical_system', 'vun'):
        np.testing.assert_array_equal(cm[col].astype(str), manifest[col].astype(str))
    for s, rows in splits.items():
        if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows):
            raise ValueError(f'Invalid split indices: {s}')
        if len(rows) != protocol['counts'][s] or np.any((rows < 0) | (rows >= len(cm))):
            raise ValueError(f'Split count/bounds mismatch: {s}')
        if not cm.iloc[rows].vun.eq(True).all():
            raise ValueError('Split contains non-VUN candidates')
    for a, b in combinations(splits, 2):
        if np.intersect1d(splits[a], splits[b]).size:
            raise ValueError(f'{a}/{b} row leakage')
        if set(cm.iloc[splits[a]].chemical_system) & set(cm.iloc[splits[b]].chemical_system):
            raise ValueError(f'{a}/{b} chemistry overlap; investigate before evaluation')
    sources = protocol['source_sha256']
    if sha(files['split_manifest.csv']) != sources['split_manifest.csv']:
        raise ValueError('Original split manifest hash mismatch')
    if sha(files['validation_rows.npy']) != sources['training_validation_rows.npy']:
        raise ValueError('Original validation split hash mismatch')
    old = io.BytesIO()
    np.save(old, np.sort(np.r_[splits['train'], splits['reserved']]), allow_pickle=False)
    if hashlib.sha256(old.getvalue()).hexdigest() != sources['training_train_rows.npy']:
        raise ValueError('Fit plus reserved does not reproduce the original training array')
    blockers, warnings = [], []
    for graph in ('physical', 'cosine'):
        if not (c['graphs']/f'{graph}.npz').is_file():
            blockers.append(f"Graph not built: {graph}. Run the notebook graph-construction cell or CLI graphs stage.")
        if not (c['training_protocols']/f'train_protocol_{graph}.json').is_file():
            warnings.append(f'Missing historical training protocol: {graph}; graph reconstruction is recorded separately.')
    for _, _, _, folder in checkpoints(c):
        if not (folder/'best.pt').is_file():
            blockers.append(f'Missing checkpoint: {folder.name}/best.pt')
        if not (folder/'complete.json').is_file():
            warnings.append(f'Historical completion record unavailable: {folder.name}; use metadata in best.pt.')
    if (c['graphs']/'graph_build.json').is_file():
        from .inference.graphs import verify_graphs
        verify_graphs(c)
    result = {'candidate_count': len(cm), 'reference_count': len(rm),
              'split_counts': {s: len(v) for s,v in splits.items()},
              'row_and_chemical_system_overlap': 0, 'source_protocol_hashes_match': True,
              'input_sha256': {name: sha(path) for name,path in files.items() if path.is_file()},
              'ready_for_gnn': not blockers, 'blockers': blockers, 'provenance_notes': warnings,
              'structural_leakage_note': 'Exact-composition duplicates cannot cross disjoint chemical systems.',
              'interpretation': 'Previously evaluated engineering holdout; frozen-model proxy comparison.'}
    c['output'].mkdir(parents=True, exist_ok=True)
    audit_path = c['output']/'input_audit.json'
    # Preserve the exact bytes of an already-sealed audit when its content is
    # unchanged (for example, a Windows CRLF receipt opened on Linux).
    if audit_path.is_file():
        try:
            if read_json(audit_path) == result:
                return result
        except (OSError, ValueError):
            pass
    write_json(audit_path, result)
    return result


def batch_rows(rows, seed=2026, size=600):
    rows = np.random.default_rng(seed).permutation(np.sort(rows))
    return pd.DataFrame({'candidate_row': rows, 'batch': np.arange(len(rows))//size,
                         'batch_position': np.arange(len(rows)) % size})


def load_structures(c):
    """Read the verified SoC bundle; never reconstruct structures on the laptop."""
    from .common import verify
    output = c['output']
    missing = [name for name in ('candidates.csv','structures.sqlite','structures_complete.json')
               if not (output/name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Incomplete downloaded structure bundle in {output}. Missing: {", ".join(missing)}. '
            'Wait for the SoC structures job to finish, then download these missing files. '
            'Existing CSV/SQLite files alone do not establish completed preparation.')
    previous = verify(output, 'structures_complete.json')
    report = audit(c)
    if previous['inputs'] != report['input_sha256'] or previous['seed'] != c['seed'] or previous['batch_size'] != c['batch_size']:
        raise ValueError('Downloaded structure bundle does not match the current inputs or batching settings')
    return pd.read_csv(output/'candidates.csv', keep_default_na=False)


def validate_structure_snapshot(candidates, metadata, reserved_rows, database_path, seed, size):
    """Validate every saved identity, batch, structure, and hash without release JSONs."""
    import json
    import sqlite3
    from contextlib import closing
    from pymatgen.core import Structure
    expected = batch_rows(reserved_rows, seed, size).merge(
        metadata.rename_axis('candidate_row').reset_index(), on='candidate_row', validate='one_to_one')
    if not candidates.candidate_row.is_unique or not candidates.id.is_unique:
        raise ValueError('Duplicate candidate row or structure ID')
    left = candidates.sort_values('candidate_row').reset_index(drop=True)
    right = expected.sort_values('candidate_row').reset_index(drop=True)
    for column in right.columns:
        np.testing.assert_array_equal(left[column].astype(str), right[column].astype(str),
                                      err_msg=f'Downloaded candidate {column} does not match original metadata/batches')
    lookup = {row.id:row for row in candidates.itertuples(index=False)}
    for row in candidates.itertuples(index=False):
        if row.id != f'candidate:{row.eval_folder}:{row.idx}' or row.turn != row.batch:
            raise ValueError(f'Inconsistent original identity or batch: {row.id}')
    path = Path(database_path).resolve()
    count = 0
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if [row[0] for row in db.execute('PRAGMA integrity_check')] != ['ok']:
            raise ValueError('SQLite integrity check failed')
        if db.execute('SELECT COUNT(*) FROM structures').fetchone()[0] != len(expected):
            raise ValueError('Database structure count does not match reserved population')
        for record in db.execute('SELECT * FROM structures ORDER BY id'):
            if record['id'] not in lookup or record['kind'] != 'candidate':
                raise ValueError(f'Unexpected database identity: {record["id"]}')
            row = lookup[record['id']]
            raw = json.loads(record['structure_json'])
            if digest(raw) != record['structure_hash'] or record['structure_hash'] != row.structure_hash:
                raise ValueError(f'Structure checksum mismatch: {row.id}')
            structure = Structure.from_dict(raw)
            if not structure.is_ordered or not len(structure):
                raise ValueError(f'Empty/disordered structure: {row.id}')
            if not np.isfinite(structure.frac_coords).all() or not np.isfinite(structure.lattice.matrix).all():
                raise ValueError(f'Nonfinite structure coordinates: {row.id}')
            if (structure.composition.chemical_system != row.chemical_system or
                    record['chemical_system'] != row.chemical_system or
                    structure.composition.reduced_formula != row.formula or record['formula'] != row.formula or
                    len(structure) != row.num_atoms or record['natoms'] != row.num_atoms):
                raise ValueError(f'Structure composition/atom count mismatch: {row.id}')
            count += 1
            if count % 5000 == 0:
                print(f'Verified {count:,}/{len(expected):,} downloaded structures', flush=True)
    return {'verified_structures':count, 'database_integrity':'ok',
            'original_candidate_ids_and_splits_verified':True, 'batch_assignments_verified':True,
            'all_structure_hashes_and_compositions_verified':True}


def recover_structure_receipt(c):
    """Create an explicitly local verification receipt, never a claimed cluster job record."""
    from .common import verify
    output = c['output']
    if (output/'structures_complete.json').exists():
        return verify(output,'structures_complete.json')
    products = [output/'candidates.csv', output/'structures.sqlite']
    before = {path.name:sha(path) for path in products}
    files = input_files(c)
    report = audit(c)
    result = validate_structure_snapshot(
        pd.read_csv(products[0],keep_default_na=False),
        pd.read_csv(files['candidate_metadata.csv'],keep_default_na=False),
        np.load(files['reserved_rows.npy'],allow_pickle=False),products[1],c['seed'],c['batch_size'])
    if before != {path.name:sha(path) for path in products}:
        raise ValueError('Structure files changed during verification; wait for writers/transfers to stop')
    seal(output,'structures_complete.json',products,inputs=report['input_sha256'],
         seed=c['seed'],batch_size=c['batch_size'],release_sources={},
         receipt_origin='local_full_content_validation',validation=result,
         provenance_note='Original SoC completion receipt unavailable. All local structures, IDs, batches and saved hashes verified; original release-file hashes and cluster job completion are not certified.')
    return read_json(output/'structures_complete.json')


def prepare_structures(c):
    """Can run on SoC before the notebook if release structures are remote-only."""
    from pymatgen.core import Structure
    report = audit(c)
    output = c['output']
    if (output/'structures_complete.json').exists():
        return load_structures(c)
    cm = pd.read_csv(input_files(c)['candidate_metadata.csv'], keep_default_na=False)
    rows = np.load(input_files(c)['reserved_rows.npy'], allow_pickle=False)
    table = batch_rows(rows, c['seed'], c['batch_size']).merge(
        cm.rename_axis('candidate_row').reset_index(), on='candidate_row', validate='one_to_one')
    records, source_hashes = [], {}
    with database(output/'structures.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS structures (id TEXT PRIMARY KEY, kind TEXT, chemical_system TEXT, formula TEXT, natoms INTEGER, structure_hash TEXT, structure_json TEXT)')
        for folder, group in table.groupby('eval_folder', sort=True):
            path = c['release_root']/str(folder)/'relaxed_mattersim.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing candidate structures: {path}; run structures on SoC and download the bundle')
            source_hashes[str(folder)] = sha(path)
            raw = pd.read_json(path)
            raw.index = raw.index.astype(str)
            if not raw.index.is_unique:
                raise ValueError('Duplicate release row IDs')
            for row in group.itertuples(index=False):
                data = raw.loc[str(row.idx)]
                structure = Structure.from_dict(data['structure'])
                if not structure.is_ordered or not len(structure) or structure.composition.chemical_system != row.chemical_system:
                    raise ValueError(f'Structure mismatch at {folder}:{row.idx}')
                if bool(data.get('exception', False)):
                    raise ValueError(f'Reserved structure has relaxation exception: {folder}:{row.idx}')
                value = structure.as_dict()
                identifier = f'candidate:{folder}:{row.idx}'
                record = row._asdict()
                record.update(id=identifier, formula=structure.composition.reduced_formula,
                              structure_hash=digest(value), num_atoms=len(structure), turn=row.batch)
                records.append(record)
                db.execute('INSERT OR REPLACE INTO structures VALUES (?,?,?,?,?,?,?)',
                           (identifier, 'candidate', row.chemical_system, record['formula'], len(structure),
                            record['structure_hash'], __import__('json').dumps(value)))
            db.commit()
    frame = pd.DataFrame(records).sort_values(['batch','batch_position'])
    frame.to_csv(output/'candidates.csv', index=False)
    seal(output, 'structures_complete.json', [output/'candidates.csv', output/'structures.sqlite'],
         inputs=report['input_sha256'], release_sources=source_hashes, seed=c['seed'], batch_size=c['batch_size'])
    return frame
