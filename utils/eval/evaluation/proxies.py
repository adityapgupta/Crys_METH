"""MEGNet/CHGNet single-point inference with immutable identities and SQLite recovery."""
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from .common import database, digest, read_json, write_json, sha, seal, verify


def prepare_phases(c):
    """Import all relevant fixed MP2023 polymorphs, before model energy evaluation."""
    from pymatgen.core import Structure
    from reference_baselines.reference_set_comparison.phase_bank import lmdb_entries
    from reference_baselines.reference_set_comparison.indexing import subsystems
    out = c['output']
    verify(out,'selection_complete.json')
    if (out/'phases_complete.json').exists():
        record = verify(out,'phases_complete.json')
        if record['hull_database_sha256'] != sha(c['hull_database']):
            raise ValueError('The fixed phase source changed; choose a new output directory')
        return record
    # Read only the trusted, supplied Crys-JEPA MP2023 LMDB (contains pickle).
    source_hash = sha(c['hull_database'])
    candidates = pd.read_csv(out/'candidates.csv',keep_default_na=False)
    wanted = {s for system in candidates.chemical_system.unique() for s in subsystems(system)}
    records, errors = [], []
    with database(out/'phases.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS structures (id TEXT PRIMARY KEY, kind TEXT, chemical_system TEXT, formula TEXT, natoms INTEGER, structure_hash TEXT, structure_json TEXT)')
        for key, system, formula, raw in lmdb_entries(c['hull_database'],wanted):
            identifier = 'phase:'+key
            try:
                structure = raw if isinstance(raw,Structure) else Structure.from_dict(raw)
                if not structure.is_ordered or not len(structure) or structure.composition.chemical_system != system:
                    raise ValueError('Invalid phase structure/system')
                value, count = structure.as_dict(), len(structure)
                formula = structure.composition.reduced_formula
            except Exception as exc:
                errors.append({'id':identifier,'error':f'{type(exc).__name__}: {exc}'})
                value, count = {'invalid':identifier}, 0
            structure_hash = digest(value)
            db.execute('INSERT OR REPLACE INTO structures VALUES (?,?,?,?,?,?,?)',
                       (identifier,'phase',system,formula,count,structure_hash,json.dumps(value)))
            records.append(dict(id=identifier,chemical_system=system,formula=formula,structure_hash=structure_hash))
            if len(records)%10000 == 0:
                db.commit()
                print(f'Imported {len(records)} fixed phases',flush=True)
    if not records:
        raise ValueError('Empty MP2023 phase bank')
    pd.DataFrame(records).to_csv(out/'phase_manifest.csv',index=False)
    write_json(out/'phase_import_errors.json',errors)
    seal(out,'phases_complete.json',[out/'phases.sqlite',out/'phase_manifest.csv',out/'phase_import_errors.json'],
         hull_database_sha256=source_hash,phase_count=len(records),selection_independent=True,
         generated_candidates_in_hull=False,dft_polymorph_pruning=False)
    return read_json(out/'phases_complete.json')


class CHGNetAdapter:
    energy_kind = 'total_energy_per_atom'

    def __init__(self,c):
        from chgnet.model.model import CHGNet
        self.model = CHGNet.load(model_name=c['chgnet_model'],use_device=c['device'])
        self.model.eval()

    def predict(self,structures):
        result = self.model.predict_structure(structures,task='e',batch_size=len(structures))
        if isinstance(result,dict):
            result = [result]
        return np.array([float(row['e']) for row in result])


def adapter(c,name):
    if name == 'chgnet':
        return CHGNetAdapter(c)
    if name == 'megnet':
        from reference_baselines.reference_set_comparison.proxies import MEGNetAdapter
        return MEGNetAdapter({'directory':str(c['megnet_directory'])},c['device'])
    raise ValueError('Choose megnet or chgnet')


def identity(c,name,model):
    import hashlib
    import inspect
    h = hashlib.sha256()
    # Hash actual loaded tensors, including buffers, not a path or mutable model name.
    for key,value in sorted(model.model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    packages = {}
    for package in ('torch','numpy','pymatgen','pymatgen-core','chgnet','matgl','ase','torch-geometric'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'name':name,'weights_sha256':h.hexdigest(),'packages':packages,
            'adapter_source_sha256':sha(Path(__file__)),
            'implementation_sha256':sha(Path(inspect.getfile(type(model.model)))),
            'megnet_adapter_sha256':sha(Path(inspect.getfile(type(model)))) if name=='megnet' else None,
            'energy_kind':model.energy_kind,'geometry':'single point on stored geometry',
            'model_name':c['chgnet_model'] if name=='chgnet' else 'MEGNet-Eform-MP-2018.6.1',
            'extra_energy_corrections':'none'}


def robust_predict(model,structures):
    try:
        values = np.asarray(model.predict(structures),dtype=float)
        if values.shape != (len(structures),) or not np.isfinite(values).all():
            raise ValueError('Nonfinite prediction or wrong energy shape')
        return [('ok',float(v),'') for v in values]
    except Exception:
        values = []
        for s in structures:
            try:
                v = np.asarray(model.predict([s]),dtype=float)
                if v.shape != (1,) or not np.isfinite(v[0]):
                    raise ValueError('Nonfinite singleton energy')
                values.append(('ok',float(v[0]),''))
            except Exception as exc:
                values.append(('prediction_error',None,f'{type(exc).__name__}: {exc}'))
        return values


def smoke(c,name):
    from pymatgen.core import Structure
    out = c['output']
    verify(out,'selection_complete.json')
    model = adapter(c,name)
    model_identity = identity(c,name,model)
    with database(out/'structures.sqlite') as db:
        raw = db.execute('SELECT structure_json FROM structures ORDER BY id LIMIT 4').fetchall()
    structures = [Structure.from_dict(json.loads(row[0])) for row in raw]
    if not structures:
        raise ValueError('No smoke structures')
    batch = np.asarray(model.predict(structures))
    alone = np.array([model.predict([s])[0] for s in structures])
    if not np.isfinite(batch).all():
        raise ValueError('Nonfinite smoke energies')
    np.testing.assert_allclose(batch,alone,rtol=1e-4,atol=1e-5)
    replicated = float(model.predict([structures[0]*(2,1,1)])[0])
    np.testing.assert_allclose(replicated,alone[0],rtol=1e-4,atol=1e-3)
    path = out/f'{name}_smoke.json'
    if path.exists() and read_json(path)['fingerprint'] != digest(model_identity):
        raise ValueError('Smoke identity changed; choose a new output/cache run before changing environments')
    write_json(path,{'passed':True,'identity':model_identity,'fingerprint':digest(model_identity),
                     'predictions_eV_per_atom':alone.tolist(),'replicated_cell_energy':replicated})
    return model_identity


def predict(c,name,max_items=None,retry_errors=False):
    from pymatgen.core import Structure
    out = c['output']
    verify(out,'selection_complete.json'); verify(out,'phases_complete.json')
    model = adapter(c,name)
    model_identity = identity(c,name,model)
    token = digest(model_identity)
    if read_json(out/f'{name}_smoke.json')['fingerprint'] != token:
        raise ValueError('Run the real-weight smoke test in this exact environment first')
    c['cache'].mkdir(parents=True,exist_ok=True)
    # One writer per proxy cache, even when different runs share this cache directory.
    lock = c['cache']/f'{name}.lock'
    try:
        handle = lock.open('x')
    except FileExistsError:
        raise RuntimeError(f'Cache locked: {lock}. Verify old job stopped before manually removing a stale lock.')
    try:
        handle.write(str(__import__('os').getpid())); handle.close()
        _predict(c,name,model,model_identity,token,max_items,retry_errors,Structure)
    finally:
        lock.unlink()


def _predict(c,name,model,model_identity,token,max_items,retry_errors,Structure):
    out = c['output']
    candidates = pd.read_csv(out/'evaluation_candidates.csv',keep_default_na=False)
    phases = pd.read_csv(out/'phase_manifest.csv',keep_default_na=False)
    manifests = {'candidate':candidates,'phase':phases}
    with database(c['cache']/f'{name}.sqlite') as cache:
        cache.execute('CREATE TABLE IF NOT EXISTS predictions (fingerprint TEXT, structure_hash TEXT, status TEXT, energy_per_atom REAL, error TEXT, PRIMARY KEY(fingerprint,structure_hash))')
        done = {row[0] for row in cache.execute('SELECT structure_hash FROM predictions WHERE fingerprint=?'+(" AND status='ok'" if retry_errors else ''),(token,))}
        needed = set(candidates.structure_hash)|set(phases.structure_hash)
        pending = needed-done
        total = 0
        batch, hashes = [], []
        def flush():
            nonlocal total
            if not batch:
                return
            for key,(status,value,error) in zip(hashes,robust_predict(model,batch)):
                cache.execute('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)',(token,key,status,value,error))
            cache.commit(); total += len(batch)
            print(f'{name}: committed {total}; pending {len(pending)}',flush=True)
            batch.clear(); hashes.clear()
        for kind,filename in [('candidate','structures.sqlite'),('phase','phases.sqlite')]:
            with database(out/filename) as source:
                for row in source.execute('SELECT * FROM structures ORDER BY id'):
                    key = row['structure_hash']
                    if key not in pending:
                        continue
                    if max_items is not None and total+len(batch) >= max_items:
                        break
                    pending.remove(key)
                    try:
                        raw = json.loads(row['structure_json'])
                        if digest(raw) != key:
                            raise ValueError('Structure checksum mismatch')
                        structure = Structure.from_dict(raw)
                    except Exception as exc:
                        cache.execute('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)',
                                      (token,key,'structure_error',None,str(exc)))
                        cache.commit(); total += 1
                        continue
                    batch.append(structure); hashes.append(key)
                    if len(batch) >= c['prediction_batch_size']:
                        flush()
                flush()
        exports = []
        for kind,manifest in manifests.items():
            values = []
            for row in manifest.itertuples():
                p = cache.execute('SELECT status,energy_per_atom,error FROM predictions WHERE fingerprint=? AND structure_hash=?',(token,row.structure_hash)).fetchone()
                values.append({'id':row.id,'structure_hash':row.structure_hash,
                               'status':p[0] if p else 'pending','energy_per_atom':p[1] if p else None,
                               'error':p[2] if p else ''})
            path = out/f'{name}_{kind}_predictions.csv'
            pd.DataFrame(values).to_csv(path,index=False); exports.append(path)
        remaining = [key for key in needed if not cache.execute('SELECT 1 FROM predictions WHERE fingerprint=? AND structure_hash=?',(token,key)).fetchone()]
    seal(out,f'{name}_predictions_complete.json',exports,complete=not remaining,pending=len(remaining),
         fingerprint=token,identity=model_identity,selection_sha256=sha(out/'selection_complete.json'),
         phases_sha256=sha(out/'phases_complete.json'))
    if remaining:
        raise RuntimeError(f'{len(remaining)} pending predictions; rerun predict to resume. No hull stage should follow this partial run.')


def calculate_hulls(c,name):
    from .hulls import label_candidates
    out = c['output']
    verify(out,'selection_complete.json'); verify(out,'phases_complete.json')
    receipt = verify(out,f'{name}_predictions_complete.json')
    if not receipt['complete'] or receipt['selection_sha256'] != sha(out/'selection_complete.json') or receipt['phases_sha256'] != sha(out/'phases_complete.json'):
        raise ValueError('Incomplete or stale prediction set')
    hull_receipt = out/f'{name}_hulls_complete.json'
    if hull_receipt.is_file():
        cached = verify(out,hull_receipt.name)
        if cached['predictions_sha256'] != sha(out/f'{name}_predictions_complete.json'):
            raise ValueError('Cached hull labels use a different prediction set')
        return pd.read_csv(out/f'{name}_labels.csv',keep_default_na=False)
    candidates = pd.read_csv(out/'evaluation_candidates.csv',keep_default_na=False)
    phases = pd.read_csv(out/'phase_manifest.csv',keep_default_na=False)
    predictions = []
    for kind,manifest in [('candidate',candidates),('phase',phases)]:
        table = pd.read_csv(out/f'{name}_{kind}_predictions.csv',keep_default_na=False)
        matched = manifest[['id','structure_hash']].merge(table[['id','structure_hash']],on='id',validate='one_to_one',suffixes=('_a','_b'))
        if len(matched)!=len(manifest) or not matched.structure_hash_a.eq(matched.structure_hash_b).all():
            raise ValueError('Prediction identities do not match required structures')
        predictions.append(table)
    labels,coverage = label_candidates(candidates,phases,*predictions,receipt['identity']['energy_kind'])
    labels.to_csv(out/f'{name}_labels.csv',index=False)
    coverage.to_csv(out/f'{name}_hull_coverage.csv',index=False)
    seal(out,f'{name}_hulls_complete.json',[out/f'{name}_labels.csv',out/f'{name}_hull_coverage.csv'],
         predictions_sha256=sha(out/f'{name}_predictions_complete.json'),
         threshold_rule='<0.1 eV/atom',candidate_entries_in_hull=False)
    return labels
