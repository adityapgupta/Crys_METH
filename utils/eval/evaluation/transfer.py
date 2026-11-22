"""Snapshot only completed scientific products for the notebook; omit full caches."""
import shutil
from .common import database, verify, seal


def export_results(c):
    out = c['output']
    verify(out,'selection_complete.json'); verify(out,'phases_complete.json')
    for name in ('megnet','chgnet'):
        receipt = verify(out,f'{name}_predictions_complete.json')
        if not receipt['complete']:
            raise ValueError(f'{name} predictions remain pending')
    dest = out/'returned'
    if dest.exists():
        raise FileExistsError('Returned snapshot already exists; verify it or choose a new export directory')
    dest.mkdir()
    products = ['candidates.csv','evaluation_candidates.csv','scores.csv','score_manifest.json',
                'selections.csv','selection_requests.csv','duplicate_pairs.csv','workload.json','input_audit.json',
                'structures_complete.json','selection_complete.json','phases_complete.json','phase_manifest.csv','phase_import_errors.json']
    for name in ('megnet','chgnet'):
        products += [f'{name}_smoke.json',f'{name}_predictions_complete.json',
                     f'{name}_candidate_predictions.csv',f'{name}_phase_predictions.csv']
    for name in products:
        shutil.copy2(out/name,dest/name)
    for name in ('structures.sqlite',):
        with database(out/name) as src, database(dest/name) as dst:
            src.backup(dst)
    # SQLite backup is logically equivalent but may have different bytes: refresh DB receipts.
    from .common import read_json, write_json, sha
    for receipt_name in ('structures_complete.json','selection_complete.json','phases_complete.json'):
        record = read_json(dest/receipt_name)
        if receipt_name == 'phases_complete.json':
            record['cluster_phase_database_sha256'] = record['files'].pop('phases.sqlite')
            record['note'] = 'Phase structures remain on SoC; formulas and same-model energies suffice for local hull calculations.'
        for rel in record['files']:
            record['files'][rel] = sha(dest/rel)
        write_json(dest/receipt_name,record)
    for name in ('megnet','chgnet'):
        record = read_json(dest/f'{name}_predictions_complete.json')
        record['selection_sha256'] = sha(dest/'selection_complete.json')
        record['phases_sha256'] = sha(dest/'phases_complete.json')
        write_json(dest/f'{name}_predictions_complete.json',record)
    seal(dest,'transfer_complete.json',[f for f in dest.iterdir() if f.is_file()],
         original_selection_receipt_sha256=sha(out/'selection_complete.json'),
         note='Consistent SQLite snapshots; caches and weights remain on SoC. Local original input/model bundle is retained.')
    return dest
