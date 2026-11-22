"""Batch-local uniqueness and equal selection budgets, frozen before proxy inference."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from .common import read_json, write_json, database, seal, verify, sha


def duplicate_pairs(candidates, structure_db):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher()
    pairs = []
    with database(structure_db) as db:
        for _, frame in candidates.groupby(['batch','formula'],sort=True):
            previous = []
            for row in frame.sort_values('candidate_row').itertuples():
                value = db.execute('SELECT structure_json FROM structures WHERE id=?',(row.id,)).fetchone()
                structure = Structure.from_dict(json.loads(value[0]))
                for other, other_structure in previous:
                    if matcher.fit(structure,other_structure):
                        pairs.append((int(row.candidate_row),other))
                previous.append((int(row.candidate_row),structure))
    return pairs


def unique_order(order, adjacency):
    kept, seen = [], set()
    for row in order:
        row = int(row)
        if not (adjacency.get(row,set()) & seen):
            kept.append(row)
            seen.add(row)
    return kept


def make_selections(scores, methods, fractions, pairs):
    adjacency = {}
    for a,b in pairs:
        adjacency.setdefault(int(a),set()).add(int(b))
        adjacency.setdefault(int(b),set()).add(int(a))
    records, requests = [], []
    for batch, frame in scores.groupby('batch',sort=True):
        for method in methods:
            values = pd.to_numeric(frame[method],errors='coerce')
            finite = frame.loc[np.isfinite(values)].copy()
            finite[method] = values.loc[finite.index]
            order = finite.sort_values([method,'candidate_row'],kind='stable').candidate_row.tolist()
            for policy, ranked in [('raw',order),('unique_refill',unique_order(order,adjacency))]:
                for fraction in fractions:
                    budget = int(np.floor(fraction*len(frame)))
                    selected = ranked[:budget]
                    request = dict(batch=int(batch), method=method, policy=policy, fraction=fraction,
                                   population=len(frame), budget=budget, selected_count=len(selected),
                                   unfilled_count=budget-len(selected), scoreable_count=len(finite))
                    requests.append(request)
                    records.extend({**request,'candidate_row':row,'rank':i+1} for i,row in enumerate(selected))
    return pd.DataFrame(records), pd.DataFrame(requests)


def export_selections(c):
    output = c['output']
    verify(output,'structures_complete.json')
    if (output/'selection_complete.json').exists():
        record = verify(output,'selection_complete.json')
        if record['fractions'] != c['fractions'] or record['seed'] != c['seed'] or record['batch_size'] != c['batch_size']:
            raise ValueError('Selections are frozen with different settings; use a new output directory')
        return read_json(output/'workload.json')
    scores = pd.read_csv(output/'scores.csv',keep_default_na=False)
    methods = read_json(output/'score_manifest.json')['methods']
    pairs = duplicate_pairs(scores,output/'structures.sqlite')
    selections, requests = make_selections(scores,methods,c['fractions'],pairs)
    selections.to_csv(output/'selections.csv',index=False)
    requests.to_csv(output/'selection_requests.csv',index=False)
    pd.DataFrame(pairs,columns=['a','b']).to_csv(output/'duplicate_pairs.csv',index=False)
    wanted = set(selections.candidate_row)
    identity_columns = ['candidate_row','id','eval_folder','idx','batch','turn','chemical_system','formula','structure_hash','num_atoms']
    scores.loc[scores.candidate_row.isin(wanted),identity_columns].to_csv(output/'evaluation_candidates.csv',index=False)
    # Input bundle contains no atomistic outcomes. Raw and unique policies share one union cache.
    files = [output/name for name in ('scores.csv','score_manifest.json','candidates.csv',
             'structures.sqlite','structures_complete.json','input_audit.json','selections.csv',
             'selection_requests.csv','duplicate_pairs.csv','evaluation_candidates.csv')]
    manifest = read_json(output/'score_manifest.json')
    write_json(output/'workload.json',{'heldout_candidates':len(scores),'selected_union':len(wanted),
                                     'methods':len(methods),'fractions':c['fractions'],
                                     'phase_predictions':'Additional; determined by all eligible chemical systems',
                                     'policies':['raw','unique_refill']})
    files.append(output/'workload.json')
    seal(output,'selection_complete.json',files,split='reserved',seed=c['seed'],
         batch_size=c['batch_size'],methods=methods,fractions=c['fractions'],
         scores_sha256=sha(output/'scores.csv'),checkpoint_records=manifest['checkpoints'])
    return read_json(output/'workload.json')
