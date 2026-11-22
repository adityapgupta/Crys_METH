"""One model-native hull calculation for MEGNet and CHGNet."""
from itertools import combinations
import json
import numpy as np
import pandas as pd


def subsystems(system):
    elements = sorted(set(system.split('-')))
    return ['-'.join(sub) for n in range(1,len(elements)+1) for sub in combinations(elements,n)]


def label_candidates(candidates,phases,candidate_predictions,phase_predictions,energy_kind):
    from reference_baselines.core.thermodynamics import FixedHull
    if energy_kind not in ('formation_energy_per_atom','total_energy_per_atom'):
        raise ValueError('Energy convention must be explicit')
    cp = candidate_predictions.set_index('id',verify_integrity=True)
    pp = phase_predictions.set_index('id',verify_integrity=True)
    groups = {system:frame for system,frame in phases.groupby('chemical_system')}
    records,audits = [],[]
    for system,frame in candidates.groupby('chemical_system',sort=True):
        parts = [groups[sub] for sub in subsystems(system) if sub in groups]
        bank = pd.concat(parts,ignore_index=True) if parts else phases.iloc[:0]
        missing = [element for element in system.split('-') if element not in groups]
        failed = [row.id for row in bank.itertuples() if row.id not in pp.index
                  or pp.loc[row.id,'status']!='ok'
                  or not np.isfinite(float(pp.loc[row.id,'energy_per_atom']))]
        solver = None
        if not missing and not failed:
            refs = [{'entry_id':row.id,'formula':row.formula,
                     'energy_per_atom':float(pp.loc[row.id,'energy_per_atom'])} for row in bank.itertuples()]
            solver = FixedHull(system.split('-'),refs)
        audits.append(dict(chemical_system=system,phase_count=len(bank),complete=solver is not None,
                           missing_elements=json.dumps(missing),failed_phase_ids=json.dumps(failed)))
        for row in frame.itertuples(index=False):
            record = row._asdict()
            record.update(energy_kind=energy_kind,hull_status='ok',hull_error='',
                          energy_per_atom=np.nan,hull_energy_per_atom=np.nan,
                          e_above_hull_proxy=np.nan,decomposition='',hull_method='')
            if row.id not in cp.index or cp.loc[row.id,'status']!='ok':
                record.update(hull_status='candidate_prediction_error',hull_error='Candidate prediction failed or absent')
            elif solver is None:
                record.update(hull_status='incomplete_reference_hull',hull_error='One or more fixed reference phases are missing or failed')
            else:
                try:
                    energy = float(cp.loc[row.id,'energy_per_atom'])
                    if not np.isfinite(energy):
                        raise ValueError('Nonfinite candidate energy')
                    hull,products,method = solver.minimum(row.formula)
                    record.update(energy_per_atom=energy,hull_energy_per_atom=hull,
                                  e_above_hull_proxy=energy-hull,decomposition=json.dumps(products),hull_method=method)
                except Exception as exc:
                    record.update(hull_status='hull_error',hull_error=f'{type(exc).__name__}: {exc}')
            records.append(record)
    return pd.DataFrame(records).sort_values('candidate_row'),pd.DataFrame(audits)
