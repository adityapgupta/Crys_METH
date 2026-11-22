"""Seed-aware summaries and paired uncertainty for the 1% notebook analysis."""
import re
import numpy as np
import pandas as pd


def family_name(method):
    return re.sub(r'_seed\d+$', '', method)


def display_name(family):
    if family == 'chemical': return 'Chemical baseline'
    if family == 'random_candidates': return 'Random control'
    match = re.fullmatch(r'knn_(euclidean|cosine)_(\d+)', family)
    if match: return f'kNN {match[1]}, k={match[2]}'
    if family == 'no_graph_physical': return 'No-graph readout'
    kind, graph = family.split('_', 1)
    return f'{dict(gcn="GCN", gat="GAT", sage="GraphSAGE")[kind]} / {graph}'


def summarize_families(run_metrics):
    frame = run_metrics.copy()
    frame['family'] = frame.method.map(family_name)
    rows = []
    for (proxy, family), group in frame.groupby(['proxy','family'], sort=True):
        if group.budget.nunique() != 1: raise ValueError('Unequal seed budgets')
        rows.append(dict(proxy=proxy, family=family, label=display_name(family),
                         n_runs=len(group), requested_slots=int(group.budget.iloc[0]),
                         stable_mean=float(group.stable_count.mean()),
                         valid_mean=float(group.valid_count.mean()),
                         unknown_mean=float(group.unknown_count.mean()),
                         yield_mean=float(group.yield_lower.mean()),
                         yield_upper=float(group.yield_upper.mean()),
                         seed_min=float(group.yield_lower.min()),
                         seed_max=float(group.yield_lower.max()),
                         precision_mean=float(group.precision.mean()),
                         union_stable=int(group.union_stable.iloc[0]),
                         union_unknown=int(group.union_unknown.iloc[0]),
                         discovery_share=float(group.union_discovery_fraction.mean())))
    result = pd.DataFrame(rows)
    baseline = result[result.family.eq('chemical')][['proxy','yield_mean','yield_upper']].rename(
        columns={'yield_mean':'chemical_yield','yield_upper':'chemical_upper'})
    result = result.merge(baseline,on='proxy',validate='many_to_one')
    result['gain_vs_chemical'] = result.yield_mean-result.chemical_yield
    result['missing_label_gain_min'] = result.yield_mean-result.chemical_upper
    result['missing_label_gain_max'] = result.yield_upper-result.chemical_yield
    return result


def top_five_plus_chemical(families, count=5):
    """Separate proxy rankings; ties prefer smaller k, then method ID."""
    records=[]
    for proxy,frame in families.groupby('proxy',sort=True):
        alternatives=frame[~frame.family.eq('chemical')].copy()
        alternatives['_k']=alternatives.family.str.extract(r'^knn_\w+_(\d+)$')[0].astype(float).fillna(np.inf)
        chosen=alternatives.sort_values(['yield_mean','_k','family'],ascending=[False,True,True],kind='stable').head(count).drop(columns='_k')
        chosen=chosen.assign(display_rank=np.arange(1,len(chosen)+1))
        records.append(pd.concat([frame[frame.family.eq('chemical')].assign(display_rank=0),chosen],ignore_index=True))
    return pd.concat(records,ignore_index=True)


def family_intervals(selections,candidates,labels_by_proxy,draws=2000,seed=2026):
    """Paired chemistry bootstrap: each draw averages seed-specific family rates.

    Selections are fixed and budgets filled. Shared resampling weights preserve
    dependence between seeds. These are unadjusted descriptive intervals.
    """
    methods=sorted(selections.method.unique())
    families={f:[i for i,m in enumerate(methods) if family_name(m)==f]
              for f in sorted({family_name(m) for m in methods})}
    chemical=methods.index('chemical')
    chemistry=candidates.set_index('candidate_row',verify_integrity=True).chemical_system
    systems=sorted(chemistry.unique());codes={s:i for i,s in enumerate(systems)}
    selected={m:selections.loc[selections.method.eq(m),'candidate_row'].to_numpy() for m in methods}
    counts=np.zeros((len(methods),len(systems)));indices={}
    for i,m in enumerate(methods):
        indices[m]=np.array([codes[s] for s in chemistry.loc[selected[m]]])
        counts[i]=np.bincount(indices[m],minlength=len(systems))
    records=[]
    for proxy,labels in labels_by_proxy.items():
        labels=labels.set_index('candidate_row',verify_integrity=True)
        good=np.zeros_like(counts)
        for i,m in enumerate(methods):
            if not set(selected[m]).issubset(labels.index): raise ValueError('Missing selected label')
            y=labels.loc[selected[m]];energy=pd.to_numeric(y.e_above_hull_proxy,errors='coerce')
            passing=(y.hull_status.eq('ok')&np.isfinite(energy)&energy.lt(.1)).to_numpy()
            good[i]=np.bincount(indices[m],weights=passing,minlength=len(systems))
        rng=np.random.default_rng(seed);differences={f:[] for f in families}
        for start in range(0,draws,128):
            w=rng.multinomial(len(systems),np.full(len(systems),1/len(systems)),size=min(128,draws-start))
            denominator=counts@w.T
            rates=np.divide(good@w.T,denominator,out=np.full_like(denominator,np.nan),where=denominator>0)
            for family,which in families.items():
                d=np.mean(rates[which],axis=0)-rates[chemical]
                differences[family].extend(d[np.isfinite(d)].tolist())
        for family,values in differences.items():
            low,high=np.quantile(values,[.025,.975]) if values else (np.nan,np.nan)
            records.append(dict(proxy=proxy,family=family,ci_low=low,ci_high=high,draws=draws,
                                resample_unit='chemical_system',interval='unadjusted descriptive 95%; seed rates averaged'))
    return pd.DataFrame(records)


def check_counts(run_metrics,selections,requests,labels,proxy):
    indexed=labels.set_index('candidate_row',verify_integrity=True)
    for method,group in selections.groupby('method',sort=True):
        y=indexed.loc[group.candidate_row];energy=pd.to_numeric(y.e_above_hull_proxy,errors='coerce')
        valid=y.hull_status.eq('ok')&np.isfinite(energy);stable=valid&energy.lt(.1)
        s=run_metrics[run_metrics.proxy.eq(proxy)&run_metrics.method.eq(method)].iloc[0]
        budget=requests.loc[requests.method.eq(method),'budget'].sum()
        if (int(stable.sum()),int(valid.sum()),len(group),int(budget)) != (int(s.stable_count),int(s.valid_count),int(s.selected_count),int(s.budget)):
            raise ValueError(f'Focus counts do not reconcile: {proxy}/{method}')


def interpretation(families,findings,coverage):
    leaders={info['numerical_leader'] for info in findings.values()}
    opening=('The ranking depends on the proxy; these results do not establish one method as best across both evaluators.'
             if len(leaders)>1 else
             'Both proxies have the same numerical leader. Numerical ranking alone does not establish superiority; inspect paired intervals and missing-label bounds below.')
    lines=['### Is there a clear winner at 1%?',opening]
    for proxy in ('megnet','chgnet'):
        frame=families[families.proxy.eq(proxy)];info=findings[proxy]
        leader=frame[frame.family.eq(info['numerical_leader'])].iloc[0]
        lines.append(f'- **{proxy.upper()}:** numerical leader **{leader.label}**, {leader.yield_mean:.1%}; chemical gives {info["chemical_stable"]}/{info["requested_slots"]} = {info["chemical_yield"]:.1%}.')
        if not info['alternatives_with_ci_above_chemical']:
            lines.append('  No alternative has a descriptive paired 95% interval wholly above chemical.')
        if info['chemical_beats_all_alternative_missing_label_upper_bounds']:
            lines.append('  Chemical remains ahead even if all unknown labels of every alternative pass. This is a missing-label sensitivity result, not a confidence interval.')
        if len(info['cutoff_ties'])>1:
            names=', '.join(display_name(f) for f in info['cutoff_ties'])
            lines.append(f'  The top-five cutoff has tied methods: {names}. Smaller k, then method ID, breaks display ties; no performance difference is established between ties.')
    neural=families[families.family.str.startswith(('gcn_','gat_','sage_'))]
    for proxy,frame in neural.groupby('proxy',sort=True):
        best=frame.sort_values('yield_mean',ascending=False).iloc[0]
        lines.append(f'- Best GNN **by three-seed mean** under {proxy.upper()}: {best.label}, {best.yield_mean:.1%}, seed range {best.seed_min:.1%} to {best.seed_max:.1%}. Seed range is not a confidence interval.')
    for row in coverage.itertuples():
        if row.unknown_hull_labels:
            lines.append(f'- **Coverage:** {row.proxy.upper()} has {row.unknown_hull_labels:,} unknown hull labels in the full evaluated union; {row.failed_phase_predictions:,} phase predictions failed. Unknowns remain visible.')
    lines.append('Top-five rankings are exploratory choices from the evaluated data. Intervals are unadjusted for multiple methods; lack of a clear advantage does not establish equivalence. Proxy labels are not DFT confirmation.')
    return '\n\n'.join(lines)
