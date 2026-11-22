"""Shared metric denominators, paired intervals, and figures for both proxies."""
from itertools import combinations
import numpy as np
import pandas as pd
from .common import verify, sha, seal, write_json


def success_metrics(selected,labels,budget,threshold=.1):
    labels = labels.reindex(list(selected))
    energy = pd.to_numeric(labels.e_above_hull_proxy,errors='coerce')
    valid = labels.hull_status.eq('ok') & np.isfinite(energy)
    passing = valid & (energy.le(0) if threshold==0 else energy.lt(threshold))
    n, good = int(valid.sum()), int(passing.sum())
    unknown = len(selected)-n
    return dict(stable_count=good,valid_count=n,unknown_count=unknown,
                unfilled_count=budget-len(selected),
                precision=good/n if n else np.nan,
                yield_lower=good/budget if budget else np.nan,
                yield_upper=(good+unknown+budget-len(selected))/budget if budget else np.nan,
                coverage=n/len(selected) if len(selected) else np.nan)


def compute_metrics(selections,requests,labels,thresholds=(.1,.05,0)):
    labels = labels.set_index('candidate_row',verify_integrity=True)
    groups = {key:frame.candidate_row.tolist() for key,frame in selections.groupby(['batch','method','policy','fraction'])}
    union = {key:np.unique(frame.candidate_row) for key,frame in selections.groupby(['batch','policy','fraction'])}
    records = []
    for row in requests.itertuples(index=False):
        selected = groups.get((row.batch,row.method,row.policy,row.fraction),[])
        for threshold in thresholds:
            u = union.get((row.batch,row.policy,row.fraction),[])
            union_metrics = success_metrics(u,labels,len(u),threshold)
            metrics = success_metrics(selected,labels,row.budget,threshold)
            records.append({**row._asdict(),**metrics,'threshold':threshold,
                            'union_stable':union_metrics['stable_count'],
                            'union_unknown':union_metrics['unknown_count'],
                            'union_selected':len(u),
                            'union_discovery_fraction':metrics['stable_count']/union_metrics['stable_count'] if union_metrics['stable_count'] else np.nan})
    return pd.DataFrame(records)


def aggregate_metrics(table):
    keys = ['proxy','method','policy','fraction','threshold']
    counts = ['budget','selected_count','stable_count','valid_count','unknown_count','unfilled_count',
              'union_stable','union_unknown','union_selected']
    result = table.groupby(keys,as_index=False)[counts].sum()
    result['precision'] = result.stable_count/result.valid_count.replace(0,np.nan)
    result['yield_lower'] = result.stable_count/result.budget.replace(0,np.nan)
    result['yield_upper'] = (result.stable_count+result.unknown_count+result.unfilled_count)/result.budget.replace(0,np.nan)
    result['coverage'] = result.valid_count/result.selected_count.replace(0,np.nan)
    result['union_discovery_fraction'] = result.stable_count/result.union_stable.replace(0,np.nan)
    return result


def paired_intervals(selections,candidates,labels,draws=2000,seed=2026):
    """Fixed selections, paired chemical-system block bootstrap; descriptive CIs."""
    labels = labels.set_index('candidate_row',verify_integrity=True)
    chemistry = candidates.set_index('candidate_row').chemical_system
    systems = sorted(chemistry.unique())
    code = {s:i for i,s in enumerate(systems)}
    rng = np.random.default_rng(seed)
    records = []
    primary = selections[selections.policy.eq('unique_refill')]
    for fraction,group in primary.groupby('fraction'):
        contributions = {}
        for method,frame in group.groupby('method'):
            ids = frame.candidate_row.to_numpy()
            y = labels.reindex(ids)
            energy = pd.to_numeric(y.e_above_hull_proxy,errors='coerce')
            passed = (y.hull_status.eq('ok') & np.isfinite(energy) & energy.lt(.1)).to_numpy()
            ix = np.array([code[s] for s in chemistry.loc[ids]])
            # Success per selected slot: unknowns are retained as unconfirmed discoveries.
            contributions[method] = (np.bincount(ix,weights=passed,minlength=len(systems)),
                                     np.bincount(ix,minlength=len(systems)))
        if 'chemical' not in contributions:
            continue
        differences = {method:[] for method in contributions if method!='chemical'}
        for _ in range(draws):
            weights = np.bincount(rng.integers(0,len(systems),size=len(systems)),minlength=len(systems))
            bg,bn = contributions['chemical']
            if not (weights@bn):
                continue
            baseline = (weights@bg)/(weights@bn)
            for method,(good,count) in contributions.items():
                if method!='chemical' and weights@count:
                    differences[method].append((weights@good)/(weights@count)-baseline)
        for method,values in differences.items():
            low,high = np.quantile(values,[.025,.975]) if values else (np.nan,np.nan)
            records.append(dict(method=method,fraction=fraction,ci_low=low,ci_high=high,
                                metric='confirmed stable per selected slot',unit='chemical_system',draws=draws))
    return pd.DataFrame(records)


def analyse(c):
    out = c['output']
    verify(out,'selection_complete.json')
    selections = pd.read_csv(out/'selections.csv')
    requests = pd.read_csv(out/'selection_requests.csv')
    candidates = pd.read_csv(out/'candidates.csv',keep_default_na=False)
    tables,intervals = [], []
    for proxy in ('megnet','chgnet'):
        verify(out,f'{proxy}_hulls_complete.json')
        labels = pd.read_csv(out/f'{proxy}_labels.csv',keep_default_na=False)
        tables.append(compute_metrics(selections,requests,labels).assign(proxy=proxy))
        intervals.append(paired_intervals(selections,candidates,labels,c['bootstrap_draws'],c['seed']).assign(proxy=proxy))
    table = pd.concat(tables,ignore_index=True)
    summary = aggregate_metrics(table)
    table.to_csv(out/'batch_metrics.csv',index=False)
    summary.to_csv(out/'aggregate_metrics.csv',index=False)
    pd.concat(intervals,ignore_index=True).to_csv(out/'paired_intervals.csv',index=False)
    # Summaries of GNN seeds describe repeated trained models, never a new ensemble.
    neural = summary[summary.method.str.contains('_seed')].copy()
    neural['family'] = neural.method.str.replace(r'_seed\d+$','',regex=True)
    neural.groupby(['proxy','family','policy','fraction','threshold'])[['yield_lower','precision']].agg(['mean','min','max']).to_csv(out/'seed_summary.csv')
    overlaps = []
    for key,frame in selections.groupby(['batch','policy','fraction']):
        sets = {method:set(group.candidate_row) for method,group in frame.groupby('method')}
        for a,b in combinations(sorted(sets),2):
            overlaps.append(dict(batch=key[0],policy=key[1],fraction=key[2],method_a=a,method_b=b,
                                 intersection=len(sets[a]&sets[b]),union=len(sets[a]|sets[b])))
    pd.DataFrame(overlaps).to_csv(out/'overlap.csv',index=False)
    seal(out,'analysis_complete.json',[out/f for f in ('batch_metrics.csv','aggregate_metrics.csv','paired_intervals.csv','seed_summary.csv','overlap.csv')],
         selections_sha256=sha(out/'selection_complete.json'),
         label_receipts={p:sha(out/f'{p}_hulls_complete.json') for p in ('megnet','chgnet')})
    plot_results(out,table,summary)
    return table,summary


def plot_results(out,table,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = out/'figures'
    directory.mkdir(exist_ok=True)
    # Keep 30+ methods readable by faceting proxies and budgets, rather than overlaying lines.
    for proxy in ('megnet','chgnet'):
        for fraction in (.01,.05,.07):
            view = table[table.proxy.eq(proxy)&table.policy.eq('unique_refill')&table.threshold.eq(.1)&table.fraction.eq(fraction)]
            matrix = view.pivot(index='method',columns='batch',values='yield_lower')*100
            fig,ax = plt.subplots(figsize=(13,10),layout='constrained')
            im = ax.imshow(matrix.to_numpy(),aspect='auto',vmin=0,vmax=100,cmap='viridis')
            ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7)
            ax.set_xlabel('Fixed shuffled batch'); ax.set_ylabel('Method / seed')
            ax.set_title(f'{proxy.upper()} | top {fraction:.0%} | confirmed stable per requested slot')
            fig.colorbar(im,ax=ax,label='Percent')
            fig.savefig(directory/f'{proxy}_batch_{int(fraction*100)}.pdf'); plt.close(fig)
        view = summary[summary.proxy.eq(proxy)&summary.policy.eq('unique_refill')&summary.threshold.eq(.1)]
        for metric,title in [('yield_lower','Confirmed stable per requested slot (%)'),
                             ('union_discovery_fraction','Share of stable discoveries in evaluated union (%)')]:
            matrix = view.pivot(index='method',columns='fraction',values=metric).sort_index()*100
            ax = matrix.plot.barh(figsize=(10,11),width=.8)
            ax.set_xlabel(title); ax.set_ylabel('Method / seed'); ax.set_title(proxy.upper())
            ax.legend(title='Selected fraction'); ax.figure.tight_layout()
            ax.figure.savefig(directory/f'{proxy}_{metric}.pdf')
            ax.figure.savefig(directory/f'{proxy}_{metric}.png',dpi=140)
            plt.close(ax.figure)
    write_json(directory/'definitions.json',{'primary':'unique refill; energy above hull <0.1 eV/atom',
               'yield':'confirmed stable / requested slots; unknown outcomes remain in denominator',
               'discovery':'stable selected / stable in union of methods at the SAME budget and policy; not population recall',
               'cross_batch_repeats':'Counted as separate selection opportunities; batches are disjoint candidate rows.'})
