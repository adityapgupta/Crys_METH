"""Export the historical eSEN appendix from the corrected saved batch analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, read_json, sha, seal


def historical_appendix():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = ROOT/'data/eval/report/assets'
    saved_assets = dest/'historical_provenance.json'
    if saved_assets.is_file():
        record = read_json(saved_assets)
        required = {'historical_esen.csv', 'historical_esen_per_batch.csv',
                    'historical_esen_table.tex', 'historical_esen_budgets.pdf',
                    'historical_esen_budgets.png', 'historical_esen_counts.pdf',
                    'historical_esen_counts.png', 'historical_esen_batches.pdf'}
        if required.issubset(record.get('files', {})):
            for name in required:
                path = dest/name
                if not path.is_file() or sha(path) != record['files'][name]:
                    raise ValueError(f'Cached historical appendix asset changed: {name}')
            return pd.read_csv(dest/'historical_esen.csv')
    source = ROOT/'outputs/gnn/reference_set_comparison/comparison_20_29/reports/analysis_batchwise'
    record = read_json(source/'analysis_provenance.json')
    filename = 'selected_scope_metrics.csv'
    if sha(source/filename) != record['derived_tables'][filename]:
        raise ValueError('Historical eSEN table does not match its saved provenance')
    table = pd.read_csv(source/filename)
    methods = ['chemical','knn_5','knn_10','knn_15','knn_25','random_candidates']
    table = table[table.proxy.eq('esen')&table.policy.eq('unique_refill')&table.threshold_eV_per_atom.eq(.1)&table.method.isin(methods)].copy()
    if table.empty or not table.scope.eq('batch_aggregate').all():
        raise ValueError('Need corrected per-batch eSEN results')
    dest.mkdir(parents=True,exist_ok=True)
    table.to_csv(dest/'historical_esen.csv',index=False)
    lines = ['\\begin{tabular}{lrrrrr}','\\toprule',
             'Method & Budget & Stable & Valid & Unknown & Slots \\\\', '\\midrule']
    for row in table.sort_values(['fraction','method']).itertuples():
        name = row.method.replace('_',r'\_')
        lines.append(f'{name} & {row.fraction:.0%}'.replace('%',r'\%')+
                     f' & {row.stable_count} & {row.evaluated_count} & {row.unknown_count} & {row.budget} '+r'\\')
    lines += ['\\bottomrule','\\end{tabular}']
    (dest/'historical_esen_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    colors = plt.get_cmap('tab10').colors
    fig,ax = plt.subplots(figsize=(7.2,3.8),layout='constrained')
    for i,method in enumerate(methods):
        data = table[table.method.eq(method)].sort_values('fraction')
        ax.plot(data.fraction*100,data.stable_yield_lower_bound*100,marker='o',label=method,color=colors[i])
        ax.fill_between(data.fraction*100,data.stable_yield_lower_bound*100,data.stable_yield_upper_bound*100,alpha=.1,color=colors[i])
    ax.set(xlabel='Selected fraction per original folder (%)',ylabel='Confirmed stable / requested slots (%)',
           title='Historical eSEN evaluation: folders 20–29',xticks=[1,5,10],ylim=(0,100))
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2)
    fig.savefig(dest/'historical_esen_budgets.pdf'); fig.savefig(dest/'historical_esen_budgets.png',dpi=170); plt.close(fig)
    primary = table[table.fraction.eq(.01)].set_index('method').reindex(methods).dropna(subset=['budget'])
    x = np.arange(len(primary))
    fig,ax = plt.subplots(figsize=(7.2,3.8),layout='constrained')
    ax.bar(x,primary.stable_count,label='Confirmed stable')
    ax.bar(x,primary.evaluated_count-primary.stable_count,bottom=primary.stable_count,label='Known non-passing')
    ax.bar(x,primary.unknown_count,bottom=primary.evaluated_count,label='Unknown')
    ax.set_xticks(x,primary.index,rotation=25,ha='right')
    ax.set(ylabel='Selected candidates',title='Historical eSEN: top 1% separately per folder')
    ax.legend(fontsize=8)
    fig.savefig(dest/'historical_esen_counts.pdf'); fig.savefig(dest/'historical_esen_counts.png',dpi=170); plt.close(fig)
    # Per-folder results are exported with the original report and verified by its receipt.
    report = source.parent
    from .common import verify as verify_bundle
    saved = read_json(report/'complete.json')
    outputs = {row['name']:row['sha256'] for row in saved['outputs']}
    if sha(report/'batch_metrics.csv') != outputs['batch_metrics.csv']:
        raise ValueError('Historical per-batch metrics changed')
    batches = pd.read_csv(report/'batch_metrics.csv')
    batches = batches[batches.proxy.eq('esen')&batches.method.isin(methods)&batches.policy.eq('unique_refill')&batches.fraction.eq(.01)&batches.threshold_eV_per_atom.eq(.1)]
    batches.to_csv(dest/'historical_esen_per_batch.csv',index=False)
    fig,ax = plt.subplots(figsize=(7.2,3.8),layout='constrained')
    for method,group in batches.groupby('method'):
        group = group.sort_values('group')
        ax.plot(group['group'],100*group.stable_count/group.budget,marker='.',label=method)
    ax.set(xlabel='Original release folder',ylabel='Confirmed stable / requested slots (%)',ylim=(0,100),
           title='Historical eSEN: per-folder top-1% yield')
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2)
    fig.savefig(dest/'historical_esen_batches.pdf'); plt.close(fig)
    files = [p for p in dest.iterdir() if p.name.startswith('historical_esen')]
    seal(dest,'historical_provenance.json',files,source_sha256=sha(source/filename),
         source_analysis_receipt_sha256=sha(source/'analysis_provenance.json'),
         original_folders=[20,29],budgets=[.01,.05,.10],new_proxy_inference=False)
    return table
