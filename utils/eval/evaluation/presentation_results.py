"""Best-observed reporting view; keep selection metadata explicit in CSV/receipts."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,verify,sha,seal


def select_best_reporting_runs(summary):
    """One observed run for each GNN architecture and one k per kNN metric.

    Rank by confirmed stable/requested slots. Ties prefer fewer unknowns,
    smaller k, then method ID. This is outcome-based reporting, not validation
    selection or an estimate averaged across random seeds.
    """
    frame=summary[summary.policy.eq('unique_refill')&summary.fraction.eq(.01)&summary.threshold.eq(.1)].copy()
    categories=[('Chemical baseline',r'^chemical$'),('Euclidean kNN',r'^knn_euclidean_\d+$'),
                ('Cosine kNN',r'^knn_cosine_\d+$'),('GCN',r'^gcn_(physical|cosine)_seed\d+$'),
                ('GraphSAGE',r'^sage_(physical|cosine)_seed\d+$'),('GAT',r'^gat_(physical|cosine)_seed\d+$')]
    records=[]
    for proxy,rows in frame.groupby('proxy',sort=True):
        if rows.budget.nunique()!=1:raise ValueError('Reporting requires equal method budgets')
        for order,(category,pattern) in enumerate(categories):
            options=rows[rows.method.str.match(pattern)].copy()
            if options.empty:raise ValueError(f'Missing category {proxy}/{category}')
            options['_k']=pd.to_numeric(options.method.str.extract(r'^knn_\w+_(\d+)$')[0],errors='coerce').fillna(np.inf)
            best=options.sort_values(['yield_lower','unknown_count','_k','method'],ascending=[False,True,True,True],kind='stable').iloc[0]
            row=best.drop(labels=['_k']).to_dict()
            row.update(category=category,plot_order=order,selection_rule='best observed confirmed yield; not a seed average')
            if category.endswith('kNN'):
                row['plot_label']=f'{category} (k={int(best["_k"])})'
            else:row['plot_label']=category
            records.append(row)
    return pd.DataFrame(records)


def export_presentation_results(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out=c['output'];record=verify(out,'analysis_complete.json')
    verify(out,'selection_complete.json')
    for proxy in ('megnet','chgnet'):
        verify(out,f'{proxy}_hulls_complete.json')
        if record['label_receipts'][proxy]!=sha(out/f'{proxy}_hulls_complete.json'):
            raise ValueError('Reporting analysis does not match current labels')
    selected=select_best_reporting_runs(pd.read_csv(out/'aggregate_metrics.csv'))
    # Crys_METH keeps shareable figures under docs/assets.
    assets=ROOT/'docs/assets';assets.mkdir(exist_ok=True)
    selected.to_csv(assets/'atomistic_1pct_best_runs.csv',index=False)
    products=[assets/'atomistic_1pct_best_runs.csv']
    colors={'Chemical baseline':'#18334C','Euclidean kNN':'#B47A22','Cosine kNN':'#456EA8',
            'GCN':'#17816D','GraphSAGE':'#17816D','GAT':'#17816D'}
    for proxy,frame in selected.groupby('proxy',sort=True):
        frame=frame.sort_values('plot_order').reset_index(drop=True)
        with plt.rc_context({'font.size':16,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':16}):
            fig,ax=plt.subplots(figsize=(11.8,3.7))
            fig.subplots_adjust(left=.25,right=.985,bottom=.22,top=.98)
            y=np.arange(len(frame))
            ax.barh(y,100*frame.yield_lower,color=[colors[k] for k in frame.category],height=.65)
            if frame.unknown_count.gt(0).any():
                ax.barh(y,100*(frame.yield_upper-frame.yield_lower),left=100*frame.yield_lower,
                        height=.65,facecolor='#F6DBA6',edgecolor='#95651A',hatch='////')
            for i,row in frame.iterrows():
                ax.text(2,i,f'{row.yield_lower:.1%}  ({int(row.stable_count)}/{int(row.budget)})',
                        va='center',color='white',fontsize=16)
                if row.unknown_count:
                    ax.text(100*row.yield_upper+1,i,f'{int(row.unknown_count)} unknown',va='center',fontsize=12,color='#6e501c')
            ax.set(yticks=y,yticklabels=frame.plot_label,xlim=(0,108),xticks=[0,20,40,60,80,100],
                   xlabel='Confirmed stable selections / requested slots (%)')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            if frame.unknown_count.gt(0).any():
                fig.legend(handles=[Patch(facecolor='#F6DBA6',edgecolor='#95651A',hatch='////',label='Unknown-label extension')],
                           loc='lower right',bbox_to_anchor=(.99,-.01),frameon=False,fontsize=10)
            for extension in ('pdf','png'):
                path=assets/f'{proxy}_heldout_1pct_best_runs.{extension}'
                fig.savefig(path,dpi=190);products.append(path)
            plt.close(fig)
    seal(assets,'atomistic_1pct_best_runs_provenance.json',products,
         source_analysis_sha256=sha(out/'analysis_complete.json'),source_sha256=sha(Path(__file__)),
         budget=.01,policy='unique_refill',threshold='strict <0.1 eV/atom',
         reporting='Best observed run per GCN/GAT/GraphSAGE over physical/cosine and 3 seeds; best k per kNN metric. Descriptive, chosen using evaluation outcomes.',
         candidate_population=35332,requested_slots_per_method=353)
    return selected
