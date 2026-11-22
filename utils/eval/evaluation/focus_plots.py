"""Small, explicitly labelled 1%-budget figures: five alternatives and chemical."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from .focus_stats import family_name


def color(family):
    if family=='chemical': return '#244b70'
    return '#c17a28' if family.startswith(('knn_','random_')) else '#27856a'


def save(fig,directory,name):
    import matplotlib.pyplot as plt
    for ext in ('png','pdf'): fig.savefig(directory/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def plot_top_five(directory,proxy,table,runs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    frame=table.sort_values('display_rank').reset_index(drop=True)
    fig,(left,right)=plt.subplots(1,2,figsize=(14.8,5.8),gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(left=.20,right=.975,top=.82,bottom=.25,wspace=.22)
    y=np.arange(len(frame));colors=[color(f) for f in frame.family]
    left.barh(y,frame.yield_mean*100,color=colors,height=.59)
    left.barh(y,(frame.yield_upper-frame.yield_mean)*100,left=frame.yield_mean*100,
              color='#f8dba0',edgecolor='#95651a',hatch='////',height=.59)
    for i,row in frame.iterrows():
        seeds=runs.loc[runs.method.map(family_name).eq(row.family),'yield_lower']*100
        if row.n_runs>1: left.scatter(seeds,np.full(len(seeds),i),s=19,color='#17222c',zorder=4)
        label=f'{row.yield_mean:.1%}  ({row.stable_mean:.0f}/{row.requested_slots})' if row.n_runs==1 else f'{row.yield_mean:.1%}  (3-seed mean)'
        left.text(2,i,label,color='white',va='center',fontsize=10)
    left.set(yticks=y,yticklabels=frame.label,xlim=(0,100),xlabel='Confirmed stable / requested slots (%)',title='Selected stability yield')
    left.invert_yaxis();left.grid(axis='x',alpha=.18);left.set_axisbelow(True)
    for i,row in frame.iterrows():
        right.plot([row.ci_low*100,row.ci_high*100],[i,i],color=colors[i],linewidth=2)
        right.scatter(row.gain_vs_chemical*100,i,color=colors[i],s=40,zorder=3)
    bound=max(10,float(np.nanmax(np.abs(frame[['ci_low','ci_high']].to_numpy())))*100+2)
    right.set(yticks=y,yticklabels=['']*len(frame),ylim=(len(frame)-.5,-.5),xlim=(-bound,bound),
              xlabel='Gain versus chemical (percentage points)',title='Paired 95% descriptive intervals')
    right.axvline(0,color='#52616f',linestyle='--',linewidth=1);right.grid(axis='x',alpha=.18)
    for ax in (left,right):
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10)
    fig.suptitle(f'{proxy.upper()} | top 1% | five best alternatives + chemical',fontsize=16,fontweight='bold',y=.96)
    legends=[Patch(color='#244b70',label='Chemical'),Patch(color='#c17a28',label='kNN / random control')]
    if frame.n_runs.gt(1).any(): legends.append(Patch(color='#27856a',label='GNN / readout; dots = seeds'))
    if (frame.yield_upper>frame.yield_mean+1e-12).any():
        legends.append(Patch(facecolor='#f8dba0',edgecolor='#95651a',hatch='////',label='Unknown-label extension'))
    fig.legend(handles=legends,loc='lower left',bbox_to_anchor=(.20,.095),ncol=4,frameon=False,fontsize=9)
    fig.text(.20,.025,'Top five selected by observed yield. Intervals are unadjusted for method selection.',fontsize=9,color='#475663')
    save(fig,directory,f'{proxy}_top5_1pct')


def plot_discoveries(directory,proxy,table):
    import matplotlib.pyplot as plt
    frame=table.sort_values('display_rank').reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(9.5,4.8));fig.subplots_adjust(left=.29,right=.95,top=.82,bottom=.23)
    y=np.arange(len(frame))
    ax.barh(y,frame.discovery_share*100,color=[color(f) for f in frame.family],height=.6)
    for i,row in frame.iterrows():
        ax.text(row.discovery_share*100+.25,i,f'{row.discovery_share:.1%}',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=frame.label,xlim=(0,frame.discovery_share.max()*125),
           xlabel='Share of confirmed stable crystals in the 1% selection union (%)')
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f'{proxy.upper()} | discoveries at the 1% budget',fontsize=15,fontweight='bold')
    fig.text(.29,.05,f'Union: {int(frame.union_stable.iloc[0]):,} confirmed stable; {int(frame.union_unknown.iloc[0]):,} unknown labels.\nMethods can overlap. This is not recall over the full held-out dataset.',fontsize=9,color='#475663')
    save(fig,directory,f'{proxy}_top5_1pct_discoveries')


def plot_batches(directory,proxy,table,batch):
    import matplotlib.pyplot as plt
    frame=table.sort_values('display_rank').reset_index(drop=True)
    grouped=batch.groupby(['family','batch'],as_index=False).agg(stable=('stable_count','mean'),budget=('budget','first'))
    baseline=grouped[grouped.family.eq('chemical')].sort_values('batch')
    rolling_baseline=100*baseline.stable.rolling(5,min_periods=1).sum()/baseline.budget.rolling(5,min_periods=1).sum()
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,(_,row) in zip(axes.flat,frame.iterrows()):
        data=grouped[grouped.family.eq(row.family)].sort_values('batch')
        ax.scatter(data.batch,100*data.stable/data.budget,s=12,color=color(row.family),alpha=.35)
        ax.plot(baseline.batch,rolling_baseline,color='#52616f',linestyle='--',linewidth=1.2,label='Chemical, rolling 5 batches')
        smooth=100*data.stable.rolling(5,min_periods=1).sum()/data.budget.rolling(5,min_periods=1).sum()
        ax.plot(data.batch,smooth,color=color(row.family),linewidth=1.8,label='Method, rolling 5 batches')
        ax.set(title=row.label,ylim=(0,103));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    for ax in axes[:,0]: ax.set_ylabel('Confirmed stable / slots (%)')
    for ax in axes[-1,:]: ax.set_xlabel('Fixed shuffled batch index')
    handles,labels=axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle(f'{proxy.upper()} | top 1% per batch | top five + chemical',fontsize=15,fontweight='bold')
    fig.tight_layout(rect=(0,.045,1,.94))
    save(fig,directory,f'{proxy}_top5_1pct_batches')
