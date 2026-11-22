"""Verified, notebook-facing 1% analysis; no model inference or re-selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import verify,sha,seal,write_json
from .focus_stats import (family_name,summarize_families,top_five_plus_chemical,
                          family_intervals,check_counts,interpretation)
from .focus_plots import plot_top_five,plot_discoveries,plot_batches


def analyse_one_percent(c,top_n=5):
    out=c['output']
    verify(out,'transfer_complete.json');verify(out,'selection_complete.json')
    analysis=verify(out,'analysis_complete.json')
    summary=pd.read_csv(out/'aggregate_metrics.csv')
    run_metrics=summary[summary.policy.eq('unique_refill')&summary.fraction.eq(.01)&summary.threshold.eq(.1)].copy()
    selections=pd.read_csv(out/'selections.csv')
    selections=selections[selections.policy.eq('unique_refill')&selections.fraction.eq(.01)].copy()
    requests=pd.read_csv(out/'selection_requests.csv')
    requests=requests[requests.policy.eq('unique_refill')&requests.fraction.eq(.01)].copy()
    if not requests.selected_count.eq(requests.budget).all():
        raise ValueError('Inspect unfilled slots before using filled-budget focus intervals')
    candidates=pd.read_csv(out/'candidates.csv',keep_default_na=False)
    labels={}
    for proxy in ('megnet','chgnet'):
        verify(out,f'{proxy}_hulls_complete.json')
        if analysis['label_receipts'][proxy]!=sha(out/f'{proxy}_hulls_complete.json'):
            raise ValueError('Cached aggregate analysis uses different proxy labels')
        labels[proxy]=pd.read_csv(out/f'{proxy}_labels.csv',keep_default_na=False)
        check_counts(run_metrics,selections,requests,labels[proxy],proxy)
    families=summarize_families(run_metrics)
    ci=family_intervals(selections,candidates,labels,c.get('bootstrap_draws',2000),c.get('seed',2026))
    families=families.merge(ci,on=['proxy','family'],validate='one_to_one')
    top=top_five_plus_chemical(families,top_n)
    directory=out/'focus_1pct';directory.mkdir(exist_ok=True)
    families.to_csv(directory/'all_method_families.csv',index=False)
    top.to_csv(directory/'top5_plus_chemical.csv',index=False)
    run_metrics.to_csv(directory/'individual_seed_results.csv',index=False)
    ci.to_csv(directory/'family_intervals.csv',index=False)
    batch=pd.read_csv(out/'batch_metrics.csv')
    batch=batch[batch.policy.eq('unique_refill')&batch.fraction.eq(.01)&batch.threshold.eq(.1)].copy()
    batch['family']=batch.method.map(family_name)
    details={}
    for proxy,frame in families.groupby('proxy',sort=True):
        baseline=frame[frame.family.eq('chemical')].iloc[0]
        alternatives=frame[~frame.family.eq('chemical')]
        winner=frame.sort_values(['yield_mean','family'],ascending=[False,True]).iloc[0]
        chosen=top[top.proxy.eq(proxy)&~top.family.eq('chemical')]
        tied=alternatives.loc[np.isclose(alternatives.yield_mean,chosen.yield_mean.min(),rtol=0,atol=1e-12),'family'].tolist()
        details[proxy]={'numerical_leader':winner.family,'leader_yield':winner.yield_mean,
                        'chemical_yield':baseline.yield_mean,'chemical_stable':int(baseline.stable_mean),
                        'requested_slots':int(baseline.requested_slots),
                        'alternatives_with_ci_above_chemical':alternatives.loc[alternatives.ci_low.gt(0),'family'].tolist(),
                        'chemical_beats_all_alternative_missing_label_upper_bounds':bool(baseline.yield_mean>alternatives.yield_upper.max()),
                        'cutoff_ties':tied,'top_alternatives':chosen.family.tolist()}
        plot_top_five(directory,proxy,top[top.proxy.eq(proxy)],run_metrics[run_metrics.proxy.eq(proxy)])
        plot_discoveries(directory,proxy,top[top.proxy.eq(proxy)])
        plot_batches(directory,proxy,top[top.proxy.eq(proxy)],batch[batch.proxy.eq(proxy)])
    coverage=[]
    for proxy in ('megnet','chgnet'):
        phases=pd.read_csv(out/f'{proxy}_phase_predictions.csv',keep_default_na=False);y=labels[proxy]
        coverage.append(dict(proxy=proxy,phase_predictions=len(phases),failed_phase_predictions=int(phases.status.ne('ok').sum()),
                             candidate_union=len(y),valid_hull_labels=int(y.hull_status.eq('ok').sum()),
                             unknown_hull_labels=int(y.hull_status.ne('ok').sum())))
    coverage=pd.DataFrame(coverage);coverage.to_csv(directory/'coverage.csv',index=False)
    write_json(directory/'findings.json',details)
    write_json(directory/'definitions.json',{
        'budget':.01,'policy':'unique_refill','threshold':'energy above same-model fixed hull <0.1 eV/atom',
        'ranking':'mean confirmed stable / requested slots, separately by proxy',
        'gnn_summary':'average three seed-specific results, not an ensemble or best seed',
        'ties':'smaller k, then method ID; cutoff ties have equal observed yield',
        'unknowns':'retained in denominator; hatched extension assumes all unknowns pass',
        'inference':'exploratory top-five ranking; descriptive CIs unadjusted for multiple comparisons',
        'union_discovery':'share of confirmed stable discoveries in same-budget union; not population recall'})
    text=interpretation(families,details,coverage)
    write_json(directory/'interpretation.json',{'markdown':text})
    seal(directory,'focus_complete.json',[p for p in directory.iterdir() if p.is_file() and p.name!='focus_complete.json'],
         source_analysis_receipt_sha256=sha(out/'analysis_complete.json'),
         source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('focus_stats.py'),Path(__file__).with_name('focus_plots.py'))})
    return {'directory':directory,'families':families,'top':top,'runs':run_metrics,
            'coverage':coverage,'findings':details,'interpretation':text}
