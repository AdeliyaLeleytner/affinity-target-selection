"""Local-label curves and matched increments from saved out-of-fold predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import use_style, save, PALETTE, WIDE
ROOT=Path(__file__).resolve().parents[1]
METHODS={
 'boltz_affinity':'Boltz-2 affinity','boltz_binary':'Boltz-2 binder',
 'boltzina_vina_far_affinity':'Boltzina far affinity','boltzina_vina_far_binary':'Boltzina far binder',
 'boltzina_vina_zero_affinity':'Boltzina zero affinity','boltzina_vina_zero_binary':'Boltzina zero binder',
 'gnina':'GNINA on Vina pose','nesso_affinity':'Nesso affinity','nesso_binary':'Nesso binder',
 'raw_ad4':'AutoDock4','raw_gnina_own':'GNINA own search','vina':'Vina'}
FRACTIONS=[.125,.25,.5,1.]

def intervals(frame, groupmap):
    """Resample whole scaffold groups; preserve all within-group molecules."""
    queries=sorted(frame['query'].unique()); groups=np.array([groupmap[q] for q in queries])
    names,idx=np.unique(groups,return_inverse=True); rng=np.random.default_rng(20261004)
    draws=rng.integers(0,len(names),(2000,len(names)))
    rows=[]
    for (fraction,stratum),sub in frame.groupby(['fraction','stratum']):
        pivot=sub.pivot(index='query',columns='method',values='concordance').reindex(queries)
        for method in pivot.columns:
            v=pivot[method].to_numpy(); reference=pivot['chemistry'].to_numpy()
            for comparison,values in [('absolute',v),('minus_chemistry',v-reference)]:
                valid=np.isfinite(values); sums=np.bincount(idx,weights=np.nan_to_num(values),minlength=len(names));counts=np.bincount(idx,weights=valid,minlength=len(names))
                denominator=counts[draws].sum(axis=1);boot=np.divide(sums[draws].sum(axis=1),denominator,out=np.full(len(draws),np.nan),where=denominator>0)
                lo,hi=np.nanpercentile(boot,[2.5,97.5])
                rows.append(dict(fraction=fraction,stratum=stratum,method=method,comparison=comparison,mean=np.nanmean(values),low=lo,high=hi,queries=int(valid.sum()),groups=int(np.unique(idx[valid]).size)))
    return pd.DataFrame(rows)

def main():
    use_style(); derived=[]
    for policy in ['source_representative','assay_envelope']:
        base=ROOT/'results/label_value/SPD'/policy/'new_molecule_known_targets'
        frame=pd.read_csv(base/'known_query_metrics.csv.gz',usecols=['fraction','stratum','method','query','concordance'])
        frame=frame[frame.stratum.isin(['all_determinate','both_exact'])]
        input_=np.load(ROOT/'results/label_value/prepared'/('spd_'+policy+'_inputs.npz'))
        features=np.load(ROOT/'results/label_value/prepared'/('SPD_'+policy+'_features.npz'))
        groupmap=dict(zip(input_['molecule_names'],features['scaffold_groups']))
        stats=intervals(frame,groupmap); stats['policy']=policy;derived.append(stats)
        fig,axes=plt.subplots(2,2,figsize=(WIDE,6.6))
        selected=[('chemistry','Chemistry', '#595959'),('plus__nesso_affinity','+ Nesso affinity',PALETTE['blue']),('plus__boltz_binary','+ Boltz-2 binder',PALETTE['orange'])]
        for c,stratum in enumerate(['all_determinate','both_exact']):
            ax=axes[0,c]
            for method,label,color in selected:
                sub=stats[(stats.method==method)&(stats.stratum==stratum)&(stats.comparison=='absolute')].sort_values('fraction')
                x=np.arange(4); ax.plot(x,100*sub['mean'],marker='o',label=label,color=color)
                ax.fill_between(x,100*sub.low,100*sub.high,color=color,alpha=.10,lw=0)
            ax.set_xticks(range(4),['⅛','¼','½','1']);ax.set_xlabel('Fraction of training\nscaffold groups');ax.set_ylabel('Target-order concordance (%)')
            ax.set_ylim(45,85);ax.text(0,1.04,('a  All determinable pairs' if c==0 else 'b  Two exact measurements'),transform=ax.transAxes,weight='bold')
        axes[0,0].legend(loc='lower right',fontsize=7)
        for c,fraction in enumerate([.25,1.]):
            ax=axes[1,c];ax.axvline(0,color='#999999',lw=.8)
            for j,(method,label) in enumerate(METHODS.items()):
                for k,(stratum,color) in enumerate([('all_determinate',PALETTE['blue']),('both_exact',PALETTE['orange'])]):
                    s=stats[(stats.method=='plus__'+method)&(stats.stratum==stratum)&(stats.fraction==fraction)&(stats.comparison=='minus_chemistry')].iloc[0]
                    y=len(METHODS)-1-j + (.13 if k==0 else -.13)
                    ax.plot([100*s.low,100*s.high],[y,y],color=color,lw=.9)
                    ax.plot(100*s['mean'],y,'o',color=color,ms=2.5)
            ax.set_yticks(range(len(METHODS)),list(METHODS.values())[::-1] if c==0 else [])
            ax.set_ylabel('Added output' if c==0 else ' ')
            ax.set_ylim(-.6,len(METHODS)-.4);ax.grid(axis='y',visible=False);ax.grid(axis='x',visible=True)
            ax.set_xlabel('Gain over chemistry (percentage points)');ax.set_xlim(-9,12)
            ax.text(0,1.04,('c  Quarter training budget' if c==0 else 'd  Full training budget'),transform=ax.transAxes,weight='bold')
        axes[1,1].plot([],[],color=PALETTE['blue'],marker='o',label='All pairs');axes[1,1].plot([],[],color=PALETTE['orange'],marker='o',label='Two exact');axes[1,1].legend(loc='lower right',fontsize=7)
        fig.subplots_adjust(left=.245,right=.985,top=.94,bottom=.08,wspace=.32,hspace=.43)
        save(fig, ROOT/'figs'/('label_value_'+policy),close=False);fig.savefig(ROOT/'figs'/('label_value_'+policy+'.png'),dpi=180);plt.close(fig)
    pd.concat(derived).to_csv(ROOT/'results/label_value/scaffold_bootstrap.csv',index=False)
    (ROOT/'figs/label_value_caption.txt').write_text('Local labels change the incremental value of a native score. Top: chemistry and chemistry augmented by one output, across nested training-scaffold fractions. Bottom: every output added separately to the same chemistry model, with the change in molecule-macro target ordering at quarter and full budgets. Blue denotes all interval-determinate pairs; orange denotes two-exact pairs. Lines and whiskers are conditional 95% percentile intervals from 2,000 paired scaffold-group bootstrap samples of fixed five-fold predictions, without refitting; they do not include split, training or inference variation. Representative-assay and envelope policies are shown in separate figures.\n')
if __name__=='__main__':main()
