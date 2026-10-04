"""DAVIS local target-transfer performance and the effect of averaging units."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_style import use_style,save,PALETTE,WIDE
ROOT=Path(__file__).resolve().parents[1]

def main():
 use_style();root=ROOT/'results/davis_cold_audit';d=pd.read_csv(root/'learning_curves_bootstrap.csv');pooled=pd.read_csv(root/'pooled_pair_weighted_bootstrap.csv')
 methods=[('baseline','Chemistry + sequence','#595959'),('plus__boltz_binder','+ Boltz-2 binder',PALETTE['orange']),('representation_plus_chemistry','+ Interaction representation',PALETTE['blue'])]
 fig=plt.figure(figsize=(WIDE,5.7));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1]);axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];bottom=fig.add_subplot(gs[1,:])
 for c,stratum in enumerate(['all_determinate','both_exact']):
  ax=axes[c]
  for method,label,color in methods:
   s=d[(d.method==method)&(d.stratum==stratum)&(d.contrast=='absolute')].sort_values('fraction');x=np.arange(len(s))
   ax.plot(x,100*s.estimate,color=color,label=label,marker='o');ax.fill_between(x,100*s.bootstrap_low,100*s.bootstrap_high,color=color,alpha=.10,lw=0)
  ax.set_xticks(range(4),['⅛','¼','½','1']);ax.set_xlabel('Fraction of training\nprotein groups');ax.set_ylabel('Target-order concordance (%)');ax.set_ylim(45,92)
  ax.text(0,1.04,'a  All determinable pairs' if c==0 else 'b  Two exact measurements',transform=ax.transAxes,weight='bold')
 axes[0].legend(loc='lower right',fontsize=6.5)
 bottom.axvline(0,color='#999999',lw=.8)
 candidates=[('plus__boltz_affinity','Affinity score'),('plus__boltz_binder','Binder score'),('representation_readout','Representation alone'),('representation_plus_chemistry','Representation + chemistry')]
 for j,(method,label) in enumerate(candidates):
  for k,(table,color,category) in enumerate([(d,PALETTE['blue'],'Equal molecule weight'),(pooled,PALETTE['orange'],'Equal comparison weight')]):
   s=table[(table.method==method)&(table.stratum=='both_exact')&(table.fraction==1)&(table.contrast=='versus_baseline')].iloc[0];y=3-j+(.12 if k==0 else -.12)
   bottom.plot([100*s.bootstrap_low,100*s.bootstrap_high],[y,y],color=color,lw=1.3);bottom.plot(100*s.estimate,y,'o',color=color,ms=4,label=category if j==0 else None)
 bottom.set_yticks(range(4),[x[1] for x in candidates][::-1]);bottom.set_ylim(-.6,3.65);bottom.set_xlabel('Gain over chemistry + sequence (percentage points)');bottom.set_ylabel('');bottom.grid(axis='y',visible=False);bottom.grid(axis='x',visible=True)
 bottom.text(0,1.07,'c  Same quantitative comparisons, different averaging questions',transform=bottom.transAxes,weight='bold');fig.legend(*bottom.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.62,.015),ncol=2,fontsize=7)
 fig.subplots_adjust(left=.25,right=.985,bottom=.16,top=.94,wspace=.33,hspace=.50)
 save(fig,ROOT/'figs/target_transfer',close=False);fig.savefig(ROOT/'figs/target_transfer.png',dpi=180);plt.close(fig)
 (ROOT/'figs/target_transfer_caption.txt').write_text('Locally unseen targets retain score information, but averaging changes the quantitative conclusion. DAVIS uses47identity-matched molecules and434sequence groups defining442constructs. Top: five-fold target-transfer curves for all determinable or two-exact pairs; targets are compared only within the same held-out fold. Bottom: full-budget increments on the identical two-exact comparisons under equal molecule versus equal comparison weighting. Points and bands/whiskers are conditional95%intervals from5,000paired molecule-bootstrap resamples, without refitting or target resampling.\n')
if __name__=='__main__':main()
