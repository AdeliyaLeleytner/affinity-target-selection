"""Compact kinase-panel readouts and the matched four-scalar control."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import use_style,save,PALETTE,WIDE
ROOT=Path(__file__).resolve().parents[1]

def main():
 use_style();path=ROOT/'results/davis_known_audit';d=pd.read_csv(path/'learning_curves.csv');c=pd.read_csv(path/'paired_contrasts.csv')
 d=d[d.bootstrap_unit=='scaffold'];c=c[c.bootstrap_unit=='scaffold']
 fig,axes=plt.subplots(2,2,figsize=(WIDE,5.5),sharex=True)
 methods=[('chemistry','Chemistry','#595959','-'),('native__boltz_binder','Native binder',PALETTE['orange'],'--'),('scalar_members_readout','Four scalar members',PALETTE['orange'],'-'),('representation_pca4_readout','Four representation PCs',PALETTE['blue'],'-')]
 for col,stratum in enumerate(['all_determinate','both_exact']):
  ax=axes[0,col]
  for method,label,color,ls in methods:
   sub=d[(d.method==method)&(d.stratum==stratum)].sort_values('fraction');x=np.arange(4)
   ax.plot(x,100*sub.macro_concordance,label=label,color=color,ls=ls,marker='o' if not method.startswith('native__') else None)
   if method=='representation_pca4_readout':ax.fill_between(x,100*sub.macro_ci_low,100*sub.macro_ci_high,color=color,alpha=.12,lw=0)
  ax.set_ylim(48,92);ax.set_ylabel('Target-order concordance (%)');ax.text(0,1.05,('a  All determinable pairs' if col==0 else 'b  Two exact measurements'),transform=ax.transAxes,weight='bold')
  ax=axes[1,col];sub=c[(c.candidate=='representation_pca4_readout')&(c.reference=='scalar_members_readout')&(c.stratum==stratum)].sort_values('fraction')
  y=100*sub.macro_difference.to_numpy();lo=100*sub.macro_ci_low.to_numpy();hi=100*sub.macro_ci_high.to_numpy();x=np.arange(4)
  ax.axhline(0,color='#888888',lw=.8);ax.errorbar(x,y,yerr=np.stack([y-lo,hi-y]),fmt='o-',capsize=3,color=PALETTE['blue'])
  bounds=c[(c.candidate=='representation_pca4_readout')&(c.reference=='scalar_members_readout')]
  ax.set_ylim(np.floor(100*bounds.macro_ci_low.min()-.3),np.ceil(100*bounds.macro_ci_high.max()+.3));ax.set_ylabel('Gain over four scalar members (pp)');ax.text(0,1.05,('c  Matched dimension' if col==0 else 'd  Matched dimension'),transform=ax.transAxes,weight='bold')
  ax.set_xticks(range(4),['⅛','¼','½','1']);ax.set_xlabel('Fraction of training\nscaffold groups')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.57),ncol=2,fontsize=9)
 fig.subplots_adjust(left=.13,right=.98,top=.93,bottom=.13,wspace=.33,hspace=.57)
 save(fig,ROOT/'figs/cached_readouts',close=False);fig.savefig(ROOT/'figs/cached_readouts.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
