"""All-panel testing budgets on the subset with an interval-certified best target."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import use_style,save,PALETTE,WIDE
ROOT=Path(__file__).resolve().parents[1]

def main():
 use_style();d=pd.read_csv(ROOT/'results/decision_curves/curves.csv')
 methods=[('chemistry','Chemistry','#595959','-'),('plus__nesso_affinity','+ Nesso affinity',PALETTE['blue'],'-'),('plus__boltz_binary','+ Boltz-2 binder',PALETTE['orange'],'-'),('native__boltz_binary','Native Boltz-2 binder',PALETTE['orange'],'--'),('native__nesso_affinity','Native Nesso affinity',PALETTE['blue'],'--')]
 fig,axes=plt.subplots(2,2,figsize=(WIDE,5.6),sharex=True,sharey=True)
 for row,policy in enumerate(['source_representative','assay_envelope']):
  for col,budget in enumerate([.25,1.]):
   ax=axes[row,col]
   for method,label,color,ls in methods:
    sub=d[(d.policy==policy)&(d.method==method)&(d.local_label_fraction==(0 if method.startswith('native__') else budget))].sort_values('assay_budget_k')
    ax.plot(sub.assay_budget_k,100*sub.certified_macro_expected_recovery,label=label,color=color,ls=ls,marker=None)
    if method=='chemistry':ax.fill_between(sub.assay_budget_k,100*sub.bootstrap_recovery_p025,100*sub.bootstrap_recovery_p975,color=color,alpha=.13,lw=0)
   ax.set_xticks([1,3,5,7,10]);ax.set_ylim(0,103);ax.set_xlim(1,10)
   ax.set_xlabel('Targets tested per molecule (k)');ax.set_ylabel('Certified best-target recovery (%)')
   n=int(sub.certified_molecules.iloc[0]);ax.text(0,1.05,f'{chr(97+row*2+col)}  '+('Representative' if row==0 else 'Assay envelope')+f', n={n}\n'+('Quarter' if col==0 else 'Full')+' training budget',transform=ax.transAxes,weight='bold')
 axes[0,0].legend(loc='lower right',fontsize=6.5)
 fig.subplots_adjust(left=.11,right=.98,top=.88,bottom=.11,hspace=.52,wspace=.28)
 save(fig,ROOT/'figs/decision_curves',close=False);fig.savefig(ROOT/'figs/decision_curves.png',dpi=180);plt.close(fig)
 (ROOT/'figs/decision_curves_caption.txt').write_text('Pair ordering and recovery of the best panel target are distinct decisions. Curves show the expected fraction of molecules whose top k predictions include at least one target certified best by the reported intervals, with exact prediction ties averaged uniformly. Every panel size k=1,...,10 is shown. The source-representative policy certifies205of909score-complete molecules; the assay envelope certifies198. Solid curves use the same nested scaffold-training budgets; dashed curves are fixed native outputs with no local-label fitting. Grey bands are conditional95%scaffold-bootstrap intervals for chemistry; all paired differences and method intervals are in Source Data. This retrospectively certifiable subset does not identify performance for unresolved profiles or establish an adaptive assay stopping rule.\n')
if __name__=='__main__':main()
