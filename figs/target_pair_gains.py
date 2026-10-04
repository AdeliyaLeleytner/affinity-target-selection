"""All-output SPD transfer contrasts against the reference and native scoring."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import use_style,save,PALETTE,WIDE
from label_value import METHODS
ROOT=Path(__file__).resolve().parents[1]

def main():
 use_style();data=pd.read_csv(ROOT/'results/target_pairs/audit/primary_comparison.csv')
 fig,axes=plt.subplots(2,2,figsize=(WIDE,6.5))
 for row,policy in enumerate(['source_representative','assay_envelope']):
  for col,anchor in enumerate(['baseline','native']):
   ax=axes[row,col];ax.axvline(0,color='#888888',lw=.8)
   for j,(method,label) in enumerate(METHODS.items()):
    for k,(stratum,color) in enumerate([('all_determinate',PALETTE['blue']),('both_exact',PALETTE['orange'])]):
     s=data[(data.policy==policy)&(data.output==method)&(data.stratum==stratum)].iloc[0];y=11-j+(.13 if k==0 else -.13)
     ax.plot([s[f'gain_vs_{anchor}_low_pp'],s[f'gain_vs_{anchor}_high_pp']],[y,y],color=color,lw=1)
     ax.plot(s[f'gain_vs_{anchor}_pp'],y,'o',color=color,ms=3)
   ax.set_yticks(range(12),list(METHODS.values())[::-1] if col==0 else []);ax.set_ylabel(' ' if col else '')
   ax.set_ylim(-.6,11.6);ax.grid(axis='y',visible=False);ax.grid(axis='x',visible=True)
   low=data[f'gain_vs_{anchor}_low_pp'].min();high=data[f'gain_vs_{anchor}_high_pp'].max();ax.set_xlim(np.floor(low-1),np.ceil(high+1))
   ax.set_xlabel('Gain over '+('chemistry + sequence' if col==0 else 'the same native output')+'\n(percentage points)')
   ax.text(0,1.05,f'{chr(97+row*2+col)}  '+('Representative policy' if row==0 else 'Assay envelope'),transform=ax.transAxes,weight='bold')
 for label,color in [('All determinable pairs',PALETTE['blue']),('Two exact measurements',PALETTE['orange'])]:axes[0,1].plot([],[],color=color,marker='o',label=label)
 fig.legend(*axes[0,1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.61,.012),ncol=1,fontsize=9)
 fig.subplots_adjust(left=.25,right=.985,bottom=.17,top=.94,wspace=.29,hspace=.40)
 save(fig,ROOT/'figs/target_pair_gains',close=False);fig.savefig(ROOT/'figs/target_pair_gains.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
