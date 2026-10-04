"""Generate supplementary comparison tables directly from released metrics."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SCORES={'boltz_affinity':'Boltz-2 affinity','boltz_binary':'Boltz-2 binder','boltz_binder':'Boltz-2 binder','nesso_affinity':'Nesso affinity','nesso_binary':'Nesso binder','gnina':'GNINA on Vina pose','raw_ad4':'AutoDock4','raw_gnina_own':'GNINA own search','vina':'Vina','boltzina_vina_far_affinity':'Boltzina far affinity','boltzina_vina_far_binary':'Boltzina far binder','boltzina_vina_zero_affinity':'Boltzina zero affinity','boltzina_vina_zero_binary':'Boltzina zero binder'}
OTHER={'chemistry':'Chemistry','baseline':'Chemistry + sequence','target_prior':'Target prior','representation_readout':'Representation','representation_plus_chemistry':'Representation + chemistry','representation_pca4_readout':'Representation, four PCs','scalar_members_readout':'Four scalar members'}

def label(method):
 if method in OTHER:return OTHER[method]
 for prefix,name in [('native__','Native '),('calibrated__','Calibrated '),('availability__','Mask: '),('plus__','Augment: '),('mean__','Profile mean: '),('mean_relative__','Mean + relative: ')]:
  if method.startswith(prefix):return name+SCORES.get(method[len(prefix):],method[len(prefix):].replace('_',' '))
 return SCORES.get(method,method.replace('_',' '))

def table(caption, headers, rows):
 spec='l'+'r'*(len(headers)-1)
 s=[r'\small',r'\begin{longtable}{'+spec+'}',r'\caption{'+caption+r'}\\',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',r'\endfirsthead',r'\caption[]{Continued.}\\',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',r'\endhead']
 s += [' & '.join(map(str,row))+(r'\\*' if str(row[0]).startswith(('Calibrated ', 'Mask: ')) else r'\\') for row in rows]
 s += [r'\bottomrule',r'\end{longtable}',r'\normalsize']
 return '\n'.join(s)

def fmt(value):return '--' if pd.isna(value) else f'{100*value:.1f}'

def main():
 parts=[r'''\documentclass[pdflatex,sn-nature]{sn-jnl}
\usepackage{graphicx,booktabs,longtable,amsmath}
\newcommand{\widefigure}[1]{\makebox[\linewidth][c]{\includegraphics[width=160mm]{#1}}}
\raggedbottom
\begin{document}
\title{Supplementary information: Local labels and measurement resolution shape the value of affinity scores}
\author{\fnm{Anonymous}\sur{Authors}}
\maketitle
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\section*{Reading the comparisons}
Every table is generated from the released source-data CSVs. Concordance values are percentages; no statistical threshold defines a winner. Native outputs receive no local fitting, while calibrated and augmented outputs use the stated local labels. Molecule-macro and comparison-pooled estimators answer different weighting questions. Assay bounds, unresolved pairs and missing observations are preserved. All-budget query distributions, bootstrap contrasts and fitting diagnostics accompany the tables in machine-readable form.
''']
 budgets=pd.read_csv(ROOT/'results/label_value/actual_label_budgets.csv')
 rows=[]
 for _,r in budgets.iterrows():
  def span(k):
   a,b=int(r[k+'_min']),int(r[k+'_max']);return str(a) if a==b else f'{a}--{b}'
  rows.append(['Representative' if r.policy=='source_representative' else 'Envelope',f'{r.fraction:g}',span('training_groups'),span('training_molecules'),span('known_training_labels'),span('exact_training_labels')])
 parts.append(table('Actual SPD local training budgets. Ranges span five outer folds. Fraction refers to eligible scaffold groups, not a fraction of individual cells.',['Policy','Fraction','Groups','Molecules','Measured','Exact'],rows))
 parts.append(r'\clearpage\section*{Native output leaderboards}')
 native=pd.read_csv(ROOT/'results/native_baseline/native_summary.csv')
 for policy in ['source_representative','assay_envelope']:
  d=native[(native.dataset=='SPD')&(native.policy==policy)];rows=[]
  for method in SCORES:
   if method=='boltz_binder':continue
   sub=d[d.method==method]
   def val(axis,stratum):return fmt(sub[(sub.axis==axis)&(sub.stratum==stratum)].iloc[0]['macro'])
   rows.append([label(method),val('ligand','all_determinate'),val('ligand','both_exact'),val('target','all_determinate'),val('target','both_exact')])
  parts.append(table('SPD native concordance: '+policy.replace('_',' ')+'. Common finite-native support.',['Output','Ligands','Ligands exact','Targets','Targets exact'],rows))
 parts.append(r'\clearpage\section*{Local score calibration and augmentation}')
 for policy in ['source_representative','assay_envelope']:
  d=pd.read_csv(ROOT/'results/label_value/SPD'/policy/'new_molecule_known_targets/known_aggregate_metrics.csv');rows=[]
  for method in d.method.drop_duplicates():
   if method.startswith('native__'):continue
   sub=d[d.method==method]
   values=[fmt(sub[(sub.fraction==fraction)&(sub.stratum==stratum)].iloc[0].macro_concordance) for stratum in ['all_determinate','both_exact'] for fraction in [.25,1.]]
   rows.append([label(method),*values])
  if policy != 'source_representative': parts.append(r'\clearpage')
  parts.append(table('SPD fitted concordance: '+policy.replace('_',' ')+'. Quarter and full nested scaffold-group budgets. Native rows are in the preceding table.',['Arm',r'All, $1/4$','All, full',r'Exact, $1/4$','Exact, full'],rows))
 parts.append(r'''\clearpage
\begin{figure}[p]\centering\widefigure{figs/label_value_assay_envelope.pdf}
\caption{\textbf{The assay-envelope sensitivity retains the local-data pattern.} Same model, partition, metric and conditional scaffold-bootstrap construction as the representative-policy figure in the main article. The envelope changes measurement support as well as values:377 molecules supply determinable target pairs, including252 with two-exact pairs.}\end{figure}
\clearpage
\begin{figure}[p]\centering\widefigure{figs/score_component_mean.pdf}
\caption{\textbf{The profile mean alone does not retain the full augmentation benefit.} Mean-only augmentation minus the corresponding availability-only control, with identical local labels and chemistry features. Both policies, all budgets and both measurement strata are retained. Whiskers are conditional95\% paired scaffold-bootstrap intervals, complementing the mean-plus-relative contrast in the main article.}\end{figure}
\clearpage
\section*{DAVIS readout controls}
''')
 for regime,kind in [('new_targets_known_molecules','cold'),('new_molecule_known_targets','known')]:
  path=ROOT/'results/label_value/DAVIS/exact_identity'/regime/(kind+'_aggregate_metrics.csv')
  if not path.exists():continue
  d=pd.read_csv(path);metric='macro' if kind=='cold' else 'macro_concordance';rows=[]
  for method in d.method.drop_duplicates():
   sub=d[d.method==method]
   values=[fmt(sub[(sub.fraction==fraction)&(sub.stratum==stratum)].iloc[0][metric]) for stratum in ['all_determinate','both_exact'] for fraction in [.25,1.]]
   rows.append([label(method),*values])
  parts.append(table('DAVIS '+regime.replace('_',' ')+'. Identity-matched47-molecule cohort; held-target comparisons remain within the same model context.',['Arm',r'All, $1/4$','All, full',r'Exact, $1/4$','Exact, full'],rows))
 parts.append(r'''\clearpage
\begin{figure}[p]\centering\widefigure{figs/target_pair_gains.pdf}
\caption{\textbf{The comparison anchor changes the interpretation of transfer gains.} All 45 held-target-pair fits use the remaining eight targets, and target comparisons never combine different model contexts. Each native output is augmented separately. Left: increments over chemistry plus sequence. Right: increments over the same native score. Whiskers are conditional 95\% paired whole-scaffold bootstrap intervals; both measurement policies and strata are shown. These are fixed-panel local transfer results, not a pretraining-exclusion test.}\end{figure}
\clearpage\section*{Exhaustive SPD target-pair transfer}''')
 for policy in ['source_representative','assay_envelope']:
  path=ROOT/'results/target_pairs/SPD'/policy/'new_targets_known_molecules/cold_aggregate_metrics.csv'
  if not path.exists():continue
  d=pd.read_csv(path);rows=[]
  for method in d.method.drop_duplicates():
   sub=d[d.method==method];a=sub[sub.stratum=='all_determinate'].iloc[0];e=sub[sub.stratum=='both_exact'].iloc[0]
   rows.append([label(method),fmt(a['macro']),fmt(e['macro']),int(a['queries']),int(a['pairs'])])
  parts.append(table('All45 target pairs withheld separately: '+policy.replace('_',' ')+'. Each fit uses the remaining eight targets. Pair credits are pooled within a molecule before averaging molecules.',['Arm','All','Exact','Molecules','Pairs'],rows))
 parts.append(r'''\section*{Reproducibility and provenance}
Raw measurement sources, dataset identities, prediction channels, versions, acquisition settings and available model/source hashes are documented in \texttt{data/provenance/}. The released prediction arrays permit metric reproduction without native inference. \texttt{scripts/replay\_results.py} independently enumerates ordered pairs and checks aggregate concordance; numerical diagnostics are retained for every fitted model. Original native inference and local downstream fitting are separate computational costs. The configurations specify technical hyperparameters, not scientific effect-size or success criteria.
\end{document}
''')
 text='\n\n'.join(parts)
 for a,b in [('Identity-matched47','Identity-matched 47'),('All45','All 45'),('values:377','values: 377'),('including252','including 252'),('conditional95','conditional 95')]:text=text.replace(a,b)
 (ROOT/'supplementary.tex').write_text(text)
 print(ROOT/'supplementary.tex')

if __name__=='__main__':main()
