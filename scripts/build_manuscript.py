"""Build the result table, figure, and manuscript using relative project paths."""
from pathlib import Path
import csv
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "boltz_affinity": "Boltz-2 affinity", "boltz_binary": "Boltz-2 binder",
    "boltzina_vina_far_affinity": "Boltzina far affinity", "boltzina_vina_far_binary": "Boltzina far binder",
    "boltzina_vina_zero_affinity": "Boltzina zero affinity", "boltzina_vina_zero_binary": "Boltzina zero binder",
    "gnina": "GNINA on Vina pose", "nesso_affinity": "Nesso affinity", "nesso_binary": "Nesso binder",
    "raw_ad4": "AutoDock4", "raw_gnina_own": "GNINA own search", "vina": "Vina",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()
    with (ROOT / "results/native_baseline/native_summary.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["dataset"] == "SPD" and r["policy"] == "source_representative"]
    lookup = {(r["method"], r["axis"], r["stratum"]): float(r["macro"] or "nan") for r in rows}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Output & Ligand order & Target order & Target order, exact \\", r"\midrule"]
    for method, label in LABELS.items():
        values = [f"{100 * lookup[method, axis, stratum]:.1f}" for axis, stratum in
                  [("ligand", "all_determinate"), ("target", "all_determinate"), ("target", "both_exact")]]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "results/native_baseline/score_table.tex").write_text("\n".join(lines) + "\n")
    if not args.skip_figures:
        for script in ['native_head_comparison.py', 'label_value.py',
                       'graphical_abstract.py', 'target_transfer.py', 'cached_readouts.py', 'target_pair_gains.py',
                       'decision_curves.py', 'score_component_diagnostic.py']:
            subprocess.run([sys.executable, str(ROOT / 'figs' / script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_supplement.py')], cwd=ROOT, check=True)
    (ROOT / "build").mkdir(exist_ok=True)
    for name in ['paper', 'supplementary']:
        with (ROOT / 'build' / (name+'_build.log')).open('w') as log:
            for _ in range(3):
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name+'.tex'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(ROOT / "paper.pdf")


if __name__ == "__main__":
    main()
