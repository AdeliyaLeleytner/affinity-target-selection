"""Graphical abstract. Run here, or copy into final figs/ and run there.

Inputs use the final project layout: figs/structure, figs/native_* and results/.
Only the structural image is raster; plots, labels and lines remain vector.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
parser = argparse.ArgumentParser()
parser.add_argument("--project-root", type=Path, default=HERE.parent if HERE.name == "figs" else HERE)
parser.add_argument("--learning-table", type=Path, default=Path("results/label_value/SPD/source_representative/new_molecule_known_targets/known_aggregate_metrics.csv"))
args = parser.parse_args()
root = args.project_root.resolve()
sys.path.insert(0, str(root / "figs"))
from plot_style import use_style, save, PALETTE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

use_style()
plt.rcParams.update({"font.size": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8.5, "lines.linewidth": 1.45})
blue, orange, grey, ink = PALETTE["blue"], PALETTE["orange"], "#747D86", "#26333C"
paths = {
    "native": root / "figs/native_head_comparison_source.json",
    "learning": root / args.learning_table,
    "structure": root / "figs/structure/nr3c1_dexamethasone_full_transparent.png",
    "structure_provenance": root / "figs/structure/provenance.json",
}
native = json.loads(paths["native"].read_text())
with paths["learning"].open() as handle:
    learning = list(csv.DictReader(handle))
native_rows = [r for r in native["rows"] if r["stratum"] in ("all_determinate", "both_exact")]
curves = {}
for method in ("chemistry", "plus__nesso_affinity"):
    rows = sorted((r for r in learning if r["method"] == method and r["stratum"] == "all_determinate"),
                  key=lambda r: float(r["fraction"]))
    assert len(rows) == 4 and all(r["policy"] == "source_representative" for r in rows)
    curves[method] = {"budget_percent": [100 * float(r["fraction"]) for r in rows],
                      "concordance_percent": [100 * float(r["macro_concordance"]) for r in rows],
                      "evaluable_molecules": [int(r["evaluable_queries"]) for r in rows]}
assert curves["chemistry"]["budget_percent"] == curves["plus__nesso_affinity"]["budget_percent"]
assert curves["chemistry"]["evaluable_molecules"] == curves["plus__nesso_affinity"]["evaluable_molecules"]

fig = plt.figure(figsize=(6.75, 3.55), facecolor="white")
fig.text(.035, .945, "When do affinity scores add value?", fontsize=14, weight="bold", color=ink)
fig.text(.035, .885, "Ranking targets for the same molecule", fontsize=10.5, color=grey)
fig.add_artist(Line2D([.035, .975], [.850, .850], transform=fig.transFigure, color="#D6DDE1", lw=.7))

headers = [(.035, "a  The molecular decision", "One molecule; compare proteins"),
           (.365, "b  Measurement resolution", "Native Boltz outputs"),
           (.715, "c  Local-label budget", "Locally fitted SPD models")]
for x, title, subtitle in headers:
    fig.text(x, .788, title, fontsize=10.3, weight="bold", color=ink)
    fig.text(x, .735, subtitle, fontsize=8.4, color=grey)

# Only transparent canvas padding is cropped. The whole source-rendered receptor
# and ligand are retained; no coordinates, chemical features or contacts change.
im = Image.open(paths["structure"]).convert("RGBA")
bbox = im.getbbox()
pad = 35
bounds = (max(0, bbox[0]-pad), max(0, bbox[1]-pad), min(im.width,bbox[2]+pad), min(im.height,bbox[3]+pad))
ax_image = fig.add_axes([.020, .335, .305, .355])
ax_image.imshow(im.crop(bounds)); ax_image.set_axis_off()
fig.text(.035, .252, "NR3C1–dexamethasone", fontsize=9.2, color=ink)
fig.text(.035, .197, "Experimental reference · 1M2Z", fontsize=8.4, color=grey)
fig.text(.035, .151, "Not a model prediction", fontsize=8.2, color=grey)

# Native scores: conditional bootstrap intervals are supplied by the verified
# source artifact. Filled/open symbols encode measurement support, not methods.
ax_native = fig.add_axes([.365, .335, .247, .350])
for row in native_rows:
    exact = row["stratum"] == "both_exact"
    y = (1 if row["dataset"] == "SPD" else 0) + (-.18 if exact else .18)
    v, lo, hi = (row[k] for k in ("difference_pp", "bootstrap_low_pp", "bootstrap_high_pp"))
    ax_native.errorbar(v, y, xerr=[[v-lo],[hi-v]], fmt="o", ms=4.8,
                       color=blue, mfc="white" if exact else blue, mec=blue,
                       capsize=2, elinewidth=.85, markeredgewidth=1.0, zorder=3)
    ax_native.annotate(f"{v:+.1f}", (v,y), xytext=(0,6),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=8.6, color=blue,
                       bbox=dict(facecolor="white", edgecolor="none", pad=.2))
ax_native.axvline(0, color="#AEB7BD", lw=.8, zorder=0)
ax_native.set(xlim=(-3,14.5), ylim=(-.67,1.67), xticks=[0,5,10], yticks=[0,1],
              yticklabels=["DAVIS","SPD"], xlabel="Binder − affinity (pp)")
ax_native.grid(False)
ax_native.spines["left"].set_visible(False)
ax_native.tick_params(axis="y", length=0, pad=4)
handles = [Line2D([],[],marker="o",ls="",color=blue,mfc=blue,ms=4.5,label="All determinate pairs"),
           Line2D([],[],marker="o",ls="",color=blue,mfc="white",ms=4.5,label="Two exact measurements")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.348,.228),
           fontsize=8.1, labelspacing=.55, borderpad=0, handletextpad=.4)

# Learning curves are descriptive OOF values from one fixed grouped partition.
# No uncertainty band is invented from the four label budgets.
ax_fit = fig.add_axes([.715, .335, .260, .350])
x = np.array(curves["chemistry"]["budget_percent"])
base = np.array(curves["chemistry"]["concordance_percent"])
plus = np.array(curves["plus__nesso_affinity"]["concordance_percent"])
ax_fit.plot(x, base, "s--", color=grey, ms=3.9, label="Chemistry")
ax_fit.plot(x, plus, "o-", color=orange, ms=4.0, label="Chemistry + Nesso affinity")
ax_fit.set(xlim=(3,110), ylim=(60,78), xticks=[12.5,25,50,100], yticks=[60,65,70,75],
           xticklabels=["12.5","25","50","100"], xlabel="Training-group budget (%)",
           ylabel="Target-order concordance (%)")
ax_fit.grid(axis="y", color="#E5E8EB", lw=.5)
ax_fit.get_xticklabels()[0].set_ha("right")
for index, offset in [(1,(6,-1)),(3,(-2,14))]:
    mid=(base[index]+plus[index])/2
    ax_fit.plot([x[index],x[index]],[base[index],plus[index]],color=orange,lw=.7)
    ax_fit.annotate(f"+{plus[index]-base[index]:.1f} pp", (x[index],mid),xytext=offset,
                    textcoords="offset points",fontsize=8.3,color="#8D6500",
                    ha="right" if index==3 else "left",va="center",
                    bbox=dict(facecolor="white",edgecolor="none",pad=.5))
fig.legend(handles=[Line2D([],[],color=orange,marker="o",ms=3.5,lw=1.4,label="Chemistry + Nesso affinity"),
                    Line2D([],[],color=grey,marker="s",ls="--",ms=3.5,lw=1.2,label="Chemistry")],
           loc="upper left",bbox_to_anchor=(.700,.228),fontsize=8.1,labelspacing=.55,
           borderpad=0,handlelength=1.1,handletextpad=.5)

fig.text(.035,.047,"Evaluate the added value at the relevant measurement resolution and label budget.",
         fontsize=9.2, color=ink)

source_ledger = {"figure_size_inches":[6.75,3.55], "native":native_rows, "native_uncertainty":native["bootstrap"],
                 "learning":curves,"learning_scope":"SPD source-representative policy; new molecules/known targets; 5-fold OOF, one fixed seeded scaffold-group partition. Curves are descriptive, not confidence bands.",
                 "structure_scope":"Experimental 1M2Z, NR3C1 chain A–DEX 301. Context illustration only, not a prediction or evidence of score mechanism.",
                 "structure_crop_pixels":list(bounds),
                 "inputs":{k:{"relative_path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}}
(HERE/"graphical_abstract_source.json").write_text(json.dumps(source_ledger,indent=2)+"\n")
save(fig, str(HERE/"graphical_abstract"), formats=("pdf","svg","png"))
