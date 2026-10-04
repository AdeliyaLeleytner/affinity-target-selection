"""Publication figures for the saved score-component audit; no resampling/fits.

From the public project root:
    python figs/score_component_diagnostic.py
Default input: results/score_components/audit/paired_scaffold_increments.csv
Default output: figs/score_component_relative.{pdf,svg,png} and mean companion.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import BASELINE, PALETTE, WIDE, save, use_style


OUTPUTS = (("nesso_affinity", "Nesso affinity"), ("boltz_binary", "Boltz binder"))
STRATA = (("all_determinate", "All determinate pairs"), ("both_exact", "Two exact measurements"))
POLICIES = (("source_representative", "Source representative", "blue", "o", "-", -0.045),
            ("assay_envelope", "Assay envelope", "orange", "s", "--", 0.045))
BUDGETS = (0.125, 0.25, 0.5, 1.0)


def caption(contrast):
    if contrast == "mean_relative_minus_mean":
        lead = "Conditional gains from target-relative score variation vary with label budget and endpoint support."
        definition = "The contrast is mean-plus-relative minus mean-only, with the same chemistry baseline and original missingness indicator in both arms."
    else:
        lead = "The profile mean alone does not consistently improve target ordering beyond the availability control."
        definition = "The contrast is mean-only minus the matched availability control; both arms include the same chemistry baseline."
    return (lead + " Panels a and c show Nesso affinity; b and d show Boltz binder. "
            "Panels a and b include all interval-determinate target pairs; c and d include pairs with two exact measurements. "
            "Points are paired increments in mean per-molecule target-order concordance, in percentage points. "
            + definition + " Error bars are 2.5th–97.5th percentiles from 2,000 paired scaffold-group bootstrap resamples of fixed out-of-fold predictions. "
            "This uses one fixed five-fold allocation; the intervals do not represent retraining or seed variability and are not simultaneous bands. "
            "Representative/envelope support is 373/377 molecules (302/304 scaffold groups) for all determinate pairs, and 261/252 molecules (211/207 groups) for two exact measurements. "
            "Training budgets are fractions of the available outer-training scaffold groups. Symbols are displaced slightly horizontally for legibility; lines only connect the evaluated budgets. "
            "The contrast measures conditional predictive information, not a physical mechanism. The profile mean requires the original native score profile, so this diagnostic establishes no inference-cost saving.")


def build(table, contrast, output_root, y_limits, y_ticks):
    subset = table[(table.contrast == contrast) & table.output.isin([x[0] for x in OUTPUTS])]
    fig, axes = plt.subplots(2, 2, figsize=(WIDE, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.98, bottom=.15, top=.805, wspace=.16, hspace=.58)
    positions = np.arange(len(BUDGETS), dtype=float)
    rows = []
    for column, (output, label) in enumerate(OUTPUTS):
        for row, (stratum, stratum_label) in enumerate(STRATA):
            ax = axes[row, column]
            ax.axhline(0, color=BASELINE, linewidth=.85, linestyle=(0, (3, 2)), zorder=1)
            for policy, policy_label, color_name, marker, line, offset in POLICIES:
                frame = subset[(subset.policy == policy) & (subset.output == output) & (subset.stratum == stratum)]
                frame = frame.set_index("fraction").reindex(BUDGETS)
                required = ["paired_macro_increment", "bootstrap_macro_p025", "bootstrap_macro_p975"]
                if len(frame) != len(BUDGETS) or not np.isfinite(frame[required].to_numpy()).all():
                    raise ValueError(f"Incomplete source rows: {contrast}/{output}/{stratum}/{policy}")
                values, low, high = (frame[column_name].to_numpy() * 100 for column_name in required)
                if np.any(low > values) or np.any(high < values):
                    raise ValueError("Displayed percentile limits must contain the source point estimates")
                color = PALETTE[color_name]
                ax.errorbar(positions + offset, values, yerr=np.vstack((values - low, high - values)),
                            color=color, marker=marker, linestyle=line, linewidth=1.15,
                            markersize=4.5, markerfacecolor=color if marker == "o" else "white",
                            markeredgewidth=1.0, elinewidth=.85, capsize=2.3, capthick=.85, zorder=3)
                for fraction, (_, record) in zip(BUDGETS, frame.iterrows()):
                    rows.append({"contrast": contrast, "output": output, "stratum": stratum,
                                 "policy": policy, "fraction": fraction,
                                 "increment_pp": float(record.paired_macro_increment * 100),
                                 "bootstrap_low_pp": float(record.bootstrap_macro_p025 * 100),
                                 "bootstrap_high_pp": float(record.bootstrap_macro_p975 * 100),
                                 "molecules": int(record.queries), "scaffold_groups": int(record.scaffold_groups),
                                 "pairs": int(record.pairs)})
            letter = chr(ord("a") + row * 2 + column)
            ax.text(0, 1.035, f"{letter}  {stratum_label}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8.5, fontweight="bold")
            ax.set_xlim(-.25, 3.25)
            ax.set_ylim(*y_limits)
            ax.set_yticks(y_ticks)
            ax.set_xticks(positions, ["1/8", "1/4", "1/2", "1"])
            ax.grid(axis="x", visible=False)
            if column == 0:
                comparison = "(mean + relative) − mean" if contrast == "mean_relative_minus_mean" else "mean − availability"
                ax.set_ylabel("Concordance increment (pp)\n" + comparison, fontsize=8.5)
            if row == 1:
                ax.set_xlabel("Training scaffold-group fraction", fontsize=8.5)
        center = (axes[0, column].get_position().x0 + axes[0, column].get_position().x1) / 2
        fig.text(center, .97, label, ha="center", va="top", fontsize=10, fontweight="bold")
    handles = [Line2D([0], [0], color=PALETTE[c], marker=m, linestyle=line, markersize=4.5,
                      markerfacecolor=PALETTE[c] if m == "o" else "white", markeredgewidth=1.0,
                      label=label) for _, label, c, m, line, _ in POLICIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .925),
               ncol=2, fontsize=8.2, handlelength=2.3, columnspacing=2.2)
    fig.canvas.draw()
    sizes = [artist.get_fontsize() for artist in fig.findobj(mpl.text.Text)
             if artist.get_visible() and artist.get_text().strip()]
    if min(sizes) < 8:
        raise AssertionError("Printed figure contains text below 8 pt")
    stem = output_root / ("score_component_relative" if contrast == "mean_relative_minus_mean" else "score_component_mean")
    save(fig, str(stem), formats=("pdf", "svg"), close=False)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    stem.with_name(stem.name + "_caption.txt").write_text(caption(contrast) + "\n")
    return {"figure": stem.name, "width_inches": WIDE, "height_inches": 4.4,
            "minimum_font_pt": min(sizes), "shared_y_limits_pp": y_limits,
            "palette": {"source_representative": PALETTE["blue"], "assay_envelope": PALETTE["orange"]},
            "points": rows, "caption": caption(contrast)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("results/score_components/audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("figs"))
    args = parser.parse_args()
    source = args.audit_dir / "paired_scaffold_increments.csv"
    table = pd.read_csv(source)
    selected = table[table.output.isin([x[0] for x in OUTPUTS])]
    expected = len(OUTPUTS) * len(STRATA) * len(POLICIES) * len(BUDGETS) * 2
    if len(selected) != expected or selected.duplicated(["contrast", "output", "stratum", "policy", "fraction"]).any():
        raise ValueError("Source table has duplicate or missing figure rows")
    use_style()
    mpl.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8.2,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 8.5})
    # Both figures use the same scale, derived from all selected confidence
    # limits rather than different ranges that exaggerate one contrast.
    low = min(0., selected.bootstrap_macro_p025.min() * 100)
    high = max(0., selected.bootstrap_macro_p975.max() * 100)
    y_limits = [float(2 * np.floor((low - .5) / 2)), float(2 * np.ceil((high + .5) / 2))]
    y_ticks = np.arange(5 * np.ceil(y_limits[0] / 5), y_limits[1] + .01, 5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = [build(table, contrast, args.output_dir, y_limits, y_ticks)
               for contrast in ("mean_relative_minus_mean", "mean_minus_availability")]
    metadata = {"source_relative_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "estimator": "paired difference in macro mean per-molecule target-order concordance",
                "uncertainty": "2,000 paired scaffold-bootstrap replicates;2.5th and97.5th percentiles from existing CSV",
                "bootstrap_recomputed": False, "figures": figures}
    (args.output_dir / "score_component_figure_data.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figures": [x["figure"] for x in figures], "source_rows": len(selected),
                      "width_inches": WIDE, "minimum_font_pt": min(x["minimum_font_pt"] for x in figures),
                      "shared_y_limits_pp": y_limits, "bootstrap_recomputed": False}))


if __name__ == "__main__":
    main()
