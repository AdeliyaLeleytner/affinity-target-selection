"""Plot head differences from complete, saved per-molecule contributions."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import PALETTE, WIDE, save, use_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRATA = ["all_determinate", "both_exact", "one_bounded"]
LABELS = ["All determinable pairs", "Two exact measurements", "One bounded measurement"]
DRAW_COUNT = 5000
SEED = 20261004


def main():
    data = pd.read_csv(ROOT / "results/native_baseline/query_contributions.csv.gz")
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.8), sharex=True, sharey=True)
    output = []
    specs = [("SPD", "source_representative", "boltz_binary"),
             ("DAVIS", "all_source_molecules", "boltz_binder")]
    for column, (dataset, policy, binder) in enumerate(specs):
        frame = data[(data.dataset == dataset) & (data.policy == policy) & (data.axis == "target")]
        query_ids = sorted(frame["query"].unique())
        draws = np.random.default_rng(SEED).integers(0, len(query_ids), (DRAW_COUNT, len(query_ids)))
        ax = axes[column]
        ax.axvline(0, color="#8A8A8A", linewidth=0.8, zorder=1)
        for index, stratum in enumerate(STRATA):
            values = frame[frame.stratum == stratum].pivot(index="query", columns="method", values="concordance").reindex(query_ids)
            difference = (values[binder] - values["boltz_affinity"]).to_numpy()
            finite = np.isfinite(difference)
            sampled = difference[draws]
            n = np.isfinite(sampled).sum(axis=1)
            boot = np.divide(np.nansum(sampled, axis=1), n,
                             out=np.full(DRAW_COUNT, np.nan), where=n > 0)
            mean = np.nanmean(difference) * 100
            lo, hi = np.nanpercentile(boot, [2.5, 97.5]) * 100
            color = PALETTE["blue"] if column == 0 else PALETTE["orange"]
            y = 2 - index
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6)
            ax.plot(mean, y, "o", color=color, markersize=5)
            ax.annotate(f"{mean:+.1f} pp", (mean, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8)
            output.append({"dataset": dataset, "policy": policy, "stratum": stratum,
                           "molecules": int(finite.sum()), "difference_pp": mean,
                           "bootstrap_low_pp": lo, "bootstrap_high_pp": hi})
        ax.set_yticks([2, 1, 0], LABELS)
        ax.set_ylim(-0.5, 2.65)
        ax.set_xlabel("Binder − affinity concordance (pp)")
        ax.set_ylabel("Comparison type" if column == 0 else " ")
        ax.text(0, 1.03, f"{'a' if column == 0 else 'b'}  {dataset}", transform=ax.transAxes,
                fontweight="bold", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xticks([-5, 0, 5, 10, 15])
        ax.set_xlim(-6, 17)
    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.22, top=0.86, wspace=0.18)
    save(fig, HERE / "native_head_comparison")
    fig.savefig(HERE / "native_head_comparison.png", dpi=180)
    (HERE / "native_head_comparison_source.json").write_text(json.dumps({
        "estimator": "Paired difference in mean per-molecule target-order concordance",
        "bootstrap": {"draws": DRAW_COUNT, "seed": SEED, "unit": "molecule", "interval": [2.5, 97.5],
                      "scope": "Conditional on fixed target panels and cached predictions; excludes training and inference variation."},
        "rows": output}, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
