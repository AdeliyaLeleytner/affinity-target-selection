"""Exploratory, inference-free SPD target-selection curves from frozen OOF files.

Requires Python 3.9+ and NumPy. No project imports, fitted models, external
services, activity thresholds, or training operations are used.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
import time

import numpy as np


POLICIES = ("source_representative", "assay_envelope")


def interval_best_sets(lower, upper, upper_open):
    """Return guaranteed-best G, possible-best P, and the missing-label mask.

    Missing bounds are replaced by [-inf,+inf]. Lower endpoints are closed.
    A finite L_j >= every other U_k certifies j as a maximizer, allowing ties.
    P contains targets not strictly dominated by another interval: L_k > U_j,
    or L_k == U_j when U_j is open, rules j out as a possible maximizer.
    """
    lower, upper = np.asarray(lower, float).copy(), np.asarray(upper, float).copy()
    opened = np.asarray(upper_open, bool).copy()
    if lower.ndim != 2 or lower.shape != upper.shape or lower.shape != opened.shape:
        raise ValueError("Interval arrays must share molecule-by-target axes")
    missing = np.isnan(lower) | np.isnan(upper)
    lower[missing], upper[missing], opened[missing] = -np.inf, np.inf, False
    if (np.any(lower > upper) or np.any((lower == upper) & opened)
            or np.isposinf(lower).any() or np.isneginf(upper).any()):
        raise ValueError("Empty or invalid assay interval")
    guaranteed, possible = np.zeros_like(opened), np.zeros_like(opened)
    for q, (lo, hi, op) in enumerate(zip(lower, upper, opened)):
        sufficient = lo[:, None] >= hi[None, :]
        np.fill_diagonal(sufficient, True)
        guaranteed[q] = np.isfinite(lo) & sufficient.all(axis=1)
        strict = ((lo[:, None] > hi[None, :])
                  | ((lo[:, None] == hi[None, :]) & op[None, :]))
        np.fill_diagonal(strict, False)
        possible[q] = ~strict.any(axis=0)
    if np.any(guaranteed & ~possible) or np.any(~possible.any(axis=1)):
        raise ValueError("Best-set construction violates interval consistency")
    return guaranteed, possible, missing


def hit_probabilities(scores, gold):
    """Exact top-k hit probabilities for k=1..T under uniform score-tie order."""
    scores, gold = np.asarray(scores, float), np.asarray(gold, bool)
    if scores.ndim != 1 or scores.shape != gold.shape or not np.isfinite(scores).all():
        raise ValueError("A finite full-panel score vector and matching gold mask are required")
    ordered = np.sort(scores)[::-1]
    result = np.zeros(len(scores), dtype=float)
    if not gold.any():
        return result
    for k, boundary in enumerate(ordered, 1):
        above, tied = scores > boundary, scores == boundary
        if (above & gold).any():
            result[k - 1] = 1.0
            continue
        size, good = int(tied.sum()), int((tied & gold).sum())
        slots = k - int(above.sum())
        misses = comb(size - good, slots) if slots <= size - good else 0
        result[k - 1] = 1.0 - misses / comb(size, slots)
    return result


def assays_spent_by_cap(recovery):
    """E[min(T_first_hit,k)] for k=1..T, from the complete hit CDF."""
    recovery = np.asarray(recovery, float)
    return np.arange(1, recovery.shape[-1] + 1) - np.concatenate(
        [np.zeros(recovery.shape[:-1] + (1,)), np.cumsum(recovery[..., :-1], axis=-1)], axis=-1)


def cluster_bootstrap_means(curves, groups, replicates, seed):
    """Paired scaffold resampling of fixed predictions, all variants and k coupled.

    ``curves`` is variant x eligible molecule x k. Each replicate samples the
    observed eligible scaffold groups with replacement and keeps every member
    of each sampled group. Ratio-of-sums preserves molecule-weighted means.
    """
    curves = np.asarray(curves, float)
    groups = np.asarray(groups, str)
    if curves.ndim != 3 or curves.shape[1] != len(groups):
        raise ValueError("Bootstrap molecule axes disagree")
    if replicates <= 0 or len(groups) == 0:
        return None
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique), curves.shape[0], curves.shape[2]), float)
    np.add.at(sums, inverse, curves.transpose(1, 0, 2))
    sizes = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=replicates)
    denominator = counts @ sizes
    means = (counts @ sums.reshape(len(unique), -1)) / denominator[:, None]
    return means.reshape(replicates, curves.shape[0], curves.shape[2])


def _read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path, records):
    if not records:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(_clean(row) for row in records)


def _interval(values):
    if values is None:
        return (None, None)
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def _log_provenance(path):
    if path is None:
        return {"log_supplied": False, "completed_blocks_checked_in": "known_summary.json"}
    evidence = []
    failures = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("COMPARISON_BLOCK "):
                entry = json.loads(line.split(" ", 1)[1])
                if entry.get("dataset") == "SPD" and entry.get("regime") == "new_molecule_known_targets":
                    result = entry["result"]
                    evidence.append({"line": line_number, "policy": entry["policy"],
                                     "status": result.get("status"), "fits": result.get("fits"),
                                     "finite_oof_cells": result.get("finite_oof_cells")})
            if "RuntimeError: Interval-regression numerical failure" in line:
                failures.append(line_number)
    if {e["policy"] for e in evidence if e["status"] == "complete"} != set(POLICIES):
        raise ValueError("Supplied log does not confirm both completed SPD known-target blocks")
    return {"log_supplied": True, "path": str(path), "sha256": _sha256(path),
            "completed_SPD_blocks": evidence, "overall_run_failure_lines": failures,
            "scope": "Completed SPD blocks only; the originating multi-block run was not globally successful."}


def analyze(input_root, output_root, bootstrap_replicates=2000, bootstrap_seed=20261004,
            run_id=None, source_revision=None, run_log=None):
    started = time.perf_counter()
    input_root, output_root = Path(input_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"analysis": "exploratory decision curves after completed pairwise analysis",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_root": str(input_root), "originating_run_id": run_id,
              "source_revision": source_revision,
              "originating_run_scope": "Completed input SPD blocks, with per-block completion checked.",
              "log_provenance": _log_provenance(run_log),
              "new_fits": 0, "new_inference": 0, "external_retrieval": 0,
              "endpoint": "Recovery of at least one certified maximizer of protocol-specific assay potency; no clinical-activity or thermodynamic-affinity claim.",
              "G": "Targets with finite lower bound at least every other target's upper bound; exact ties allowed.",
              "P": "Targets not strictly dominated by another assay interval, accounting for open upper bounds.",
              "missing_labels": "Either missing endpoint replaces the full interval with [-inf,+inf]; missing is never inactive.",
              "conditional_population": "Nonempty G and all 12 native methods finite at every one of the 10 targets.",
              "tie_rule": "Uniform permutations within exact prediction ties; hit=1 if gold above boundary, else 1-C(B-g,r)/C(B,r).",
              "cost": "E[min(T_first_G,k)] = k-sum(F_G(s),s=1..k-1); full expected first-G rank is the k=10 value.",
              "broad_bounds": "On all score-complete molecules, mean P(hit G) <= true maximum recovery <= mean P(hit P). These can be nonsharp and are not confidence intervals.",
              "k10_bound_caveat": "For G-empty molecules the requested set lower bound stays 0 at k=10, although testing the full finite panel necessarily includes a maximizer. The set bound is deliberately not sharpened.",
              "native_reference": "Native outputs are static zero-local-label references, included once at local_label_fraction=0; comparing them with multiple chemistry budgets does not refit them.",
              "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                            "unit": "eligible scaffold group, sampled with replacement; all methods and k share each resample",
                            "interval": "2.5th and 97.5th percentiles of paired bootstrap means/differences",
                            "limitations": "Conditional on fixed eligibility and fitted OOF predictions; no refitting, selection correction, simultaneous-band guarantee, p-values or success thresholds."},
              "inputs": [], "policies": {}, "all_variant_summaries": []}
    all_queries, all_curves, all_costs, all_differences = [], [], [], []
    probability_path = output_root / "per_query_probabilities.csv.gz"
    probability_fields = ["policy", "method", "kind", "local_label_fraction", "query_index",
                          "molecule", "outer_fold", "certified_eligible", "expected_assays_until_first_G"]
    probability_fields += [f"p_hit_G_k{k}" for k in range(1, 11)]
    probability_fields += [f"p_hit_P_k{k}" for k in range(1, 11)]
    probability_fields += [f"expected_assays_spent_until_G_or_cap_{k}" for k in range(1, 11)]
    with gzip.open(probability_path, "wt", newline="") as probability_handle:
        probability_writer = csv.DictWriter(probability_handle, fieldnames=probability_fields)
        probability_writer.writeheader()
        for policy_index, policy in enumerate(POLICIES):
            block = input_root / "SPD" / policy / "new_molecule_known_targets"
            paths = [input_root / "prepared" / f"spd_{policy}_inputs.npz",
                     block / "known_oof.npz", block / "known_summary.json"]
            labels, oof = _read_npz(paths[0]), _read_npz(paths[1])
            source_summary = json.loads(paths[2].read_text())
            if source_summary.get("status") != "complete":
                raise ValueError(f"SPD {policy} is not a completed block")
            for key in ("molecule_names", "target_names"):
                if not np.array_equal(labels[key], oof[key]):
                    raise ValueError(f"Prepared versus OOF {key} mismatch")
            if (not np.array_equal(labels["methods"], oof["native_methods"])
                    or not np.array_equal(labels["scores"], oof["native_scores"], equal_nan=True)):
                raise ValueError("Prepared versus OOF native score mismatch")
            if labels["lower"].shape != (930, 10) or labels["scores"].shape != (12, 930, 10):
                raise ValueError("Unexpected SPD cohort: expected 930 molecules, ten targets and twelve native outputs")
            if not np.isfinite(oof["predictions"]).all():
                raise ValueError("Incomplete OOF predictions cannot use method-specific eligibility")
            common = np.isfinite(labels["scores"]).all(axis=0)
            if not np.array_equal(common, oof["common_native_finite"]):
                raise ValueError("Stored common native support disagrees with score arrays")
            G, P, missing = interval_best_sets(labels["lower"], labels["upper"], labels["upper_open"])
            complete = common.all(axis=1)
            eligible = complete & G.any(axis=1)
            indices = np.flatnonzero(complete)
            eligible_local = eligible[indices]
            names, target_names = labels["molecule_names"], labels["target_names"]
            state_rows = []
            for q in range(930):
                if not complete[q]:
                    reason = "excluded_native_score_incomplete"
                elif eligible[q]:
                    reason = "eligible_certified_maximum"
                elif missing[q].any():
                    reason = "unresolved_with_missing_labels"
                else:
                    reason = "unresolved_interval_overlap"
                state_rows.append({"policy": policy, "query_index": q, "molecule": str(names[q]),
                                   "outer_fold": int(oof["fold_ids"][q]),
                                   "scaffold_group": str(oof["scaffold_groups"][q]),
                                   "native_score_complete": bool(complete[q]), "certified_eligible": bool(eligible[q]),
                                   "status": reason, "missing_label_targets": int(missing[q].sum()),
                                   "native_incomplete_targets": int((~common[q]).sum()),
                                   "G_size": int(G[q].sum()), "G_targets": ";".join(target_names[G[q]]),
                                   "P_size": int(P[q].sum()), "P_targets": ";".join(target_names[P[q]])})
            all_queries.extend(state_rows)
            _write_csv(output_root / f"certified_eligible_{policy}.csv", [r for r in state_rows if r["certified_eligible"]])
            variants = []
            for mi, method in enumerate(oof["methods"]):
                for fi, fraction in enumerate(oof["fractions"]):
                    variants.append({"method": str(method), "kind": "trained_oof", "fraction": float(fraction),
                                     "scores": oof["predictions"][mi, fi, indices]})
            for mi, method in enumerate(oof["native_methods"]):
                variants.append({"method": "native__" + str(method), "kind": "static_native_zero_local_labels",
                                 "fraction": 0., "scores": oof["native_scores"][mi, indices]})
            pg = np.empty((len(variants), len(indices), 10), float)
            pp = np.empty_like(pg)
            for vi, variant in enumerate(variants):
                for local, q in enumerate(indices):
                    pg[vi, local] = hit_probabilities(variant["scores"][local], G[q])
                    pp[vi, local] = hit_probabilities(variant["scores"][local], P[q])
                spent = assays_spent_by_cap(pg[vi])
                for local, q in enumerate(indices):
                    record = {"policy": policy, "method": variant["method"], "kind": variant["kind"],
                              "local_label_fraction": variant["fraction"], "query_index": int(q),
                              "molecule": str(names[q]), "outer_fold": int(oof["fold_ids"][q]),
                              "certified_eligible": bool(eligible[q]),
                              "expected_assays_until_first_G": float(spent[local, -1]) if eligible[q] else None}
                    record.update({f"p_hit_G_k{k+1}": float(pg[vi, local, k]) for k in range(10)})
                    record.update({f"p_hit_P_k{k+1}": float(pp[vi, local, k]) for k in range(10)})
                    record.update({f"expected_assays_spent_until_G_or_cap_{k+1}": float(spent[local, k]) if eligible[q] else None
                                   for k in range(10)})
                    probability_writer.writerow(record)
            conditional = pg[:, eligible_local]
            n_eligible = int(eligible.sum())
            boot = cluster_bootstrap_means(conditional, oof["scaffold_groups"][eligible], bootstrap_replicates,
                                           bootstrap_seed + policy_index)
            means = conditional.mean(axis=1) if n_eligible else np.full((len(variants), 10), np.nan)
            costs = assays_spent_by_cap(conditional)
            bootstrap_costs = assays_spent_by_cap(boot) if boot is not None else None
            for vi, variant in enumerate(variants):
                base = {"policy": policy, "method": variant["method"], "kind": variant["kind"],
                        "local_label_fraction": variant["fraction"], "certified_molecules": n_eligible,
                        "score_complete_molecules": int(complete.sum())}
                for k in range(10):
                    ci = _interval(boot[:, vi, k] if boot is not None else None)
                    spent_ci = _interval(bootstrap_costs[:, vi, k] if boot is not None else None)
                    all_curves.append({**base, "assay_budget_k": k+1,
                                       "certified_macro_expected_recovery": means[vi, k],
                                       "certified_expected_recovered_count": float(conditional[vi, :, k].sum()),
                                       "bootstrap_recovery_p025": ci[0], "bootstrap_recovery_p975": ci[1],
                                       "certified_expected_assays_spent_to_hit_or_cap": float(costs[vi, :, k].mean()) if n_eligible else None,
                                       "bootstrap_capped_assays_p025": spent_ci[0], "bootstrap_capped_assays_p975": spent_ci[1],
                                       "whole_population_lower_recovery_G": float(pg[vi, :, k].mean()),
                                       "whole_population_upper_recovery_P": float(pp[vi, :, k].mean()),
                                       "whole_population_lower_expected_count": float(pg[vi, :, k].sum()),
                                       "whole_population_upper_expected_count": float(pp[vi, :, k].sum())})
                expected = costs[vi, :, -1]
                ci = _interval(bootstrap_costs[:, vi, -1] if boot is not None else None)
                cost_row = {**base, "expected_assays_until_first_G": float(expected.mean()) if n_eligible else None,
                            "query_expected_assays_q10": float(np.quantile(expected, .1)) if n_eligible else None,
                            "query_expected_assays_q50": float(np.quantile(expected, .5)) if n_eligible else None,
                            "query_expected_assays_q90": float(np.quantile(expected, .9)) if n_eligible else None,
                            "bootstrap_mean_p025": ci[0], "bootstrap_mean_p975": ci[1]}
                all_costs.append(cost_row)
                report["all_variant_summaries"].append({**cost_row, "certified_recovery_k1_to_k10": means[vi],
                                                       "broad_lower_k1_to_k10": pg[vi].mean(axis=0),
                                                       "broad_upper_k1_to_k10": pp[vi].mean(axis=0)})
            chemistry = {v["fraction"]: i for i, v in enumerate(variants) if v["method"] == "chemistry"}
            if set(chemistry) != set(float(f) for f in oof["fractions"]):
                raise ValueError("Every training fraction needs the chemistry comparison")
            for vi, variant in enumerate(variants):
                comparisons = list(chemistry.items()) if variant["fraction"] == 0 else [(variant["fraction"], chemistry[variant["fraction"]])]
                for comparator_fraction, bi in comparisons:
                    delta = means[vi] - means[bi]
                    delta_cost = costs[vi, :, -1] - costs[bi, :, -1]
                    boot_delta = boot[:, vi] - boot[:, bi] if boot is not None else None
                    cost_interval = _interval(bootstrap_costs[:, vi, -1] - bootstrap_costs[:, bi, -1] if boot is not None else None)
                    for k in range(10):
                        ci = _interval(boot_delta[:, k] if boot_delta is not None else None)
                        all_differences.append({"policy": policy, "method": variant["method"], "kind": variant["kind"],
                                                "local_label_fraction": variant["fraction"], "comparator": "chemistry",
                                                "comparator_label_fraction": comparator_fraction, "assay_budget_k": k+1,
                                                "certified_molecules": n_eligible, "paired_recovery_difference": delta[k],
                                                "paired_expected_recovered_count_difference": delta[k] * n_eligible,
                                                "bootstrap_recovery_difference_p025": ci[0], "bootstrap_recovery_difference_p975": ci[1],
                                                "paired_expected_assays_difference": float(delta_cost.mean()) if n_eligible else None,
                                                "bootstrap_assays_difference_p025": cost_interval[0],
                                                "bootstrap_assays_difference_p975": cost_interval[1]})
            counts = {reason: sum(row["status"] == reason for row in state_rows)
                      for reason in ("eligible_certified_maximum", "unresolved_with_missing_labels",
                                     "unresolved_interval_overlap", "excluded_native_score_incomplete")}
            report["policies"][policy] = {"all_molecules": 930, "target_count": 10, "native_method_count": 12,
                                         "trained_method_count": len(oof["methods"]), "training_fractions": oof["fractions"],
                                         "score_complete": int(complete.sum()), "status_counts": counts,
                                         "certified_among_all_molecules": int(G.any(axis=1).sum()),
                                         "certified_but_score_incomplete": int((G.any(axis=1) & ~complete).sum()),
                                         "eligible_scaffold_groups": len(np.unique(oof["scaffold_groups"][eligible])),
                                         "eligible_molecule_names": names[eligible],
                                         "G_size_distribution_on_eligible": {str(g): int((G[eligible].sum(axis=1) == g).sum()) for g in np.unique(G[eligible].sum(axis=1))},
                                         "source_worker_status": source_summary["status"], "source_worker_fits": source_summary.get("fits")}
            for path in paths:
                report["inputs"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
            print("POLICY_COMPLETE", json.dumps(_clean({"policy": policy, **report["policies"][policy],
                                                        "eligible_molecule_names": "saved in report.json and eligibility CSV"})), flush=True)
    _write_csv(output_root / "query_sets.csv", all_queries)
    _write_csv(output_root / "curves.csv", all_curves)
    _write_csv(output_root / "expected_assays.csv", all_costs)
    _write_csv(output_root / "paired_differences_to_chemistry.csv", all_differences)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["outputs"] = {name: {"bytes": (output_root / name).stat().st_size, "sha256": _sha256(output_root / name)}
                         for name in ["query_sets.csv", "curves.csv", "expected_assays.csv",
                                      "paired_differences_to_chemistry.csv", "per_query_probabilities.csv.gz"]}
    report["script_sha256"] = _sha256(Path(__file__))
    (output_root / "report.json").write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
    print("DECISION_CURVES_COMPLETE", json.dumps({"output_root": str(output_root),
                                                  "curve_rows": len(all_curves), "difference_rows": len(all_differences),
                                                  "elapsed_seconds": report["elapsed_seconds"], "new_fits": 0}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("build/decision_curves"), type=Path)
    parser.add_argument("--bootstrap-replicates", default=2000, type=int)
    parser.add_argument("--bootstrap-seed", default=20261004, type=int)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--run-log", type=Path)
    args = parser.parse_args()
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be nonnegative")
    analyze(**vars(args))


if __name__ == "__main__":
    main()
