"""Independently audit saved score-component predictions; no fitting or retrieval.

The CLI accepts separate source, prepared-data and output roots. Input records
may be plain text or gzip-compressed; manifests contain only relative paths.
"""
import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits




def resolve_input(path):
    """Prefer the named input, otherwise its .gz public-package counterpart."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def open_text(path):
    path = resolve_input(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_text(path):
    with open_text(path) as handle:
        return handle.read()


def read_json(path):
    with open_text(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    with open_text(path) as handle:
        return [json.loads(line) for line in handle]


POLICIES = ("source_representative", "assay_envelope")
STRATA = ("all_determinate", "both_exact", "one_bounded", "both_bounded")
TOLERANCE = 1e-12  # Floating-point CSV replay only; not a scientific cutoff.


def independent_replay(lower, upper, opened, support, predictions):
    """Rebuild interval-order credits without importing the experiment's code."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    opened, support, predictions = np.asarray(opened, bool), np.asarray(support, bool), np.asarray(predictions)
    if predictions.ndim != 3 or predictions.shape[1:] != lower.shape:
        raise ValueError("Prediction axes must be variant,molecule,target")
    if lower.shape != upper.shape or lower.shape != opened.shape or lower.shape != support.shape:
        raise ValueError("Observation/support axes disagree")
    if not np.isfinite(predictions[:, support]).all():
        raise ValueError("Nonfinite predictions on shared native support")
    n_variants, n_queries, n_targets = predictions.shape
    counts = np.zeros((n_queries, 4), dtype=int)
    credits = np.zeros((n_variants, n_queries, 4), dtype=float)
    for q in range(n_queries):
        exact = np.isfinite(lower[q]) & (lower[q] == upper[q]) & ~opened[q]
        # Enumerate target pairs directly; the original implementation's
        # pair_order/evaluate helpers are deliberately not imported.
        for a in range(n_targets):
            for b in range(a + 1, n_targets):
                if not (support[q, a] and support[q, b]):
                    continue
                if any(np.isnan(value) for value in (lower[q, a], upper[q, a], lower[q, b], upper[q, b])):
                    continue
                a_above = lower[q, a] > upper[q, b] or (lower[q, a] == upper[q, b] and opened[q, b])
                b_above = lower[q, b] > upper[q, a] or (lower[q, b] == upper[q, a] and opened[q, a])
                if a_above and b_above:
                    raise ValueError("Inconsistent interval order")
                if not (a_above or b_above):
                    continue
                direction = 1 if a_above else -1
                credit = (np.sign(predictions[:, q, a] - predictions[:, q, b]) * direction + 1) / 2
                stratum = 1 if exact[a] and exact[b] else (2 if exact[a] != exact[b] else 3)
                counts[q, [0, stratum]] += 1
                credits[:, q, 0] += credit
                credits[:, q, stratum] += credit
    concordance = np.divide(credits, counts[None], out=np.full_like(credits, np.nan), where=counts[None] > 0)
    return counts, credits, concordance


def paired_scaffold_bootstrap(deltas, credit_deltas, pair_counts, groups, replicates, seed):
    """Coupled, molecule-weighted macro and pair-weighted pooled increments."""
    deltas, credit_deltas, pair_counts = np.asarray(deltas), np.asarray(credit_deltas), np.asarray(pair_counts)
    if deltas.shape != credit_deltas.shape or deltas.shape[0] != len(pair_counts):
        raise ValueError("Bootstrap axes disagree")
    if np.any(pair_counts <= 0) or not np.isfinite(deltas).all():
        raise ValueError("Bootstrap input must contain only evaluable queries")
    unique, inverse = np.unique(np.asarray(groups, str), return_inverse=True)
    size = np.bincount(inverse, minlength=len(unique))
    pairs = np.bincount(inverse, weights=pair_counts, minlength=len(unique))
    sums = np.zeros((len(unique), deltas.shape[1]))
    credit_sums = np.zeros_like(sums)
    np.add.at(sums, inverse, deltas)
    np.add.at(credit_sums, inverse, credit_deltas)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=replicates)
    macro = (weights @ sums) / (weights @ size)[:, None]
    pooled = (weights @ credit_sums) / (weights @ pairs)[:, None]
    return macro, pooled


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _sha256(path):
    digest = hashlib.sha256()
    with resolve_input(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _dump(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")


def _check_numbers(actual, expected, name, exact=False):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.array_equal(np.isnan(actual), np.isnan(expected)):
        raise AssertionError(f"{name}: shape or missingness mismatch")
    use = ~np.isnan(expected)
    error = float(np.max(np.abs(actual[use] - expected[use]))) if use.any() else 0.
    if error > (0 if exact else TOLERANCE):
        raise AssertionError(f"{name}: maximum absolute error {error}")
    return error


def _verify_query_csv(path, methods, fractions, names, folds, counts, credits, concordance):
    nm, nf, nq, ns = len(methods), len(fractions), len(names), len(STRATA)
    maps = [{str(x): i for i, x in enumerate(methods)}, {float(x): i for i, x in enumerate(fractions)},
            {str(x): i for i, x in enumerate(names)}, {str(x): i for i, x in enumerate(STRATA)}]
    seen = np.zeros(nm * nf * nq * ns, bool)
    errors = dict(pairs=0., eligible_pairs=0., credit=0., concordance=0.)
    rows = 0
    columns = ["method", "fraction", "query", "outer_fold", "stratum", "status", "pairs", "eligible_pairs", "credit", "concordance"]
    for chunk in pd.read_csv(resolve_input(path), usecols=columns, chunksize=50000):
        mapped = [chunk[col].map(mapping) for col, mapping in zip(("method", "fraction", "query", "stratum"), maps)]
        if any(x.isna().any() for x in mapped):
            raise AssertionError("Unrecognized query metric key")
        mi, fi, qi, si = [x.to_numpy(dtype=int) for x in mapped]
        vi = mi * nf + fi
        linear = (vi * nq + qi) * ns + si
        if seen[linear].any() or len(np.unique(linear)) != len(linear):
            raise AssertionError("Duplicate query metric row")
        seen[linear] = True
        if not chunk.status.eq("available").all() or not np.array_equal(chunk.outer_fold.to_numpy(), folds[qi]):
            raise AssertionError("Metric status/fold mismatch")
        for field, expected in (("pairs", counts[qi, si]), ("eligible_pairs", counts[qi, si]),
                                ("credit", credits[vi, qi, si]), ("concordance", concordance[vi, qi, si])):
            errors[field] = max(errors[field], _check_numbers(chunk[field], expected, field, exact=field != "concordance"))
        rows += len(chunk)
    if not seen.all():
        raise AssertionError("Missing query metric rows")
    return {"rows": rows, "maximum_absolute_replay_errors": errors}


def _verify_aggregate_csv(path, methods, fractions, counts, credits, concordance):
    table = pd.read_csv(resolve_input(path))
    lookup = {(m, float(f), s): (mi * len(fractions) + fi, si)
              for mi, m in enumerate(methods) for fi, f in enumerate(fractions) for si, s in enumerate(STRATA)}
    seen, replay, errors = set(), [], {}
    for row in table.to_dict("records"):
        key = (row["method"], float(row["fraction"]), row["stratum"])
        if key in seen or key not in lookup:
            raise AssertionError("Unexpected or duplicate aggregate key")
        seen.add(key)
        vi, si = lookup[key]
        valid = counts[:, si] > 0
        pairs, credit = int(counts[:, si].sum()), float(credits[vi, :, si].sum())
        expected = {"queries": len(counts), "evaluable_queries": int(valid.sum()), "unavailable_queries": 0,
                    "eligible_pairs": pairs, "pairs": pairs, "credit": credit,
                    "macro_concordance": float(concordance[vi, valid, si].mean()) if valid.any() else np.nan,
                    "pooled_concordance": credit / pairs if pairs else np.nan}
        for field, value in expected.items():
            error = _check_numbers([row[field]], [value], field, exact=field not in ("macro_concordance", "pooled_concordance"))
            errors[field] = max(errors.get(field, 0), error)
        replay.append({"method": key[0], "fraction": key[1], "stratum": key[2], **expected})
    if seen != set(lookup):
        raise AssertionError("Missing aggregate rows")
    return {"rows": len(table), "maximum_absolute_replay_errors": errors}, replay


def _fit_audit(path, expected_count):
    seen, messages, stages = set(), Counter(), Counter()
    iterations, gradients, fit_seconds = [], [], []
    with open_text(path) as handle:
        for line in handle:
            row = json.loads(line)
            d = row.get("diagnostics", {})
            if row.get("status") != "fitted" or d.get("success") is not True:
                raise AssertionError("A fit record is not explicitly successful")
            for field in ("objective", "gradient_max", "alpha", "fit_seconds"):
                if not np.isfinite(d[field]):
                    raise AssertionError(f"Nonfinite fit diagnostic: {field}")
            if d["alpha"] != row["ridge"] / 10:
                raise AssertionError("Unexpected ridge/target normalization")
            key = tuple(row.get(k) for k in ("stage", "outer_fold", "fraction", "inner_fold", "method", "ridge"))
            if key in seen:
                raise AssertionError("Duplicated fit record")
            seen.add(key)
            messages[d["message"]] += 1
            stages[row["stage"]] += 1
            iterations.append(d["iterations"])
            gradients.append(d["gradient_max"])
            fit_seconds.append(d["fit_seconds"])
    if len(seen) != expected_count:
        raise AssertionError("Fit-log count differs from block summary")
    return {"records": len(seen), "success_true": len(seen), "failed": 0,
            "unique_fit_keys": len(seen), "stage_counts": dict(stages), "termination_messages": dict(messages),
            "maximum_iterations": max(iterations), "maximum_gradient": max(gradients),
            "sum_fit_seconds": sum(fit_seconds)}


def _identity_rows(current, reference, policy):
    for key in ("molecule_names", "target_names", "fold_ids", "scaffold_groups", "fractions", "native_methods"):
        if not np.array_equal(current[key], reference[key]):
            raise AssertionError(f"Original comparison alignment mismatch: {key}")
    for key in ("native_scores", "native_missing", "common_native_finite"):
        if not np.array_equal(current[key], reference[key], equal_nan=True):
            raise AssertionError(f"Original native-score identity mismatch: {key}")
    rows = []
    for mi, method in enumerate(current["methods"]):
        if method not in reference["methods"]:
            continue
        oi = list(reference["methods"]).index(method)
        for fi, fraction in enumerate(current["fractions"]):
            a, b = current["predictions"][mi, fi], reference["predictions"][oi, fi]
            rows.append({"policy": policy, "method": str(method), "fraction": float(fraction),
                         "cells": a.size, "exact_prediction_identity": bool(np.array_equal(a, b, equal_nan=True)),
                         "different_finite_cells": int(np.count_nonzero(a != b)),
                         "maximum_absolute_difference": float(np.max(np.abs(a-b))),
                         "selected_ridge_identity": bool(np.array_equal(current["selected_ridge"][mi, fi],
                                                                          reference["selected_ridge"][oi, fi], equal_nan=True))})
    return rows


def audit(component_root, reference_root, output_root, bootstrap_replicates=2000,
          bootstrap_seed=20261004, run_id=None, revision=None, run_log=None, prepared_root=None):
    started = time.perf_counter()
    component_root, reference_root, output_root = Path(component_root), Path(reference_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(prepared_root) if prepared_root else reference_root / "prepared"
    summary = {"analysis": "independent retrospective numerical replay of saved score-component predictions",
               "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "revision": revision,
               "new_fits": 0, "retrieval": 0, "numerical_threads": 1, "numerical_replay_tolerance": TOLERANCE,
               "tolerance_meaning": "Floating-point CSV round-trip check only; not a scientific effect threshold",
               "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed,
                             "unit": "scaffold groups of evaluable molecules, resampled with replacement",
                             "coupling": "Every contrast, output and label budget uses the same resample within each policy/stratum",
                             "intervals": "2.5th and97.5th percentiles; conditional on fixed OOF predictions; no refits or p-values"},
               "policies": {}, "files": [], "log_completion_evidence": []}
    if run_log:
        for line_no, line in enumerate(read_text(run_log).splitlines(), 1):
            if line.startswith("COMPARISON_BLOCK "):
                entry = json.loads(line.split(" ", 1)[1])
                if entry.get("dataset") == "SPD" and entry.get("regime") == "new_molecule_known_targets":
                    summary["log_completion_evidence"].append({"line": line_no, "policy": entry["policy"],
                                                                 "status": entry["result"].get("status"),
                                                                 "fits": entry["result"].get("fits")})
        if {x["policy"] for x in summary["log_completion_evidence"] if x["status"] == "complete"} != set(POLICIES):
            raise AssertionError("Run log does not verify both completed SPD blocks")
    all_aggregates, identities, increments = [], [], []
    query_path = output_root / "paired_query_increments.csv.gz"
    query_fields = ["policy", "stratum", "output", "fraction", "contrast", "query", "scaffold_group", "outer_fold",
                    "pairs", "candidate_concordance", "reference_concordance", "paired_difference", "credit_difference"]
    with gzip.open(query_path, "wt", newline="") as handle:
        query_writer = csv.DictWriter(handle, fieldnames=query_fields)
        query_writer.writeheader()
        for pi, policy in enumerate(POLICIES):
            relative = Path("SPD") / policy / "new_molecule_known_targets"
            block, old_block = component_root / relative, reference_root / relative
            labels_path = prepared_root / f"spd_{policy}_inputs.npz"
            labels = _load(labels_path)
            current, old = _load(block / "known_oof.npz"), _load(old_block / "known_oof.npz")
            info = read_json(block / "known_summary.json")
            if info["status"] != "complete" or not np.isfinite(current["predictions"]).all():
                raise AssertionError("Completed, finite component predictions are required")
            if (not np.array_equal(labels["methods"], current["native_methods"])
                    or not np.array_equal(labels["scores"], current["native_scores"], equal_nan=True)):
                raise AssertionError("Shared prepared native scores differ from the saved component inputs")
            for key in ("molecule_names", "target_names"):
                if not np.array_equal(labels[key], current[key]):
                    raise AssertionError("Label-to-prediction identity mismatch")
            support = np.isfinite(labels["scores"]).all(axis=0)
            if not np.array_equal(support, current["common_native_finite"]):
                raise AssertionError("Common finite-native support mismatch")
            identities.extend(_identity_rows(current, old, policy))
            fractions = current["fractions"]
            methods = list(map(str, current["methods"])) + ["native__" + str(x) for x in current["native_methods"]]
            n, t = labels["lower"].shape
            native = np.repeat(current["native_scores"][:, None], len(fractions), axis=1)
            predictions = np.concatenate((current["predictions"], native), axis=0).reshape(-1, n, t)
            counts, credits, concordance = independent_replay(labels["lower"], labels["upper"], labels["upper_open"], support, predictions)
            query_check = _verify_query_csv(block / "known_query_metrics.csv", methods, fractions, current["molecule_names"],
                                             current["fold_ids"], counts, credits, concordance)
            aggregate_check, aggregate = _verify_aggregate_csv(block / "known_aggregate_metrics.csv", methods, fractions,
                                                               counts, credits, concordance)
            all_aggregates.extend({"policy": policy, **row} for row in aggregate)
            fits = _fit_audit(block / "known_fits.jsonl", info["fits"])
            contrast_specs = []
            for output in current["native_methods"]:
                for fi, fraction in enumerate(fractions):
                    for contrast, candidate, comparator in (("mean_relative_minus_mean", "mean_relative__", "mean__"),
                                                            ("mean_minus_availability", "mean__", "availability__")):
                        ci = methods.index(candidate + str(output)) * len(fractions) + fi
                        bi = methods.index(comparator + str(output)) * len(fractions) + fi
                        contrast_specs.append((str(output), float(fraction), contrast, ci, bi))
            for si in (0, 1):
                valid = counts[:, si] > 0
                queries = np.flatnonzero(valid)
                delta = np.column_stack([concordance[ci, valid, si] - concordance[bi, valid, si] for _, _, _, ci, bi in contrast_specs])
                credit_delta = np.column_stack([credits[ci, valid, si] - credits[bi, valid, si] for _, _, _, ci, bi in contrast_specs])
                groups = current["scaffold_groups"][valid]
                boot, pooled_boot = paired_scaffold_bootstrap(delta, credit_delta, counts[valid, si], groups,
                                                               bootstrap_replicates, bootstrap_seed + pi * 100 + si)
                for column, (output, fraction, contrast, ci, bi) in enumerate(contrast_specs):
                    lo, hi = np.quantile(boot[:, column], [.025, .975])
                    plo, phi = np.quantile(pooled_boot[:, column], [.025, .975])
                    increments.append({"policy": policy, "stratum": STRATA[si], "output": output, "fraction": fraction,
                                       "contrast": contrast, "queries": int(valid.sum()), "scaffold_groups": len(set(groups)),
                                       "pairs": int(counts[valid, si].sum()),
                                       "candidate_macro": float(concordance[ci, valid, si].mean()),
                                       "reference_macro": float(concordance[bi, valid, si].mean()),
                                       "paired_macro_increment": float(delta[:, column].mean()),
                                       "bootstrap_macro_p025": float(lo), "bootstrap_macro_p975": float(hi),
                                       "paired_pooled_increment": float(credit_delta[:, column].sum() / counts[valid, si].sum()),
                                       "bootstrap_pooled_p025": float(plo), "bootstrap_pooled_p975": float(phi)})
                    for local, q in enumerate(queries):
                        query_writer.writerow({"policy": policy, "stratum": STRATA[si], "output": output,
                                               "fraction": fraction, "contrast": contrast,
                                               "query": str(current["molecule_names"][q]),
                                               "scaffold_group": str(current["scaffold_groups"][q]),
                                               "outer_fold": int(current["fold_ids"][q]), "pairs": int(counts[q, si]),
                                               "candidate_concordance": float(concordance[ci, q, si]),
                                               "reference_concordance": float(concordance[bi, q, si]),
                                               "paired_difference": float(delta[local, column]),
                                               "credit_difference": float(credit_delta[local, column])})
            summary["policies"][policy] = {"per_query_replay": query_check, "aggregate_replay": aggregate_check,
                                            "fit_audit": fits, "methods_including_native": len(methods),
                                            "stored_oof_shape": list(current["predictions"].shape),
                                            "support": {s: {"queries": int((counts[:, j] > 0).sum()), "pairs": int(counts[:, j].sum())}
                                                        for j, s in enumerate(STRATA)}}
            source_files = [("prepared", labels_path, prepared_root),
                            ("component", block / "known_oof.npz", component_root),
                            ("reference", old_block / "known_oof.npz", reference_root),
                            *[("component", block / name, component_root) for name in
                              ("known_summary.json", "known_fits.jsonl", "known_query_metrics.csv", "known_aggregate_metrics.csv")]]
            for root_kind, path, base in source_files:
                path = resolve_input(path)
                summary["files"].append({"root": root_kind, "relative_path": str(path.relative_to(base)),
                                         "bytes": path.stat().st_size, "sha256": _sha256(path)})
            print("POLICY_AUDITED", json.dumps(_clean({"policy": policy, **summary["policies"][policy]})), flush=True)
    pd.DataFrame(all_aggregates).to_csv(output_root / "replayed_aggregate_concordance.csv", index=False)
    pd.DataFrame(identities).to_csv(output_root / "prediction_identity_vs_original.csv", index=False)
    pd.DataFrame(increments).to_csv(output_root / "paired_scaffold_increments.csv", index=False)
    summary["prediction_identity"] = {"method_fraction_policy_comparisons": len(identities),
                                       "all_exact": all(row["exact_prediction_identity"] for row in identities),
                                       "all_selected_ridge_identical": all(row["selected_ridge_identity"] for row in identities),
                                       "maximum_absolute_difference": max(row["maximum_absolute_difference"] for row in identities)}
    summary["increments"] = {"rows": len(increments), "contrasts": ["mean_relative_minus_mean", "mean_minus_availability"],
                              "strata": ["all_determinate", "both_exact"], "outputs": 12, "budgets": 4,
                              "interpretation": "Predictive increments under fixed arm definitions and independent tuning; no causal mechanism or equivalence conclusion."}
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["threadpools"] = [{k:v for k,v in x.items() if k in ("user_api", "internal_api", "num_threads", "version")}
                              for x in threadpool_info()]
    summary["status"] = "replay_verified"
    summary["script_sha256"] = _sha256(__file__)
    _dump(output_root / "audit_summary.json", summary)
    print("AUDIT_COMPLETE", json.dumps(_clean({"status": summary["status"], "identity": summary["prediction_identity"],
                                               "increment_rows": len(increments), "elapsed_seconds": summary["elapsed_seconds"]})), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, help="Shared prepared data; defaults to REFERENCE_ROOT/prepared")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    parser.add_argument("--run-id")
    parser.add_argument("--revision")
    parser.add_argument("--run-log", type=Path)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap-replicates must be positive")
    with threadpool_limits(limits=1):
        audit(**vars(args))


if __name__ == "__main__":
    main()
