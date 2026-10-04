"""Label-budget learning on sequence-group-held-out target contexts.

Ranking pairs are formed only within an outer target fold. Their credits are
then pooled per molecule; predictions from different fitted models are never
compared to construct a whole-panel ranking.
The all-target-pairs extension evaluates each pair in its own fitted context
and stores separate predictions for every context.
"""
from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .features import MolecularKernel, grouped_folds, nested_group_subsets
from .metrics import pair_order
from .models import GridIntervalRidge, KernelBasis, PairScaler


@dataclass
class ProteinKernelBasis:
    mean: np.ndarray
    scale: np.ndarray
    normalized_train: np.ndarray
    basis: KernelBasis

    @staticmethod
    def normalize(values):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)

    @classmethod
    def fit(cls, train):
        train = np.asarray(train, float)
        if train.ndim != 2 or not len(train) or not np.isfinite(train).all():
            raise ValueError("Finite training-target embeddings are required")
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale == 0] = 1
        normalized = cls.normalize((train - mean) / scale)
        kernel = (1 + normalized @ normalized.T) / 2
        return cls(mean, scale, normalized, KernelBasis.fit(kernel))

    @property
    def training(self):
        return self.basis.training

    def transform(self, values):
        values = np.asarray(values, float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Finite evaluation-target embeddings are required")
        normalized = self.normalize((values - self.mean) / self.scale)
        return self.basis.transform((1 + normalized @ self.normalized_train.T) / 2)


def target_allocations(groups, fractions, outer_folds=5, inner_folds=3, seed=0,
                       cold_split="sequence_folds"):
    """Outcome-independent, nested whole-sequence-group budgets.

    Fractional group counts round up, a technical allocation convention. Targets
    sharing an exact sequence always enter the same outer and inner folds.
    ``all_target_pairs`` is the complete ten-target SPD extension: it requires
    ten distinct sequence identities and full training budget, returning None
    instead of a misleading per-target fold vector. Each pair has its own
    context ID and the same three-fold inner allocation procedure on eight
    training targets. Duplicate-sequence aliases fail explicitly.
    """
    groups = np.asarray(groups, dtype=str)
    fractions = [float(x) for x in fractions]
    if not fractions or fractions != sorted(set(fractions)) or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Fractions must be unique, increasing and in (0, 1]")
    if cold_split == "sequence_folds":
        folds = grouped_folds(groups, outer_folds, seed)
        outer = [(fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
                 for fold in range(outer_folds)]
    elif cold_split == "all_target_pairs":
        if len(groups) != 10 or len(np.unique(groups)) != 10:
            raise ValueError("All-target-pairs evaluation requires ten distinct sequence hashes; aliases are not permitted")
        if fractions != [1.0]:
            raise ValueError("All-target-pairs evaluation uses only the full eight-target training budget")
        folds = None
        outer = []
        for context_id, pair in enumerate(combinations(range(len(groups)), 2)):
            test = np.asarray(pair, dtype=int)
            train = np.flatnonzero(~np.isin(groups, groups[test]))
            if len(train) != 8:
                raise ValueError("Held-out sequence aliases changed the eight-target training partition")
            outer.append((context_id, train, test))
    else:
        raise ValueError(f"Unknown cold-target split: {cold_split}")
    contexts = []
    for fold, train_pool, test in outer:
        total_groups = len(np.unique(groups[train_pool]))
        counts = [int(np.ceil(f * total_groups)) for f in fractions]
        subsets = nested_group_subsets(groups[train_pool], counts, seed + 1009 * (fold + 1))
        for fraction, count, subset in zip(fractions, counts, subsets):
            train = train_pool[subset]
            context = {"outer_fold": fold, "fraction": fraction,
                       "training_groups": count, "available_training_groups": total_groups,
                       "train_indices": train, "test_indices": test}
            if count < inner_folds:
                context["skip_reason"] = "Fewer training sequence groups than inner folds"
                context["inner"] = []
            else:
                inner = grouped_folds(groups[train], inner_folds, seed + 2003 * (fold + 1))
                context["inner"] = [(train[inner != k], train[inner == k])
                                    for k in range(inner_folds)]
            contexts.append(context)
    return folds, contexts


def _store_context_prediction(predictions, key, prediction, shape):
    """Write one context once; never overwrite another fit of the same target."""
    if key in predictions:
        raise ValueError(f"Duplicate held-target context prediction: {key}")
    prediction = np.asarray(prediction)
    if prediction.shape != shape:
        raise ValueError("Held-target context prediction has unexpected axes")
    predictions[key] = prediction.copy()


def evaluate_context(prediction, lower, upper, upper_open, support, molecule_names,
                     *, dataset, policy, method, fraction, fold, errors=True):
    """Complete per-molecule contributions from one held-out target context."""
    prediction = np.asarray(prediction, float)
    lower, upper, upper_open = np.asarray(lower), np.asarray(upper), np.asarray(upper_open)
    if prediction.shape != lower.shape or upper.shape != lower.shape or support.shape != lower.shape:
        raise ValueError("Prediction, observation and support axes disagree")
    if np.any(support & ~np.isfinite(prediction)):
        raise ValueError("Nonfinite predictions on the shared evaluation support")
    pairs = np.stack(np.triu_indices(lower.shape[1], 1), axis=1)
    a, b = pairs.T
    truth = pair_order(lower, upper, upper_open, pairs)
    rows = []
    for i, name in enumerate(molecule_names):
        finite = support[i] & np.isfinite(prediction[i])
        use = (truth[i] != 0) & finite[a] & finite[b]
        exact = np.isfinite(lower[i]) & (lower[i] == upper[i]) & ~upper_open[i]
        known = (finite & ~np.isnan(lower[i]) & ~np.isnan(upper[i])
                 & (np.isfinite(lower[i]) | np.isfinite(upper[i])))
        masks = {"all_determinate": use, "both_exact": use & exact[a] & exact[b],
                 "one_bounded": use & (exact[a] ^ exact[b]),
                 "both_bounded": use & ~exact[a] & ~exact[b]}
        credits = (np.sign(prediction[i, a] - prediction[i, b]) * truth[i] + 1) / 2
        distance = (np.maximum(lower[i, known] - prediction[i, known], 0)
                    + np.maximum(prediction[i, known] - upper[i, known], 0))
        exact_use = known & exact
        for stratum, mask in masks.items():
            count = int(mask.sum())
            error_use = errors and stratum == "all_determinate"
            error_cells = int(known.sum()) if error_use else 0
            exact_cells = int(exact_use.sum()) if error_use else 0
            interval_sum = float(distance.sum()) if error_use else 0.0
            exact_sum = float(np.abs(prediction[i, exact_use] - lower[i, exact_use]).sum()) if error_use else 0.0
            rows.append({"dataset": dataset, "policy": policy,
                         "regime": "new_targets_known_molecules", "axis": "target",
                         "method": method, "fraction": fraction, "outer_fold": fold,
                         "query": str(name), "stratum": stratum, "pairs": count,
                         "credit": float(credits[mask].sum()),
                         "concordance": float(credits[mask].mean()) if count else np.nan,
                         "observed_support_cells": int(known.sum()) if stratum == "all_determinate" else 0,
                         "error_cells": error_cells, "exact_error_cells": exact_cells,
                         "interval_absolute_error_sum": interval_sum,
                         "exact_absolute_error_sum": exact_sum,
                         "interval_mae": interval_sum / error_cells if error_cells else np.nan,
                         "exact_mae": exact_sum / exact_cells if exact_cells else np.nan})
    return rows


def pool_contexts(records):
    """Pool already-formed pair contributions per molecule, never predictions."""
    if not records:
        return [], []
    frame = pd.DataFrame(records)
    group = ["dataset", "policy", "regime", "axis", "method", "fraction", "stratum"]
    sum_columns = ["pairs", "credit", "observed_support_cells", "error_cells",
                   "exact_error_cells", "interval_absolute_error_sum", "exact_absolute_error_sum"]
    molecules = frame.groupby(group + ["query"], sort=False)[sum_columns].sum().reset_index()
    molecules["concordance"] = molecules.credit / molecules.pairs.replace(0, np.nan)
    molecules["interval_mae"] = molecules.interval_absolute_error_sum / molecules.error_cells.replace(0, np.nan)
    molecules["exact_mae"] = molecules.exact_absolute_error_sum / molecules.exact_error_cells.replace(0, np.nan)
    summaries = []
    for key, rows in molecules.groupby(group, sort=False):
        valid = rows[rows.pairs > 0]
        error_n, exact_n = int(rows.error_cells.sum()), int(rows.exact_error_cells.sum())
        summaries.append(dict(zip(group, key), macro=float(valid.concordance.mean()),
                              pooled=float(valid.credit.sum() / valid.pairs.sum()) if len(valid) else np.nan,
                              queries=len(valid), pairs=int(valid.pairs.sum()),
                              query_q10=float(valid.concordance.quantile(.1)),
                              query_q50=float(valid.concordance.quantile(.5)),
                              query_q90=float(valid.concordance.quantile(.9)),
                              observed_support_cells=int(rows.observed_support_cells.sum()),
                              error_cells=error_n, exact_error_cells=exact_n,
                              interval_mae=float(rows.interval_absolute_error_sum.sum() / error_n) if error_n else np.nan,
                              exact_mae=float(rows.exact_absolute_error_sum.sum() / exact_n) if exact_n else np.nan))
    return molecules.to_dict("records"), summaries


def _macro(records):
    _, summary = pool_contexts(records)
    values = [r["macro"] for r in summary if r["stratum"] == "all_determinate"]
    return values[0] if values else np.nan


def _checked_prediction(model, molecular, target, pairs):
    prediction = model.predict(molecular, target, pairs)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Fitted cold-target model returned nonfinite predictions")
    return prediction


@dataclass
class Partition:
    train: np.ndarray
    evaluate: np.ndarray
    target_train: np.ndarray
    target_evaluate: np.ndarray
    pair_cache: dict = field(default_factory=dict)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_cold(data, out, config):
    """Fit nested target-budget comparisons or a training-only numerical pilot."""
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mode = config.get("mode", "full")
    if mode not in ("technical_pilot", "full"):
        raise ValueError("Unknown cold-target learning mode")
    pilot = mode == "technical_pilot"
    cold_split = config.get("cold_split", "sequence_folds")
    all_pairs = cold_split == "all_target_pairs"
    fractions = [1.0] if pilot else config.get("fractions", [.125, .25, .5, 1.0])
    ridge = sorted(set(float(x) for x in config.get("ridge", [.001, .03, 1.0])))
    if not ridge or min(ridge) <= 0:
        raise ValueError("Positive ridge candidates required")
    pilot_ridge = float(config.get("pilot_ridge", .03))
    if pilot and pilot_ridge not in ridge:
        raise ValueError("Pilot ridge must be one of the documented ridge candidates")
    lower, upper = np.asarray(data["lower"]), np.asarray(data["upper"])
    opened, scores = np.asarray(data["upper_open"]), np.asarray(data["scores"])
    names, target_names = np.asarray(data["molecule_names"]), np.asarray(data["target_names"])
    methods = [str(m) for m in data["methods"]]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Distinct native method names are required")
    groups, esm = np.asarray(data["sequence_sha1"], dtype=str), np.asarray(data["esm_embeddings"])
    if lower.shape != (len(names), len(target_names)) or scores.shape != (len(methods), *lower.shape):
        raise ValueError("Cold-target data axes disagree")
    if len(groups) != len(target_names) or len(esm) != len(target_names):
        raise ValueError("Target feature axes disagree")
    native_support = np.isfinite(scores).all(axis=0)
    feature = data["molecule_features"]
    molecular_kernel = MolecularKernel.fit(feature.fingerprints, feature.descriptors)
    molecular = KernelBasis.fit((1 + molecular_kernel.kernel(feature.fingerprints, feature.descriptors)) / 2).training
    ones = np.ones((len(names), 1))
    folds, allocations = target_allocations(groups, fractions, config.get("outer_folds", 5),
                                            config.get("inner_folds", 3), config.get("seed", 0),
                                            cold_split=cold_split)
    if pilot:
        allocations = [a for a in allocations if a["outer_fold"] == 0]
    specs = [("baseline", "chemistry", None, False), ("target_prior", "ones", None, False)]
    for j, method in enumerate(methods):
        specs += [("calibrated__" + method, "ones", ("score", j), True),
                  ("availability__" + method, "chemistry", ("missing", j), True),
                  ("plus__" + method, "chemistry", ("score", j), True)]
    if data.get("representations") is not None:
        specs += [("representation_readout", "ones", ("representations", None), False),
                  ("representation_plus_chemistry", "chemistry", ("representations", None), False)]
    if data.get("scalar_members") is not None:
        specs.append(("scalar_members_readout", "ones", ("scalar_members", None), False))
    if pilot:
        affinity = config.get("pilot_affinity_method", "boltz_affinity")
        binder = config.get("pilot_binder_method", "boltz_binder" if "boltz_binder" in methods else "boltz_binary")
        wanted = {"baseline", "plus__" + affinity, "plus__" + binder, "representation_readout"}
        specs = [s for s in specs if s[0] in wanted]

    pair_sources = {("score", j): np.moveaxis(scores[j:j+1], 0, -1) for j in range(len(methods))}
    for key in ["representations", "scalar_members"]:
        if data.get(key) is not None:
            pair_sources[(key, None)] = np.asarray(data[key])
    records, fits, selections, skipped, unavailable = [], [], [], [], []
    predictions, native_predictions = {}, {}
    fit_path, selection_path = out / "cold_fits.jsonl", out / "cold_selections.jsonl"

    def emit(path, item):
        with path.open("a") as handle:
            handle.write(json.dumps(_jsonable(item), allow_nan=False) + "\n")

    if fit_path.exists() or selection_path.exists():
        raise FileExistsError("Cold-target output already contains fit records")

    def partition(train, evaluate):
        transform = ProteinKernelBasis.fit(esm[train])
        return Partition(train, evaluate, transform.training,
                         transform.transform(esm[evaluate]) if len(evaluate) else np.empty((0, transform.training.shape[1])))

    def pair_features(part, source):
        if source is None:
            return None, None
        cache_key = ("score", source[1]) if source[0] == "missing" else source
        if cache_key not in part.pair_cache:
            raw = pair_sources[cache_key]
            scaler = PairScaler.fit(raw[:, part.train])
            train, missing_train = scaler.transform(raw[:, part.train])
            test, missing_test = scaler.transform(raw[:, part.evaluate])
            part.pair_cache[cache_key] = (train, test, missing_train, missing_test)
        train, test, missing_train, missing_test = part.pair_cache[cache_key]
        if source[0] == "missing":
            return missing_train, missing_test
        if source[0] == "score":
            return np.concatenate([train, missing_train], axis=-1), np.concatenate([test, missing_test], axis=-1)
        return train, test

    def fit_one(spec, part, candidate, metadata):
        method, chemistry, source, dependent = spec
        mol = molecular if chemistry == "chemistry" else ones
        train_pairs, test_pairs = pair_features(part, source)
        model = GridIntervalRidge(alpha=candidate / len(part.train),
                                  target_dependent_pair=dependent, maxiter=config.get("maxiter", 500))
        try:
            model.fit(mol, part.target_train, lower[:, part.train], upper[:, part.train], train_pairs)
        except Exception as error:
            failure = dict(metadata, method=method, ridge_candidate=candidate,
                           training_target_count=len(part.train), status="failed",
                           error=str(error), diagnostics=getattr(model, "diagnostics_", None))
            emit(fit_path, failure)
            raise
        item = dict(metadata, method=method, ridge_candidate=candidate,
                    training_target_count=len(part.train), training_sequence_groups=len(np.unique(groups[part.train])),
                    **model.diagnostics_)
        fits.append(item)
        emit(fit_path, item)
        return model, mol, test_pairs

    for allocation in allocations:
        fold, fraction = allocation["outer_fold"], allocation["fraction"]
        train, test = allocation["train_indices"], allocation["test_indices"]
        if allocation.get("skip_reason"):
            skipped.append({"outer_fold": fold, "fraction": fraction, "reason": allocation["skip_reason"]})
            continue
        # Cache each target transform once per partition across every model and alpha.
        outer = partition(train, np.array([], dtype=int) if pilot else test)
        inner = [] if pilot else [partition(a, b) for a, b in allocation["inner"]]
        if not pilot:
            for j, method in enumerate(methods):
                if all_pairs:
                    key = f"native__{method}__fraction_{fraction:g}__context_{fold}"
                    _store_context_prediction(native_predictions, key, scores[j][:, test], (len(names), 2))
                records.extend(evaluate_context(scores[j][:, test], lower[:, test], upper[:, test], opened[:, test],
                                                native_support[:, test], names, dataset=data["name"], policy=data["policy"],
                                                method="native__" + method, fraction=fraction, fold=fold, errors=False))
        for spec in specs:
            inner_values = []
            for candidate in ([] if pilot else ridge):
                validation_rows = []
                for inner_fold, part in enumerate(inner):
                    model, mol, pair = fit_one(spec, part, candidate,
                                              {"stage": "inner", "outer_fold": fold, "fraction": fraction,
                                               "inner_fold": inner_fold})
                    predicted = _checked_prediction(model, mol, part.target_evaluate, pair)
                    validation_rows.extend(evaluate_context(predicted, lower[:, part.evaluate], upper[:, part.evaluate],
                                                            opened[:, part.evaluate], native_support[:, part.evaluate], names,
                                                            dataset=data["name"], policy=data["policy"], method=spec[0],
                                                            fraction=fraction, fold=inner_fold, errors=False))
                inner_values.append({"ridge_candidate": candidate, "macro": _macro(validation_rows)})
            finite = [x for x in inner_values if np.isfinite(x["macro"])]
            if not pilot and not finite:
                item = {"method": spec[0], "outer_fold": fold, "fraction": fraction,
                        "chosen_ridge": None, "inner_scores": inner_values,
                        "selection": "unavailable_no_determinate_inner_validation_order",
                        "training_targets": len(train), "training_groups": allocation["training_groups"]}
                unavailable.append(item)
                selections.append(item)
                emit(selection_path, item)
                continue  # No fabricated tuning choice, outer fit, or held-out metric.
            choice = (pilot_ridge if pilot else
                      max(finite, key=lambda x: (x["macro"], x["ridge_candidate"]))["ridge_candidate"])
            selection = {"method": spec[0], "outer_fold": fold, "fraction": fraction,
                         "chosen_ridge": choice, "inner_scores": inner_values,
                         "selection": "training_only_technical_pilot" if pilot else
                         "highest_context_safe_molecule_macro; larger_ridge_on_exact_tie",
                         "training_targets": len(train), "training_groups": allocation["training_groups"]}
            selections.append(selection)
            emit(selection_path, selection)
            model, mol, pair = fit_one(spec, outer, choice,
                                      {"stage": "pilot" if pilot else "outer", "outer_fold": fold, "fraction": fraction})
            if pilot:
                continue  # No held-out label evaluation or held-out prediction in pilot mode.
            predicted = _checked_prediction(model, mol, outer.target_evaluate, pair)
            key = f"{spec[0]}__fraction_{fraction:g}"
            if all_pairs:
                _store_context_prediction(predictions, key + f"__context_{fold}", predicted, (len(names), 2))
            else:
                predictions.setdefault(key, np.full(lower.shape, np.nan))[:, test] = predicted
            records.extend(evaluate_context(predicted, lower[:, test], upper[:, test], opened[:, test],
                                            native_support[:, test], names, dataset=data["name"], policy=data["policy"],
                                            method=spec[0], fraction=fraction, fold=fold))
        print("COLD_CONTEXT_COMPLETE", json.dumps({"dataset": data["name"], "outer_fold": fold,
                                                  "fraction": fraction, "fits": len(fits), "pilot": pilot}), flush=True)
    context_ids = np.asarray(sorted({a["outer_fold"] for a in allocations}), dtype=int)
    context_lookup = {a["outer_fold"]: a["test_indices"] for a in allocations}
    expected_contexts = len(context_ids)
    split_record = {"target_names": target_names, "sequence_sha1": groups, "target_fold": folds,
                    "cold_split": cold_split, "context_ids": context_ids,
                    "context_targets": {str(k): context_lookup[k] for k in context_ids},
                    "allocations": allocations, "budget_rounding": "ceil(fraction * outer_training_sequence_groups)",
                    "molecules": "All supplied molecules are known and remain fixed at every target budget."}
    (out / "cold_splits.json").write_text(json.dumps(_jsonable(split_record), indent=2) + "\n")
    if not pilot:
        if all_pairs:
            for row in records:
                pair = context_lookup[row["outer_fold"]]
                row.update(context_id=row["outer_fold"], cold_split=cold_split,
                           target_1=str(target_names[pair[0]]), target_2=str(target_names[pair[1]]))
        molecules, aggregate = pool_contexts(records)
        coverage = {}
        for row in records:
            coverage.setdefault((row["method"], row["fraction"]), set()).add(row["outer_fold"])
        for row in aggregate:
            row["evaluated_target_folds"] = len(coverage[(row["method"], row["fraction"])])
            row["expected_target_folds"] = expected_contexts
            row["evaluated_target_contexts"] = row["evaluated_target_folds"]
            row["expected_target_contexts"] = expected_contexts
            row["context_coverage_complete"] = row["evaluated_target_folds"] == row["expected_target_folds"]
        pd.DataFrame(records).to_csv(out / "cold_query_metrics.csv", index=False)
        pd.DataFrame(molecules).to_csv(out / "cold_molecule_metrics.csv", index=False)
        pd.DataFrame(aggregate).to_csv(out / "cold_aggregate_metrics.csv", index=False)
        archive = dict(predictions, molecule_names=names.astype(str),
                       target_names=target_names.astype(str), sequence_sha1=groups,
                       native_common_support=native_support, cold_split=cold_split)
        if all_pairs:
            context_targets = np.stack([context_lookup[k] for k in context_ids])
            archive.update(native_predictions)
            archive.update(context_ids=context_ids, context_targets=context_targets,
                           context_target_names=target_names[context_targets].astype(str),
                           context_sequence_sha1=groups[context_targets],
                           context_native_common_support=np.stack([native_support[:, p] for p in context_targets]),
                           native_methods=np.asarray(methods, dtype=str),
                           prediction_layout="One method/fraction/context array per fit, with axes molecule,held_target (2).",
                           comparison_scope="Compare only the two targets in the same context; never stitch target predictions across contexts.")
        else:
            archive.update(target_fold=folds,
                           comparison_scope="Compare targets only when target_fold is equal; never rank a stitched matrix.")
        np.savez_compressed(out / "cold_oof.npz", **archive)
    summary = {"dataset": data["name"], "policy": data["policy"], "mode": mode,
               "regime": "new_targets_known_molecules", "fits": len(fits), "models": [s[0] for s in specs],
               "fractions": fractions, "skipped_contexts": skipped, "unavailable_arms": unavailable,
               "cold_split": cold_split, "expected_target_contexts": expected_contexts,
               "elapsed_seconds": time.perf_counter() - started,
               "test_labels_evaluated": bool(records), "test_predictions_saved": bool(predictions),
               "prediction_comparison_scope": "Within held-target context only; pool pair credits per molecule before macro averaging.",
               "context_identifier": "outer_fold is a compatibility field naming the held-target context; pair-mode metrics also expose context_id.",
               "error_observation_unit": "molecule-target-context; each target appears in nine contexts" if all_pairs else "molecule-target",
               "molecular_feature_scope": "All known molecules; no held-target labels used in feature fitting.",
               "target_and_pair_feature_scope": "Fitted only on the training target partition, separately inside inner folds.",
               "native_common_finite_cells": int(native_support.sum()),
               "native_inference_runs": 0, "paid_compute": 0}
    (out / "cold_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary
