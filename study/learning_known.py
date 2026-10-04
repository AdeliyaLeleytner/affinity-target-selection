"""Nested scaffold-group learning curves for new molecules / known targets.

All ridge selection is confined to inner validation groups. Fractions describe
whole training-scaffold groups, with group order independent of outcomes.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import time

import numpy as np

from .features import MolecularKernel, grouped_folds, nested_group_subsets
from .metrics import pair_order
from .models import GridIntervalRidge, KernelBasis, PairScaler


STRATA = ("all_determinate", "both_exact", "one_bounded", "both_bounded")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _append_json(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(value), allow_nan=False) + "\n")


def _write_csv(path, records):
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _known(lower, upper):
    return (~np.isnan(lower) & ~np.isnan(upper)
            & (np.isfinite(lower) | np.isfinite(upper)))


def _order_masks(lower, upper, opened, support, pairs):
    a, b = pairs.T
    truth = pair_order(lower[None], upper[None], opened[None], pairs)[0]
    usable = (truth != 0) & support[a] & support[b]
    exact = np.isfinite(lower) & (lower == upper) & ~opened
    return truth, (usable, usable & exact[a] & exact[b],
                   usable & (exact[a] ^ exact[b]),
                   usable & ~exact[a] & ~exact[b])


def evaluate_queries(predictions, lower, upper, upper_open, support, query_names,
                     fold_ids, *, dataset, policy, method, fraction,
                     error_metrics=True):
    """Per-molecule target ordering and errors on a predeclared common support.

    Four rows per molecule contain determinate target-pair strata. Cell-level
    interval and exact MAEs appear only in ``all_determinate`` rows. Set
    ``error_metrics=False`` for native ranking scores without an affinity-unit
    interpretation. An unavailable model is reported, never silently evaluated
    on a smaller method-specific finite-prediction support.
    """
    predictions = np.asarray(predictions, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    upper_open, support = np.asarray(upper_open, bool), np.asarray(support, bool)
    if predictions.ndim != 2 or any(x.shape != predictions.shape for x in (lower, upper, upper_open, support)):
        raise ValueError("Metric arrays must share molecule-by-target axes")
    if len(query_names) != len(predictions) or len(fold_ids) != len(predictions):
        raise ValueError("Query names or fold IDs do not match metric rows")
    pairs = np.stack(np.triu_indices(predictions.shape[1], 1), axis=1)
    a, b = pairs.T
    rows = []
    for q, prediction in enumerate(predictions):
        truth, masks = _order_masks(lower[q], upper[q], upper_open[q], support[q], pairs)
        available = bool(np.isfinite(prediction[support[q]]).all())
        credit = (np.sign(prediction[a] - prediction[b]) * truth + 1) / 2
        observed = support[q] & _known(lower[q], upper[q])
        exact = observed & np.isfinite(lower[q]) & (lower[q] == upper[q]) & ~upper_open[q]
        for stratum, mask in zip(STRATA, masks):
            count = int(mask.sum()) if available else 0
            total = float(credit[mask].sum()) if count else 0.0
            row = {"dataset": dataset, "policy": policy, "fraction": fraction,
                   "method": method, "query": str(query_names[q]),
                   "outer_fold": int(fold_ids[q]), "stratum": stratum,
                   "status": "available" if available else "unavailable_model",
                   "eligible_pairs": int(mask.sum()), "pairs": count, "credit": total,
                   "concordance": total / count if count else None,
                   "error_scale": "affinity_units" if error_metrics else "native_rank_only",
                   "eligible_labels": None, "labels": None, "interval_abs_error": None,
                   "interval_mae": None, "exact_labels": None,
                   "exact_abs_error": None, "exact_mae": None}
            if stratum == "all_determinate" and error_metrics:
                count_labels = int(observed.sum()) if available else 0
                count_exact = int(exact.sum()) if available else 0
                residual = (np.minimum(prediction[observed] - lower[q, observed], 0)
                            + np.maximum(prediction[observed] - upper[q, observed], 0))
                error = float(np.abs(residual).sum()) if count_labels else 0.0
                exact_error = float(np.abs(prediction[exact] - lower[q, exact]).sum()) if count_exact else 0.0
                row.update(eligible_labels=int(observed.sum()), labels=count_labels,
                           interval_abs_error=error,
                           interval_mae=error / count_labels if count_labels else None,
                           exact_labels=count_exact, exact_abs_error=exact_error,
                           exact_mae=exact_error / count_exact if count_exact else None)
            rows.append(row)
    return rows


def aggregate_queries(records):
    """Macro molecule means and pooled pair/cell estimates with explicit support."""
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ("dataset", "policy", "fraction", "method", "stratum"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in groups.items():
        concordances = [r["concordance"] for r in rows if r["concordance"] is not None]
        pairs, credit = sum(r["pairs"] for r in rows), sum(r["credit"] for r in rows)
        row = dict(zip(("dataset", "policy", "fraction", "method", "stratum"), key))
        row.update(queries=len(rows), evaluable_queries=len(concordances),
                   unavailable_queries=sum(r["status"] != "available" for r in rows),
                   eligible_pairs=sum(r["eligible_pairs"] for r in rows), pairs=pairs,
                   credit=credit, macro_concordance=float(np.mean(concordances)) if concordances else None,
                   pooled_concordance=credit / pairs if pairs else None,
                   labels=None, exact_labels=None, error_queries=None, exact_error_queries=None,
                   macro_interval_mae=None, pooled_interval_mae=None,
                   macro_exact_mae=None, pooled_exact_mae=None)
        if any(r["labels"] is not None for r in rows):
            labels = sum(r["labels"] or 0 for r in rows)
            exact_labels = sum(r["exact_labels"] or 0 for r in rows)
            means = [r["interval_mae"] for r in rows if r["interval_mae"] is not None]
            exact_means = [r["exact_mae"] for r in rows if r["exact_mae"] is not None]
            row.update(labels=labels, exact_labels=exact_labels,
                       error_queries=len(means), exact_error_queries=len(exact_means),
                       macro_interval_mae=float(np.mean(means)) if means else None,
                       pooled_interval_mae=sum(r["interval_abs_error"] or 0 for r in rows) / labels if labels else None,
                       macro_exact_mae=float(np.mean(exact_means)) if exact_means else None,
                       pooled_exact_mae=sum(r["exact_abs_error"] or 0 for r in rows) / exact_labels if exact_labels else None)
        result.append(row)
    return result


class _ValidationOrder:
    """Cache validation pair identities only within one current partition."""
    def __init__(self, lower, upper, opened, support):
        pairs = np.stack(np.triu_indices(lower.shape[1], 1), axis=1)
        a, b = pairs.T
        self.support = support
        self.queries = []
        for q in range(len(lower)):
            truth, masks = _order_masks(lower[q], upper[q], opened[q], support[q], pairs)
            use = masks[0]
            self.queries.append((a[use].astype(np.int32), b[use].astype(np.int32), truth[use]))

    def score(self, predictions):
        if not np.isfinite(predictions[self.support]).all():
            raise ValueError("Non-finite model prediction on common native support")
        scores, pairs = [], 0
        for prediction, (a, b, truth) in zip(predictions, self.queries):
            if len(a):
                scores.append(float(np.mean((np.sign(prediction[a] - prediction[b]) * truth + 1) / 2)))
                pairs += len(a)
        return {"sum_concordance": float(sum(scores)), "queries": len(scores), "pairs": pairs,
                "mean_concordance": float(np.mean(scores)) if scores else None}


def split_plan(groups, config):
    """Return deterministic outer/budget/inner partitions without accepting labels."""
    groups = np.asarray(groups)
    fractions = [float(value) for value in config["fractions"]]
    if (not fractions or fractions != sorted(set(fractions))
            or any(not np.isfinite(x) or not 0 < x <= 1 for x in fractions)):
        raise ValueError("Fractions must be unique, increasing values in (0,1]")
    folds = grouped_folds(groups, n_splits=int(config["outer_folds"]), seed=int(config["seed"]))
    plans = []
    for outer in range(int(config["outer_folds"])):
        training = np.flatnonzero(folds != outer)
        testing = np.flatnonzero(folds == outer)
        count = len(set(groups[training]))
        budgets = [int(math.ceil(fraction * count)) for fraction in fractions]
        subsets = nested_group_subsets(groups[training], budgets, seed=int(config["seed"]) + outer)
        for fraction, budget, local in zip(fractions, budgets, subsets):
            selected = training[local]
            plan = {"outer_fold": outer, "fraction": fraction,
                    "outer_training_indices": training.tolist(), "test_indices": testing.tolist(),
                    "training_indices": selected.tolist(), "outer_training_groups": count,
                    "budget_groups": budget, "training_molecules": len(selected),
                    "inner": [], "status": "available"}
            if budget < int(config["inner_folds"]):
                plan["status"] = "insufficient_groups_for_inner_folds"
            else:
                inner = grouped_folds(groups[selected], n_splits=int(config["inner_folds"]),
                                      seed=int(config["seed"]) + 1000 + outer)
                for fold in range(int(config["inner_folds"])):
                    plan["inner"].append({"fold": fold,
                                          "training_indices": selected[inner != fold].tolist(),
                                          "validation_indices": selected[inner == fold].tolist()})
            plans.append(plan)
    return folds, plans


@dataclass(frozen=True)
class _Arm:
    name: str
    kind: str
    native: int = -1


def score_components(scores):
    """Label-free molecule-wide mean and target-relative native-score profiles.

    The mean uses all available targets for the same molecule and scorer. Both
    returned profiles retain the original cell mask; an entirely unavailable
    profile remains unavailable. This transformation still requires acquiring
    the original target-profile scores.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or not scores.shape[1]:
        raise ValueError("Scores must have molecule-by-target axes")
    finite = np.isfinite(scores)
    counts = finite.sum(axis=1, keepdims=True)
    average = np.divide(np.where(finite, scores, 0).sum(axis=1, keepdims=True), counts,
                        out=np.full((len(scores), 1), np.nan), where=counts > 0)
    mean = np.where(finite, average, np.nan)
    relative = np.subtract(scores, average, out=np.full_like(scores, np.nan), where=finite)
    return mean, relative


def _arms(data, config=None):
    family = (config or {}).get("arm_family", "standard")
    if family not in ("standard", "score_components"):
        raise ValueError(f"Unknown known-target arm family: {family}")
    result = [_Arm("target_prior", "prior"), _Arm("chemistry", "chemistry")]
    reported = [a.name for a in result]
    aliases, masks = {}, {}
    for j, method in enumerate(data["methods"]):
        calibration = _Arm("calibrated__" + method, "calibrated", j)
        availability = _Arm("availability__" + method, "availability", j)
        plus = _Arm("plus__" + method, "plus", j)
        if family == "standard":
            result.append(calibration)
        key = np.packbits(~np.isfinite(data["scores"][j])).tobytes()
        if key in masks:
            aliases[availability.name] = masks[key]
        else:
            masks[key] = availability.name
            result.append(availability)
        result.append(plus)
        if family == "standard":
            reported.extend((calibration.name, availability.name, plus.name))
        else:
            mean = _Arm("mean__" + method, "mean", j)
            mean_relative = _Arm("mean_relative__" + method, "mean_relative", j)
            result.extend((mean, mean_relative))
            reported.extend((availability.name, plus.name, mean.name, mean_relative.name))
    if family == "score_components":
        return result, aliases, reported
    if data.get("representations") is not None:
        for name, kind in (("representation_readout", "representation"),
                           ("representation_plus_chemistry", "representation_plus"),
                           ("representation_pca4_readout", "representation_pca4")):
            result.append(_Arm(name, kind))
            reported.append(name)
    if data.get("scalar_members") is not None:
        result.append(_Arm("scalar_members_readout", "scalar_members"))
        reported.append("scalar_members_readout")
    return result, aliases, reported


class _Partition:
    """Fitted preprocessing cached only for this exact training partition."""
    def __init__(self, data, training, prediction):
        self.data = data
        self.training, self.prediction = np.asarray(training, int), np.asarray(prediction, int)
        self.target = np.eye(data["lower"].shape[1])
        self.ones = (np.ones((len(self.training), 1)), np.ones((len(self.prediction), 1)))
        self.chemical = None
        self.pairs = {}
        self.component_profiles = {}

    def chemistry(self):
        if self.chemical is None:
            f = self.data["molecule_features"]
            kernel = MolecularKernel.fit(f.fingerprints[self.training], f.descriptors[self.training])
            gram = (1 + kernel.against_train(f.fingerprints[self.training], f.descriptors[self.training])) / 2
            basis = KernelBasis.fit(gram)
            cross = (1 + kernel.against_train(f.fingerprints[self.prediction], f.descriptors[self.prediction])) / 2
            self.chemical = (basis.training, basis.transform(cross))
        return self.chemical

    def score_pairs(self, index):
        key = ("score", index)
        if key not in self.pairs:
            raw = self.data["scores"][index, ..., None]
            scaler = PairScaler.fit(raw[self.training])
            train, missing_train = scaler.transform(raw[self.training])
            predict, missing_predict = scaler.transform(raw[self.prediction])
            self.pairs[key] = (np.concatenate((train, missing_train), axis=-1),
                               np.concatenate((predict, missing_predict), axis=-1))
        return self.pairs[key]

    def component_pairs(self, index, include_relative):
        key = ("mean_relative" if include_relative else "mean", index)
        if key not in self.pairs:
            if index not in self.component_profiles:
                self.component_profiles[index] = score_components(self.data["scores"][index])
            mean, relative = self.component_profiles[index]
            raw = np.stack((mean, relative), axis=-1) if include_relative else mean[..., None]
            scaler = PairScaler.fit(raw[self.training])
            train = scaler.transform(raw[self.training])[0]
            predict = scaler.transform(raw[self.prediction])[0]
            # M and R have the same source mask. Append that mask once, rather
            # than duplicating the missingness coefficient in the two-channel arm.
            missing = (~np.isfinite(self.data["scores"][index]))[..., None].astype(float)
            self.pairs[key] = (np.concatenate((train, missing[self.training]), axis=-1),
                               np.concatenate((predict, missing[self.prediction]), axis=-1))
        return self.pairs[key]

    def representations(self):
        if "representations" not in self.pairs:
            raw = self.data["representations"]
            scaler = PairScaler.fit(raw[self.training])
            self.pairs["representations"] = (scaler.transform(raw[self.training])[0],
                                               scaler.transform(raw[self.prediction])[0])
        return self.pairs["representations"]

    def scalar_members(self):
        if "scalar_members" not in self.pairs:
            raw = self.data["scalar_members"]
            scaler = PairScaler.fit(raw[self.training])
            self.pairs["scalar_members"] = (scaler.transform(raw[self.training])[0],
                                             scaler.transform(raw[self.prediction])[0])
        return self.pairs["scalar_members"]

    def representation_pca4(self):
        if "representation_pca4" not in self.pairs:
            training, prediction = self.representations()
            flat = training.reshape(-1, training.shape[-1])
            center = flat.mean(axis=0)
            centered = flat - center
            # Fixed four-component dimension control; no variance-explained
            # cutoff and no labels. Standardization before and after PCA uses
            # only this partition's training cells.
            _, vectors = np.linalg.eigh(centered.T @ centered)
            components = vectors[:, -4:][:, ::-1]
            projected_train = (training - center) @ components
            projected_predict = (prediction - center) @ components
            scaler = PairScaler.fit(projected_train)
            self.pairs["representation_pca4"] = (scaler.transform(projected_train)[0],
                                                  scaler.transform(projected_predict)[0])
        return self.pairs["representation_pca4"]

    def features(self, arm):
        chemical = arm.kind in ("chemistry", "availability", "plus", "mean",
                                "mean_relative", "representation_plus")
        molecular = self.chemistry() if chemical else self.ones
        if arm.kind in ("calibrated", "plus"):
            pairs = self.score_pairs(arm.native)
        elif arm.kind in ("mean", "mean_relative"):
            pairs = self.component_pairs(arm.native, include_relative=arm.kind == "mean_relative")
        elif arm.kind == "availability":
            missing = (~np.isfinite(self.data["scores"][arm.native]))[..., None].astype(float)
            pairs = (missing[self.training], missing[self.prediction])
        elif arm.kind in ("representation", "representation_plus"):
            pairs = self.representations()
        elif arm.kind == "representation_pca4":
            pairs = self.representation_pca4()
        elif arm.kind == "scalar_members":
            pairs = self.scalar_members()
        else:
            pairs = (None, None)
        return molecular, pairs, arm.kind in ("calibrated", "availability", "plus", "mean", "mean_relative")


def _fit_arm(data, partition, arm, ridge, config, out, context, predict=True):
    molecular, pairs, dependent = partition.features(arm)
    indices = partition.training
    lower, upper = data["lower"][indices], data["upper"][indices]
    known = _known(lower, upper)
    exact = known & np.isfinite(lower) & (lower == upper) & ~data["upper_open"][indices]
    record = {"dataset": data["name"], "policy": data["policy"], **context,
              "method": arm.name, "ridge": float(ridge),
              "training_molecules": len(indices),
              "training_groups": len(set(data["molecule_features"].scaffold_groups[indices])),
              "known_training_labels": int(known.sum()), "exact_training_labels": int(exact.sum()),
              "censored_or_interval_training_labels": int((known & ~exact).sum())}
    model = GridIntervalRidge(alpha=float(ridge) / data["lower"].shape[1],
                              target_dependent_pair=dependent,
                              maxiter=int(config["maxiter"]))
    print(f"{data['name']} {data['policy']} known outer={context['outer_fold']} "
          f"fraction={context['fraction']:g} {context['stage']} "
          f"inner={context.get('inner_fold', '-')} {arm.name} ridge={ridge:g}", flush=True)
    try:
        model.fit(molecular[0], partition.target, lower, upper, pairs[0])
    except (RuntimeError, ValueError) as error:
        record.update(status="failed", error=str(error), diagnostics=getattr(model, "diagnostics_", None))
        _append_json(out / "known_fits.jsonl", record)
        raise
    record.update(status="fitted", diagnostics=model.diagnostics_)
    _append_json(out / "known_fits.jsonl", record)
    if not predict:
        return None, record
    prediction = model.predict(molecular[1], partition.target, pairs[1])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite predictions from {arm.name}")
    return prediction, record


def _validate_data(data):
    shape = data["lower"].shape
    if len(shape) != 2 or data["upper"].shape != shape or data["upper_open"].shape != shape:
        raise ValueError("Label arrays must share molecule-by-target axes")
    if data["scores"].shape != (len(data["methods"]),) + shape:
        raise ValueError("Native scores do not align with labels and method names")
    if len(set(data["methods"])) != len(data["methods"]) or len(data["methods"]) == 0:
        raise ValueError("Native method names must be nonempty and unique")
    if len(data["molecule_names"]) != shape[0] or len(set(data["molecule_names"])) != shape[0]:
        raise ValueError("Molecule names must uniquely identify each row")
    features = data["molecule_features"]
    if features.fingerprints.shape[0] != shape[0] or features.descriptors.shape[0] != shape[0]:
        raise ValueError("Molecular features have mismatched rows")
    if (not np.isfinite(features.descriptors).all()
            or np.any((features.fingerprints != 0) & (features.fingerprints != 1))):
        raise ValueError("Cached molecular descriptors must be finite and fingerprints binary")
    if any(len(values) != shape[0] for values in (features.identities, features.scaffolds,
                                                features.scaffold_groups)):
        raise ValueError("Molecular identity/scaffold metadata have mismatched rows")
    if data.get("representations") is not None:
        reps = data["representations"]
        if reps.ndim != 3 or reps.shape[:2] != shape or reps.shape[-1] < 4 or not np.isfinite(reps).all():
            raise ValueError("Cached representations must be finite and align with the label grid")
    if data.get("scalar_members") is not None:
        members = data["scalar_members"]
        if members.shape != shape + (4,) or not np.isfinite(members).all():
            raise ValueError("Four scalar members must be finite and align with the label grid")


def run_known(data, out, config):
    """Run numerical pilot or complete nested known-target learning curves."""
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"mode": "full", "fractions": [0.125, 0.25, 0.5, 1.0],
              "outer_folds": 5, "inner_folds": 3, "seed": 20261004,
              "ridge": [0.001, 0.03, 1.0], "maxiter": 500, **config}
    if config["mode"] not in ("technical_pilot", "full"):
        raise ValueError("Unknown learning mode")
    if not config["ridge"] or any(not np.isfinite(x) or x <= 0 for x in config["ridge"]):
        raise ValueError("Ridge candidates must be finite and positive")
    _validate_data(data)
    arms, aliases, reported = _arms(data, config)
    # Pilot split construction uses features only; no held-out label values are
    # sliced, summarized, predicted or evaluated anywhere in its branch.
    split_config = {**config, "fractions": [1.0]} if config["mode"] == "technical_pilot" else config
    fold_ids, plans = split_plan(data["molecule_features"].scaffold_groups, split_config)
    _write_json(out / "known_splits.json", {"config": split_config, "plans": plans})
    for name in ("known_fits.jsonl", "known_selections.jsonl"):
        (out / name).write_text("")
    summary = {"dataset": data["name"], "policy": data["policy"], "regime": "new_molecules_known_targets",
               "mode": config["mode"], "config": config, "availability_aliases": aliases,
               "arm_family": config.get("arm_family", "standard"),
               "ridge_definition": "model alpha = listed ridge / number of known targets",
               "pca4_definition": "PCA on training-standardized representation cells; first four PCs restandardized using training cells only",
               "selection_metric": "macro within-molecule determinate target-pair concordance on common finite-native support",
               "status": "provisional" if config["mode"] == "technical_pilot" else "complete"}
    if config.get("arm_family") == "score_components":
        summary["score_components"] = {
            "mean": "Per-scorer mean over available target scores for each molecule; original cell mask retained.",
            "relative": "Original score minus that molecule's profile mean; original cell mask retained.",
            "pair_features": {"plus": 2, "mean": 2, "mean_relative": 3, "availability": 1},
            "missingness": "Exactly one original per-cell missingness indicator in each arm.",
            "scaling": "PairScaler fitted on the current training partition, separately for each arm.",
            "interpretation": "Predictive sufficiency of molecule-wide summaries versus conditional target-relative variation; no biological causality or absence-of-interaction-information claim.",
            "acquisition_cost": "The mean uses all available original target-profile scores; no inference-cost saving is established."}
    if config["mode"] == "technical_pilot":
        affinity = config.get("pilot_affinity_method", "boltz_affinity")
        binder = config.get("pilot_binder_method", "boltz_binary" if "boltz_binary" in data["methods"] else "boltz_binder")
        if config.get("arm_family") == "score_components":
            names = ["chemistry", "plus__" + affinity, "mean__" + affinity,
                     "mean_relative__" + affinity, "plus__" + binder]
        else:
            names = ["chemistry", "calibrated__" + affinity, "plus__" + affinity, "plus__" + binder]
        if config.get("arm_family", "standard") == "standard" and data.get("representations") is not None:
            names.append("representation_readout")
        lookup = {arm.name: arm for arm in arms}
        if any(name not in lookup for name in names):
            raise ValueError(f"Pilot native heads are absent: {names}")
        plan = next(p for p in plans if p["outer_fold"] == 0)
        partition = _Partition(data, plan["training_indices"], [])
        context = {"outer_fold": 0, "fraction": 1.0, "stage": "technical_pilot"}
        diagnostics = []
        for name in names:
            _, record = _fit_arm(data, partition, lookup[name], 0.03, config, out, context, predict=False)
            diagnostics.append(record)
        summary.update(fits=diagnostics, test_metrics_evaluated=False,
                       elapsed_seconds=time.perf_counter() - started)
        _write_json(out / "known_summary.json", summary)
        return summary

    fractions = [float(x) for x in config["fractions"]]
    ridge_values = sorted(set(float(x) for x in config["ridge"]))
    n, t = data["lower"].shape
    oof = np.full((len(reported), len(fractions), n, t), np.nan)
    selected_ridge = np.full((len(reported), len(fractions), int(config["outer_folds"])), np.nan)
    method_pos = {name: i for i, name in enumerate(reported)}
    common = np.isfinite(data["scores"]).all(axis=0)
    unavailable, fit_count, selections = [], 0, []
    for plan in plans:
        fraction, outer = plan["fraction"], plan["outer_fold"]
        fi = fractions.index(fraction)
        context = {"outer_fold": outer, "fraction": fraction}
        if plan["status"] != "available":
            record = {**context, "status": plan["status"]}
            unavailable.append(record)
            _append_json(out / "known_selections.jsonl", record)
            continue
        candidates = {(arm.name, ridge): {"sum_concordance": 0., "queries": 0, "pairs": 0, "inner": []}
                      for arm in arms for ridge in ridge_values}
        block_unavailable = None
        for inner in plan["inner"]:
            train, valid = np.asarray(inner["training_indices"]), np.asarray(inner["validation_indices"])
            if not _known(data["lower"][train], data["upper"][train]).any():
                block_unavailable = "inner_training_has_no_informative_labels"
                break
            partition = _Partition(data, train, valid)
            order = _ValidationOrder(data["lower"][valid], data["upper"][valid],
                                     data["upper_open"][valid], common[valid])
            inner_context = {**context, "stage": "inner", "inner_fold": inner["fold"]}
            for arm in arms:
                for ridge in ridge_values:
                    prediction, _ = _fit_arm(data, partition, arm, ridge, config, out, inner_context)
                    fit_count += 1
                    score = order.score(prediction)
                    candidate = candidates[(arm.name, ridge)]
                    for field in ("sum_concordance", "queries", "pairs"):
                        candidate[field] += score[field]
                    candidate["inner"].append({"fold": inner["fold"], **score})
            del partition
        if block_unavailable is not None:
            record = {**context, "status": block_unavailable}
            unavailable.append(record)
            _append_json(out / "known_selections.jsonl", record)
            continue
        winners = {}
        for arm in arms:
            options = []
            for ridge in ridge_values:
                item = candidates[(arm.name, ridge)]
                mean = item["sum_concordance"] / item["queries"] if item["queries"] else None
                options.append({"ridge": ridge, "macro_concordance": mean, **item})
            eligible = [option for option in options if option["macro_concordance"] is not None]
            if not eligible:
                record = {**context, "method": arm.name, "status": "unestimable_inner_concordance", "candidates": options}
                unavailable.append(record)
            else:
                winner = max(eligible, key=lambda option: (option["macro_concordance"], option["ridge"]))
                winners[arm.name] = winner["ridge"]
                record = {**context, "method": arm.name, "status": "selected",
                          "selected_ridge": winner["ridge"], "selected_alpha": winner["ridge"] / t,
                          "candidates": options}
            selections.append(record)
            _append_json(out / "known_selections.jsonl", record)
        final_partition = _Partition(data, plan["training_indices"], plan["test_indices"])
        for arm in arms:
            if arm.name not in winners:
                continue
            prediction, _ = _fit_arm(data, final_partition, arm, winners[arm.name], config, out,
                                     {**context, "stage": "outer_final"})
            fit_count += 1
            oof[method_pos[arm.name], fi, plan["test_indices"]] = prediction
            selected_ridge[method_pos[arm.name], fi, outer] = winners[arm.name]
        del final_partition
    for alias, canonical in aliases.items():
        oof[method_pos[alias]] = oof[method_pos[canonical]]
        selected_ridge[method_pos[alias]] = selected_ridge[method_pos[canonical]]
    np.savez_compressed(out / "known_oof.npz", predictions=oof, methods=np.asarray(reported),
                        fractions=np.asarray(fractions), selected_ridge=selected_ridge,
                        native_scores=data["scores"], native_methods=np.asarray(data["methods"]),
                        common_native_finite=common, native_missing=~np.isfinite(data["scores"]),
                        fold_ids=fold_ids, molecule_names=np.asarray(data["molecule_names"], dtype=str),
                        scaffold_groups=data["molecule_features"].scaffold_groups,
                        target_names=np.asarray(data.get("target_names", [str(i) for i in range(t)]), dtype=str))
    rows = []
    metric_args = (data["lower"], data["upper"], data["upper_open"], common,
                   data["molecule_names"], fold_ids)
    for fi, fraction in enumerate(fractions):
        for name in reported:
            rows.extend(evaluate_queries(oof[method_pos[name], fi], *metric_args,
                                         dataset=data["name"], policy=data["policy"],
                                         method=name, fraction=fraction))
        for j, name in enumerate(data["methods"]):
            rows.extend(evaluate_queries(data["scores"][j], *metric_args,
                                         dataset=data["name"], policy=data["policy"],
                                         method="native__" + name, fraction=fraction,
                                         error_metrics=False))
    aggregate = aggregate_queries(rows)
    _write_csv(out / "known_query_metrics.csv", rows)
    _write_csv(out / "known_aggregate_metrics.csv", aggregate)
    summary.update(methods=reported, fits=fit_count, selected_models=len(selections),
                   unavailable=unavailable, test_metrics_evaluated=True,
                   oof_shape=list(oof.shape), oof_axes=["method", "fraction", "molecule", "target"],
                   finite_oof_cells=int(np.isfinite(oof).sum()), common_native_cells=int(common.sum()),
                   elapsed_seconds=time.perf_counter() - started)
    if unavailable:
        summary["status"] = "complete_with_unavailable_blocks"
    _write_json(out / "known_summary.json", summary)
    return summary
