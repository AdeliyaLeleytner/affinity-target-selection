import json
import contextlib
import io
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from study.features import molecule_features
from study.learning_cold import (ProteinKernelBasis, evaluate_context, pool_contexts,
                                 run_cold, target_allocations, _store_context_prediction)
from study import learning_run


class ContextTaggedModel:
    """No optimization: distinct fit IDs test the prediction archive plumbing."""
    counter = 0

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha

    def fit(self, molecular, target, lower, upper, pairs=None):
        type(self).counter += 1
        self.identifier = type(self).counter
        known = ~np.isnan(lower) & ~np.isnan(upper)
        self.diagnostics_ = {"success": True, "alpha": self.alpha,
                             "known_labels": int(known.sum()),
                             "exact_labels": int((known & (lower == upper)).sum()),
                             "iterations": 0, "fit_seconds": 0.0}
        return self

    def predict(self, molecular, target, pairs=None):
        return np.broadcast_to(self.identifier + np.arange(len(target), dtype=float),
                               (len(molecular), len(target))).copy()


class ColdTargetTests(unittest.TestCase):
    def test_all_45_pairs_and_inner_training_exclude_held_sequences(self):
        groups = np.array([f"s{i}" for i in range(10)])
        folds, contexts = target_allocations(groups, [1.], seed=17, cold_split="all_target_pairs")
        self.assertIsNone(folds)
        self.assertEqual(len(contexts), 45)
        self.assertEqual([tuple(c["test_indices"]) for c in contexts],
                         list(combinations(range(10), 2)))
        self.assertEqual([c["outer_fold"] for c in contexts], list(range(45)))
        for context in contexts:
            self.assertEqual(len(context["train_indices"]), 8)
            self.assertEqual(len(context["inner"]), 3)
            held = set(groups[context["test_indices"]])
            self.assertFalse(held & set(groups[context["train_indices"]]))
            validation = []
            for train, valid in context["inner"]:
                self.assertFalse(held & set(groups[np.r_[train, valid]]))
                self.assertFalse(set(groups[train]) & set(groups[valid]))
                self.assertEqual(set(train) | set(valid), set(context["train_indices"]))
                validation.extend(valid.tolist())
            self.assertEqual(sorted(validation), sorted(context["train_indices"]))
        aliases = groups.copy()
        aliases[-1] = aliases[0]
        with self.assertRaisesRegex(ValueError, "ten distinct sequence hashes"):
            target_allocations(aliases, [1.], cold_split="all_target_pairs")
        with self.assertRaisesRegex(ValueError, "full eight-target"):
            target_allocations(groups, [.5, 1.], cold_split="all_target_pairs")

    def test_legacy_five_fold_allocation_is_unchanged(self):
        groups = np.array([f"s{i}" for i in range(10)])
        implicit_folds, implicit = target_allocations(groups, [1.], seed=17)
        explicit_folds, explicit = target_allocations(groups, [1.], seed=17,
                                                       cold_split="sequence_folds")
        np.testing.assert_array_equal(implicit_folds, explicit_folds)
        self.assertEqual(len(implicit), 5)
        for old, new in zip(implicit, explicit):
            np.testing.assert_array_equal(old["train_indices"], new["train_indices"])
            np.testing.assert_array_equal(old["test_indices"], new["test_indices"])
            for left, right in zip(old["inner"], new["inner"]):
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[1], right[1])

    def test_sequence_groups_and_nested_budgets(self):
        groups = np.array([f"s{i}" for i in range(30)] + ["s0", "s1"])
        folds, contexts = target_allocations(groups, [.5, 1.0], seed=17)
        self.assertEqual(folds[0], folds[-2])
        self.assertEqual(folds[1], folds[-1])
        for fold in range(5):
            small, large = [x for x in contexts if x["outer_fold"] == fold]
            self.assertTrue(set(small["train_indices"]) <= set(large["train_indices"]))
            np.testing.assert_array_equal(small["test_indices"], large["test_indices"])
        for context in contexts:
            held = set(groups[context["test_indices"]])
            self.assertFalse(held & set(groups[context["train_indices"]]))
            for train, valid in context["inner"]:
                self.assertFalse(set(groups[train]) & set(groups[valid]))
                self.assertFalse(held & set(groups[np.r_[train, valid]]))

    def test_target_preprocessing_uses_training_only_and_zero_vectors(self):
        train = np.array([[1., 2., 0.], [3., 2., 0.], [2., 2., 0.]])
        fit = ProteinKernelBasis.fit(train)
        np.testing.assert_array_equal(fit.mean, [2, 2, 0])
        self.assertTrue(np.isfinite(fit.training).all())
        before = fit.training.copy()
        fit.transform(np.array([[1e8, -1e8, 50.]]))
        np.testing.assert_array_equal(fit.training, before)
        np.testing.assert_allclose(fit.training @ fit.training.T,
                                   (1 + fit.normalized_train @ fit.normalized_train.T) / 2,
                                   atol=1e-12)

    def test_never_forms_pairs_between_target_contexts(self):
        rows = []
        for fold, y, prediction in [(0, [0., 1.], [100., 101.]),
                                    (1, [2., 3.], [0., 1.])]:
            y, prediction = np.array([y]), np.array([prediction])
            rows += evaluate_context(prediction, y, y, np.zeros_like(y, bool),
                                     np.ones_like(y, bool), ["m"], dataset="toy", policy="point",
                                     method="model", fraction=1., fold=fold)
        molecules, summary = pool_contexts(rows)
        row = next(x for x in summary if x["stratum"] == "all_determinate")
        self.assertEqual(row["pairs"], 2)  # A stitched four-target ranking would contain six.
        self.assertEqual(row["macro"], 1.)
        self.assertEqual(len([x for x in molecules if x["stratum"] == "all_determinate"]), 1)

    def test_bounds_and_common_native_support(self):
        lower = np.array([[2., -np.inf, np.nan, 1.]])
        upper = np.array([[2., 2., np.nan, 1.]])
        opened = np.array([[False, True, False, False]])
        prediction = np.array([[3., 0., 5., 8.]])
        support = np.array([[True, True, True, False]])
        rows = evaluate_context(prediction, lower, upper, opened, support, ["m"],
                                dataset="toy", policy="bounds", method="fit", fraction=1., fold=0)
        all_row = next(x for x in rows if x["stratum"] == "all_determinate")
        self.assertEqual(all_row["pairs"], 1)
        self.assertEqual(all_row["credit"], 1)
        self.assertEqual(all_row["error_cells"], 2)
        self.assertEqual(all_row["interval_mae"], .5)
        self.assertEqual(all_row["exact_mae"], 1.)
        prediction[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite predictions"):
            evaluate_context(prediction, lower, upper, opened, support, ["m"],
                             dataset="toy", policy="bounds", method="fit", fraction=1., fold=0)

    def test_all_pair_credits_match_manual_common_support(self):
        lower = np.full((2, 10), np.nan)
        lower[0, :3] = [3., 2., 1.]
        lower[1, :3] = [3., -np.inf, 1.]
        upper = lower.copy()
        upper[1, 1] = 2.
        opened = np.zeros_like(lower, bool)
        prediction = np.zeros_like(lower)
        prediction[0, :3] = [3., 0., 1.]
        prediction[1, :3] = [0., 2., 1.]
        support = np.ones_like(lower, bool)
        support[0, 1] = False
        records = []
        _, contexts = target_allocations([f"s{i}" for i in range(10)], [1.], cold_split="all_target_pairs")
        for context in contexts:
            targets = context["test_indices"]
            records.extend(evaluate_context(prediction[:, targets], lower[:, targets], upper[:, targets],
                                            opened[:, targets], support[:, targets], ["a", "b"],
                                            dataset="toy", policy="bounds", method="model", fraction=1.,
                                            fold=context["outer_fold"]))
        molecules, summaries = pool_contexts(records)
        rows = [r for r in molecules if r["stratum"] == "all_determinate"]
        self.assertEqual([r["pairs"] for r in rows], [1, 2])
        self.assertEqual([r["credit"] for r in rows], [1., 0.])
        total = next(r for r in summaries if r["stratum"] == "all_determinate")
        self.assertEqual(total["pairs"], 3)
        self.assertEqual(total["macro"], .5)
        self.assertAlmostEqual(total["pooled"], 1/3)
        # Cell errors concern separate model contexts; each target is evaluated
        # in nine pairs. They must not be described as 45 unique observations.
        self.assertEqual(total["observed_support_cells"], (2 + 3) * 9)

    def test_context_storage_rejects_overwrite_and_copies_values(self):
        values = np.array([[1., 2.]])
        predictions = {}
        _store_context_prediction(predictions, "arm__context_0", values, (1, 2))
        values[:] = 99
        np.testing.assert_array_equal(predictions["arm__context_0"], [[1, 2]])
        with self.assertRaisesRegex(ValueError, "Duplicate held-target context"):
            _store_context_prediction(predictions, "arm__context_0", values, (1, 2))

    def test_pair_archive_has_separate_fit_for_each_target_context(self):
        rng = np.random.default_rng(11)
        lower = np.arange(30, dtype=float).reshape(3, 10) / 10
        scores = lower[None].copy()
        scores[0, 0, 0] = np.nan
        data = {"name": "toy", "policy": "point", "lower": lower, "upper": lower,
                "upper_open": np.zeros((3, 10), bool), "scores": scores, "methods": ["score"],
                "molecule_names": np.array(["a", "b", "c"]),
                "target_names": np.array([f"t{i}" for i in range(10)]),
                "sequence_sha1": np.array([f"s{i}" for i in range(10)]),
                "esm_embeddings": rng.normal(size=(10, 6)),
                "molecule_features": molecule_features(["CC", "CCC", "CCO"])}
        ContextTaggedModel.counter = 0
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("study.learning_cold.GridIntervalRidge", ContextTaggedModel):
                summary = run_cold(data, directory, {"fractions": [1.], "ridge": [.03],
                                                     "cold_split": "all_target_pairs"})
            self.assertEqual(summary["expected_target_contexts"], 45)
            self.assertEqual(summary["fits"], 900)
            with np.load(Path(directory) / "cold_oof.npz", allow_pickle=False) as saved:
                self.assertNotIn("baseline__fraction_1", saved.files)
                self.assertNotIn("target_fold", saved.files)
                self.assertEqual(saved["context_targets"].shape, (45, 2))
                self.assertEqual(saved["context_native_common_support"].shape, (45, 3, 2))
                self.assertEqual(len([k for k in saved.files if k.startswith("baseline__")]), 45)
                for key in saved.files:
                    if "__context_" in key:
                        self.assertEqual(saved[key].shape, (3, 2))
                first = saved["baseline__fraction_1__context_0"]
                second = saved["baseline__fraction_1__context_1"]
                self.assertFalse(np.array_equal(first[:, 0], second[:, 0]))
                np.testing.assert_array_equal(saved["context_targets"][:2], [[0, 1], [0, 2]])
            table = pd.read_csv(Path(directory) / "cold_aggregate_metrics.csv")
            overall = table[table.stratum.eq("all_determinate")]
            self.assertTrue(overall.pairs.eq(126).all())
            self.assertTrue(overall.expected_target_contexts.eq(45).all())
            self.assertTrue(overall.evaluated_target_contexts.eq(45).all())
            self.assertTrue(overall.context_coverage_complete.all())

    def test_run_blocks_filter_dispatches_only_exact_requested_blocks(self):
        datasets = [{"name": "SPD", "policy": "source_representative"},
                    {"name": "SPD", "policy": "assay_envelope"},
                    {"name": "DAVIS", "policy": "exact_identity"}]
        requested = ["SPD/source_representative/new_targets_known_molecules",
                     "SPD/assay_envelope/new_targets_known_molecules"]
        config = {"mode": "full", "numerical_threads": 1, "block_workers": 1,
                  "regimes": ["new_molecule_known_targets", "new_targets_known_molecules"],
                  "run_blocks": requested}
        dispatched = []
        def execute(data, regime, destination, config):
            dispatched.append(f"{data['name']}/{data['policy']}/{regime}")
            return {"dataset": data["name"], "policy": data["policy"], "regime": regime}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch("study.learning_run.prepare", return_value=datasets):
                with patch("study.learning_run.execute_block", side_effect=execute):
                    learning_run.run(SimpleNamespace(data_root=Path("unused")), Path(directory), config, None, None)
            self.assertEqual(dispatched, requested)
            self.assertFalse((Path(directory) / "DAVIS").exists())
        with tempfile.TemporaryDirectory() as directory:
            with patch("study.learning_run.prepare", return_value=datasets):
                with self.assertRaisesRegex(ValueError, "Unknown run_blocks IDs"):
                    learning_run.run(SimpleNamespace(data_root=Path("unused")), Path(directory),
                                     {**config, "run_blocks": ["SPD/wrong/new_targets_known_molecules"]}, None, None)

    def test_pilot_does_not_predict_or_evaluate_held_labels(self):
        rng = np.random.default_rng(5)
        data = {"name": "toy", "policy": "point", "lower": rng.normal(size=(3, 10)),
                "molecule_names": np.array(["a", "b", "c"]),
                "target_names": np.array([f"t{i}" for i in range(10)]),
                "sequence_sha1": np.array([f"s{i}" for i in range(10)]),
                "esm_embeddings": rng.normal(size=(10, 6)),
                "methods": ["boltz_affinity", "boltz_binder"],
                "scores": rng.normal(size=(2, 3, 10)),
                "molecule_features": molecule_features(["CC", "CCC", "CCO"])}
        data["upper"] = data["lower"].copy()
        data["upper_open"] = np.zeros((3, 10), bool)
        with tempfile.TemporaryDirectory() as directory:
            with patch("study.learning_cold.GridIntervalRidge.predict", side_effect=AssertionError("Pilot predicted")):
                with patch("study.learning_cold.evaluate_context", side_effect=AssertionError("Pilot evaluated")):
                    summary = run_cold(data, directory, {"mode": "technical_pilot", "maxiter": 1000})
            self.assertFalse(summary["test_labels_evaluated"])
            self.assertFalse((Path(directory) / "cold_oof.npz").exists())
            self.assertEqual(summary["fits"], 3)
            records = [json.loads(x) for x in (Path(directory) / "cold_fits.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["stage"] == "pilot" for x in records))
            self.assertTrue(all(x["ridge_candidate"] == .03 for x in records))
            self.assertTrue(all(x["alpha"] == .03 / x["training_target_count"] for x in records))

    def test_tiny_full_run_keeps_context_and_missing_score_support(self):
        rng = np.random.default_rng(11)
        lower = np.arange(30, dtype=float).reshape(3, 10) / 10
        scores = lower[None].copy()
        scores[0, 0, 0] = np.nan
        data = {"name": "toy", "policy": "point", "lower": lower, "upper": lower,
                "upper_open": np.zeros((3, 10), bool), "scores": scores, "methods": ["score"],
                "molecule_names": np.array(["a", "b", "c"]),
                "target_names": np.array([f"t{i}" for i in range(10)]),
                "sequence_sha1": np.array([f"s{i}" for i in range(10)]),
                "esm_embeddings": rng.normal(size=(10, 6)),
                "molecule_features": molecule_features(["CC", "CCC", "CCO"])}
        with tempfile.TemporaryDirectory() as directory:
            summary = run_cold(data, directory, {"fractions": [1.], "ridge": [.03], "maxiter": 1000})
            self.assertEqual(summary["fits"], 100)
            table = pd.read_csv(Path(directory) / "cold_aggregate_metrics.csv")
            overall = table[table.stratum.eq("all_determinate")]
            self.assertTrue(overall.pairs.eq(14).all())
            with np.load(Path(directory) / "cold_oof.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["baseline__fraction_1"].shape, (3, 10))
                self.assertTrue(np.isfinite(saved["baseline__fraction_1"]).all())
                self.assertEqual(len(np.unique(saved["target_fold"])), 5)

    def test_no_inner_order_is_explicitly_unavailable_without_outer_fit(self):
        rng = np.random.default_rng(12)
        data = {"name": "toy", "policy": "bounded", "lower": np.full((2, 10), -np.inf),
                "upper": np.zeros((2, 10)), "upper_open": np.ones((2, 10), bool),
                "scores": rng.normal(size=(1, 2, 10)), "methods": ["score"],
                "molecule_names": np.array(["a", "b"]),
                "target_names": np.array([f"t{i}" for i in range(10)]),
                "sequence_sha1": np.array([f"s{i}" for i in range(10)]),
                "esm_embeddings": rng.normal(size=(10, 6)),
                "molecule_features": molecule_features(["CC", "CCC"])}
        with tempfile.TemporaryDirectory() as directory:
            summary = run_cold(data, directory, {"fractions": [1.], "ridge": [.03]})
            self.assertEqual(len(summary["unavailable_arms"]), 25)
            self.assertFalse(summary["test_predictions_saved"])
            fits = [json.loads(x) for x in (Path(directory) / "cold_fits.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["stage"] == "inner" for x in fits))
            selections = [json.loads(x) for x in (Path(directory) / "cold_selections.jsonl").read_text().splitlines()]
            self.assertTrue(all(x["chosen_ridge"] is None for x in selections))
            table = pd.read_csv(Path(directory) / "cold_query_metrics.csv")
            self.assertTrue(table.method.eq("native__score").all())


if __name__ == "__main__":
    unittest.main()
