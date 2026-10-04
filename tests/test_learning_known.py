import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from study.features import MoleculeFeatures
from study.learning_known import (_Arm, _Partition, _arms, aggregate_queries,
                                  evaluate_queries, run_known, score_components, split_plan)
from study.learning_run import selected_blocks


def toy_data(n=9):
    rng = np.random.default_rng(19)
    names = np.array([f"m{i}" for i in range(n)])
    features = MoleculeFeatures(
        fingerprints=rng.integers(0, 2, size=(n, 8), dtype=np.uint8),
        descriptors=rng.normal(size=(n, 6)), identities=names,
        scaffolds=names, scaffold_groups=names)
    labels = np.tile([8., 7., 6.], (n, 1)) + np.arange(n)[:, None] / 10
    return {"name": "toy", "policy": "exact", "lower": labels.copy(),
            "upper": labels.copy(), "upper_open": np.zeros(labels.shape, bool),
            "scores": np.stack([labels * 2 + 3, labels / 10]),
            "methods": ["boltz_affinity", "boltz_binder"],
            "molecule_names": names, "molecule_features": features}


class GuardedRows:
    """Fail immediately if pilot touches any held-out outcome value."""
    def __init__(self, array, forbidden):
        self.array, self.forbidden = array, set(forbidden)
        self.shape = array.shape

    def __getitem__(self, index):
        first = index[0] if isinstance(index, tuple) else index
        indices = np.atleast_1d(np.arange(len(self.array))[first])
        if self.forbidden.intersection(indices.tolist()):
            raise AssertionError("Pilot accessed held-out outcome values")
        return self.array[index]


class KnownLearning(unittest.TestCase):
    def test_group_budgets_and_nested_fold_isolation(self):
        groups = np.repeat([f"g{i}" for i in range(20)], [1, 2] * 10)
        config = {"fractions": [.25, .5, 1.], "outer_folds": 5,
                  "inner_folds": 3, "seed": 72}
        folds, plans = split_plan(groups, config)
        for outer in range(5):
            previous = set()
            for plan in [p for p in plans if p["outer_fold"] == outer]:
                training, test = np.array(plan["training_indices"]), np.array(plan["test_indices"])
                selected = set(groups[training])
                self.assertEqual(len(selected), plan["budget_groups"])
                self.assertTrue(previous.issubset(selected))
                previous = selected
                self.assertFalse(selected & set(groups[test]))
                validation_rows = []
                for inner in plan["inner"]:
                    inner_train = np.array(inner["training_indices"])
                    inner_valid = np.array(inner["validation_indices"])
                    self.assertFalse(set(groups[inner_train]) & set(groups[inner_valid]))
                    self.assertFalse(set(groups[test]) & set(groups[inner_train]))
                    self.assertFalse(set(groups[test]) & set(groups[inner_valid]))
                    self.assertEqual(set(inner_train) | set(inner_valid), set(training))
                    validation_rows.extend(inner_valid.tolist())
                self.assertEqual(sorted(validation_rows), sorted(training.tolist()))
        np.testing.assert_array_equal(folds, split_plan(groups, config)[0])

    def test_rank_and_error_support_have_analytic_values(self):
        lower = np.array([[8., 7., 6., 5.], [7., -np.inf, 5., np.nan]])
        upper = np.array([[8., 7., 6., 5.], [7., 6., 5., np.nan]])
        prediction = np.array([[8., 5., 100., 6.], [7., 6., 6., 100.]])
        support = np.array([[True, True, False, True], [True, True, True, True]])
        rows = evaluate_queries(prediction, lower, upper, np.zeros(lower.shape, bool),
                                support, ["a", "b"], [0, 1], dataset="toy", policy="test",
                                method="model", fraction=1.)
        all_rows = [r for r in rows if r["stratum"] == "all_determinate"]
        self.assertEqual([r["pairs"] for r in all_rows], [3, 2])
        np.testing.assert_allclose([r["concordance"] for r in all_rows], [2/3, 1])
        np.testing.assert_allclose([r["interval_mae"] for r in all_rows], [1, 1/3])
        np.testing.assert_allclose([r["exact_mae"] for r in all_rows], [1, .5])
        strata_b = {r["stratum"]: r for r in rows if r["query"] == "b"}
        self.assertEqual(strata_b["both_exact"]["pairs"], 1)
        self.assertEqual(strata_b["one_bounded"]["pairs"], 1)
        self.assertEqual(strata_b["both_bounded"]["pairs"], 0)
        aggregate = next(r for r in aggregate_queries(rows) if r["stratum"] == "all_determinate")
        self.assertAlmostEqual(aggregate["macro_concordance"], 5/6)
        self.assertAlmostEqual(aggregate["pooled_concordance"], 4/5)
        self.assertAlmostEqual(aggregate["macro_exact_mae"], .75)
        self.assertAlmostEqual(aggregate["pooled_exact_mae"], .8)
        self.assertEqual(aggregate["labels"], 6)
        self.assertEqual(aggregate["exact_labels"], 5)

    def test_native_scale_and_unavailable_predictions(self):
        labels = np.array([[3., 2., 1.]])
        kwargs = dict(dataset="toy", policy="exact", method="native", fraction=1.)
        rows = evaluate_queries(np.array([[9., 9., 0.]]), labels, labels,
                                np.zeros(labels.shape, bool), np.ones(labels.shape, bool),
                                ["a"], [0], error_metrics=False, **kwargs)
        self.assertAlmostEqual(rows[0]["concordance"], 5/6)
        self.assertIsNone(rows[0]["interval_mae"])
        self.assertIsNone(rows[0]["exact_mae"])
        unavailable = evaluate_queries(np.array([[9., np.nan, 0.]]), labels, labels,
                                       np.zeros(labels.shape, bool), np.ones(labels.shape, bool),
                                       ["a"], [0], **kwargs)
        self.assertEqual(unavailable[0]["status"], "unavailable_model")
        self.assertEqual(unavailable[0]["eligible_pairs"], 3)
        self.assertEqual(unavailable[0]["pairs"], 0)
        self.assertIsNone(unavailable[0]["concordance"])

    def test_partition_preprocessing_never_fits_prediction_rows(self):
        data = toy_data()
        rng = np.random.default_rng(4)
        data["representations"] = rng.normal(size=(9, 3, 6))
        data["scalar_members"] = rng.normal(size=(9, 3, 4))
        training, validation = [0, 1, 2, 3, 4], [5, 6, 7, 8]
        first = _Partition(data, training, validation)
        arms = [_Arm("chem", "chemistry"), _Arm("plus", "plus", 0),
                _Arm("mean", "mean", 0), _Arm("mean_relative", "mean_relative", 0),
                _Arm("pca", "representation_pca4"), _Arm("scalar", "scalar_members")]
        fitted = [(a, first.features(a)) for a in arms]
        data["molecule_features"].descriptors[validation] *= 1000
        data["scores"][:, validation] += 10000
        data["representations"][validation] += 10000
        data["scalar_members"][validation] -= 10000
        second = _Partition(data, training, validation)
        for arm, (molecular, pairs, dependent) in fitted:
            new_molecular, new_pairs, new_dependent = second.features(arm)
            np.testing.assert_allclose(molecular[0], new_molecular[0], atol=1e-12)
            if pairs[0] is not None:
                np.testing.assert_allclose(pairs[0], new_pairs[0], atol=1e-12)
            self.assertEqual(dependent, new_dependent)

    def test_pilot_cannot_read_held_out_labels_or_evaluate(self):
        data = toy_data()
        config = {"mode": "technical_pilot", "outer_folds": 3, "inner_folds": 2,
                  "fractions": [1.], "seed": 9, "maxiter": 1000}
        folds, _ = split_plan(data["molecule_features"].scaffold_groups, config)
        forbidden = np.flatnonzero(folds == 0)
        for name in ("lower", "upper", "upper_open"):
            data[name] = GuardedRows(data[name], forbidden)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch("study.learning_known.evaluate_queries", side_effect=AssertionError("Pilot evaluated test metrics")):
                summary = run_known(data, Path(folder), config)
            self.assertFalse(summary["test_metrics_evaluated"])
            self.assertEqual(summary["status"], "provisional")
            self.assertEqual(len(summary["fits"]), 4)
            self.assertFalse((Path(folder) / "known_oof.npz").exists())
            self.assertFalse((Path(folder) / "known_query_metrics.csv").exists())

    def test_full_oof_and_tuning_ties(self):
        data = toy_data()
        config = {"mode": "full", "outer_folds": 3, "inner_folds": 2,
                  "fractions": [1.], "seed": 9, "maxiter": 1000,
                  "ridge": [.01, .1]}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder)
            summary = run_known(data, out, config)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["availability_aliases"],
                             {"availability__boltz_binder": "availability__boltz_affinity"})
            with np.load(out / "known_oof.npz", allow_pickle=False) as saved:
                self.assertTrue(np.isfinite(saved["predictions"]).all())
                self.assertEqual(saved["predictions"].shape[1:], (1, 9, 3))
                prior = list(saved["methods"]).index("target_prior")
                np.testing.assert_array_equal(saved["selected_ridge"][prior], [[.1, .1, .1]])
                a = list(saved["methods"]).index("availability__boltz_affinity")
                b = list(saved["methods"]).index("availability__boltz_binder")
                np.testing.assert_array_equal(saved["predictions"][a], saved["predictions"][b])
            selections = [json.loads(line) for line in (out / "known_selections.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["status"] == "selected" for row in selections))
            self.assertTrue(all(len(row["candidates"]) == 2 for row in selections))

    def test_availability_dedup_uses_exact_masks_not_counts(self):
        data = toy_data()
        data["scores"][0, 0, 0] = np.nan
        data["scores"][1, 1, 0] = np.nan
        self.assertEqual(_arms(data)[1], {})
        data["scores"] = np.concatenate((data["scores"], data["scores"][:1]), axis=0)
        data["methods"].append("same_availability")
        self.assertEqual(_arms(data)[1],
                         {"availability__same_availability": "availability__boltz_affinity"})

    def test_unestimable_tuning_is_reported_without_outer_selection(self):
        data = toy_data(n=6)
        data["lower"][:] = 5
        data["upper"][:] = 5
        config = {"mode": "full", "outer_folds": 3, "inner_folds": 2,
                  "fractions": [1.], "seed": 9, "maxiter": 1000,
                  "ridge": [.03]}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder)
            summary = run_known(data, out, config)
            self.assertEqual(summary["status"], "complete_with_unavailable_blocks")
            self.assertTrue(all(r["status"] == "unestimable_inner_concordance"
                                for r in summary["unavailable"]))
            with np.load(out / "known_oof.npz", allow_pickle=False) as saved:
                self.assertTrue(np.isnan(saved["predictions"]).all())
            fit_rows = [json.loads(line) for line in (out / "known_fits.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["stage"] == "inner" for row in fit_rows))

    def test_score_decomposition_reconstructs_and_preserves_missingness(self):
        raw = np.array([[1., 2., np.nan, np.inf], [2., -1., 5., 3.],
                        [np.nan, -np.inf, np.nan, np.nan]])
        original = raw.copy()
        mean, relative = score_components(raw)
        finite = np.isfinite(raw)
        np.testing.assert_allclose((mean + relative)[finite], raw[finite], atol=1e-14)
        np.testing.assert_array_equal(np.isnan(mean), ~finite)
        np.testing.assert_array_equal(np.isnan(relative), ~finite)
        np.testing.assert_allclose(mean[0, :2], [1.5, 1.5])
        np.testing.assert_allclose(mean[1], [2.25] * 4)
        self.assertTrue(np.isnan(mean[2]).all())
        np.testing.assert_array_equal(raw, original)
        permutation = [2, 0, 3, 1]
        permuted_mean, permuted_relative = score_components(raw[:, permutation])
        np.testing.assert_allclose(permuted_mean, mean[:, permutation], equal_nan=True)
        np.testing.assert_allclose(permuted_relative, relative[:, permutation], equal_nan=True)

    def test_component_arm_family_and_default_standard_arms(self):
        data = toy_data()
        data["representations"] = np.ones((9, 3, 6))
        data["scalar_members"] = np.ones((9, 3, 4))
        self.assertEqual(_arms(data), _arms(data, {"arm_family": "standard"}))
        standard_names = _arms(data)[2]
        self.assertEqual(len(standard_names), 12)
        self.assertIn("calibrated__boltz_affinity", standard_names)
        self.assertIn("representation_pca4_readout", standard_names)
        arms, aliases, reported = _arms(data, {"arm_family": "score_components"})
        self.assertEqual(len(reported), 10)
        self.assertEqual(len(arms), 9)  # Two identical availability masks are fitted once.
        self.assertEqual(aliases, {"availability__boltz_binder": "availability__boltz_affinity"})
        self.assertFalse(any("calibrated" in name or "representation" in name or "members" in name
                             for name in reported))
        self.assertIn("mean__boltz_affinity", reported)
        self.assertIn("mean_relative__boltz_binder", reported)
        with self.assertRaisesRegex(ValueError, "Unknown known-target arm family"):
            _arms(data, {"arm_family": "typo"})

    def test_component_scaling_and_exactly_one_missingness_column(self):
        data = toy_data(n=3)
        data["scores"][0] = np.array([[1., 2., 3.], [3., 4., 5.], [100., 101., 102.]])
        partition = _Partition(data, [0, 1], [2])
        mean = partition.features(_Arm("mean", "mean", 0))
        full = partition.features(_Arm("mean_relative", "mean_relative", 0))
        raw = partition.features(_Arm("plus", "plus", 0))
        availability = partition.features(_Arm("availability", "availability", 0))
        self.assertEqual([x[1][0].shape[-1] for x in (mean, full, raw, availability)], [2, 3, 2, 1])
        self.assertTrue(all(x[2] for x in (mean, full, raw, availability)))
        np.testing.assert_allclose(mean[1][0][..., 0], [[-1.] * 3, [1.] * 3])
        np.testing.assert_allclose(mean[1][1][..., 0], [[98.] * 3])
        np.testing.assert_allclose(full[1][0][..., 0], mean[1][0][..., 0])
        for features in (full, raw, availability):
            np.testing.assert_allclose(features[0][0], mean[0][0])
        data["scores"][0, 0, 1] = np.nan
        data["scores"][0, 2] = np.nan
        partition = _Partition(data, [0, 1], [2])
        for kind in ("mean", "mean_relative"):
            _, pair, _ = partition.features(_Arm(kind, kind, 0))
            self.assertTrue(np.isfinite(pair[0]).all() and np.isfinite(pair[1]).all())
            np.testing.assert_array_equal(pair[0][..., -1], [[0, 1, 0], [0, 0, 0]])
            np.testing.assert_array_equal(pair[1][..., -1], [[1, 1, 1]])

    def test_component_full_run_preserves_budgets_labels_and_common_support(self):
        data = toy_data()
        data["scores"][:, 0, 0] = np.nan
        config = {"mode": "full", "arm_family": "score_components", "outer_folds": 3,
                  "inner_folds": 2, "fractions": [.5, 1.], "seed": 9,
                  "maxiter": 1000, "ridge": [.03]}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder)
            summary = run_known(data, out, config)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["arm_family"], "score_components")
            self.assertEqual(summary["common_native_cells"], 26)
            with np.load(out / "known_oof.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["predictions"].shape, (10, 2, 9, 3))
                self.assertTrue(np.isfinite(saved["predictions"]).all())
                np.testing.assert_array_equal(saved["common_native_finite"], np.isfinite(data["scores"]).all(axis=0))
            fits = [json.loads(line) for line in (out / "known_fits.jsonl").read_text().splitlines()]
            context_counts = {}
            for row in fits:
                key = tuple(row.get(k) for k in ("outer_fold", "fraction", "stage", "inner_fold"))
                counts = tuple(row[k] for k in ("training_molecules", "known_training_labels", "exact_training_labels"))
                context_counts.setdefault(key, set()).add(counts)
                self.assertEqual(row["known_training_labels"], row["training_molecules"] * 3)
            self.assertTrue(all(len(values) == 1 for values in context_counts.values()))
            plans = json.loads((out / "known_splits.json").read_text())["plans"]
            self.assertEqual(plans, split_plan(data["molecule_features"].scaffold_groups, config)[1])

    def test_exact_run_block_filter(self):
        datasets = [{"name": "SPD", "policy": "source_representative"},
                    {"name": "SPD", "policy": "assay_envelope"},
                    {"name": "DAVIS", "policy": "exact_identity"}]
        config = {"mode": "full", "regimes": ["new_molecule_known_targets", "new_targets_known_molecules"]}
        self.assertEqual(len(selected_blocks(datasets, config)), 6)
        wanted = ["SPD/source_representative/new_molecule_known_targets",
                  "SPD/assay_envelope/new_molecule_known_targets"]
        selected = selected_blocks(datasets, {**config, "run_blocks": wanted})
        self.assertEqual([(d["name"], d["policy"], r) for d, r in selected],
                         [("SPD", "source_representative", "new_molecule_known_targets"),
                          ("SPD", "assay_envelope", "new_molecule_known_targets")])
        for invalid in [["SPD"], [wanted[0], wanted[0]], [], "SPD"]:
            with self.assertRaises(ValueError):
                selected_blocks(datasets, {**config, "run_blocks": invalid})


if __name__ == "__main__":
    unittest.main()
