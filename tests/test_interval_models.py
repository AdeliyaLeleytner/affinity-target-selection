import unittest
import numpy as np
from scipy.optimize import check_grad
from study.models import GridIntervalRidge, KernelBasis, PairScaler


class IntervalModels(unittest.TestCase):
    def test_exact_solution_matches_closed_form(self):
        ligand = np.array([[1., 0.], [1., 1.], [1., -1.]])
        target = np.eye(2)
        values = np.array([[2., 4.], [3., 6.], [1., 2.]])
        model = GridIntervalRidge(alpha=.1).fit(ligand, target, values, values)
        design = np.array([np.r_[1, np.outer(l, t).ravel()] for l in ligand for t in target])
        penalty = np.diag([0.] + [.1] * (design.shape[1]-1))
        solution = np.linalg.solve(design.T @ design / len(design) + penalty,
                                   design.T @ values.ravel() / len(design))
        np.testing.assert_allclose(model.predict(ligand, target).ravel(), design @ solution, atol=2e-5)

    def test_censoring_is_not_an_exact_label(self):
        ligand = np.ones((3, 1))
        target = np.ones((1, 1))
        lo = np.array([[1.], [1.], [-np.inf]])
        hi = np.array([[1.], [1.], [9.]])
        model = GridIntervalRidge(.01).fit(ligand, target, lo, hi)
        np.testing.assert_allclose(model.predict(ligand, target), 1, atol=1e-6)

    def test_unbounded_cell_does_not_change_effective_sample_size(self):
        x = np.ones((3, 1))
        target = np.ones((1, 1))
        lo, hi = np.array([[1.], [2.], [-np.inf]]), np.array([[1.], [2.], [np.inf]])
        a = GridIntervalRidge(.1).fit(x, target, lo, hi)
        b = GridIntervalRidge(.1).fit(x[:2], target, lo[:2], hi[:2])
        self.assertEqual(a.diagnostics_["known_labels"], 2)
        np.testing.assert_allclose(a.predict(x, target), b.predict(x, target), atol=1e-6)

    def test_pair_gradient(self):
        rng = np.random.default_rng(8)
        molecular, target, pairs = rng.normal(size=(3, 2)), rng.normal(size=(2, 2)), rng.normal(size=(3, 2, 2))
        lo = np.array([[1., -np.inf], [2., np.nan], [0., 1.]])
        hi = np.array([[1., .5], [2., np.nan], [0., 1.]])
        known = ~np.isnan(lo)
        for dependent in [False, True]:
            model = GridIntervalRidge(.03, target_dependent_pair=dependent)
            model.mol_dim, model.target_dim, model.pair_dim = 2, 2, 2
            point = rng.normal(size=1+4+(4 if dependent else 2))
            args = (molecular, target, pairs, lo, hi, known)
            error = check_grad(lambda x: model._objective(x, *args)[0],
                               lambda x: model._objective(x, *args)[1], point)
            self.assertLess(error, 1e-5)

    def test_identity_fast_path_preserves_objective_and_gradient(self):
        rng = np.random.default_rng(31)
        molecular, target, pairs = rng.normal(size=(4, 2)), np.eye(3), rng.normal(size=(4, 3, 2))
        lo = rng.normal(size=(4, 3))
        hi = lo.copy()
        lo[0, 0] = -np.inf
        for dependent in [False, True]:
            model = GridIntervalRidge(.02, target_dependent_pair=dependent)
            model.mol_dim, model.target_dim, model.pair_dim = 2, 3, 2
            theta = rng.normal(size=1+6+(6 if dependent else 2))
            args = (molecular, target, pairs, lo, hi, np.ones(lo.shape, bool))
            slow = model._objective(theta, *args)
            fast = model._objective(theta, *args, target_identity=True)
            np.testing.assert_allclose(fast[0], slow[0], atol=1e-12)
            np.testing.assert_allclose(fast[1], slow[1], atol=1e-12)

    def test_fallback_converges_without_changing_ridge_solution(self):
        ligand = np.array([[1., 0.], [1., 1.], [1., -1.]])
        target = np.eye(2)
        y = np.array([[2., 4.], [3., 6.], [1., 2.]])
        short = GridIntervalRidge(.1, maxiter=1).fit(ligand, target, y, y)
        reference = GridIntervalRidge(.1).fit(ligand, target, y, y)
        self.assertEqual(short.diagnostics_["fallback"]["method"], "dense_active_set_newton")
        np.testing.assert_allclose(short.predict(ligand, target), reference.predict(ligand, target), atol=2e-5)

    def test_kernel_projection_and_pair_scaler(self):
        features = np.array([[1., 1.], [1., 2.], [2., 1.]])
        kernel = features @ features.T
        basis = KernelBasis.fit(kernel)
        np.testing.assert_allclose(basis.training @ basis.training.T, kernel, atol=1e-12)
        np.testing.assert_allclose(basis.transform(kernel), basis.training, atol=1e-12)
        train = np.array([[[1., np.nan]], [[3., np.nan]]])
        scaler = PairScaler.fit(train)
        transformed, missing = scaler.transform(np.array([[[5., 9.]]]))
        np.testing.assert_array_equal(scaler.median, [2., 0.])
        np.testing.assert_allclose(transformed, [[[3., 9.]]])
        self.assertFalse(missing.any())


if __name__ == "__main__":
    unittest.main()
