import unittest

import numpy as np
from scipy.optimize import Bounds, minimize

from study.solvers import solve_dense_interval


def constrained_reference(design, lower, upper, alpha):
    """Independent QP: latent exact responses constrained inside each interval."""
    informative = ~np.isnan(lower) & ~np.isnan(upper) & (np.isfinite(lower) | np.isfinite(upper))
    design, lower, upper = design[informative], lower[informative], upper[informative]
    n, p = design.shape
    initial_response = np.maximum(np.minimum(np.zeros(n), upper), lower)

    def objective(point):
        parameters, response = point[:p], point[p:]
        residual = design @ parameters - response
        loss = residual @ residual / n + alpha * (parameters[1:] @ parameters[1:])
        gradient = np.r_[2 * design.T @ residual / n, -2 * residual / n]
        gradient[1:p] += 2 * alpha * parameters[1:]
        return loss, gradient

    result = minimize(objective, np.r_[np.zeros(p), initial_response], jac=True,
                      method="SLSQP", bounds=Bounds(np.r_[np.full(p, -np.inf), lower],
                                                     np.r_[np.full(p, np.inf), upper]),
                      options={"ftol": 1e-13, "maxiter": 2000})
    if not result.success:
        raise AssertionError(result.message)
    return result


class DenseIntervalSolverTests(unittest.TestCase):
    def test_exact_data_matches_independent_closed_form(self):
        rng = np.random.default_rng(8)
        design = np.c_[np.ones(15), rng.normal(size=(15, 6))]
        y = rng.normal(size=15)
        alpha = .03
        penalty = np.diag(np.r_[0., np.full(6, alpha)])
        expected = np.linalg.solve(design.T @ design / len(y) + penalty,
                                   design.T @ y / len(y))
        result = solve_dense_interval(design, y, y, alpha, rng.normal(size=7), gtol=1e-10)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, expected, atol=1e-11, rtol=1e-10)
        self.assertEqual(result.diagnostics["active_labels"], len(y))
        self.assertLessEqual(result.nit, 2)

    def test_mixed_intervals_match_constrained_quadratic_program(self):
        design = np.c_[np.ones(8), np.arange(-4., 4.), [1., -1., 2., 0., 3., 1., -2., .5]]
        lower = np.array([-2., -np.inf, .1, .4, -np.inf, 1., .9, 2.])
        upper = np.array([-2., -.5, .8, .4, .3, np.inf, 1.4, 2.])
        result = solve_dense_interval(design, lower, upper, .07, [4., -2., 3.], gtol=1e-10)
        reference = constrained_reference(design, lower, upper, .07)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.fun, reference.fun, places=10)
        np.testing.assert_allclose(result.x, reference.x[:3], atol=2e-6)
        self.assertLessEqual(np.max(np.abs(result.jac)), 1e-10)
        self.assertTrue(np.all(np.diff(result.diagnostics["objective_history"]) < 1e-12))

    def test_missing_and_fully_unbounded_rows_do_not_change_loss_weight(self):
        design = np.array([[1., -1.], [1., 0.], [1., 2.], [1., 100.], [1., -100.]])
        lower = np.array([1., -np.inf, 3., np.nan, -np.inf])
        upper = np.array([1., 2., 3., np.nan, np.inf])
        full = solve_dense_interval(design, lower, upper, .2)
        subset = solve_dense_interval(design[:3], lower[:3], upper[:3], .2)
        self.assertTrue(full.success)
        self.assertEqual(full.diagnostics["known_labels"], 3)
        self.assertEqual(full.diagnostics["ignored_labels"], 2)
        np.testing.assert_allclose(full.x, subset.x, atol=1e-12)
        self.assertEqual(full.fun, subset.fun)

    def test_inactive_data_hessian_retains_free_intercept_without_new_penalty(self):
        design = np.array([[1., -1.], [1., 0.], [1., 1.]])
        lower, upper = np.full(3, -np.inf), np.full(3, 10.)
        result = solve_dense_interval(design, lower, upper, .05, [2., .5], gtol=1e-12)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [2., 0.], atol=1e-12)
        self.assertEqual(result.fun, 0.)
        self.assertGreater(result.diagnostics["inactive_hessian_steps"], 0)
        # Any feasible intercept is a valid optimum; the solver need not select 0.
        at_optimum = solve_dense_interval(design, lower, upper, .05, [-200., 0.])
        self.assertTrue(at_optimum.success)
        self.assertEqual(at_optimum.nit, 0)
        self.assertEqual(at_optimum.x[0], -200.)

    def test_small_ridge_and_rank_deficient_design(self):
        rng = np.random.default_rng(14)
        base = rng.normal(size=(20, 5))
        design = np.c_[np.ones(20), base, base, np.ones(20), np.zeros(20)]
        y = rng.normal(size=20)
        lower, upper = y.copy(), y.copy()
        lower[::3] = -np.inf
        upper[1::4] = np.inf
        result = solve_dense_interval(design, lower, upper, .001 / 442,
                                      rng.normal(size=design.shape[1]), gtol=1e-9)
        reference = constrained_reference(design, lower, upper, .001 / 442)
        self.assertTrue(result.success, result.message)
        # The generic constrained solver is less accurate in nearly flat ridge
        # directions; require agreement and that Newton is not worse.
        self.assertLessEqual(abs(result.fun - reference.fun), 1e-8)
        self.assertLessEqual(result.fun, reference.fun + 1e-11)
        self.assertLessEqual(np.max(np.abs(result.jac)), 1e-9)

    def test_no_informative_rows_and_invalid_input_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "No informative"):
            solve_dense_interval(np.ones((2, 1)), [-np.inf, np.nan], [np.inf, np.nan], .1)
        with self.assertRaisesRegex(ValueError, "intercept"):
            solve_dense_interval(np.zeros((2, 1)), [1., 2.], [1., 2.], .1)
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            solve_dense_interval(np.ones((2, 1)), [2., 2.], [1., 2.], .1)
        result = solve_dense_interval(np.ones((2, 1)), [1., 2.], [1., 2.], .1, maxiter=0)
        self.assertFalse(result.success)
        self.assertEqual(result.status, 1)
        self.assertGreater(np.max(np.abs(result.jac)), 1e-7)


if __name__ == "__main__":
    unittest.main()
