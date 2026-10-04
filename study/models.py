"""Convex interval regression on separable molecular/protein kernels."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize


@dataclass
class KernelBasis:
    vectors: np.ndarray
    roots: np.ndarray
    training: np.ndarray

    @classmethod
    def fit(cls, kernel):
        kernel = np.asarray(kernel, float)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Square training kernel required")
        if not np.allclose(kernel, kernel.T, atol=1e-12, rtol=0):
            raise ValueError("Asymmetric kernel")
        values, vectors = np.linalg.eigh(kernel)
        tolerance = np.finfo(float).eps * len(values) * max(float(values[-1]), 1)
        if values[0] < -100 * tolerance:
            raise ValueError("Kernel is not positive semidefinite")
        # Remove numerical null directions only; this is not a tuned rank cutoff.
        use = values > tolerance
        roots = np.sqrt(values[use])
        vectors = vectors[:, use]
        return cls(vectors, roots, vectors * roots)

    def transform(self, cross_kernel):
        return (np.asarray(cross_kernel) @ self.vectors) / self.roots


@dataclass
class PairScaler:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train):
        flat = np.asarray(train, float).reshape(-1, train.shape[-1])
        median = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0
                           for x in flat.T])
        filled = np.where(np.isfinite(flat), flat, median)
        mean, scale = filled.mean(axis=0), filled.std(axis=0)
        scale[scale == 0] = 1
        return cls(median, mean, scale)

    def transform(self, values):
        values = np.asarray(values, float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        return (filled - self.mean) / self.scale, missing.astype(float)


class GridIntervalRidge:
    """Fit an interval-aware model with explicit, regularized feature components.

    Molecular and target features yield a bilinear function. Pair features can
    have shared coefficients (representation readouts) or target-feature-dependent
    coefficients (single-score calibration). The global intercept is unpenalized.
    Every observed training cell contributes equal loss; missing labels contribute
    nothing. Bounds are not substituted as exact observations.
    """

    def __init__(self, alpha=0.01, target_dependent_pair=False, maxiter=500):
        if alpha <= 0:
            raise ValueError("Strictly positive regularization required")
        self.alpha = alpha
        self.target_dependent_pair = target_dependent_pair
        self.maxiter = maxiter

    def _unpack(self, parameters):
        stop = 1 + self.mol_dim * self.target_dim
        matrix = parameters[1:stop].reshape(self.mol_dim, self.target_dim)
        pair = parameters[stop:]
        if self.target_dependent_pair:
            pair = pair.reshape(self.pair_dim, self.target_dim)
        return parameters[0], matrix, pair

    def _predict(self, parameters, molecular, target, pairs, target_identity=False):
        intercept, matrix, pair = self._unpack(parameters)
        n, r = molecular.shape
        t, s = target.shape
        if target_identity:
            prediction = intercept + molecular @ matrix
        elif r * s * t + n * r * t < n * r * s + n * s * t:
            prediction = intercept + molecular @ (matrix @ target.T)
        else:
            prediction = intercept + (molecular @ matrix) @ target.T
        if self.pair_dim:
            if self.target_dependent_pair:
                slopes = pair.T if target_identity else target @ pair.T
                prediction += np.einsum("ntd,td->nt", pairs, slopes, optimize=True)
            else:
                prediction += np.einsum("ntd,d->nt", pairs, pair, optimize=True)
        return prediction

    def _objective(self, parameters, molecular, target, pairs, lower, upper, known, target_identity=False):
        prediction = self._predict(parameters, molecular, target, pairs, target_identity)
        residual = np.where(known, np.minimum(prediction - lower, 0) + np.maximum(prediction - upper, 0), 0)
        count = known.sum()
        loss = np.square(residual).sum() / count + self.alpha * np.square(parameters[1:]).sum()
        derivative = (2 / count) * residual
        gradient = np.zeros_like(parameters)
        gradient[0] = derivative.sum()
        stop = 1 + self.mol_dim * self.target_dim
        gradient[1:stop] = (molecular.T @ derivative if target_identity else molecular.T @ derivative @ target).ravel()
        if self.pair_dim:
            if self.target_dependent_pair:
                if target_identity:
                    gradient[stop:] = np.einsum("nt,ntd->dt", derivative, pairs, optimize=True).ravel()
                else:
                    gradient[stop:] = np.einsum("nt,ntd,tk->dk", derivative, pairs, target, optimize=True).ravel()
            else:
                gradient[stop:] = np.einsum("nt,ntd->d", derivative, pairs, optimize=True)
        gradient[1:] += 2 * self.alpha * parameters[1:]
        return float(loss), gradient

    def fit(self, molecular, target, lower, upper, pairs=None):
        started = time.perf_counter()
        molecular, target = np.asarray(molecular, float), np.asarray(target, float)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if lower.shape != (len(molecular), len(target)) or upper.shape != lower.shape:
            raise ValueError("Feature and label axes disagree")
        known = ~np.isnan(lower) & ~np.isnan(upper) & (np.isfinite(lower) | np.isfinite(upper))
        if not known.any() or np.any(lower[known] > upper[known]):
            raise ValueError("Missing or inconsistent training observations")
        self.mol_dim, self.target_dim = molecular.shape[1], target.shape[1]
        if pairs is None:
            pairs = np.empty(lower.shape + (0,))
        pairs = np.asarray(pairs, float)
        if pairs.shape[:2] != lower.shape or pairs.ndim != 3 or not np.isfinite(pairs).all():
            raise ValueError("Pair features must be finite and align with training grid")
        self.pair_dim = pairs.shape[-1]
        size = 1 + self.mol_dim * self.target_dim + self.pair_dim * (self.target_dim if self.target_dependent_pair else 1)
        initial = np.zeros(size)
        exact = known & np.isfinite(lower) & (lower == upper)
        finite_endpoints = np.r_[lower[known & np.isfinite(lower)], upper[known & np.isfinite(upper)]]
        if not len(finite_endpoints):
            raise ValueError("Training observations have no finite bounds")
        initial[0] = np.median(lower[exact]) if exact.any() else np.median(finite_endpoints)
        # Exact invertible variable scaling from the all-observed quadratic
        # curvature. It changes numerical conditioning, never the objective,
        # feature space, regularization or training observations.
        diagonal = np.empty(size)
        diagonal[0] = 2
        stop = 1 + self.mol_dim * self.target_dim
        diagonal[1:stop] = ((2 / known.sum()) *
                            (np.square(molecular).T @ known.astype(float) @ np.square(target))).ravel()
        if self.pair_dim:
            if self.target_dependent_pair:
                diagonal[stop:] = ((2 / known.sum()) * np.einsum(
                    "nt,ntd,tk->dk", known, np.square(pairs), np.square(target), optimize=True)).ravel()
            else:
                diagonal[stop:] = (2 / known.sum()) * np.einsum(
                    "nt,ntd->d", known, np.square(pairs), optimize=True)
        diagonal[1:] += 2 * self.alpha
        variable_scale = 1 / np.sqrt(diagonal)
        target_identity = target.shape[0] == target.shape[1] and np.array_equal(target, np.eye(len(target)))

        def scaled_objective(coordinates):
            value, gradient = self._objective(coordinates * variable_scale,
                                               molecular, target, pairs, lower, upper, known, target_identity)
            return value, gradient * variable_scale

        result = minimize(scaled_objective, initial / variable_scale,
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40})
        primary_iterations = int(result.nit)
        fallback = None
        if not result.success and self.maxiter > 0:
            design_bytes = int(known.sum()) * size * 8
            if size <= 2048 and design_bytes <= 192 * 1024**2:
                from .solvers import solve_dense_interval
                i, j = np.nonzero(known)
                design = np.empty((len(i), size))
                design[:, 0] = 1
                design[:, 1:stop] = np.einsum("ir,is->irs", molecular[i], target[j]).reshape(len(i), -1)
                if self.pair_dim:
                    pair_rows = pairs[i, j]
                    design[:, stop:] = (np.einsum("id,ik->idk", pair_rows, target[j]).reshape(len(i), -1)
                                          if self.target_dependent_pair else pair_rows)
                solved = solve_dense_interval(design, lower[known], upper[known], self.alpha,
                                               initial=result.x * variable_scale, maxiter=100, gtol=1e-7)
                fallback = {"method": "dense_active_set_newton", **solved.diagnostics}
                result = solved
                result.x = result.x / variable_scale
                result.jac = result.jac * variable_scale
            elif result.status == 1:
                result = minimize(scaled_objective, result.x, method="L-BFGS-B", jac=True,
                                  options={"maxiter": 20000, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40})
                fallback = {"method": "continued_LBFGS", "additional_iteration_cap": 20000}
        self.parameters_ = result.x * variable_scale
        _, objective_gradient = self._objective(self.parameters_, molecular, target, pairs, lower, upper, known, target_identity)
        self.diagnostics_ = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                             "objective": float(result.fun), "gradient_max": float(np.max(np.abs(objective_gradient))),
                             "scaled_gradient_max": float(np.max(np.abs(result.jac))),
                             "preconditioner": "diagonal all-observed Hessian; objective unchanged",
                             "primary_iterations": primary_iterations, "fallback": fallback,
                             "parameters": size, "known_labels": int(known.sum()),
                             "exact_labels": int(exact.sum()), "alpha": self.alpha,
                             "fit_seconds": time.perf_counter() - started}
        if not result.success:
            raise RuntimeError(f"Interval-regression numerical failure: {self.diagnostics_}")
        return self

    def predict(self, molecular, target, pairs=None):
        if pairs is None:
            pairs = np.empty((len(molecular), len(target), 0))
        target = np.asarray(target)
        identity = target.shape[0] == target.shape[1] and np.array_equal(target, np.eye(len(target)))
        return self._predict(self.parameters_, np.asarray(molecular), target, np.asarray(pairs), identity)
