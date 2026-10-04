"""Dense numerical solvers for the unchanged interval-ridge objective."""
import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lstsq
from scipy.optimize import OptimizeResult


def solve_dense_interval(design, lower, upper, alpha, initial=None, maxiter=100,
                         gtol=1e-7):
    """Damped active-set Newton for squared distance to observation intervals.

    Minimize ``mean(distance(design @ x, [lower, upper]) ** 2)`` over
    informative rows plus ``alpha * sum(x[1:] ** 2)``. Column zero must be
    an intercept of ones and is unpenalized. Rows with either endpoint missing,
    and fully unbounded intervals, do not enter the mean's denominator.

    Every exact observation contributes curvature, including at zero residual.
    For proper intervals, the active Hessian uses only predictions outside the
    interval. Diagonal scaling changes the linear-system coordinates only;
    no damping penalty, objective modification, or statistical cutoff is added.
    The caller must bound the size of the dense design and Hessian.
    """
    design = np.asarray(design, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if design.ndim != 2 or not design.shape[0] or not design.shape[1]:
        raise ValueError("A nonempty two-dimensional design is required")
    if lower.shape != (len(design),) or upper.shape != lower.shape:
        raise ValueError("One lower and upper bound is required per design row")
    if not np.all(design[:, 0] == 1):
        raise ValueError("Design column zero must be an intercept of ones")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be finite and strictly positive")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    present = ~np.isnan(lower) & ~np.isnan(upper)
    if np.any(present & ((lower > upper) | np.isposinf(lower) | np.isneginf(upper))):
        raise ValueError("Inconsistent or empty observation intervals")
    known = present & (np.isfinite(lower) | np.isfinite(upper))
    if not known.any():
        raise ValueError("No informative observation intervals; mean loss is undefined")
    original_count = len(design)
    design, lower, upper = design[known], lower[known], upper[known]
    if not np.isfinite(design).all():
        raise ValueError("Informative design rows must be finite")
    count, width = design.shape
    exact = np.isfinite(lower) & (lower == upper)
    x = np.zeros(width) if initial is None else np.asarray(initial, dtype=float).copy()
    if x.shape != (width,) or not np.isfinite(x).all():
        raise ValueError("A finite initial parameter vector of the design width is required")
    weight = 2.0 / count
    evaluations = 0

    def objective(point):
        nonlocal evaluations
        evaluations += 1
        predicted = design @ point
        residual = np.minimum(predicted - lower, 0) + np.maximum(predicted - upper, 0)
        value = float(residual @ residual / count + alpha * (point[1:] @ point[1:]))
        gradient = weight * (design.T @ residual)
        gradient[1:] += 2 * alpha * point[1:]
        active = exact | (predicted < lower) | (predicted > upper)
        return value, gradient, active

    value, gradient, active = objective(x)
    diagnostics = {"solver": "damped_active_set_newton", "total_rows": original_count,
                   "known_labels": count, "ignored_labels": original_count - count,
                   "exact_labels": int(exact.sum()), "alpha": float(alpha),
                   "gtol": float(gtol), "cholesky_solves": 0, "least_squares_solves": 0,
                   "inactive_hessian_steps": 0, "diagonal_gradient_steps": 0,
                   "line_search_reductions": 0, "step_lengths": [],
                   "objective_history": [value],
                   "gradient_inf_history": [float(np.max(np.abs(gradient)))]}
    status, message, iterations = 1, "Maximum iterations reached", 0
    for iteration in range(maxiter + 1):
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            status, message = 3, "Nonfinite objective or gradient"
            break
        if np.max(np.abs(gradient)) <= gtol:
            status, message = 0, "Objective gradient tolerance reached"
            break
        if iteration == maxiter:
            break
        if not active.any():
            # The intercept has zero gradient and curvature. The penalized
            # block has Hessian 2*alpha*I, so retain this free intercept gauge.
            direction = np.r_[0.0, -x[1:]]
            diagonal = np.r_[1.0, np.full(width - 1, 2 * alpha)]
            diagnostics["inactive_hessian_steps"] += 1
        else:
            active_design = design[active]
            hessian = weight * (active_design.T @ active_design)
            penalized = np.arange(1, width)
            hessian[penalized, penalized] += 2 * alpha
            diagonal = np.diag(hessian).copy()
            scale = 1 / np.sqrt(diagonal)
            scaled_hessian = scale[:, None] * hessian * scale[None, :]
            rhs = -gradient * scale
            try:
                factor = cho_factor(scaled_hessian, lower=True, check_finite=False)
                direction = scale * cho_solve(factor, rhs, check_finite=False)
                diagnostics["cholesky_solves"] += 1
            except LinAlgError:
                # Solve the same Newton equation if finite-precision
                # factorization fails; do not add a new ridge penalty.
                direction = scale * lstsq(scaled_hessian, rhs, check_finite=False,
                                          lapack_driver="gelsy")[0]
                diagnostics["least_squares_solves"] += 1
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
            direction = -gradient / diagonal
            slope = float(gradient @ direction)
            diagnostics["diagonal_gradient_steps"] += 1
        step, accepted = 1.0, False
        # This is a numerical line-search budget, not a scientific stopping rule.
        for reduction in range(60):
            proposed = x + step * direction
            trial_value, trial_gradient, trial_active = objective(proposed)
            roundoff = 8 * np.finfo(float).eps * max(1.0, abs(value))
            if (np.isfinite(trial_value) and np.isfinite(trial_gradient).all()
                    and trial_value <= value + 1e-4 * step * slope + roundoff):
                accepted = True
                break
            step *= .5
        diagnostics["line_search_reductions"] += reduction
        if not accepted or np.array_equal(proposed, x):
            status, message = 2, "Armijo line search failed to make numerical progress"
            break
        x, value, gradient, active = proposed, trial_value, trial_gradient, trial_active
        iterations += 1
        diagnostics["step_lengths"].append(step)
        diagnostics["objective_history"].append(value)
        diagnostics["gradient_inf_history"].append(float(np.max(np.abs(gradient))))
    diagnostics["active_labels"] = int(active.sum())
    diagnostics["gradient_max"] = float(np.max(np.abs(gradient)))
    return OptimizeResult(x=x, fun=value, jac=gradient, success=status == 0,
                          status=status, message=message, nit=iterations,
                          nfev=evaluations, njev=evaluations, diagnostics=diagnostics)
