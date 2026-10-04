"""Interval order without activity cutoffs or substituted censoring values."""
import numpy as np


def pair_order(lower, upper, upper_open, pairs):
    a, b = pairs.T
    known = ~(np.isnan(lower[:, a]) | np.isnan(upper[:, a]) |
              np.isnan(lower[:, b]) | np.isnan(upper[:, b]))
    left = ((lower[:, a] > upper[:, b]) |
            ((lower[:, a] == upper[:, b]) & upper_open[:, b]))
    right = ((lower[:, b] > upper[:, a]) |
             ((lower[:, b] == upper[:, a]) & upper_open[:, a]))
    if np.any(known & left & right):
        raise ValueError("Inconsistent interval bounds")
    return np.where(known, left.astype(np.int8) - right, 0)


def evaluate(scores, lower, upper, upper_open, methods, query_ids, axis, dataset, policy):
    """Return complete per-query records; all methods share a finite-score mask."""
    if axis == "ligand":
        scores = scores.transpose(0, 2, 1)
        lower, upper, upper_open = lower.T, upper.T, upper_open.T
    n_method, n_query, n_candidate = scores.shape
    pairs = np.stack(np.triu_indices(n_candidate, 1), axis=1)
    a, b = pairs.T
    # One query at a time bounds memory for large target panels.
    rows = []
    for q in range(n_query):
        truth = pair_order(lower[q:q+1], upper[q:q+1], upper_open[q:q+1], pairs)[0]
        finite = np.isfinite(scores[:, q, :]).all(axis=0)
        support = (truth != 0) & finite[a] & finite[b]
        exact = np.isfinite(lower[q]) & (lower[q] == upper[q]) & ~upper_open[q]
        masks = {"all_determinate": support,
                 "both_exact": support & exact[a] & exact[b],
                 "one_bounded": support & (exact[a] ^ exact[b]),
                 "both_bounded": support & ~exact[a] & ~exact[b]}
        pred = np.sign(scores[:, q, a] - scores[:, q, b])
        credit = (pred * truth[None] + 1) / 2
        for stratum, use in masks.items():
            count = int(use.sum())
            for j, method in enumerate(methods):
                total = float(credit[j, use].sum()) if count else 0.0
                rows.append({"dataset": dataset, "policy": policy, "axis": axis,
                             "stratum": stratum, "query": str(query_ids[q]),
                             "method": str(method), "pairs": count, "credit": total,
                             "concordance": total / count if count else np.nan})
    return rows
