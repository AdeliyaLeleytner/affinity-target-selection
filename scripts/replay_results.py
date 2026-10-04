"""Reconstruct ranking metrics from released predictions, without fitting models.

This verifier deliberately does not import the fitting or metric implementation.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRATA = ['all_determinate', 'both_exact', 'one_bounded', 'both_bounded']


def read_table(path):
    return pd.read_csv(path if path.exists() else path.with_suffix(path.suffix + '.gz'))


def contributions(prediction, lower, upper, opened, support):
    n, t = lower.shape
    a, b = np.triu_indices(t, 1)
    known = ~(np.isnan(lower[:, a]) | np.isnan(upper[:, a]) |
              np.isnan(lower[:, b]) | np.isnan(upper[:, b]))
    positive = (lower[:, a] > upper[:, b]) | ((lower[:, a] == upper[:, b]) & opened[:, b])
    negative = (lower[:, b] > upper[:, a]) | ((lower[:, b] == upper[:, a]) & opened[:, a])
    assert not np.any(known & positive & negative)
    truth = positive.astype(int) - negative.astype(int)
    usable = known & (truth != 0) & support[:, a] & support[:, b]
    exact = np.isfinite(lower) & (lower == upper) & ~opened
    masks = [usable, usable & exact[:, a] & exact[:, b],
             usable & (exact[:, a] ^ exact[:, b]), usable & ~exact[:, a] & ~exact[:, b]]
    available = np.all(~support | np.isfinite(prediction), axis=1)
    credits = (np.sign(prediction[:, a] - prediction[:, b]) * truth + 1) / 2
    counts = np.stack([m.sum(axis=1) for m in masks])
    sums = np.stack([np.where(m, credits, 0).sum(axis=1) for m in masks])
    counts[:, ~available] = 0
    sums[:, ~available] = 0
    return counts, sums


def check_summary(row, count, credit, *, cold=False):
    usable = count > 0
    macro = np.mean(credit[usable] / count[usable]) if usable.any() else np.nan
    pooled = credit.sum() / count.sum() if count.sum() else np.nan
    np.testing.assert_allclose([row['macro' if cold else 'macro_concordance'],
                                row['pooled' if cold else 'pooled_concordance']],
                               [macro, pooled], rtol=0, atol=1e-12, equal_nan=True)
    assert int(row['pairs']) == int(count.sum())
    assert int(row['queries' if cold else 'evaluable_queries']) == int(usable.sum())


def input_data(dataset, policy, names=None):
    filename = 'davis_inputs.npz' if dataset == 'DAVIS' else 'spd_' + policy + '_inputs.npz'
    with np.load(ROOT/'results/label_value/prepared'/filename) as source:
        data = {k: source[k] for k in source.files if k != 'representations'}
    if names is not None:
        lookup = {n: i for i, n in enumerate(data['molecule_names'])}
        take = np.array([lookup[n] for n in names])
        for k in ['lower', 'upper', 'upper_open', 'molecule_names']:
            data[k] = data[k][take]
        data['scores'] = data['scores'][:, take]
    return data


def fit_integrity(base, kind):
    path = base/(kind+'_fits.jsonl')
    opener = path.open if path.exists() else lambda: gzip.open(str(path)+'.gz', 'rt')
    count = 0
    with opener() as handle:
        for line in handle:
            row = json.loads(line)
            assert row.get('success', row.get('diagnostics', {}).get('success')) is True, (path, row)
            count += 1
    return count


def check_known(path):
    base = path.parent
    dataset, policy = base.parts[-3:-1]
    with np.load(path) as z:
        names, methods, fractions = z['molecule_names'], z['methods'], z['fractions']
        data = input_data(dataset, policy, names)
        support = z['common_native_finite']
        prediction = z['predictions']
    rows = read_table(base/'known_query_metrics.csv')
    indexed = rows.set_index(['method','fraction','stratum','query']).sort_index()
    aggregate = read_table(base/'known_aggregate_metrics.csv')
    for i, method in enumerate(methods):
        for j, fraction in enumerate(fractions):
            counts, credits = contributions(prediction[i, j], data['lower'], data['upper'],
                                              data['upper_open'], support)
            for k, stratum in enumerate(STRATA):
                selected = indexed.loc[(method, fraction, stratum)].loc[names]
                np.testing.assert_array_equal(selected.pairs, counts[k])
                np.testing.assert_allclose(selected.credit, credits[k], rtol=0, atol=1e-12)
                summary = aggregate[(aggregate.method == method) & (aggregate.fraction == fraction) & (aggregate.stratum == stratum)].iloc[0]
                check_summary(summary, counts[k], credits[k])
    return {'path': str(base.relative_to(ROOT)), 'query_rows': len(rows),
            'aggregate_rows': len(aggregate), 'successful_fits': fit_integrity(base, 'known')}


def check_cold(path):
    base = path.parent
    dataset, policy = base.parts[-3:-1]
    summary = read_table(base/'cold_aggregate_metrics.csv')
    with np.load(path) as z:
        names = z['molecule_names']
        data = input_data(dataset, policy, names)
        support = z['native_common_support']
        all_pairs = 'context_targets' in z.files
        contexts = (list(zip(z['context_ids'], z['context_targets'])) if all_pairs else
                    [(f, np.flatnonzero(z['target_fold'] == f)) for f in np.unique(z['target_fold'])])
        for (method, fraction), group in summary.groupby(['method', 'fraction']):
            totals = np.zeros((2, 4, len(names)))
            for context_id, targets in contexts:
                key = f'{method}__fraction_{fraction:g}'
                if all_pairs:
                    key += f'__context_{context_id}'
                    pred = z[key]
                elif method.startswith('native__'):
                    i = list(data['methods']).index(method[len('native__'):])
                    pred = data['scores'][i][:, targets]
                else:
                    pred = z[key][:, targets]
                count, credit = contributions(pred, data['lower'][:, targets], data['upper'][:, targets],
                                                data['upper_open'][:, targets], support[:, targets])
                totals[0] += count; totals[1] += credit
            for k, stratum in enumerate(STRATA):
                row = group[group.stratum == stratum].iloc[0]
                check_summary(row, totals[0, k], totals[1, k], cold=True)
                assert bool(row.context_coverage_complete)
    return {'path': str(base.relative_to(ROOT)), 'aggregate_rows': len(summary),
            'contexts': len(contexts), 'successful_fits': fit_integrity(base, 'cold')}


def check_native():
    base = ROOT/'results/native_baseline'
    saved = pd.read_csv(base/'query_contributions.csv.gz')
    blocks = 0
    for (dataset, policy, axis), group in saved.groupby(['dataset', 'policy', 'axis']):
        indexed = group.set_index(['method','query','stratum']).sort_index()
        name = 'davis_inputs.npz' if dataset == 'DAVIS' else 'spd_'+policy+'_inputs.npz'
        with np.load(base/name) as source:
            data = {k: source[k] for k in source.files if k != 'representations'}
        if dataset == 'DAVIS' and policy == 'exact_identity':
            m = pd.read_csv(base/'davis_molecules.csv').set_index('drug_name')
            take = m.loc[data['molecule_names'], 'match_method'].eq('exact_standard_inchikey').to_numpy()
            for key in ['lower', 'upper', 'upper_open', 'molecule_names']: data[key] = data[key][take]
            data['scores'] = data['scores'][:, take]
        lo, hi, op = data['lower'], data['upper'], data['upper_open']
        scores = data['scores']; names = data['molecule_names']
        if axis == 'ligand':
            scores = scores.transpose(0, 2, 1); lo, hi, op = lo.T, hi.T, op.T; names = data['target_names']
        support = np.isfinite(scores).all(axis=0)
        for i, method in enumerate(data['methods']):
            # Bound memory for the930-molecule ligand-ranking axis.
            for q, query in enumerate(names):
                count, credit = contributions(scores[i, q:q+1], lo[q:q+1], hi[q:q+1], op[q:q+1], support[q:q+1])
                s = indexed.loc[(method, query)].loc[STRATA]
                np.testing.assert_array_equal(s.pairs, count[:, 0])
                np.testing.assert_allclose(s.credit, credit[:, 0], rtol=0, atol=1e-12)
        blocks += 1
    return {'path': 'results/native_baseline', 'query_rows': len(saved), 'blocks': blocks}


def main():
    manifest = json.loads((ROOT/'data/raw/manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT/item['file']; payload = path.read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
    results = [check_native()]
    for folder in ['label_value', 'score_components', 'target_pairs']:
        root = ROOT/'results'/folder
        for path in sorted(root.glob('**/known_oof.npz')):
            results.append(check_known(path))
        for path in sorted(root.glob('**/cold_oof.npz')):
            results.append(check_cold(path))
    output = ROOT/'build/replay_verification.json'; output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'status': 'passed', 'raw_inputs': len(manifest['files']), 'blocks': results}, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'blocks': len(results), 'report': str(output)}, indent=2))

if __name__ == '__main__': main()
