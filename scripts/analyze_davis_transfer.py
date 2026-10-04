"""Reconstruct DAVIS target-transfer comparisons and paired molecule-bootstrap tables.

Read published predictions, intervals and diagnostics; no fitting is performed.
"""
from pathlib import Path
import argparse
import hashlib
import gzip
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

FRACTIONS = [.125, .25, .5, 1.]
MAIN = ['baseline', 'plus__boltz_affinity', 'plus__boltz_binder',
        'representation_readout', 'representation_plus_chemistry']
STRATA = ['all_determinate', 'both_exact', 'one_bounded', 'both_bounded']
SEED, DRAWS = 20261004, 5000



def resolve_input(path):
    """Prefer a plain input, otherwise accept its gzip-compressed equivalent."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = path.with_name(path.name + '.gz')
    if not path.name.endswith('.gz') and compressed.is_file():
        return compressed
    raise FileNotFoundError(f'Input not found: {path} (or its .gz equivalent)')


def open_text(path):
    """Open UTF-8 text transparently, streaming gzip CSV/JSONL when supplied."""
    actual = resolve_input(path)
    if actual.suffix == '.gz':
        return gzip.open(actual, 'rt', encoding='utf-8', newline='')
    return actual.open('r', encoding='utf-8', newline='')


def read_json(path):
    with open_text(path) as handle:
        return json.load(handle)


def iter_jsonl(path):
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepared_directory(result_root, prepared_root=None):
    """Resolve a caller-specified shared prepared directory or a local default."""
    return Path(prepared_root) if prepared_root is not None else Path(result_root) / 'prepared'


def file_sha256(path):
    """Hash bytes of the actual input file, including a gzip container if used."""
    digest = hashlib.sha256()
    with resolve_input(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan, dtype=float), where=denominator > 0)


def run(result_root, output, prepared_root=None):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    block = result_root / 'DAVIS/exact_identity/new_targets_known_molecules'
    source_path = prepared_directory(result_root, prepared_root) / 'davis_inputs.npz'
    saved = pd.read_csv(resolve_input(block / 'cold_query_metrics.csv'))
    saved_macro = pd.read_csv(resolve_input(block / 'cold_aggregate_metrics.csv'))
    splits = read_json(block / 'cold_splits.json')
    summary = read_json(block / 'cold_summary.json')
    with np.load(block / 'cold_oof.npz', allow_pickle=False) as z:
        names, targets, folds = z['molecule_names'].copy(), z['target_names'].copy(), z['target_fold'].copy()
        native_support, sequence_groups = z['native_common_support'].copy(), z['sequence_sha1'].copy()
        predictions = {key: z[key].copy() for key in z.files if '__fraction_' in key}
    with np.load(source_path, allow_pickle=False) as z:
        mol_lookup = {v: i for i, v in enumerate(z['molecule_names'])}
        target_lookup = {v: i for i, v in enumerate(z['target_names'])}
        mi, ti = np.array([mol_lookup[n] for n in names]), np.array([target_lookup[t] for t in targets])
        lower, upper, opened = [z[k][np.ix_(mi, ti)] for k in ['lower', 'upper', 'upper_open']]
        native = {str(method): z['scores'][j][np.ix_(mi, ti)]
                  for j, method in enumerate(z['methods']) if method in ['boltz_affinity', 'boltz_binder']}
    assert len(names) == 47 and len(targets) == 442 and len(set(sequence_groups)) == 434
    assert np.array_equal(native_support, np.isfinite(np.stack(list(native.values()))).all(axis=0))
    assert all(np.isfinite(x).all() for x in predictions.values())
    for group in set(sequence_groups): assert len(set(folds[sequence_groups == group])) == 1
    a, b = np.triu_indices(len(targets), 1)
    inside = folds[a] == folds[b]
    a, b = a[inside], b[inside]
    known = ~(np.isnan(lower[:, a]) | np.isnan(upper[:, a]) | np.isnan(lower[:, b]) | np.isnan(upper[:, b]))
    left = (lower[:, a] > upper[:, b]) | ((lower[:, a] == upper[:, b]) & opened[:, b])
    right = (lower[:, b] > upper[:, a]) | ((lower[:, b] == upper[:, a]) & opened[:, a])
    truth = np.where(known, left.astype(np.int8) - right.astype(np.int8), 0)
    usable = (truth != 0) & native_support[:, a] & native_support[:, b]
    exact = np.isfinite(lower) & (lower == upper) & ~opened
    masks = {'all_determinate': usable,
             'both_exact': usable & exact[:, a] & exact[:, b],
             'one_bounded': usable & (exact[:, a] ^ exact[:, b]),
             'both_bounded': usable & ~exact[:, a] & ~exact[:, b]}
    counts = {key: value.sum(axis=1) for key, value in masks.items()}
    assert not counts['both_bounded'].any()
    assert np.array_equal(counts['all_determinate'], counts['both_exact'] + counts['one_bounded'])
    common_queries = (counts['both_exact'] > 0) & (counts['one_bounded'] > 0)
    def credit(prediction):
        return (np.sign(prediction[:, a] - prediction[:, b]) * truth + 1) / 2
    records, molecule_records, query_values, pair_deltas = [], [], {}, {}
    for fraction in FRACTIONS:
        methods = {key.rsplit('__fraction_', 1)[0]: value for key, value in predictions.items()
                   if key.endswith('__fraction_' + format(fraction, 'g'))}
        methods.update({'native__' + key: value for key, value in native.items()})
        base_credit = credit(methods['baseline'])
        for method, prediction in methods.items():
            credits = base_credit if method == 'baseline' else credit(prediction)
            if method in MAIN: pair_deltas[(fraction, method)] = (credits - base_credit).astype(np.float32)
            for stratum, mask in masks.items():
                total = (credits * mask).sum(axis=1)
                concordance = ratio(total, counts[stratum])
                query_values[(fraction, method, stratum)] = concordance
                for i, name in enumerate(names):
                    molecule_records.append({'fraction': fraction, 'method': method, 'stratum': stratum,
                                             'query': str(name), 'pairs': int(counts[stratum][i]),
                                             'credit': float(total[i]), 'concordance': concordance[i]})
                for fold in range(5):
                    columns = folds[a] == fold
                    fold_counts = mask[:, columns].sum(axis=1)
                    fold_credit = (credits[:, columns] * mask[:, columns]).sum(axis=1)
                    for i, name in enumerate(names):
                        records.append({'fraction': fraction, 'method': method, 'stratum': stratum,
                                        'outer_fold': fold, 'query': str(name), 'pairs': int(fold_counts[i]),
                                        'credit': float(fold_credit[i]),
                                        'concordance': fold_credit[i] / fold_counts[i] if fold_counts[i] else np.nan})
    rebuilt, rebuilt_molecules = pd.DataFrame(records), pd.DataFrame(molecule_records)
    keys = ['fraction', 'method', 'stratum', 'outer_fold', 'query']
    left_frame, right_frame = rebuilt.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    assert left_frame.index.equals(right_frame.index)
    replay_errors = {key: float(np.nanmax(np.abs(left_frame[key] - right_frame[key]))) for key in ['pairs', 'credit', 'concordance']}
    assert replay_errors['pairs'] == replay_errors['credit'] == 0
    assert replay_errors['concordance'] < 1e-14
    reconstructed_summary = []
    for (fraction, method, stratum), frame in rebuilt_molecules.groupby(['fraction', 'method', 'stratum']):
        valid = frame[frame.pairs > 0]
        reconstructed_summary.append({'fraction': fraction, 'method': method, 'stratum': stratum,
                                      'macro': valid.concordance.mean(), 'queries': len(valid),
                                      'pairs': int(valid.pairs.sum()),
                                      'pooled': valid.credit.sum()/valid.pairs.sum() if len(valid) else np.nan})
    numeric = pd.DataFrame(reconstructed_summary).set_index(['fraction', 'method', 'stratum']).sort_index()
    original = saved_macro.set_index(['fraction', 'method', 'stratum']).sort_index()
    aggregate_errors = {key: float(np.nanmax(np.abs(numeric[key] - original[key]))) for key in ['macro', 'pooled', 'pairs', 'queries']}
    assert max(aggregate_errors.values()) < 1e-14
    rebuilt_molecules.to_csv(output / 'reconstructed_molecule_metrics.csv', index=False)
    draws = np.random.default_rng(SEED).integers(0, len(names), (DRAWS, len(names)))
    boot_arrays = {'molecule_draw_indices': draws.astype(np.int16)}
    def describe(values, tag=None):
        values = np.asarray(values, float)
        sample = values[draws]
        boot = ratio(np.nansum(sample, axis=1), np.isfinite(sample).sum(axis=1))
        valid = values[np.isfinite(values)]
        if tag is not None: boot_arrays[tag] = boot
        return {'queries': len(valid), 'estimate': float(valid.mean()) if len(valid) else np.nan,
                'bootstrap_low': float(np.nanpercentile(boot, 2.5)) if len(valid) else np.nan,
                'bootstrap_high': float(np.nanpercentile(boot, 97.5)) if len(valid) else np.nan,
                'query_q10': float(np.quantile(valid, .1)) if len(valid) else np.nan,
                'query_median': float(np.median(valid)) if len(valid) else np.nan,
                'query_q90': float(np.quantile(valid, .9)) if len(valid) else np.nan}
    curves = []
    for fraction in FRACTIONS:
        for stratum in STRATA[:3]:
            baseline = query_values[(fraction, 'baseline', stratum)]
            for method in MAIN + ['native__boltz_affinity', 'native__boltz_binder']:
                values = query_values[(fraction, method, stratum)]
                for kind, numbers in [('absolute', values), ('versus_baseline', values-baseline)]:
                    tag = f'{kind}:{fraction:g}:{method}:{stratum}'
                    curves.append({'fraction': fraction, 'method': method, 'stratum': stratum,
                                   'contrast': kind, 'pairs': int(counts[stratum].sum()), **describe(numbers, tag)})
    pd.DataFrame(curves).to_csv(output / 'learning_curves_bootstrap.csv', index=False)
    pooled_curves = []
    for fraction in FRACTIONS:
        for stratum in STRATA[:3]:
            denominator = counts[stratum]
            base_credit = np.nan_to_num(query_values[(fraction, 'baseline', stratum)] * denominator)
            for method in MAIN + ['native__boltz_affinity', 'native__boltz_binder']:
                method_credit = np.nan_to_num(query_values[(fraction, method, stratum)] * denominator)
                for kind, numerator in [('absolute', method_credit), ('versus_baseline', method_credit-base_credit)]:
                    boot = ratio(numerator[draws].sum(axis=1), denominator[draws].sum(axis=1))
                    pooled_curves.append({'fraction': fraction, 'method': method, 'stratum': stratum,
                                          'contrast': kind, 'queries': int((denominator > 0).sum()),
                                          'pairs': int(denominator.sum()),
                                          'estimate': float(numerator.sum()/denominator.sum()),
                                          'bootstrap_low': float(np.percentile(boot, 2.5)),
                                          'bootstrap_high': float(np.percentile(boot, 97.5)),
                                          'estimator': 'Pooled within-fold comparison-weighted concordance; secondary to molecule-macro metric.'})
    pd.DataFrame(pooled_curves).to_csv(output / 'pooled_pair_weighted_bootstrap.csv', index=False)
    # Restrict all strata to the same molecules, then make their pair-mixture
    # weights constant across molecules using the observed pooled pair shares.
    exact_share = float(counts['both_exact'][common_queries].sum() / counts['all_determinate'][common_queries].sum())
    composition = []
    for fraction in FRACTIONS:
        for method in MAIN[1:]:
            delta = {st: query_values[(fraction, method, st)] - query_values[(fraction, 'baseline', st)] for st in STRATA[:3]}
            full_gap = np.nanmean(delta['all_determinate']) - np.nanmean(delta['both_exact'])
            restricted = {st: np.where(common_queries, v, np.nan) for st, v in delta.items()}
            common_gap = np.nanmean(restricted['all_determinate']) - np.nanmean(restricted['both_exact'])
            fixed = exact_share * restricted['both_exact'] + (1-exact_share) * restricted['one_bounded']
            settings = {**restricted, 'fixed_empirical_stratum_mix': fixed,
                        'mixture_weighting_difference': restricted['all_determinate'] - fixed,
                        'one_bounded_minus_both_exact': restricted['one_bounded'] - restricted['both_exact']}
            for label, values in settings.items():
                composition.append({'fraction': fraction, 'method': method, 'diagnostic': label,
                                    'common_query_count': int(common_queries.sum()),
                                    'pooled_exact_pair_share': exact_share,
                                    'original_all_minus_exact_gain': float(full_gap),
                                    'common_all_minus_exact_gain': float(common_gap),
                                    'query_support_contribution_to_gap': float(full_gap-common_gap),
                                    **describe(values, f'composition:{fraction:g}:{method}:{label}')})
    pd.DataFrame(composition).to_csv(output / 'common_query_and_stratum_composition.csv', index=False)
    # A second descriptive standardization balances the IDENTITIES of target
    # pairs across the two resolution strata. It is not a new primary metric.
    e, bmask = masks['both_exact'][common_queries], masks['one_bounded'][common_queries]
    ne, nb = e.sum(axis=0), bmask.sum(axis=0)
    pair_common = (ne > 0) & (nb > 0)
    pair_weight = (ne[pair_common] + nb[pair_common]).astype(float)
    pair_weight /= pair_weight.sum()
    pair_support = pd.DataFrame({'target_a': targets[a], 'target_b': targets[b], 'outer_fold': folds[a],
                                 'exact_query_observations': ne, 'one_bounded_query_observations': nb,
                                 'observed_in_both_strata': pair_common})
    pair_support.to_csv(output / 'target_pair_support.csv', index=False)
    pair_standardized = []
    for fraction in FRACTIONS:
        for method in MAIN[1:]:
            difference = pair_deltas[(fraction, method)][common_queries]
            pair_e = ratio((difference * e).sum(axis=0), ne)
            pair_b = ratio((difference * bmask).sum(axis=0), nb)
            standard_e = float(pair_weight @ pair_e[pair_common])
            standard_b = float(pair_weight @ pair_b[pair_common])
            pair_standardized.append({'fraction': fraction, 'method': method,
                                      'common_queries': int(common_queries.sum()),
                                      'target_pair_identities': int(pair_common.sum()),
                                      'exact_comparisons_retained': int(ne[pair_common].sum()),
                                      'one_bounded_comparisons_retained': int(nb[pair_common].sum()),
                                      'exact_gain_fixed_target_pair_mix': standard_e,
                                      'one_bounded_gain_fixed_target_pair_mix': standard_b,
                                      'stratum_gain_difference_fixed_target_pair_mix': standard_b-standard_e,
                                      'uncertainty':'Descriptive reweighting only; no interval computed for this secondary diagnostic.'})
    pd.DataFrame(pair_standardized).to_csv(output / 'fixed_target_pair_composition.csv', index=False)
    # Paired change in each augmentation contrast from the smallest to largest budget.
    changes = []
    for method in MAIN[1:]:
        for stratum in STRATA[:3]:
            low = query_values[(.125, method, stratum)] - query_values[(.125, 'baseline', stratum)]
            high = query_values[(1., method, stratum)] - query_values[(1., 'baseline', stratum)]
            changes.append({'method': method, 'stratum': stratum, 'contrast':'full_minus_one_eighth_gain',
                            **describe(high-low, f'budget_change:{method}:{stratum}')})
    pd.DataFrame(changes).to_csv(output / 'paired_budget_change.csv', index=False)
    fits = pd.DataFrame(iter_jsonl(block / 'cold_fits.jsonl'))
    split_rows = []
    for allocation in splits['allocations']:
        train, test = np.array(allocation['train_indices']), np.array(allocation['test_indices'])
        assert not set(sequence_groups[train]) & set(sequence_groups[test])
        observed = ~np.isnan(lower[:, train]) & ~np.isnan(upper[:, train]) & (np.isfinite(lower[:, train]) | np.isfinite(upper[:, train]))
        exact_train = observed & np.isfinite(lower[:, train]) & (lower[:, train]==upper[:, train]) & ~opened[:, train]
        row = {'outer_fold':allocation['outer_fold'], 'fraction':allocation['fraction'],
               'training_molecules':len(names), 'training_targets':len(train),
               'training_sequence_groups':len(set(sequence_groups[train])), 'known_training_labels':int(observed.sum()),
               'exact_training_labels':int(exact_train.sum()), 'bounded_training_labels':int((observed&~exact_train).sum()),
               'held_out_targets':len(test), 'held_out_sequence_groups':len(set(sequence_groups[test]))}
        checks = fits[(fits.stage=='outer')&(fits.outer_fold==row['outer_fold'])&(fits.fraction==row['fraction'])]
        assert len(checks)==len(summary['models'])
        assert checks.known_labels.eq(row['known_training_labels']).all() and checks.exact_labels.eq(row['exact_training_labels']).all()
        for itr, iva in allocation['inner']:
            assert not set(sequence_groups[itr]) & set(sequence_groups[iva])
            assert not set(sequence_groups[test]) & set(sequence_groups[itr])
        split_rows.append(row)
    for fold in range(5):
        previous=set(); tests=[]
        for arow in sorted([x for x in splits['allocations'] if x['outer_fold']==fold],key=lambda x:x['fraction']):
            current=set(arow['train_indices']); assert previous <= current; previous=current; tests.append(arow['test_indices'])
        assert all(test==tests[0] for test in tests)
    pd.DataFrame(split_rows).to_csv(output/'labels_by_outer_split.csv',index=False)
    fit_cost=fits.groupby(['method','fraction','stage']).agg(fits=('fit_seconds','size'),sum_fit_wall_seconds=('fit_seconds','sum'),
              median_fit_wall_seconds=('fit_seconds','median'),maximum_fit_wall_seconds=('fit_seconds','max'),
              maximum_gradient_inf=('gradient_max','max')).reset_index()
    fit_cost.to_csv(output/'fit_costs.csv',index=False)
    np.savez_compressed(output/'bootstrap_draws.npz', **boot_arrays, molecule_names=names)
    sources = [block/k for k in ['cold_oof.npz','cold_query_metrics.csv','cold_aggregate_metrics.csv',
                                'cold_molecule_metrics.csv','cold_fits.jsonl','cold_summary.json','cold_splits.json']]+[source_path]
    audit={'status':'completed','analysis':'fixed-prediction replay',
           'source_block':str(block),'source_sha256':{str(resolve_input(p)):file_sha256(p) for p in sources},
           'reconstruction':{'query_context_rows':len(rebuilt),'molecule_rows':len(rebuilt_molecules),
                             'query_max_errors':replay_errors,'aggregate_max_errors':aggregate_errors},
           'cohort':{'molecules':len(names),'assay_constructs':len(targets),'unique_sequence_groups':len(set(sequence_groups)),
                     'target_folds':5,'candidate_pairs_within_target_folds':len(a),'native_finite_cells':int(native_support.sum()),
                     'strata':{st:{'molecules_with_pairs':int((counts[st]>0).sum()),'pairs':int(counts[st].sum()),
                                    'query_pair_count_min_positive':int(counts[st][counts[st]>0].min()) if (counts[st]>0).any() else None,
                                    'query_pair_count_median_positive':float(np.median(counts[st][counts[st]>0])) if (counts[st]>0).any() else None}
                               for st in STRATA},
                     'common_exact_and_bounded_queries':int(common_queries.sum()),
                     'excluded_common_query_names':names[~common_queries].tolist()},
           'bootstrap':{'draws':DRAWS,'seed':SEED,'unit':'molecule','paired_across_methods_and_budgets':True,
                        'percentiles':[2.5,97.5],'scope':'Conditional on fixed folds, nested label allocations, trained predictions and target panel; does not resample training or targets. Composition-diagnostic bootstrap holds its empirical reference weights fixed.'},
           'reweighting':{'fixed_stratum_mix_exact_share':exact_share,
                         'common_target_pair_identities':int(pair_common.sum()),
                         'target_pair_weights':'Observed exact-plus-bounded query-comparison counts, fixed identically for both strata; same common molecules.',
                         'limitations':'Common-query restriction controls query identity. Fixed stratum shares control how exact/bounded pairs weight each molecule. Fixed target-pair weights control target-pair identities across strata, but the molecules contributing to each pair can still differ. None identifies a causal measurement effect.'},
           'cost':{'source_block_elapsed_wall_seconds':summary['elapsed_seconds'],'logged_fits':len(fits),
                   'summed_fit_wall_seconds':float(fits.fit_seconds.sum()),'numerical_threads_per_worker':2,
                   'scope':'Observed local fitting wall time. Not measured CPU-seconds or original native-score acquisition cost.'},
           'audit_elapsed_seconds':time.perf_counter()-started}
    dump(output/'audit.json',audit)
    print(json.dumps(clean({'reconstruction':audit['reconstruction'],'cohort':audit['cohort'],'cost':audit['cost'],
                            'audit_elapsed_seconds':audit['audit_elapsed_seconds']}),indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', '--input-root', dest='result_root', type=Path, required=True,
                        help='Root containing DAVIS/exact_identity/new_targets_known_molecules.')
    parser.add_argument('--prepared-root', type=Path,
                        help='Directory containing davis_inputs.npz; defaults to RESULT_ROOT/prepared.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Destination for reconstructed tables, bootstrap draws and audit metadata.')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.result_root, args.output_root, args.prepared_root)


if __name__ == '__main__':
    main()
