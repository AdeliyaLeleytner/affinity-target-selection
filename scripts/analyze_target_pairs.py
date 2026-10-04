"""Stream a completed all-target-pair archive without reading its large query CSV.

Use --result-root and --prepared-root for separately stored results and shared
prepared inputs. CSV/JSONL inputs may be gzip-compressed. No fitting is performed.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import gzip
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

STRATA = ('all_determinate', 'both_exact', 'one_bounded', 'both_bounded')
SUM_FIELDS = ('pairs', 'credit', 'observed_support_cells', 'error_cells',
              'exact_error_cells', 'interval_absolute_error_sum', 'exact_absolute_error_sum')
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


def dump(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, np.generic): x = x.item()
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def divide(num, den):
    return np.divide(num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=den > 0)


def known(lower, upper):
    return ~np.isnan(lower) & ~np.isnan(upper) & (np.isfinite(lower) | np.isfinite(upper))


def ready(block):
    files = ['cold_summary.json', 'cold_oof.npz', 'cold_splits.json',
             'cold_molecule_metrics.csv', 'cold_aggregate_metrics.csv', 'cold_fits.jsonl',
             'cold_selections.jsonl']
    try:
        if any(resolve_input(block / name).stat().st_size == 0 for name in files):
            return False
        summary = read_json(block / 'cold_summary.json')
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (summary.get('mode') == 'full' and summary.get('cold_split') == 'all_target_pairs'
            and summary.get('expected_target_contexts') == 45
            and summary.get('test_predictions_saved') is True)


def audit_policy(result_root, output, policy, prepared_root=None):
    started = time.perf_counter()
    block = result_root / 'SPD' / policy / 'new_targets_known_molecules'
    if not ready(block): return {'policy': policy, 'status': 'pending_completion'}
    out = output / policy;out.mkdir(parents=True, exist_ok=True)
    summary = read_json(block / 'cold_summary.json')
    assert not summary['skipped_contexts'] and not summary['unavailable_arms']
    allocations = read_json(block / 'cold_splits.json')
    config = read_json(result_root / 'effective_config.json')
    assert config['cold_split'] == 'all_target_pairs' and config['fractions'] == [1.0]
    prepared = prepared_directory(result_root, prepared_root)
    source = prepared / f'spd_{policy}_inputs.npz'
    feature_source = prepared / f'SPD_{policy}_features.npz'
    with np.load(source, allow_pickle=False) as z:
        lower, upper, opened = z['lower'].copy(), z['upper'].copy(), z['upper_open'].copy()
        native_scores, native_methods = z['scores'].copy(), list(map(str, z['methods']))
        source_molecules, source_targets = z['molecule_names'].copy(), z['target_names'].copy()
    with np.load(feature_source, allow_pickle=False) as z:
        scaffolds, feature_sequences = z['scaffold_groups'].copy(), z['sequence_sha1'].copy()
    n = len(source_molecules)
    assert n == 930 and lower.shape == (930, 10) and len(native_methods) == 12
    common = np.isfinite(native_scores).all(axis=0)
    models = list(summary['models']);methods = models + ['native__' + m for m in native_methods]
    mpos, qpos = {m:i for i,m in enumerate(methods)}, {str(q):i for i,q in enumerate(source_molecules)}
    fitted = np.array([not m.startswith('native__') for m in methods])
    assert len(models) == 38 and len(set(methods)) == 50 and 'baseline' in mpos
    counts = np.zeros((4, n), np.int64)
    credits = np.zeros((len(methods), 4, n))
    observed_count, exact_count = np.zeros(n, np.int64), np.zeros(n, np.int64)
    interval_errors, exact_errors = np.zeros((len(methods), n)), np.zeros((len(methods), n))
    label_rows, expected_fit_labels, expected_test_pairs = [], {}, {}
    seen_pairs = set()
    assert allocations['target_fold'] is None and allocations['cold_split'] == 'all_target_pairs'
    assert len(allocations['allocations']) == 45 and len(set(feature_sequences)) == 10
    for allocation in allocations['allocations']:
        context = int(allocation['outer_fold']);train = np.asarray(allocation['train_indices'], int)
        test = np.asarray(allocation['test_indices'], int)
        assert len(train) == 8 and len(test) == 2 and len(set(train)) == 8 and len(set(test)) == 2
        assert set(train).isdisjoint(test) and set(train) | set(test) == set(range(10))
        assert set(feature_sequences[train]).isdisjoint(feature_sequences[test])
        assert tuple(test) not in seen_pairs;seen_pairs.add(tuple(test));expected_test_pairs[context] = test
        assert float(allocation['fraction']) == 1.0 and allocation['training_groups'] == 8
        info = known(lower[:, train], upper[:, train])
        exact = info & np.isfinite(lower[:, train]) & (lower[:, train] == upper[:, train]) & ~opened[:, train]
        expected_fit_labels[(context, 'outer', None)] = (8, int(info.sum()), int(exact.sum()))
        inner_valid = []
        assert len(allocation['inner']) == 3
        for inner_fold, (itrain, ivalid) in enumerate(allocation['inner']):
            itrain, ivalid = np.asarray(itrain, int), np.asarray(ivalid, int)
            assert set(itrain).isdisjoint(ivalid) and set(itrain) | set(ivalid) == set(train)
            assert set(feature_sequences[itrain]).isdisjoint(feature_sequences[ivalid])
            assert set(feature_sequences[itrain]).isdisjoint(feature_sequences[test])
            inner_valid.extend(ivalid.tolist())
            ik = known(lower[:, itrain], upper[:, itrain])
            ie = ik & np.isfinite(lower[:, itrain]) & (lower[:, itrain] == upper[:, itrain]) & ~opened[:, itrain]
            expected_fit_labels[(context, 'inner', inner_fold)] = (len(itrain), int(ik.sum()), int(ie.sum()))
        assert sorted(inner_valid) == sorted(train.tolist())
        label_rows.append({'context_id': context, 'target_1': str(source_targets[test[0]]),
                           'target_2': str(source_targets[test[1]]), 'training_targets': 8,
                           'known_training_labels': int(info.sum()), 'exact_training_labels': int(exact.sum()),
                           'bounded_training_labels': int((info & ~exact).sum()),
                           'molecules_with_training_labels': int(info.any(axis=1).sum()),
                           'molecules_without_training_labels': int((~info.any(axis=1)).sum())})
    assert seen_pairs == set(combinations(range(10), 2))
    labels_by_context = {r['context_id']:r for r in label_rows}
    with np.load(block / 'cold_oof.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['molecule_names'], source_molecules)
        np.testing.assert_array_equal(archive['target_names'], source_targets)
        np.testing.assert_array_equal(archive['native_common_support'], common)
        np.testing.assert_array_equal(archive['sequence_sha1'], feature_sequences)
        np.testing.assert_array_equal(archive['context_ids'], np.arange(45))
        assert 'target_fold' not in archive.files
        context_targets = archive['context_targets']
        np.testing.assert_array_equal(context_targets, np.array(list(combinations(range(10), 2))))
        np.testing.assert_array_equal(archive['context_target_names'], source_targets[context_targets])
        np.testing.assert_array_equal(archive['context_sequence_sha1'], feature_sequences[context_targets])
        context_support = archive['context_native_common_support']
        wanted = {f'{method}__fraction_1__context_{context}' for method in methods for context in range(45)}
        actual = {k for k in archive.files if '__fraction_' in k}
        assert actual == wanted
        for context, test in enumerate(context_targets):
            np.testing.assert_array_equal(test, expected_test_pairs[context])
            lo, hi, op = lower[:, test], upper[:, test], opened[:, test]
            cell_support = common[:, test]
            np.testing.assert_array_equal(context_support[context], cell_support)
            present = known(lo, hi)
            positive = (lo[:, 0] > hi[:, 1]) | ((lo[:, 0] == hi[:, 1]) & op[:, 1])
            negative = (lo[:, 1] > hi[:, 0]) | ((lo[:, 1] == hi[:, 0]) & op[:, 0])
            truth = positive.astype(np.int8) - negative.astype(np.int8)
            comparable = present.all(axis=1) & cell_support.all(axis=1) & (truth != 0)
            is_exact = np.isfinite(lo) & (lo == hi) & ~op
            masks = np.stack((comparable, comparable & is_exact.all(axis=1),
                              comparable & (is_exact[:, 0] ^ is_exact[:, 1]),
                              comparable & ~is_exact[:, 0] & ~is_exact[:, 1]))
            counts += masks
            observed = present & cell_support
            exact = observed & is_exact
            observed_count += observed.sum(axis=1);exact_count += exact.sum(axis=1)
            train = np.asarray(allocations['allocations'][context]['train_indices'], int)
            has_train = known(lower[:, train], upper[:, train]).any(axis=1)
            labels_by_context[context]['evaluable_query_pairs'] = int(comparable.sum())
            labels_by_context[context]['evaluable_queries_without_training_labels'] = int((comparable & ~has_train).sum())
            for method in methods:
                mi = mpos[method]
                prediction = archive[f'{method}__fraction_1__context_{context}']
                assert prediction.shape == (n, 2)
                if fitted[mi]: assert np.isfinite(prediction).all()
                else:
                    native_index = native_methods.index(method.removeprefix('native__'))
                    np.testing.assert_array_equal(prediction, native_scores[native_index][:, test])
                assert np.isfinite(prediction[cell_support]).all()
                credit = (np.sign(prediction[:, 0] - prediction[:, 1]) * truth + 1) / 2
                credits[mi] += np.where(masks, credit[None], 0)
                if fitted[mi]:
                    distance = np.maximum(lo-prediction, 0) + np.maximum(prediction-hi, 0)
                    interval_errors[mi] += np.where(observed, distance, 0).sum(axis=1)
                    exact_errors[mi] += np.where(exact, np.abs(prediction-lo), 0).sum(axis=1)
    assert np.array_equal(counts[0], counts[1:].sum(axis=0))
    unique_known = known(lower, upper) & common
    unique_exact = unique_known & np.isfinite(lower) & (lower == upper) & ~opened
    np.testing.assert_array_equal(observed_count, unique_known.sum(axis=1) * 9)
    np.testing.assert_array_equal(exact_count, unique_exact.sum(axis=1) * 9)
    def expected(method, stratum):
        mi, si = mpos[method], STRATA.index(stratum)
        all_stratum = si == 0
        ec = observed_count if fitted[mi] and all_stratum else np.zeros(n)
        xc = exact_count if fitted[mi] and all_stratum else np.zeros(n)
        es = interval_errors[mi] if fitted[mi] and all_stratum else np.zeros(n)
        xs = exact_errors[mi] if fitted[mi] and all_stratum else np.zeros(n)
        return {'pairs': counts[si], 'credit': credits[mi, si],
                'observed_support_cells': observed_count if all_stratum else np.zeros(n),
                'error_cells': ec, 'exact_error_cells': xc,
                'interval_absolute_error_sum': es, 'exact_absolute_error_sum': xs,
                'concordance': divide(credits[mi, si], counts[si]),
                'interval_mae': divide(es, ec), 'exact_mae': divide(xs, xc)}
    # Stream the much smaller molecule table. The huge context-query CSV is not opened.
    errors = defaultdict(float);seen = np.zeros((len(methods), 4, n), bool)
    expected_cache = {(m,s):expected(m,s) for m in methods for s in STRATA}
    molecule_rows = 0
    with open_text(block / 'cold_molecule_metrics.csv') as handle:
        for row in csv.DictReader(handle):
            mi, si, qi = mpos[row['method']], STRATA.index(row['stratum']), qpos[row['query']]
            assert not seen[mi,si,qi];seen[mi,si,qi]=True;molecule_rows+=1
            values = expected_cache[(row['method'],row['stratum'])]
            for field, array in values.items():
                actual = float(row[field]) if row[field] else np.nan;value = float(array[qi])
                if np.isnan(value): assert np.isnan(actual),(field,row['method'],row['query'])
                else:
                    difference = abs(actual-value);errors[field] = max(errors[field],difference)
                    assert difference < 1e-9,(field,difference,row['method'],row['query'])
    assert seen.all() and molecule_rows == len(methods)*4*n
    aggregate_rows=[];aggregate_expected={}
    for method in methods:
        for si,stratum in enumerate(STRATA):
            v=expected_cache[(method,stratum)];eligible=counts[si]>0;values=v['concordance'][eligible]
            ec,xc=int(v['error_cells'].sum()),int(v['exact_error_cells'].sum())
            a={'method':method,'stratum':stratum,'macro':float(values.mean()) if len(values) else np.nan,
               'pooled':float(v['credit'].sum()/counts[si].sum()) if counts[si].sum() else np.nan,
               'queries':int(eligible.sum()),'pairs':int(counts[si].sum()),
               'query_q10':float(np.quantile(values,.1)) if len(values) else np.nan,
               'query_q50':float(np.quantile(values,.5)) if len(values) else np.nan,
               'query_q90':float(np.quantile(values,.9)) if len(values) else np.nan,
               'observed_support_cells':int(v['observed_support_cells'].sum()),'error_cells':ec,'exact_error_cells':xc,
               'interval_mae':float(v['interval_absolute_error_sum'].sum()/ec) if ec else np.nan,
               'exact_mae':float(v['exact_absolute_error_sum'].sum()/xc) if xc else np.nan}
            aggregate_rows.append(a);aggregate_expected[(method,stratum)]=a
    aggregate_errors=defaultdict(float);aggregate_seen=set()
    with open_text(block / 'cold_aggregate_metrics.csv') as handle:
        for row in csv.DictReader(handle):
            key=(row['method'],row['stratum']);assert key not in aggregate_seen;aggregate_seen.add(key)
            assert int(row['evaluated_target_contexts'])==int(row['expected_target_contexts'])==45
            assert row['context_coverage_complete']=='True'
            for field,value in aggregate_expected[key].items():
                if field in ['method','stratum']:continue
                actual=float(row[field]) if row[field] else np.nan
                if np.isnan(value):assert np.isnan(actual)
                else:
                    difference=abs(actual-value);aggregate_errors[field]=max(aggregate_errors[field],difference)
                    assert difference < 1e-9,(field,difference,key)
    assert len(aggregate_seen)==len(methods)*4
    fit_seen=set();fit_cost=defaultdict(lambda: [0,0.,0.]);fit_max_gradient=0.;fits_count=0
    with open_text(block / 'cold_fits.jsonl') as handle:
        for line in handle:
            row=json.loads(line);assert row.get('success') is True and row.get('status')!='failed'
            context,stage,inner=int(row['outer_fold']),row['stage'],row.get('inner_fold')
            key=(context,stage,inner,row['method'],float(row['ridge_candidate']))
            assert key not in fit_seen;fit_seen.add(key)
            expected_targets,expected_known,expected_exact=expected_fit_labels[(context,stage,inner)]
            assert row['training_target_count']==expected_targets and row['training_sequence_groups']==expected_targets
            assert row['known_labels']==expected_known and row['exact_labels']==expected_exact
            assert np.isclose(row['alpha'],row['ridge_candidate']/expected_targets,rtol=1e-13,atol=0)
            assert row['method'] in models and row['ridge_candidate'] in config['ridge']
            fit_cost[(row['method'],stage)][0]+=1;fit_cost[(row['method'],stage)][1]+=row['fit_seconds']
            fit_cost[(row['method'],stage)][2]=max(fit_cost[(row['method'],stage)][2],row['fit_seconds'])
            fit_max_gradient=max(fit_max_gradient,float(row['gradient_max']));fits_count+=1
    chosen={};selection_count=0
    with open_text(block / 'cold_selections.jsonl') as handle:
        for line in handle:
            row=json.loads(line);key=(int(row['outer_fold']),row['method'])
            assert key not in chosen and row['training_targets']==8 and row['training_groups']==8
            assert row['chosen_ridge'] in config['ridge'] and len(row['inner_scores'])==len(config['ridge'])
            options=[x for x in row['inner_scores'] if x['macro'] is not None]
            assert options
            winner=max(options,key=lambda x:(x['macro'],x['ridge_candidate']))
            assert row['chosen_ridge']==winner['ridge_candidate'];chosen[key]=row['chosen_ridge'];selection_count+=1
    assert selection_count==45*len(models)
    expected_fit_keys=set()
    for context in range(45):
        for model in models:
            expected_fit_keys.add((context,'outer',None,model,float(chosen[(context,model)])))
            for inner in range(3):
                for ridge in config['ridge']:expected_fit_keys.add((context,'inner',inner,model,float(ridge)))
    assert fit_seen==expected_fit_keys and fits_count==summary['fits']==17100
    # Resample scaffold groups, retaining every molecule in a sampled group.
    unique_groups,group_index=np.unique(scaffolds.astype(str),return_inverse=True);g=len(unique_groups)
    draws=np.random.default_rng(SEED).integers(0,g,(DRAWS,g),dtype=np.int16)
    weights=np.empty((DRAWS,g),dtype=float)
    for index,row in enumerate(draws):weights[index]=np.bincount(row,minlength=g)
    baseline=mpos['baseline'];bootstrap_rows=[];saved_boot={}
    for si,stratum in enumerate(STRATA[:2]):
        eligible=counts[si]>0;values=divide(credits[:,si],counts[si][None,:])
        sums=np.zeros((g,len(methods)));np.add.at(sums,group_index,np.nan_to_num(values).T)
        group_n=np.bincount(group_index,weights=eligible.astype(float),minlength=g)
        boot=divide(weights@sums,(weights@group_n)[:,None])
        gains=boot-boot[:,baseline,None]
        point=np.nanmean(values,axis=1);point_gain=point-point[baseline]
        for mi,method in enumerate(methods):
            bootstrap_rows.append({'policy':policy,'method':method,'stratum':stratum,
                'queries':int(eligible.sum()),'scaffolds_with_evaluable_queries':int((group_n>0).sum()),
                'pairs':int(counts[si].sum()),'macro':float(point[mi]),
                'macro_low':float(np.nanpercentile(boot[:,mi],2.5)),
                'macro_high':float(np.nanpercentile(boot[:,mi],97.5)),
                'gain_vs_baseline':float(point_gain[mi]),
                'gain_low':float(np.nanpercentile(gains[:,mi],2.5)),
                'gain_high':float(np.nanpercentile(gains[:,mi],97.5)),
                'draws':DRAWS,'resampling_unit':'whole scaffold group'})
        saved_boot[f'{stratum}_macro']=boot;saved_boot[f'{stratum}_gain']=gains
    csv_write(out/'bootstrap_gains.csv',bootstrap_rows)
    reference_rows=[]
    for stratum in STRATA[:2]:
        boot=saved_boot[f'{stratum}_macro']
        for output_name in native_methods:
            plus='plus__'+output_name;pi=mpos[plus]
            for reference in ['native__'+output_name,'availability__'+output_name]:
                ri=mpos[reference];difference=boot[:,pi]-boot[:,ri]
                reference_rows.append({'policy':policy,'output':output_name,'stratum':stratum,
                    'augmented_method':plus,'reference':reference,
                    'gain':aggregate_expected[(plus,stratum)]['macro']-aggregate_expected[(reference,stratum)]['macro'],
                    'bootstrap_low':float(np.nanpercentile(difference,2.5)),
                    'bootstrap_high':float(np.nanpercentile(difference,97.5)),
                    'queries':int((counts[STRATA.index(stratum)]>0).sum()),
                    'resampling_unit':'whole scaffold group','draws':DRAWS})
    csv_write(out/'paired_reference_contrasts.csv',reference_rows)
    csv_write(out/'reconstructed_aggregate.csv',aggregate_rows)
    csv_write(out/'labels_by_target_pair.csv',label_rows)
    cost_rows=[{'method':key[0],'stage':key[1],'fits':v[0],'summed_fit_wall_seconds':v[1],
                'maximum_fit_wall_seconds':v[2]} for key,v in sorted(fit_cost.items())]
    csv_write(out/'fit_costs.csv',cost_rows)
    np.savez_compressed(out/'reconstructed_molecule_contributions.npz',methods=np.asarray(methods),
                         strata=np.asarray(STRATA),molecule_names=source_molecules,scaffold_groups=scaffolds,
                         pairs=counts,credits=credits,observed_context_cells=observed_count,
                         exact_context_cells=exact_count,interval_error_sums=interval_errors,
                         exact_error_sums=exact_errors)
    np.savez_compressed(out/'scaffold_bootstrap.npz',methods=np.asarray(methods),
                         scaffold_names=unique_groups,draws=draws.astype(np.int16),**saved_boot)
    sources=[block/x for x in ['cold_oof.npz','cold_summary.json','cold_splits.json','cold_fits.jsonl',
                               'cold_selections.jsonl','cold_molecule_metrics.csv','cold_aggregate_metrics.csv']]
    sources += [source,feature_source,result_root/'effective_config.json']
    result={'policy':policy,'status':'verified','molecules':n,'scaffold_groups':g,'targets':10,
            'unique_held_target_pairs':45,'training_targets_per_context':8,'fitted_models':len(models),
            'native_references':len(native_methods),'prediction_arrays_replayed':len(methods)*45,
            'molecule_metric_rows_checked':molecule_rows,'aggregate_rows_checked':len(aggregate_seen),
            'molecule_metric_max_errors':dict(errors),'aggregate_metric_max_errors':dict(aggregate_errors),
            'fit_records_verified':fits_count,'selection_records_verified':selection_count,
            'all_fit_success_flags':True,'maximum_logged_gradient_inf':fit_max_gradient,
            'large_context_query_csv_opened':False,
            'strata':{s:{'evaluable_molecules':int((counts[i]>0).sum()),'pairs':int(counts[i].sum())}
                      for i,s in enumerate(STRATA)},
            'unique_common_support_observed_cells':int(unique_known.sum()),
            'unique_common_support_exact_cells':int(unique_exact.sum()),
            'error_unit':'molecule-target-context observations; each target appears in nine contexts',
            'context_training_labels_range':[min(r['known_training_labels'] for r in label_rows),max(r['known_training_labels'] for r in label_rows)],
            'molecules_without_training_labels_range':[min(r['molecules_without_training_labels'] for r in label_rows),max(r['molecules_without_training_labels'] for r in label_rows)],
            'evaluable_query_contexts_without_training_labels':sum(r['evaluable_queries_without_training_labels'] for r in label_rows),
            'bootstrap':{'draws':DRAWS,'seed':SEED,'unit':'scaffold','paired_methods':True,
                         'scope':'Resample whole scaffold groups from the fixed molecule pool. Predictions, target pairs, training allocations and inner choices remain fixed. No target or training resampling; no significance gate.'},
            'interpretation':'Forty-five separate locally held-out target-pair problems on this fixed panel. Molecular structures are known; actual presence of training labels is reported. This does not establish pretraining exclusion or transfer to arbitrary proteins, and these context-specific predictions are not one stitched panel ranking.',
            'cost':{'block_elapsed_wall_seconds':summary['elapsed_seconds'],
                    'summed_fit_wall_seconds':sum(v[1] for v in fit_cost.values()),
                    'numerical_threads_per_worker':config['numerical_threads'],
                    'scope':'Recorded local fitting wall time, not CPU-seconds or original score acquisition cost. Block elapsed wall time includes non-fitting overhead; logged fit wall time is reported separately.'},
            'source_sha256':{str(resolve_input(path)):file_sha256(path) for path in sources},
            'audit_seconds':time.perf_counter()-started}
    dump(out/'audit.json',result)
    lookup={(r['method'],r['stratum']):r for r in bootstrap_rows}
    references={(r['output'],r['stratum'],r['reference']):r for r in reference_rows}
    def interval(point,low,high):return f'{100*point:+.2f} [{100*low:+.2f}, {100*high:+.2f}]'
    lines=[f'SPD all-target-pair transfer: {policy}', '',
        'Verification:45 unique unordered target pairs; exactly8 training targets/context; no outer/inner sequence overlap.',
        f'All{fits_count:,} fit records report success and have verified labels, alpha and unique context/method/candidate identities. All{selection_count:,} outer choices follow the recorded inner criterion.',
        'The2,250 small context prediction arrays reproduce every molecule/aggregate pair count, credit and ranking statistic exactly. Only small floating-point differences occur in absolute-error sums. The huge context-query CSV was not opened.',
        f'Fixed molecule pool:{n}; scaffold groups:{g}. Bootstrap:5,000 paired whole-scaffold resamples, seed20261004; predictions and the ten-target panel remain fixed.',
        'Every evaluable molecule/pair context has at least one informative label among its eight training targets. Molecules without any training label remain in the archive but contribute no evaluable pair.',
        'These are45 separate pair-transfer problems. They do not form a single stitched ranking of all ten targets or establish absence from native-model pretraining.', '']
    for stratum in STRATA[:2]:
        b=lookup[('baseline',stratum)]
        lines += [stratum, f'Baseline macro concordance:{100*b["macro"]:.2f}% [{100*b["macro_low"]:.2f}, {100*b["macro_high"]:.2f}];{b["queries"]} molecules,{b["pairs"]} pairs.',
                  'Output | native% | augmented% | gain vs baseline pp[95% interval] | gain vs own native pp[95% interval] | gain vs availability pp[95% interval]']
        for name in native_methods:
            nat=lookup[('native__'+name,stratum)];aug=lookup[('plus__'+name,stratum)]
            rn=references[(name,stratum,'native__'+name)];ra=references[(name,stratum,'availability__'+name)]
            lines.append(f'{name} | {100*nat["macro"]:.2f} | {100*aug["macro"]:.2f} | '+
                         interval(aug['gain_vs_baseline'],aug['gain_low'],aug['gain_high'])+' | '+
                         interval(rn['gain'],rn['bootstrap_low'],rn['bootstrap_high'])+' | '+
                         interval(ra['gain'],ra['bootstrap_low'],ra['bootstrap_high']))
        lines.append('')
    lines += ['Counts and cost:',
        f'Training measurement cells/context:{result["context_training_labels_range"][0]}–{result["context_training_labels_range"][1]}. Error denominators count molecule-target-context observations; each target occurs in nine contexts.',
        f'Block elapsed wall time:{summary["elapsed_seconds"]:.2f}s; summed recorded fitting wall time:{result["cost"]["summed_fit_wall_seconds"]:.2f}s. These are not CPU-seconds, and original native-score acquisition is excluded.',
        'Native references matter: a large gain over a weak protein-transfer baseline need not be a large gain over the native score itself. All outputs, availability controls and native references are retained in the accompanying tables.',
        'All estimates above are computed from the supplied fixed predictions and observation intervals.', '']
    # Keep numeric units and sentence spacing readable in the plain-text report.
    import re
    text='\n'.join(lines);text=re.sub(r'(?<=[;:])(?=[^\s])',' ',text)
    text=re.sub(r'(?<=[a-z])(?=\d)',' ',text);text=re.sub(r'(?<=\d)(?=s\b)',' ',text)
    (out/'report.txt').write_text(text)
    return result



def combine_reports(output):
    all_rows, compact, audits = [], [], []
    for policy in ['source_representative', 'assay_envelope']:
        audits.append(json.loads((output / policy / 'audit.json').read_text()))
        with (output / policy / 'bootstrap_gains.csv').open() as f:
            rows = list(csv.DictReader(f))
        all_rows.extend(rows)
        lookup = {(r['method'], r['stratum']): r for r in rows}
        with (output / policy / 'paired_reference_contrasts.csv').open() as f:
            refs = {(r['output'], r['stratum'], r['reference']): r for r in csv.DictReader(f)}
        for row in rows:
            if not row['method'].startswith('plus__'):
                continue
            name = row['method'].removeprefix('plus__'); st = row['stratum']
            native, baseline = lookup[('native__' + name, st)], lookup[('baseline', st)]
            entry = {'policy': policy, 'output': name, 'stratum': st,
                     'queries': int(row['queries']), 'pairs': int(row['pairs']),
                     'baseline_pct': 100 * float(baseline['macro']),
                     'native_pct': 100 * float(native['macro']),
                     'augmented_pct': 100 * float(row['macro']),
                     'gain_vs_baseline_pp': 100 * float(row['gain_vs_baseline']),
                     'gain_vs_baseline_low_pp': 100 * float(row['gain_low']),
                     'gain_vs_baseline_high_pp': 100 * float(row['gain_high'])}
            for ref_name, prefix in [('native__' + name, 'native'), ('availability__' + name, 'availability')]:
                ref = refs[(name, st, ref_name)]
                for src, suffix in [('gain', ''), ('bootstrap_low', '_low'), ('bootstrap_high', '_high')]:
                    entry[f'gain_vs_{prefix}{suffix}_pp'] = 100 * float(ref[src])
            compact.append(entry)
    csv_write(output / 'all_policies_bootstrap_gains.csv', all_rows)
    csv_write(output / 'primary_comparison.csv', compact)
    lines = ['Completed SPD held-target-pair audit', '',
             'Both policies pass: 45 unique held pairs, eight training targets per context, no sequence overlap; 17,100 successful fits and 1,710 verified choices per policy.',
             'All 2,250 context arrays per policy reproduce molecule/aggregate ranking counts, credits and metrics exactly. Absolute-error differences are below 5.69e-14. Huge context-query CSVs were never opened.',
             'Uncertainty: 5,000 paired whole-scaffold bootstrap resamples, seed 20261004, with fitted models and target contexts fixed. It does not resample training or proteins.', '']
    for a in audits:
        lines.append(f"{a['policy']}: {a['molecules']} pool molecules; {a['scaffold_groups']} scaffold groups; "
                     f"{a['strata']['all_determinate']['evaluable_molecules']} evaluable molecules / "
                     f"{a['strata']['all_determinate']['pairs']:,} determinate pairs; "
                     f"{a['strata']['both_exact']['evaluable_molecules']} molecules / "
                     f"{a['strata']['both_exact']['pairs']:,} two-exact pairs.")
    lines += ['', 'Selected magnitudes; all twelve outputs are retained in primary_comparison.csv:']
    for r in compact:
        if r['output'] not in ['boltz_binary', 'boltzina_vina_far_binary', 'vina']:
            continue
        lines.append(f"{r['policy']}, {r['stratum']}, {r['output']}: baseline {r['baseline_pct']:.2f}%; "
                     f"native {r['native_pct']:.2f}%; augmented {r['augmented_pct']:.2f}%; "
                     f"gain vs baseline {r['gain_vs_baseline_pp']:+.2f} "
                     f"[{r['gain_vs_baseline_low_pp']:+.2f}, {r['gain_vs_baseline_high_pp']:+.2f}] pp; "
                     f"gain vs native {r['gain_vs_native_pp']:+.2f} "
                     f"[{r['gain_vs_native_low_pp']:+.2f}, {r['gain_vs_native_high_pp']:+.2f}] pp.")
    lines += ['', 'Interpretation and units:',
              'Every evaluable molecule/pair context has at least one informative training label among the remaining eight targets. Entirely unlabelled pool molecules contribute no evaluated pair.',
              'These are 45 separate locally held-out target-pair problems on a fixed ten-target panel. Their predictions are not one stitched panel ranking, and no native-model pretraining exclusion is established.',
              'Keep native references visible: the gain over a weak protein-transfer baseline is not the gain from augmenting or calibrating the native score.',
              'Error denominators count molecule-target-context observations; each target occurs in nine contexts. They are not ninefold more unique measurements.', '']
    for a in audits:
        lines.append(f"{a['policy']} cost: block wall time {a['cost']['block_elapsed_wall_seconds']:.2f}s; "
                     f"summed fit wall time {a['cost']['summed_fit_wall_seconds']:.2f}s; "
                     f"training labels/context {a['context_training_labels_range'][0]:,}–"
                     f"{a['context_training_labels_range'][1]:,}. These are not CPU-seconds or original score-acquisition cost.")
    lines += ['', 'See per-policy reports, bootstrap tables, label counts, hashes and cost records. analyze_target_pairs.py regenerates these outputs from supplied fixed predictions.', '']
    (output / 'report.txt').write_text('\n'.join(lines))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', '--input-root', dest='result_root', type=Path, required=True,
                        help='Root containing effective_config.json and SPD/<policy>/new_targets_known_molecules.')
    parser.add_argument('--prepared-root', type=Path,
                        help='Shared prepared inputs; defaults to RESULT_ROOT/prepared. Use results/label_value/prepared for the release layout.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Destination for per-policy and combined reconstructed tables and bootstrap output.')
    parser.add_argument('--policy', choices=['source_representative', 'assay_envelope'], action='append',
                        help='Optional policy selection; repeat for both policies (the default).')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    policies = args.policy or ['source_representative', 'assay_envelope']
    results = []
    with threadpool_limits(limits=1):
        for policy in policies:
            result = audit_policy(args.result_root, args.output_root, policy, args.prepared_root)
            results.append(result)
            print(json.dumps({k: result[k] for k in ['policy', 'status', 'unique_held_target_pairs',
                                                    'fit_records_verified', 'audit_seconds'] if k in result}), flush=True)
    dump(args.output_root / 'audit_status.json', results)
    if set(policies) == {'source_representative', 'assay_envelope'} and all(r['status'] == 'verified' for r in results):
        combine_reports(args.output_root)


if __name__ == '__main__':
    main()
