"""Read-only DAVIS OOF replay, split/selection audit and paired learning curves.

No project imports, model fitting, new inference or retrieval. All paths are
CLI arguments. Plain or gzip-compressed records are supported, with relative
paths in the output provenance manifest.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits



def resolve_input(path):
    """Prefer the named input, otherwise its .gz public-package counterpart."""
    path = Path(path)
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def open_text(path):
    path = resolve_input(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_text(path):
    with open_text(path) as handle:
        return handle.read()


def read_json(path):
    with open_text(path) as handle:
        return json.load(handle)


def read_jsonl(path):
    with open_text(path) as handle:
        return [json.loads(line) for line in handle]


STRATA = ("all_determinate", "both_exact", "one_bounded", "both_bounded")
TOL = 1e-12  # Numerical replay tolerance only.


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sha256(path):
    h = hashlib.sha256()
    with resolve_input(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def replay_intervals(lower, upper, opened, support, predictions):
    """Independent row-wise interval ordering; never call study.metrics."""
    lower, upper, opened, support, predictions = map(np.asarray, (lower, upper, opened, support, predictions))
    if predictions.ndim != 3 or predictions.shape[1:] != lower.shape:
        raise ValueError("Prediction axes disagree")
    if not np.isfinite(predictions[:, support]).all():
        raise ValueError("Nonfinite supported prediction")
    variants, n, targets = predictions.shape
    a, b = np.triu_indices(targets, 1)
    counts, credit = np.zeros((n, 4), int), np.zeros((variants, n, 4))
    for q in range(n):
        observed = ~np.isnan(lower[q]) & ~np.isnan(upper[q]) & support[q]
        forward = (lower[q, a] > upper[q, b]) | ((lower[q, a] == upper[q, b]) & opened[q, b])
        reverse = (lower[q, b] > upper[q, a]) | ((lower[q, b] == upper[q, a]) & opened[q, a])
        use = observed[a] & observed[b] & (forward | reverse)
        if np.any(use & forward & reverse):
            raise ValueError("Inconsistent interval order")
        aa, bb = a[use], b[use]
        truth = np.where(forward[use], 1., -1.)
        values = (np.sign(predictions[:, q, aa] - predictions[:, q, bb]) * truth[None] + 1) / 2
        exact = np.isfinite(lower[q]) & (lower[q] == upper[q]) & ~opened[q]
        masks = (np.ones(len(aa), bool), exact[aa] & exact[bb], exact[aa] ^ exact[bb], ~exact[aa] & ~exact[bb])
        for si, mask in enumerate(masks):
            counts[q, si] = int(mask.sum())
            credit[:, q, si] = values[:, mask].sum(axis=1)
    concordance = np.divide(credit, counts[None], out=np.full_like(credit, np.nan), where=counts[None] > 0)
    return counts, credit, concordance


def numeric_check(actual, expected, field, exact=False):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise AssertionError(f"{field}: axes/missingness mismatch")
    finite = np.isfinite(b)
    error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
    if error > (0 if exact else TOL):
        raise AssertionError(f"{field}: replay error {error}")
    return error


def verify_metrics(block, methods, fractions, names, folds, counts, credits, concordance):
    q = pd.read_csv(resolve_input(block / "known_query_metrics.csv"))
    mapping = ({m:i for i,m in enumerate(methods)}, {float(f):i for i,f in enumerate(fractions)},
               {str(n):i for i,n in enumerate(names)}, {s:i for i,s in enumerate(STRATA)})
    keys = [q[col].map(m) for col,m in zip(("method", "fraction", "query", "stratum"), mapping)]
    if any(x.isna().any() for x in keys):
        raise AssertionError("Unrecognized query key")
    mi, fi, qi, si = [x.to_numpy(int) for x in keys]
    vi = mi * len(fractions) + fi
    if q.duplicated(["method", "fraction", "query", "stratum"]).any() or len(q) != len(methods)*len(fractions)*len(names)*4:
        raise AssertionError("Missing or duplicate query metric rows")
    if not q.status.eq("available").all() or not np.array_equal(q.outer_fold, folds[qi]):
        raise AssertionError("Query status/fold mismatch")
    query_errors = {}
    for field, values in (("pairs", counts[qi,si]), ("eligible_pairs", counts[qi,si]),
                           ("credit", credits[vi,qi,si]), ("concordance", concordance[vi,qi,si])):
        query_errors[field] = numeric_check(q[field], values, field, exact=field!="concordance")
    agg = pd.read_csv(resolve_input(block / "known_aggregate_metrics.csv"))
    if len(agg) != len(methods)*len(fractions)*4 or agg.duplicated(["method","fraction","stratum"]).any():
        raise AssertionError("Missing/duplicate aggregate metrics")
    aggregate_errors, replayed = {}, []
    for row in agg.to_dict("records"):
        vi = mapping[0][row["method"]] * len(fractions) + mapping[1][row["fraction"]]
        si = mapping[3][row["stratum"]]
        valid = counts[:,si]>0
        pairs, credit = int(counts[:,si].sum()), float(credits[vi,:,si].sum())
        expected = {"queries": len(names), "evaluable_queries": int(valid.sum()), "unavailable_queries": 0,
                    "eligible_pairs": pairs, "pairs": pairs, "credit": credit,
                    "macro_concordance": float(concordance[vi,valid,si].mean()) if valid.any() else np.nan,
                    "pooled_concordance": credit/pairs if pairs else np.nan}
        for field,value in expected.items():
            error = numeric_check([row[field]], [value], field, exact=field not in ("macro_concordance","pooled_concordance"))
            aggregate_errors[field] = max(aggregate_errors.get(field,0), error)
        replayed.append({"method": row["method"], "fraction": row["fraction"], "stratum": row["stratum"], **expected})
    return {"query_rows":len(q),"aggregate_rows":len(agg),"query_errors":query_errors,"aggregate_errors":aggregate_errors}, replayed


def split_audit(record, oof, lower, upper, opened):
    groups, n = oof["scaffold_groups"], len(oof["molecule_names"])
    plan_lookup, training_rows = {}, []
    outer_folds, inner_folds = int(record["config"]["outer_folds"]), int(record["config"]["inner_folds"])
    known = ~np.isnan(lower)&~np.isnan(upper)&(np.isfinite(lower)|np.isfinite(upper))
    exact = known & np.isfinite(lower)&(lower==upper)&~opened
    for plan in record["plans"]:
        fold, fraction = plan["outer_fold"], plan["fraction"]
        pool, train, test = [np.asarray(plan[key],int) for key in ("outer_training_indices","training_indices","test_indices")]
        if plan["status"]!="available" or len(set(train))!=len(train):
            raise AssertionError("Unavailable/duplicate training partition")
        if set(pool)|set(test)!=set(range(n)) or set(pool)&set(test) or not set(train)<=set(pool):
            raise AssertionError("Outer index separation failure")
        if set(test)!=set(np.flatnonzero(oof["fold_ids"]==fold)):
            raise AssertionError("Saved fold identity mismatch")
        if set(groups[pool]) & set(groups[test]):
            raise AssertionError("Outer scaffold leakage")
        if len(set(groups[train])) != int(np.ceil(fraction*len(set(groups[pool])))):
            raise AssertionError("Training group budget mismatch")
        if set(train)!=set(pool[np.isin(groups[pool], groups[train])]):
            raise AssertionError("Training subset splits a scaffold group")
        validation = []
        if len(plan["inner"])!=inner_folds:
            raise AssertionError("Wrong inner-fold count")
        for inner in plan["inner"]:
            a,b=np.asarray(inner["training_indices"],int),np.asarray(inner["validation_indices"],int)
            if set(a)|set(b)!=set(train) or set(a)&set(b) or set(groups[a])&set(groups[b]):
                raise AssertionError("Inner partition/group leakage")
            if (set(a)|set(b)) & set(test):
                raise AssertionError("Outer outcomes entered an inner partition")
            validation.extend(b.tolist())
        if sorted(validation)!=sorted(train.tolist()):
            raise AssertionError("Inner validation does not cover each training row once")
        row={"outer_fold":fold,"fraction":fraction,"training_molecules":len(train),
             "training_groups":len(set(groups[train])),"known_training_labels":int(known[train].sum()),
             "exact_training_labels":int(exact[train].sum()),
             "bounded_training_labels":int((known[train]&~exact[train]).sum()),"test_molecules":len(test)}
        training_rows.append(row);plan_lookup[(fold,fraction)]=plan
    for fold in range(outer_folds):
        previous=set()
        for fraction in sorted(oof["fractions"]):
            current=set(plan_lookup[(fold,float(fraction))]["training_indices"])
            if not previous<=current:raise AssertionError("Budgets are not nested")
            previous=current
    return plan_lookup,training_rows,known,exact


def fit_and_selection_audit(block, plans, known, exact, order_counts, oof, summary):
    fits=read_jsonl(block/"known_fits.jsonl")
    seen=set();stages=Counter();fallbacks=Counter();params={};max_gradient=0.;fallback_rows=[];final_ridges={}
    for row in fits:
        d=row["diagnostics"];key=tuple(row.get(k) for k in ("stage","outer_fold","fraction","inner_fold","method","ridge"))
        if key in seen or row["status"]!="fitted" or d["success"] is not True:
            raise AssertionError("Failed or duplicated fit")
        seen.add(key);stages[row["stage"]]+=1
        if not all(np.isfinite(d[k]) for k in ("objective","gradient_max","alpha","fit_seconds")):
            raise AssertionError("Nonfinite fit diagnostic")
        plan=plans[(row["outer_fold"],row["fraction"])]
        if row["stage"]=="inner":
            inner=next(x for x in plan["inner"] if x["fold"]==row["inner_fold"]);train=inner["training_indices"]
        elif row["stage"]=="outer_final":train=plan["training_indices"]
        else:raise AssertionError("Unexpected fit stage")
        expected={"training_molecules":len(train),"training_groups":len(set(oof["scaffold_groups"][train])),
                  "known_training_labels":int(known[train].sum()),"exact_training_labels":int(exact[train].sum()),
                  "censored_or_interval_training_labels":int((known[train]&~exact[train]).sum())}
        if (any(row[k]!=v for k,v in expected.items()) or d["known_labels"]!=expected["known_training_labels"]
                or d["exact_labels"]!=expected["exact_training_labels"]):
            raise AssertionError("Recorded training counts do not match training-only labels")
        if row["stage"]=="outer_final":final_ridges[(row["method"],row["outer_fold"],row["fraction"])]=row["ridge"]
        if d["alpha"]!=row["ridge"]/known.shape[1]:raise AssertionError("Ridge normalization mismatch")
        params.setdefault(row["method"],set()).add(d["parameters"])
        max_gradient=max(max_gradient,d["gradient_max"])
        fallback=d.get("fallback")
        if fallback:
            fallbacks[fallback["method"]]+=1
            history=fallback.get("objective_history")
            if history and np.any(np.diff(history)>TOL):raise AssertionError("Fallback objective increased")
            fallback_rows.append({"method":row["method"],"outer_fold":row["outer_fold"],"fraction":row["fraction"],
                                  "stage":row["stage"],"inner_fold":row.get("inner_fold"),"ridge":row["ridge"],
                                  "fallback_method":fallback["method"],"gradient_max":d["gradient_max"],
                                  "iterations":d["iterations"],"primary_iterations":d.get("primary_iterations")})
    if len(fits)!=summary["fits"]:raise AssertionError("Fit count differs from summary")
    selections=read_jsonl(block/"known_selections.jsonl")
    choices=[];seen_choices=set()
    for row in selections:
        if row["status"]!="selected":raise AssertionError("Unselected model")
        choice_key=(row["method"],row["outer_fold"],row["fraction"])
        if choice_key in seen_choices:raise AssertionError("Duplicate selection record")
        seen_choices.add(choice_key)
        plan=plans[(row["outer_fold"],row["fraction"])]
        for c in row["candidates"]:
            if sorted(x["fold"] for x in c["inner"])!=sorted(x["fold"] for x in plan["inner"]):
                raise AssertionError("Inner candidate fold coverage mismatch")
            for item in c["inner"]:
                validation=next(x["validation_indices"] for x in plan["inner"] if x["fold"]==item["fold"])
                if item["pairs"]!=int(order_counts[validation,0].sum()) or item["queries"]!=int((order_counts[validation,0]>0).sum()):
                    raise AssertionError("Inner candidate support differs from its saved validation partition")
            if c["queries"]!=sum(x["queries"] for x in c["inner"]) or c["pairs"]!=sum(x["pairs"] for x in c["inner"]):
                raise AssertionError("Inner selection support sum mismatch")
            numeric_check([c["sum_concordance"]],[sum(x["sum_concordance"] for x in c["inner"])],"inner sum")
            numeric_check([c["macro_concordance"]],[c["sum_concordance"]/c["queries"]],"inner macro")
        winner=max(row["candidates"],key=lambda c:(c["macro_concordance"],c["ridge"]))
        if winner["ridge"]!=row["selected_ridge"] or row["selected_alpha"]!=winner["ridge"]/known.shape[1]:
            raise AssertionError("Hyperparameter selection/tie rule mismatch")
        if final_ridges[choice_key]!=winner["ridge"]:raise AssertionError("Final fit used a different ridge")
        mi=list(oof["methods"]).index(row["method"]);fi=list(oof["fractions"]).index(row["fraction"])
        if oof["selected_ridge"][mi,fi,row["outer_fold"]]!=winner["ridge"]:
            raise AssertionError("Saved OOF ridge differs from selection record")
        choices.append({"method":row["method"],"outer_fold":row["outer_fold"],"fraction":row["fraction"],
                        "ridge":winner["ridge"],"inner_macro":winner["macro_concordance"]})
    if len(selections)!=summary["selected_models"]:raise AssertionError("Selection count mismatch")
    for alias,canonical in summary["availability_aliases"].items():
        a,b=list(oof["methods"]).index(alias),list(oof["methods"]).index(canonical)
        if not np.array_equal(oof["predictions"][a],oof["predictions"][b]) or not np.array_equal(oof["selected_ridge"][a],oof["selected_ridge"][b]):
            raise AssertionError("Availability alias mismatch")
    return {"fits":len(fits),"success_true":len(fits),"failed":0,"stages":dict(stages),"selections_verified":len(selections),
            "fallbacks":dict(fallbacks),"maximum_gradient":max_gradient,
            "parameters_by_arm":{k:sorted(v) for k,v in params.items()},
            "inner_predictions_saved":False,"selection_verification_limit":"Candidate arithmetic and exact winning-ridge rule verified; inner predictions were not saved for a second metric replay."}, choices,fallback_rows


def bootstrap_curves(concordance,credit,counts,groups,replicates,seed):
    # Preserve first-occurrence group order, so singleton scaffolds yield
    # exactly the same draws as paired molecule resampling at the same seed.
    group_codes,unique=pd.factorize(np.asarray(groups,str),sort=False)
    sizes=np.bincount(group_codes,minlength=len(unique));pair_sums=np.bincount(group_codes,weights=counts,minlength=len(unique))
    sums=np.zeros((len(unique),concordance.shape[0]));credit_sums=np.zeros_like(sums)
    np.add.at(sums,group_codes,concordance.T);np.add.at(credit_sums,group_codes,credit.T)
    weights=np.random.default_rng(seed).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=replicates)
    return (weights@sums)/(weights@sizes)[:,None],(weights@credit_sums)/(weights@pair_sums)[:,None]


def audit(run_root,output_root,run_log=None,source_snapshot=None,bootstrap_replicates=2000,bootstrap_seed=20261004,
          run_id=None,revision=None,prepared_root=None):
    start=time.perf_counter();run_root,output_root=Path(run_root),Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    prepared_root=Path(prepared_root) if prepared_root else run_root/"prepared"
    block=run_root/"DAVIS/exact_identity/new_molecule_known_targets"
    oof=load_npz(block/"known_oof.npz");source=load_npz(prepared_root/"davis_inputs.npz")
    info=read_json(block/"known_summary.json");splits=read_json(block/"known_splits.json")
    if info["status"]!="complete" or info["unavailable"] or not np.isfinite(oof["predictions"]).all():raise AssertionError("Incomplete block")
    molecule_table=pd.read_csv(resolve_input(prepared_root/"davis_molecules.csv"),dtype=str)
    expected_names=set(molecule_table.loc[molecule_table.match_method.eq("exact_standard_inchikey"),"drug_name"])
    if set(oof["molecule_names"])!=expected_names:raise AssertionError("Cohort differs from exact-identity source rule")
    lookup={str(x):i for i,x in enumerate(source["molecule_names"])};indices=np.array([lookup[str(x)] for x in oof["molecule_names"]])
    if len(indices)!=47 or len(set(indices))!=47 or not np.array_equal(source["target_names"],oof["target_names"]):raise AssertionError("DAVIS axis mismatch")
    native_indices=[list(source["methods"]).index(x) for x in oof["native_methods"]]
    if not np.array_equal(source["scores"][native_indices][:,indices],oof["native_scores"],equal_nan=True):raise AssertionError("Native prediction mismatch")
    lower,upper,opened=(source[k][indices] for k in ("lower","upper","upper_open"));support=np.isfinite(oof["native_scores"]).all(axis=0)
    if not np.array_equal(support,oof["common_native_finite"]):raise AssertionError("Finite-native support mismatch")
    fractions=oof["fractions"];methods=list(map(str,oof["methods"]))+["native__"+str(x) for x in oof["native_methods"]]
    native=np.repeat(oof["native_scores"][:,None],len(fractions),axis=1)
    predictions=np.concatenate((oof["predictions"],native)).reshape(-1,len(indices),lower.shape[1])
    counts,credits,concordance=replay_intervals(lower,upper,opened,support,predictions)
    metric_check,replayed=verify_metrics(block,methods,fractions,oof["molecule_names"],oof["fold_ids"],counts,credits,concordance)
    plans,training,known,exact=split_audit(splits,oof,lower,upper,opened)
    fit_check,selection_rows,fallback_rows=fit_and_selection_audit(block,plans,known,exact,counts,oof,info)
    training_frame=pd.DataFrame(training);curves=[];contrasts=[];query_records=[]
    contrast_specs=[]
    for fi,fraction in enumerate(fractions):
        chemistry=methods.index("chemistry")*len(fractions)+fi
        for mi,method in enumerate(methods):
            if method!="chemistry":contrast_specs.append((method,"chemistry",float(fraction),mi*len(fractions)+fi,chemistry))
        for candidate in ("representation_pca4_readout","representation_readout","representation_plus_chemistry"):
            contrast_specs.append((candidate,"scalar_members_readout",float(fraction),methods.index(candidate)*len(fractions)+fi,
                                   methods.index("scalar_members_readout")*len(fractions)+fi))
    bootstrap_equal={}
    for si in (0,1):
        valid=counts[:,si]>0;query_indices=np.flatnonzero(valid);last_boot=None
        for unit in ("molecule","scaffold"):
            groups=oof["molecule_names"][valid] if unit=="molecule" else oof["scaffold_groups"][valid]
            boot,pool_boot=bootstrap_curves(concordance[:,valid,si],credits[:,valid,si],counts[valid,si],groups,
                                             bootstrap_replicates,bootstrap_seed+si)
            if last_boot is not None:bootstrap_equal[STRATA[si]]=bool(np.array_equal(last_boot,boot))
            last_boot=boot
            for mi,method in enumerate(methods):
                for fi,fraction in enumerate(fractions):
                    vi=mi*len(fractions)+fi;lo,hi=np.quantile(boot[:,vi],[.025,.975]);plo,phi=np.quantile(pool_boot[:,vi],[.025,.975])
                    native_method=method.startswith("native__");tr=training_frame[training_frame.fraction.eq(fraction)]
                    row={"method":method,"fraction":float(fraction),"local_label_fraction":0. if native_method else float(fraction),
                         "kind":"static_native_reference" if native_method else "trained_oof","stratum":STRATA[si],"bootstrap_unit":unit,
                         "queries":int(valid.sum()),"resampled_units":len(set(groups)),"pairs":int(counts[valid,si].sum()),
                         "macro_concordance":float(concordance[vi,valid,si].mean()),"macro_ci_low":float(lo),"macro_ci_high":float(hi),
                         "pooled_concordance":float(credits[vi,valid,si].sum()/counts[valid,si].sum()),"pooled_ci_low":float(plo),"pooled_ci_high":float(phi)}
                    for field in ("training_molecules","training_groups","known_training_labels","exact_training_labels","bounded_training_labels"):
                        for stat in ("min","median","max"):
                            row[field+"_"+stat]=0. if native_method else float(getattr(tr[field],stat)())
                    curves.append(row)
            for candidate,reference,fraction,ci,bi in contrast_specs:
                delta=boot[:,ci]-boot[:,bi];pdelta=pool_boot[:,ci]-pool_boot[:,bi]
                lo,hi=np.quantile(delta,[.025,.975]);plo,phi=np.quantile(pdelta,[.025,.975])
                contrasts.append({"candidate":candidate,"reference":reference,"fraction":fraction,"stratum":STRATA[si],"bootstrap_unit":unit,
                                  "queries":int(valid.sum()),"resampled_units":len(set(groups)),"pairs":int(counts[valid,si].sum()),
                                  "macro_difference":float((concordance[ci,valid,si]-concordance[bi,valid,si]).mean()),
                                  "macro_ci_low":float(lo),"macro_ci_high":float(hi),
                                  "pooled_difference":float((credits[ci,valid,si]-credits[bi,valid,si]).sum()/counts[valid,si].sum()),
                                  "pooled_ci_low":float(plo),"pooled_ci_high":float(phi)})
        for mi,method in enumerate(methods):
            for fi,fraction in enumerate(fractions):
                vi=mi*len(fractions)+fi
                for q in query_indices:
                    query_records.append({"method":method,"fraction":float(fraction),"stratum":STRATA[si],"query":str(oof["molecule_names"][q]),
                                          "scaffold_group":str(oof["scaffold_groups"][q]),"outer_fold":int(oof["fold_ids"][q]),
                                          "pairs":int(counts[q,si]),"credit":float(credits[vi,q,si]),"concordance":float(concordance[vi,q,si])})
    outputs={"learning_curves.csv":curves,"paired_contrasts.csv":contrasts,"training_label_counts.csv":training,
             "selected_ridge_verified.csv":selection_rows,"fallback_fits.csv":fallback_rows,"replayed_aggregate.csv":replayed,
             "replayed_query_concordance.csv":query_records}
    for name,rows in outputs.items():pd.DataFrame(rows).to_csv(output_root/name,index=False)
    summary={"status":"verified","run_id":run_id,"revision":revision,"new_fits":0,"new_retrieval":0,
             "cohort":{"molecules":len(indices),"scaffold_groups":len(set(oof["scaffold_groups"])),"constructs":lower.shape[1],
                        "distinct_source_sequences":len(set(source["sequence_groups"])),"known_labels":int(known.sum()),
                        "exact_labels":int(exact.sum()),"bounded_labels":int((known&~exact).sum()),"source_original_molecules":len(source["molecule_names"])},
             "metric_replay":metric_check,"fit_audit":fit_check,
             "split_audit":{"outer_folds":5,"budget_plans":len(plans),"inner_partitions":sum(len(p["inner"]) for p in plans.values()),
                            "outer_scaffold_overlap":0,"inner_scaffold_overlap":0,"all_budgets_nested":True,
                            "all_fit_training_label_counts_recomputed":True},
             "bootstrap":{"replicates":bootstrap_replicates,"seed":bootstrap_seed,"units":["molecule","scaffold"],
                          "interval_percentiles":[2.5,97.5],"all_methods_and_budgets_paired":True,
                          "molecule_scaffold_draws_identical":bootstrap_equal,
                          "scope":"Conditional on existing OOF fits and eligible queries; no refits, p-values or success thresholds."},
             "numerical_tolerance":TOL,"scaling_scope":"Fitted scaler/PCA states were not persisted. Optional source files are hashed; call-path inspection is a separate check.",
             "outputs":{name:{"rows":len(rows),"sha256":sha256(output_root/name)} for name,rows in outputs.items()},
             "log_evidence":[],"input_files":[],"source_snapshot_files":[],
             "elapsed_seconds":time.perf_counter()-start,
             "threadpools":[{k:v for k,v in x.items() if k in ("internal_api","num_threads","version")} for x in threadpool_info()]}
    if run_log:
        for number,line in enumerate(read_text(run_log).splitlines(),1):
            if line.startswith("COMPARISON_BLOCK "):
                entry=json.loads(line.split(" ",1)[1])
                if entry.get("dataset")=="DAVIS":summary["log_evidence"].append({"line":number,"status":entry["result"].get("status"),"fits":entry["result"].get("fits")})
        if not any(x["status"]=="complete" for x in summary["log_evidence"]):raise AssertionError("No completed DAVIS block in supplied log")
    source_files=[("prepared",prepared_root/"davis_inputs.npz",prepared_root),
                  ("prepared",prepared_root/"davis_molecules.csv",prepared_root),
                  *[("run",block/name,run_root) for name in
                    ("known_oof.npz","known_fits.jsonl","known_splits.json","known_selections.jsonl",
                     "known_query_metrics.csv","known_aggregate_metrics.csv","known_summary.json")]]
    for role,path,base in source_files:
        path=resolve_input(path)
        summary["input_files"].append({"root":role,"path":str(path.relative_to(base)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    if source_snapshot:
        for name in ("study/learning_known.py","study/features.py","study/models.py"):
            path=Path(source_snapshot)/name
            summary["source_snapshot_files"].append({"path":name,"sha256":sha256(path)})
    summary["script_sha256"]=sha256(__file__)
    dump(output_root/"audit_summary.json",summary)
    print("DAVIS_AUDIT_COMPLETE",json.dumps(clean({"cohort":summary["cohort"],"metric_replay":metric_check,
                                                  "fits":fit_check["fits"],"fallbacks":fit_check["fallbacks"],
                                                  "bootstrap":summary["bootstrap"],"elapsed_seconds":summary["elapsed_seconds"]})),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--prepared-root",type=Path,help="Shared prepared data; defaults to RUN_ROOT/prepared")
    p.add_argument("--run-log",type=Path);p.add_argument("--source-snapshot",type=Path)
    p.add_argument("--bootstrap-replicates",type=int,default=2000);p.add_argument("--bootstrap-seed",type=int,default=20261004)
    p.add_argument("--run-id");p.add_argument("--revision")
    args=p.parse_args()
    if args.bootstrap_replicates<1:p.error("bootstrap-replicates must be positive")
    with threadpool_limits(limits=1):audit(**vars(args))


if __name__=="__main__":main()
