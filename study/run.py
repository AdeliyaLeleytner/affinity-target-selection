"""Rebuild native-score comparisons from source measurements and predictions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .metrics import evaluate

GENES = ["KCNH2", "ADRB1", "SLC6A3", "CHRM1", "CHRM3", "DRD2",
         "SLC6A2", "SLC6A4", "NR3C1", "HRH1"]


def validate_inputs(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["files"]:
        relative = Path(item["file"]).relative_to("data/raw")
        data = (root / relative).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Input integrity mismatch: {relative}")
    return manifest


def davis(root, out, *, evaluate_native=True):
    table = pd.read_csv(root / "davis/measurements.tsv", sep="\t")
    mapping = pd.read_csv(root / "davis/ligand_mapping.csv", dtype={"pubchem_cid": str})
    if table.duplicated(["drug_name", "protein"]).any():
        raise ValueError("Duplicate DAVIS measurement key")
    molecules = table.drug_name.drop_duplicates().sort_values().tolist()
    targets = table.protein.drop_duplicates().sort_values().tolist()
    if set(molecules) != set(mapping.drug_name):
        raise ValueError("DAVIS name mapping is incomplete")
    mapped = mapping.set_index("drug_name").loc[molecules]
    mpos, tpos = {v: i for i, v in enumerate(molecules)}, {v: i for i, v in enumerate(targets)}
    cid_to_name = dict(zip(mapped.pubchem_cid, molecules))
    if len(cid_to_name) != len(molecules):
        raise ValueError("Non-bijective ligand/CID map")
    kd = table.pivot(index="drug_name", columns="protein", values="affinity").loc[molecules, targets].to_numpy()
    if not np.isfinite(kd).all() or np.any(kd <= 0) or np.any(kd > 10000):
        raise ValueError("Unexpected DAVIS measurement range")
    upper = 9 - np.log10(kd)
    lower = np.where(kd == 10000, -np.inf, upper)
    opened = np.zeros(kd.shape, bool)
    with np.load(root / "davis/affinity_out.npz", allow_pickle=False) as z:
        methods = ["boltz_affinity", "boltz_binder", "affinity_member1", "affinity_member2",
                   "binder_member1", "binder_member2"]
        rows = np.stack([-z["aff"], (z["prob1"] + z["prob2"]) / 2,
                         -z["aff1"], -z["aff2"], z["prob1"], z["prob2"]])
        scores = np.full((len(methods), len(molecules), len(targets)), np.nan)
        representations = np.full((len(molecules), len(targets), 256), np.nan, np.float32)
        cached_representations = np.concatenate([z["g_post1"], z["g_post2"]], axis=1)
        occupied = set()
        for k, (cid, target) in enumerate(zip(z["cids"], z["proteins"])):
            i, j = mpos[cid_to_name[str(cid)]], tpos[str(target)]
            if (i, j) in occupied:
                raise ValueError("Duplicate DAVIS prediction key")
            occupied.add((i, j))
            scores[:, i, j] = rows[:, k]
            representations[i, j] = cached_representations[k]
    if len(occupied) != kd.size or not np.isfinite(representations).all():
        raise ValueError("Incomplete prediction/representation grid")
    seq = table[["protein", "target_sequence"]].drop_duplicates()
    if seq.protein.duplicated().any():
        raise ValueError("One target has multiple source sequences")
    seq = seq.set_index("protein").loc[targets]
    sequence_groups = seq.groupby("target_sequence", sort=False).ngroup().to_numpy()
    duplicate_records = []
    for _, group in seq.reset_index().groupby("target_sequence"):
        names = group.protein.tolist()
        if len(names) > 1:
            vals = kd[:, [tpos[n] for n in names]]
            duplicate_records.append({"targets": names,
                                      "different_measurement_molecules": int(np.any(vals != vals[:, :1], axis=1).sum())})
    molecules_table = table[["drug_name", "compound_iso_smiles"]].drop_duplicates().set_index("drug_name").loc[molecules]
    if len(molecules_table) != len(molecules):
        raise ValueError("Ambiguous source ligand structure")
    molecules_table.join(mapped).to_csv(out / "davis_molecules.csv")
    seq.to_csv(out / "davis_targets.csv")
    np.savez_compressed(out / "davis_inputs.npz", methods=methods, scores=scores,
                        lower=lower, upper=upper, upper_open=opened,
                        representations=representations, sequence_groups=sequence_groups,
                        molecule_names=molecules, target_names=targets)
    records = []
    if evaluate_native:
        for scope, mask in [("all_source_molecules", np.ones(len(molecules), bool)),
                            ("exact_identity", mapped.match_method.to_numpy() == "exact_standard_inchikey")]:
            for axis in ["target", "ligand"]:
                records.extend(evaluate(scores[:, mask], lower[mask], upper[mask], opened[mask], methods,
                                        np.asarray(molecules)[mask] if axis == "target" else targets,
                                        axis, "DAVIS", scope))
    audit = {"molecules": len(molecules), "assay_constructs": len(targets),
             "unique_sequences": int(seq.target_sequence.nunique()), "measurement_cells": int(kd.size),
             "censored_at_source_ceiling": int((kd == 10000).sum()),
             "mapping_categories": mapped.match_method.value_counts().to_dict(),
             "duplicate_sequence_groups": duplicate_records,
             "interpretation": "Same sequence strings do not establish identical complete producer inputs; modifications and structural input provenance require inspection."}
    return records, audit


def spd(root, out, *, evaluate_native=True):
    raw = pd.read_parquet(root / "spd/measurements.parquet")
    panel = pd.read_csv(root / "spd/scored_molecules.csv")
    assays = pd.read_csv(root / "spd/assays.csv")
    # Imported binary activity and prior partition columns are intentionally not read.
    keys = panel.inchi_key.tolist()
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scored molecule")
    assay_selection = assays[assays.binding_event & assays.mapped_human_gene.isin(GENES) &
                             assays.assay_species.eq("Human (Homo sapiens)")]
    if len(assay_selection) != len(GENES) or assay_selection.mapped_human_gene.duplicated().any():
        raise ValueError("Ambiguous target-to-assay map")
    group_gene = dict(zip(assay_selection.assay_group, assay_selection.mapped_human_gene))
    rows = raw[raw.inchi_key.isin(keys) & raw.struct_match_type.eq("exact") &
               raw.assay_group.isin(group_gene)].copy()
    rows["gene"] = rows.assay_group.map(group_gene)
    v = pd.to_numeric(rows.ic50_uM, errors="raise")
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Invalid SPD IC50")
    prefix = rows["summarized prefix"].fillna("").str.strip()
    if not set(prefix).issubset({"", "=", ">"}):
        raise ValueError(f"Unhandled assay qualifiers: {set(prefix)}")
    rows["upper"] = 6 - np.log10(v)
    rows["lower"] = np.where(prefix.eq(">"), -np.inf, rows.upper)
    rows["upper_open"] = prefix.eq(">")
    mpos, tpos = {k: i for i, k in enumerate(keys)}, {g: j for j, g in enumerate(GENES)}
    with np.load(root / "spd/native_scores.npz", allow_pickle=False) as z:
        methods = list(z.files)
        scores = np.stack([z[k] for k in methods])
    if scores.shape[1:] != (len(keys), len(GENES)):
        raise ValueError("SPD cached-score axis mismatch")
    all_records = []
    for policy in ["source_representative", "assay_envelope"]:
        source = rows[rows.representative_result_drug_assay_group_pair] if policy == "source_representative" else rows
        lo = np.full((len(keys), len(GENES)), np.nan)
        hi, op = lo.copy(), np.zeros(lo.shape, bool)
        for (key, gene), group in source.groupby(["inchi_key", "gene"]):
            if policy == "source_representative" and len(group) != 1:
                raise ValueError("Multiple source representative results")
            i, j = mpos[key], tpos[gene]
            lo[i, j], hi[i, j] = group.lower.min(), group.upper.max()
            op[i, j] = bool(group.loc[group.upper.eq(hi[i, j]), "upper_open"].all())
        np.savez_compressed(out / f"spd_{policy}_inputs.npz", methods=methods, scores=scores,
                            lower=lo, upper=hi, upper_open=op, molecule_names=keys, target_names=GENES)
        if evaluate_native:
            for axis in ["target", "ligand"]:
                all_records.extend(evaluate(scores, lo, hi, op, methods, keys if axis == "target" else GENES,
                                            axis, "SPD", policy))
    panel[["inchi_key", "name", "canonical_smiles"]].to_csv(out / "spd_molecules.csv", index=False)
    assay_selection.to_csv(out / "spd_assays.csv", index=False)
    audit = {"source_assay_records": len(raw), "scored_molecules": len(keys),
             "targets": len(GENES), "matched_exact_identity_assay_records": len(rows),
             "source_representative_records": int(rows.representative_result_drug_assay_group_pair.sum()),
             "qualifiers": prefix.value_counts().to_dict(),
             "score_coverage": {m: int(np.isfinite(scores[j]).sum()) for j, m in enumerate(methods)}}
    return all_records, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("study_config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_root / config["stage"] / timestamp
    out.mkdir(parents=True, exist_ok=False)
    print("CONFIG", json.dumps(config), flush=True)
    manifest = validate_inputs(args.data_root)
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if config["stage"] == "label_value":
        from .learning_run import run
        return run(args, out, config, davis, spd)
    all_rows, audit = [], {}
    for name, function in [("davis", davis), ("spd", spd)]:
        records, audit[name] = function(args.data_root, out)
        all_rows.extend(records)
        print("DATA", name, json.dumps(audit[name]), flush=True)
    contributions = pd.DataFrame(all_rows)
    contributions.to_csv(out / "query_contributions.csv.gz", index=False)
    summary = []
    grouping = ["dataset", "policy", "axis", "stratum", "method"]
    for key, frame in contributions.groupby(grouping, sort=False):
        valid = frame[frame.pairs.gt(0)]
        summary.append(dict(zip(grouping, key), macro=valid.concordance.mean(),
                            pooled=valid.credit.sum() / valid.pairs.sum() if len(valid) else np.nan,
                            queries=len(valid), pairs=int(valid.pairs.sum()),
                            query_q10=valid.concordance.quantile(0.1),
                            query_q50=valid.concordance.quantile(0.5),
                            query_q90=valid.concordance.quantile(0.9)))
    frame = pd.DataFrame(summary)
    frame.to_csv(out / "native_summary.csv", index=False)
    audit.update(status="completed", elapsed_seconds=time.monotonic() - started,
                 new_fits=0, new_native_inference=0, paid_compute=0,
                 quantile_definition="Empirical query distribution; not confidence intervals")
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("SUMMARY", frame[(frame.axis == "target") & frame.method.isin(["boltz_affinity", "boltz_binder", "boltz_binary", "nesso_affinity", "nesso_binary"])].to_json(orient="records"), flush=True)
    print("ARTIFACT_DIRECTORY", str(out), flush=True)
    print("RUN_COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
