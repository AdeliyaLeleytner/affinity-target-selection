"""Prepare aligned inputs and dispatch the frozen label-value comparison."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import molecule_features


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def aligned_esm(root, name, target_names):
    data = read_arrays(root / name / "esm_targets.npz")
    lookup = {str(value): i for i, value in enumerate(data["target_names"])}
    indices = np.array([lookup[str(value)] for value in target_names])
    if len(set(indices)) != len(indices):
        raise ValueError("Ambiguous ESM target mapping")
    return {key: value[indices] for key, value in data.items()}


def prepare(root, out, config, davis_prepare, spd_prepare):
    destination = out / "prepared"
    destination.mkdir()
    _, davis_audit = davis_prepare(root, destination, evaluate_native=False)
    _, spd_audit = spd_prepare(root, destination, evaluate_native=False)
    datasets = []
    for policy in config["spd_policies"]:
        arrays = read_arrays(destination / f"spd_{policy}_inputs.npz")
        arrays["methods"] = list(map(str, arrays["methods"]))
        molecules = pd.read_csv(destination / "spd_molecules.csv").set_index("inchi_key").loc[arrays["molecule_names"]]
        features = molecule_features(molecules.canonical_smiles.tolist())
        esm = aligned_esm(root, "spd", arrays["target_names"])
        datasets.append(dict(arrays, name="SPD", policy=policy,
                             molecule_features=features, sequence_sha1=esm["sequence_sha1"],
                             sequences=esm["sequences"], esm_embeddings=esm["embeddings"]))
    arrays = read_arrays(destination / "davis_inputs.npz")
    molecules = pd.read_csv(destination / "davis_molecules.csv").set_index("drug_name").loc[arrays["molecule_names"]]
    identity = molecules.match_method.to_numpy() == "exact_standard_inchikey"
    if config["davis_fitting_cohort"] != "exact_identity":
        raise ValueError("Unspecified DAVIS fitting cohort")
    esm = aligned_esm(root, "davis", arrays["target_names"])
    features = molecule_features(molecules.loc[identity, "compound_iso_smiles"].tolist())
    indices = [list(arrays["methods"]).index(name) for name in ["boltz_affinity", "boltz_binder"]]
    members = [list(arrays["methods"]).index(name) for name in
               ["affinity_member1", "affinity_member2", "binder_member1", "binder_member2"]]
    datasets.append(dict(
        name="DAVIS", policy="exact_identity", lower=arrays["lower"][identity],
        upper=arrays["upper"][identity], upper_open=arrays["upper_open"][identity],
        scores=arrays["scores"][indices][:, identity], methods=list(map(str, arrays["methods"][indices])),
        scalar_members=arrays["scores"][members][:, identity].transpose(1, 2, 0),
        representations=arrays["representations"][identity], molecule_names=arrays["molecule_names"][identity],
        target_names=arrays["target_names"], molecule_features=features,
        sequence_sha1=esm["sequence_sha1"], sequences=esm["sequences"], esm_embeddings=esm["embeddings"]))
    inventory = []
    for data in datasets:
        features = data["molecule_features"]
        label_known = ~np.isnan(data["lower"]) & ~np.isnan(data["upper"])
        exact = label_known & np.isfinite(data["lower"]) & (data["lower"] == data["upper"])
        item = {"name": data["name"], "policy": data["policy"],
                "molecules": len(data["molecule_names"]), "targets": len(data["target_names"]),
                "scaffold_groups": len(set(features.scaffold_groups)),
                "sequence_groups": len(set(data["sequence_sha1"])),
                "known_labels": int(label_known.sum()), "exact_labels": int(exact.sum()),
                "methods": list(map(str, data["methods"]))}
        inventory.append(item)
        np.savez_compressed(destination / f"{data['name']}_{data['policy']}_features.npz",
                            fingerprints=features.fingerprints, descriptors=features.descriptors,
                            identities=features.identities, scaffolds=features.scaffolds,
                            scaffold_groups=features.scaffold_groups,
                            sequence_sha1=data["sequence_sha1"], esm_embeddings=data["esm_embeddings"])
    (destination / "preparation_audit.json").write_text(json.dumps({
        "source": {"davis": davis_audit, "spd": spd_audit}, "datasets": inventory,
        "label_history": "No previous binary activity columns, selected thresholds or stored folds are used.",
        "davis_identity_scope": "Fitting comparisons use47 full-identity-matched molecules. Full72 native comparisons remain separately available; no quantitative-performance criterion selected this identity set."}, indent=2) + "\n")
    print("PREPARED_DATA", json.dumps(inventory), flush=True)
    return datasets


def execute_block(data, regime, destination, config):
    # Load numerical libraries before applying limits so late-loaded BLAS
    # implementations are included in the controller's library inventory.
    from .learning_known import run_known
    from .learning_cold import run_cold
    if config.get("arm_family") == "score_components" and regime != "new_molecule_known_targets":
        raise ValueError("Score-component arms are defined only for known-target comparisons")
    local = dict(config, pilot_affinity_method="boltz_affinity",
                 pilot_binder_method="boltz_binary" if data["name"] == "SPD" else "boltz_binder")
    with threadpool_limits(limits=config["numerical_threads"]):
        if regime == "new_molecule_known_targets":
            summary = run_known(data, destination, local)
        elif regime == "new_targets_known_molecules":
            if data["name"] == "SPD":
                local["fractions"] = [1.0]
            summary = run_cold(data, destination, local)
        else:
            raise ValueError(regime)
    return {"dataset": data["name"], "policy": data["policy"], "regime": regime, "result": summary}


def selected_blocks(datasets, config):
    """Apply an optional exact dataset/policy/regime allowlist before dispatch."""
    blocks = []
    for data in datasets:
        if config["mode"] == "technical_pilot" and data["name"] == "SPD" and data["policy"] != config["spd_policies"][0]:
            continue
        for regime in config["regimes"]:
            blocks.append((data, regime))
    requested = config.get("run_blocks")
    if requested is None:
        return blocks
    if (not isinstance(requested, list) or not requested
            or any(not isinstance(x, str) for x in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("run_blocks must be a nonempty list of distinct exact block IDs")
    identifiers = [f"{data['name']}/{data['policy']}/{regime}" for data, regime in blocks]
    unknown = set(requested) - set(identifiers)
    if unknown:
        raise ValueError(f"Unknown run_blocks IDs: {sorted(unknown)}")
    return [block for block, identifier in zip(blocks, identifiers) if identifier in set(requested)]


def run(args, out, config, davis_prepare, spd_prepare):
    started = time.perf_counter()
    summaries, tasks = [], []
    with threadpool_limits(limits=config["numerical_threads"]):
        datasets = prepare(args.data_root, out, config, davis_prepare, spd_prepare)
    for data, regime in selected_blocks(datasets, config):
        destination = out / data["name"] / data["policy"] / regime
        destination.mkdir(parents=True)
        tasks.append((data, regime, destination, config))
    workers = int(config.get("block_workers", 1))
    if workers == 1:
        for task in tasks:
            summaries.append(execute_block(*task))
            print("COMPARISON_BLOCK", json.dumps(summaries[-1]), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = [executor.submit(execute_block, *task) for task in tasks]
            for completed in as_completed(pending):
                summaries.append(completed.result())
                print("COMPARISON_BLOCK", json.dumps(summaries[-1]), flush=True)
    result = {"status": "technical_pilot_complete" if config["mode"] == "technical_pilot" else "completed",
              "elapsed_seconds": time.perf_counter() - started, "blocks": summaries,
              "run_blocks": config.get("run_blocks"),
              "new_native_inference": 0, "paid_compute": 0}
    (out / "learning_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ARTIFACT_DIRECTORY", str(out), flush=True)
    print("RUN_COMPLETE", json.dumps(result), flush=True)
    return result
