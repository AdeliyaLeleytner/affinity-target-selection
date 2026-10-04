# Affinity scores for molecular target selection

Reproducible analyses for **Local labels and measurement resolution shape the value of affinity scores**.

The study asks when an interaction score adds useful information for ranking proteins for the same molecule. It compares native outputs, a local chemistry/protein reference, individual score augmentation and cached representation readouts. Source assay bounds, measurement-resolution strata, local training budgets and query weights remain explicit.

## Read the study

- `paper.pdf` and `paper.tex`: main article in the official Springer Nature author template, using the Nature reference style.
- `figs/`: vector figures, generating scripts, source-data pointers and the experimental 1M2Z structural illustration.
- `data/provenance/acquisition.md`: native acquisition protocols and the boundary between cached-analysis reproduction and inference replay.
- `data/provenance/data_sources.json`: source attribution, verified licenses, identifiers and checksums.

## Install

The recorded analysis environment uses Python 3.9, NumPy 1.26.4, pandas 2.3.1, SciPy 1.13.1, scikit-learn 1.6.1 and RDKit 2023.09.6. Git LFS is required for binary payloads in the Git repository. A downloaded anonymous archive must contain the actual LFS object bytes rather than pointer text; the restore and release-integrity checks verify this.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/restore_payloads.py
python scripts/verify_release.py
```

Large files are distributed as checksummed parts to respect the anonymous hosting service's file-size limit. Restoration reconstructs the original paths and verifies their complete SHA-256 hashes. It performs no network requests and refuses to overwrite a conflicting local file. Run the release-integrity check before editing or regenerating artifacts: it verifies every exported byte, whereas later figure/PDF regeneration can legitimately change formatting metadata.

## Verify without refitting

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/replay_results.py
```

The numerical replay checks raw-input checksums, reconstructs interval-defined pair order directly from saved measurements and predictions, verifies query credits and summary estimates, and checks successful fit diagnostics. It imports no fitting code and performs no model fitting or inference. Its report is `build/replay_verification.json`.

## Reproduce the experiments

Each command creates a timestamped directory under the specified output root. The original configurations and splits are released. Run blocks sequentially on memory-constrained machines.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m study.run --config configs/native.json --data-root data/raw --output-root build/recomputed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m study.run --config configs/label_value.json --data-root data/raw --output-root build/recomputed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m study.run --config configs/target_pairs.json --data-root data/raw --output-root build/recomputed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m study.run --config configs/score_components.json --data-root data/raw --output-root build/recomputed
```

The native block requires no fitting. The other blocks perform nested model selection and can take substantially longer; their fit-level wall times are included. The completed release combines successful comparison blocks from the original run with an objective-preserving numerical completion of the difficult DAVIS new-molecule block. `configs/davis_completion.json` reruns that block alone. The current solver includes that numerical fallback for the full experiment as well.

No GPU is needed to reproduce the released downstream analyses. Reacquiring native predictions is a separate workflow with model-specific input preparation and hardware requirements; the provenance document distinguishes the parts that can and cannot be exactly replayed from the available archive.

## Recompute uncertainty and decision summaries

These commands use the released out-of-fold arrays, perform no fitting, and regenerate the tables consumed by the figures. They preserve the recorded resampling units and seeds.

```sh
python scripts/analyze_decisions.py --input-root results/label_value --output-root results/decision_curves
python scripts/analyze_components.py --component-root results/score_components --reference-root results/label_value --prepared-root results/label_value/prepared --output-root results/score_components/audit
python scripts/analyze_davis_known.py --run-root results/label_value --prepared-root results/label_value/prepared --output-root results/davis_known_audit
python scripts/analyze_davis_transfer.py --result-root results/label_value --prepared-root results/label_value/prepared --output-root results/davis_cold_audit
python scripts/analyze_target_pairs.py --result-root results/target_pairs --prepared-root results/label_value/prepared --output-root results/target_pairs/audit
```

## Regenerate figures and PDF

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_manuscript.py
```

The build reads released result tables and generates vector PDF/SVG figures, then compiles the manuscript with pdfLaTeX. It needs TeX Live with the standard Springer Nature dependencies, including `sttools` and `threeparttable`. Original structural rendering is optional; the published image is included, together with its CIF coordinates, PyMOL script and isolated rendering environment specification in `figs/structure/`.

## Data layout and interpretation

`data/raw/` holds source measurements, explicit molecular/target axes and frozen native predictions. `results/native_baseline/` contains the source-derived native comparisons. `results/label_value/`, `results/target_pairs/` and `results/score_components/` hold the corresponding fitted predictions, splits, selected hyperparameters, query contributions and aggregate tables. `results/decision_curves/` records recovery at every panel testing budget, including support and uncertainty. `study/` is the fitting and metric implementation; `tests/` covers interval order, data separation and numerical correctness.

SPD measures protocol-specific binding-assay potency. Its representative and assay-envelope policies are separate analyses. DAVIS measures kinase affinity on assay constructs; the 47 full-identity-matched molecules define fitted comparisons, while native comparisons also report all 72 source molecules. A locally held-out molecule or protein need not be absent from a pretrained model's history.

Molecule-macro concordance weights evaluable molecules equally; pooled concordance weights their determined comparisons equally. Overlapping intervals are unresolved. Missing measurements are not inactive labels, and source ceilings are not imputed as exact values. Prediction ties receive half credit. Best-target recovery is reported only where the source intervals certify a maximum, with broader bounds for unresolved profiles. Twelve SPD outputs are not twelve independent algorithms.

The released SPD score-axis CSV retains the three identity columns actually used by this study, in their original row order. Its raw manifest records both the original source hash and the reduced-table hash. Recorded historical run-input manifests retain the original hash; the omitted columns do not enter any preparation, feature, fitting or evaluation function.
