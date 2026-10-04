# Data and native-score acquisition

The analysis uses quantitative measurements, their reported bounds, and cached native predictions. The SPD panel has **twelve output channels**, including related heads and alternative pose protocols; these are not twelve independent algorithms. `native_outputs.csv` identifies every channel, orientation and observed coverage. `data_sources.json` records input hashes, source references, model versions and acquisition settings.

## Measurements and identity joins

**SPD.** The source is [Sutherland et al. (2023)](https://doi.org/10.1038/s41467-023-40064-9), with data in [Zenodo record 8103950](https://doi.org/10.5281/zenodo.8103950). The publisher record explicitly licenses the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the original authors and identify the filtering, joins and transformations below when reusing the derived tables. Public file URLs and publisher checksums are retained in `data_sources.json`.

Rows are joined by full Standard InChIKey and the source assay-group mapping, retaining exact structure-match records for human binding assays. The 930-molecule score matrix uses this target order:

`KCNH2, ADRB1, SLC6A3, CHRM1, CHRM3, DRD2, SLC6A2, SLC6A4, NR3C1, HRH1`.

The join retains 7,463 assay records. Reported IC50 values in micromolar units become `pIC50 = 6 − log10(IC50_uM)`. Quantitative records remain points; `IC50 > c` becomes an open upper bound on pIC50. The source-representative policy uses the designated representative record. The separate assay-envelope policy spans all retained results for each molecule–target pair, preserving assay disagreement. These policies contain 6,959 and 7,142 observed cells, respectively. Unmeasured cells remain unknown; imported binary activity columns do not define these analyses.

**DAVIS.** Measurements are attributed to [Davis et al. (2011)](https://doi.org/10.1038/nbt.1990). The packaged table contains 72 molecules and 442 assay constructs, with 434 distinct recorded sequence strings. Construct identifiers are retained even when sequences coincide. Kd in nanomolar units becomes `pKd = 9 − log10(Kd_nM)`; the 22,400 cells at the source 10,000-nM reporting ceiling are represented by `pKd ≤ 5`. IC50 and Kd remain separate endpoints.

Prediction rows are joined by compound CID and assay-construct identifier. The explicit ligand map records 47 full Standard InChIKey matches, 21 connectivity-block matches and four named salt/formulation/structure-variant overrides. Native comparisons retain the full mapping and its full-InChIKey sensitivity subset separately. Fitted comparisons use the 47 full-InChIKey matches. The recorded PubChem request and cached-response hash are provided in `data_sources.json`.

## Reproducing the cached analyses

Use the repository requirements and the configuration accompanying the desired result:

```sh
python -m unittest discover -s tests -v
python -m study.run --data-root data/raw --output-root results/recomputed
```

The entry point checks input SHA-256 values before reconstructing observations and predictions. Evaluated methods share a finite-native-score mask. Native outputs are oriented so larger values indicate stronger predicted binding: continuous Boltz/Boltzina/Nesso outputs and docking energies are negated; binding probabilities and GNINA CNNaffinity retain their sign. This orientation does not convert native scores into measured pKd or pIC50.

Per-channel hashes in `native_outputs.csv` refer to independent NPY serialization. SPD arrays are 930×10 in the published molecule/target order. DAVIS arrays retain source prediction-row order; join their `cids` and `proteins` fields before constructing a grid. The stored DAVIS affinity ensemble is the mean of `aff1` and `aff2`; the binder ensemble is the mean of `prob1` and `prob2`. The cached representation concatenates `g_post1` and `g_post2`, each with 128 entries.

## Native acquisition recipes

- **Boltz-2 cofolding:** Boltz 2.2.1, PyTorch 2.5.1 and Lightning 2.5.0. Parent canonical SMILES undergo stock ligand preparation. Protein-only templates, eight pocket residues and cached homologous MSAs define the target inputs. Settings are seed 20260906, batches of 16, three recycling steps, 200 structure sampling steps, one structure sample, 200 affinity sampling steps, five affinity samples, `use_potentials`, no kernels and no affinity molecular-weight correction. Model seeding does not fully specify reference-conformer randomness.
- **Boltzina supplied-pose scoring:** Boltzina 1.0.1 with Boltz 2.2.1 and Torch 2.7.0+cu126. Saved Vina coordinates pass through the trunk and affinity heads, with diffusion disabled. Settings include seed 20260906, bf16-mixed precision, highest float32 matmul precision, no kernels and no affinity molecular-weight correction. Missing coordinates use either `(0,0,0)` or `(10000,10000,10000)` markers with their presence masks retained. The far variant is reused only when no unresolved atoms remain and the cropped inputs are identical.
- **Nesso-1:** Nesso 1.0.0, Torch 2.5.1+cu124, seed 20260906, bf16-mixed precision, batches of 128, one worker and no kernels. The model revision is `1896c84c7186c506c7efd79051480809d51098bf`; its ESM-2 dependency is separately pinned. Inputs comprise 918 prepared nominal-pH-7.4 ligand states, eight full canonical human protein sequences, NR3C1 residues 521–777, and four identical KCNH2 chains spanning residues 407–665.
- **GNINA fixed-pose rescoring:** `--score_only`, CNN rescore, two CPU threads, requested GPU device 0 and seed 20260906; the value is CNNaffinity at Vina pose rank 0. **GNINA own search** instead uses the recorded v1.3.3 binary, three CPU threads with GPU inference disabled, seed 20260906, exhaustiveness 32 and 50 modes, selecting CNNaffinity from the highest-CNNscore retained pose. CNNscore itself is a different, pose-confidence output.
- **Vina-GPU:** version 2.1 with a custom cached-input build, 8,000 threads, search depth 20, seed 20260905, five saved modes and default RILC. The selected value is the minimum reported Vina energy.
- **AD4:** AutoDock-GPU 1.6, 100 LGA runs, seed tuple 20260906/20260907/20260908, heuristics enabled, `heurmax=12000000`, autostop enabled, `lsmet=ad` and `ubmod=0`. The selected value is the minimum reported estimated binding free energy; XML and DLG records are joined by run identifier, retaining rounded-score ties.

Docking preparation uses Dimorphite-DL 2.0.2 nominal-pH-7.4 states, a documented tertiary-amide correction and parent graph/stereochemistry checks. The manifest contains one state for each of 918 prepared parents. Prepared receptor hashes and box coordinates are recorded separately. Sequence-based Nesso inputs, prepared rigid docking structures and Boltz template/MSA inputs are distinct acquisition protocols.

Cached ESM-2 features are also provided for protein baselines. DAVIS constructs are aligned by exact sequence SHA1. SPD features use the same full-sequence/domain regime described above, averaging residue embeddings without BOS/EOS tokens. Recorded model revisions, sequence hashes and dimensions are in `data_sources.json`.

## Technical limits

The cached numerical analysis is reproducible. Complete regeneration of every native inference job is not claimed: keyed exports or original export hashes verify the current score values, while some original raw predictions, prepared-input bundles and per-run records are unavailable. Checkpoint hashes described as recorded metadata have not been independently recomputed from checkpoint binaries.

The fixed-pose GNINA binary version, Vina custom-build commit/hash, and one of 9,180 finite Vina cells lacking the independent pose-table cross-check remain unresolved. DAVIS original inference versions, checkpoints and complete input settings are also unresolved; SPD Boltz runtime metadata must not be applied to that separate cache. The original DAVIS table download record and data license have not been established. Unverified data/model/software license fields remain explicitly null in `data_sources.json`.

Acquisition protocols differ in preparation, structure/sequence inputs and sampling. Their comparison is therefore not a pure architecture or pose-only intervention. Local train/test exclusion also does not establish exclusion from pretrained-model training data.
