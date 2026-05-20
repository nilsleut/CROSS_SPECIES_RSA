# CLAUDE.md — Cross-Species RSA Project

## What this project is

A NeuroAI research project comparing how well different learning rules (Backpropagation, Feedback Alignment, Predictive Coding, STDP, Random/untrained) predict neural representations across **species**: human fMRI vs. macaque electrophysiology. This is Paper 2 in a series; Paper 1 showed these learning rules against human fMRI alone.

**Core research question:** Does the learning rule ranking (BP > PC > FA > STDP ≈ Random at IT; all ≈ equal at V1) hold across species?

**Author:** Nils Leuthäuser, independent pre-university researcher (ETH Zürich BSc CS starting autumn 2027). GitHub: github.com/nilsleut. Based in Switzerland.

## Paper 1 context (the predecessor this builds on)

Paper 1 title: "Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI"

Key findings from Paper 1 (verified against rsa_results_cnn.csv, v6, 5 seeds × 3 subjects):
- Architecture (not learning) drives V1 alignment: Random ρ=0.075 > all trained models at V1
- V1 ranking: Random > STDP(0.064) > PC(0.056) >> BP(0.033) >> FA(0.012)
- IT ranking: PC(0.014) ≈ BP(0.013) > STDP(0.012) ≈ FA(0.012) > Random(0.008)
- Clean crossover: Random dominates early (V1/V2), learning rules dominate late (LOC/IT)
- Method: RSA (Representational Similarity Analysis) comparing model RDMs against THINGS-fMRI neural RDMs

**Paper 1 architecture:** Custom 3-conv CNN trained on CIFAR-10 (NOT ResNet-50). The v8 codebase defines this architecture.

Paper 1 code lives at `C:\Users\nilsl\Desktop\Projekte\learning-rules-rsa`. Key files:
- `learning_rules_v8.py` — main training/extraction pipeline (latest version, defines custom CNN + all learning rules)
- `outputs/model_weights_{rule}.pt` — saved weights per learning rule (seed 0), loadable directly
- `outputs_720/` — per-subject THINGS-fMRI RDMs (`fmri_rdm_{REGION}_sub-0N.npy`, 720×720)
- `outputs/rsa_results_cnn.csv` — per-rule per-layer RSA ρ values used in Paper 1

**Human RSA ρ values are filled in** at `run_pipeline.py → import_paper1_results()` from `rsa_results_cnn.csv` using the fixed-layer mapping (Conv1→V1/V2, Conv3→LOC, FC1→IT).

## Project structure

```
cross-species-rsa/
├── CLAUDE.md                          ← you are here
├── README.md                          ← project overview
├── requirements.txt                   ← pip dependencies
├── quickstart.py                      ← run FIRST: verifies data access
├── run_pipeline.py                    ← main 5-step pipeline with CLI
├── configs/
│   └── experiment_config.yaml         ← all hyperparameters (things_fmri_dir set here)
├── src/
│   ├── data_loader.py                 ← Brain-Score data loading (MajajHong, FreemanZiemba, THINGS)
│   ├── models.py                      ← learning rule implementations + feature extraction
│   ├── rsa_engine.py                  ← RDM construction, RSA comparison, bootstrap CIs
│   ├── cross_species_analysis.py      ← ranking conservation test, V1 invariance, interaction effects
│   └── visualization.py              ← 4 publication figures
├── results/                           ← output JSONs + CSVs
└── figures/                           ← output PDFs

# External data (not in repo)
C:\Users\nilsl\Desktop\Projekte\learning-rules-rsa\outputs_720\   ← Paper 1 THINGS-fMRI RDMs
    fmri_rdm_{V1,V2,V3,V4,LOC,IT}_sub-0{1,2,3}.npy               ← 720×720, 3 subjects
C:\Users\nilsl\Desktop\Projekte\learning-rules-rsa\outputs\       ← Paper 1 saved weights
    model_weights_Backprop.pt, model_weights_Feedback Alignment.pt,
    model_weights_Predictive Coding.pt, model_weights_STDP.pt      ← seed 0 weights for custom CNN
C:\Users\nilsl\.brainio\                                           ← brainscore S3 cache (auto)
```

## Neural datasets

| Dataset | Species | Regions | Stimuli | Key dimensions | Brain-Score ID |
|---------|---------|---------|---------|---------------|----------------|
| MajajHong2015 | Macaque | V4, IT | HVM objects (64 obj, 8 categories) | V4: 88 neurons, IT: 168 neurons, 2560 presentations | `MajajHong2015.public` |
| FreemanZiemba2013 | Macaque | V1, V2 | Textures (NOT objects) | Multi-unit recordings | `FreemanZiemba2013.public` |
| THINGS-fMRI | Human | V1, V2, V4, LOC, IT | THINGS objects | fMRI voxels, 3 subjects, 720 stimuli | Paper 1 outputs_720/ |

**Critical note:** FreemanZiemba uses **texture stimuli**, not objects. This means V1/V2 cross-species comparison is about texture processing, not object recognition. The paper should note this as a limitation. For the V4/IT comparison (MajajHong), stimuli are objects — directly comparable methodology.

MajajHong + FreemanZiemba load from S3 via `brainscore-vision`. Requires internet; no API key needed for public data. Files cached locally at `~/.brainio/`.

## Models

### CRITICAL: Architecture is custom 3-conv CNN, NOT ResNet-50

Paper 1 (v8) uses a custom 3-layer CNN trained on CIFAR-10. The architecture is defined in `learning_rules_v8.py`. The current `src/models.py` incorrectly uses ResNet-50 — this must be replaced with the custom CNN.

**Layer mapping (custom CNN → cortical regions):**
- Conv1 → V1, V2
- Conv3 → LOC (mid-level)
- FC1 → IT (high-level)

Five learning rules:
1. **Backprop (BP)** — standard SGD on CIFAR-10
2. **Feedback Alignment (FA)** — random fixed backward weights (Lillicrap et al. 2016)
3. **Predictive Coding (PC)** — iterative inference + weight update (Whittington & Bogacz 2017)
4. **STDP** — spike-timing dependent plasticity approximation
5. **Random** — untrained, random initialization (control)

**Saved weights exist** at `outputs/model_weights_{rule}.pt` (seed 0). Load these directly — do not retrain.

## Analysis pipeline

Run with: `python run_pipeline.py --step all` or step-by-step:

1. `--step download` — load MajajHong2015, FreemanZiemba2013, THINGS-fMRI; build neural RDMs; estimate noise ceilings
2. `--step extract` — load saved weights per rule into custom CNN; extract layer activations for macaque stimuli
3. `--step rsa` — build model RDMs, compute RSA (Spearman ρ) against neural RDMs, bootstrap 10K CIs
4. `--step analysis` — cross-species ranking comparison (Kendall's τ), V1 invariance test, interaction effects
5. `--step figures` — generate 4 publication figures

## Statistical framework

- **Within-species:** Spearman ρ between model and neural RDM upper triangles, bootstrap CIs (n=10000)
- **Cross-species ranking:** Kendall's τ on learning rule ρ-vectors, permutation test (n=10000)
- **V1 invariance:** ρ range across rules < 0.05 → "invariant"; pairwise bootstrap overlap tests
- **Interaction effects:** (Δρ_human - Δρ_macaque) where Δρ = rule_ρ - random_ρ

## Expected hypotheses and outcomes

- **H1 (most likely):** BP dominance at IT is conserved (τ significant, top-1 match). Story: "evolutionary conservation of learning rule sensitivity."
- **H2 (exciting):** Partial divergence, e.g. STDP matches macaque V4 better. Story: "species-specific plasticity regimes."
- **H3:** V1 invariance conserved in both species (Δρ < 0.05 in both).

## Known issues and TODOs

- [x] **Paper 1 ρ values:** verified against actual `rsa_results_cnn.csv` (2026-05-17)
- [x] **THINGS-fMRI loading:** handles both averaged and per-subject formats
- [x] **Figure bugs:** all 5 fixed (missing rules, label overlap, color encoding, region order)
- [x] **ResNet-50 replaced:** `src/models.py` now uses exact custom 3-conv CNN from v8 (C1=32, C2=64, C3=128, FC1=512). Weights loaded from `outputs/model_weights_{rule}.pt`. Layer mapping: Conv1→V1/V2, Conv2→V4, FC1→IT. Verified via log: Conv1=(n,32) shapes, distinct ρ per rule (2026-05-17).
- [x] **FreemanZiemba region bug fixed:** V1/V2 now correctly split via MultiIndex (102 vs 103 neurons, 135×135 RDM).
- [x] **Paper 1 consistency check:** `scripts/verify_paper1_consistency.py` confirmed all 20 ρ values within tolerance (δ ≤ 0.015). Pipeline consistent with Paper 1.
- [x] **V4/V4 cross-species region pair added:** Conv2→V4 mapping implemented end-to-end. THINGS-720 stimulus paths resolved once in step_extract; Conv2 features extracted per rule; human V4 RSA computed in step_rsa (Conv2 vs THINGS-fMRI V4 RDM); merge logic fixed to preserve computed V4. Config updated: `V4: "V4"` in region_mapping, `things_images_dir` added. visualization.py already had V4/V4 in REGION_ORDER_PAIRS.
- [x] **ResNet-50 pretrained baseline added:** `scripts/run_resnet50_baseline.py` — standalone script, no training. Uses `torchvision.models.resnet50(weights='IMAGENET1K_V2')` + ImageNet normalization. Layer mapping: layer1 (GAP)→V1/V2, layer2 (GAP)→V4, layer4 (GAP)→IT. Features extracted for MajajHong, FreemanZiemba, THINGS-720. RSA computed identically to main pipeline. Saves `results/rsa_results_resnet50.json` + `figures/fig5_architecture_comparison.pdf`. Run after main pipeline: `python scripts/run_resnet50_baseline.py`.
- [x] **Stimulus control analysis added:** `scripts/stimulus_control_analysis.py` — tests whether learning-rule RSA rankings are stable when applying the same model to THINGS-720 (human) vs macaque stimuli. Computes Kendall's τ between per-rule ρ vectors on both stimulus sets for each region. V4/IT use object stimuli on both sides; V1/V2 have a known texture-vs-object mismatch. Saves `results/stimulus_control.json` + `figures/fig6_stimulus_control.pdf`. Run standalone: `python scripts/stimulus_control_analysis.py`.

### Other TODOs

- [ ] **FreemanZiemba stimulus mismatch:** texture vs. object stimuli — handle in paper as limitation
- [ ] **`delta_rho_plot.png` from Paper 1** still has German axis labels needing English translation
- [ ] **Multi-subject analysis for THINGS-fMRI:** Paper 1 had Subject 03 sensitivity noted — propagate that caveat

## Windows / environment notes

Nils runs on Windows 11 + Python 3.11 (Microsoft Store). Several fixes were applied to make brainscore-vision work on Windows:

1. **brainscore_core path separator bug** — `BotoFetcher.__init__` used `os.path.join` to build S3 keys, producing backslashes on Windows → 404. Fixed in `brainscore_core/supported_data_standards/brainio/fetch.py` lines 90/93: replaced `os.path.join(*(split_path))` with `'/'.join(split_path)`.

2. **Spurious VersionId** — `version_id="null"` (string) was truthy, so boto3 received `ExtraArgs={"VersionId": "null"}`. Fixed by checking `version_id and version_id != "null"` before setting extra_args (same file, line 94).

3. **StimulusSet API change** — brainscore-vision dropped `get_image()`, replaced by `get_stimulus()`. Updated in `quickstart.py`, `src/data_loader.py`, and `run_pipeline.py`.

4. **File encoding** — every `open()` for reading or writing text must use `encoding="utf-8"` on Windows (cp1252 default can't handle `ρ`, `τ`, etc.). Fixed in `run_pipeline.py` (all JSON reads/writes) and `src/cross_species_analysis.py` (`generate_summary_report`). When adding new file I/O, always pass `encoding="utf-8"`.

5. **xarray FutureWarning** — `Score` class in `brainscore_core/metrics/__init__.py` needs `__slots__ = ()` added as first body line. Without it every import prints a FutureWarning.

If brainscore_core is updated/reinstalled, re-apply fixes #1, #2, and #5 to their respective files.

## Code style and conventions

- Python 3.11+ (Nils uses 3.11 on Windows, container uses 3.12)
- Type hints in function signatures
- Logging via `logging` module (not print statements in library code)
- Config via YAML (`configs/experiment_config.yaml`)
- Numpy for arrays, scipy for statistics, matplotlib for figures
- Brain-Score for data loading (`brainscore-vision` package)
- PyTorch for models

## Collaboration rules

- **Never run scripts autonomously.** If a script needs to be run (e.g. `python run_pipeline.py`), tell Nils what to run and wait for him to run it and paste the output. Do not use background Bash tasks for long-running pipeline steps.
- Short verification snippets (single `python -c "..."` lines to inspect data structures, check a coordinate name, etc.) are fine to run directly.

## How to help

When Nils asks for help, likely tasks include:
- **Debugging data loading** — S3 access, Brain-Score API, xarray assemblies
- **Extending the analysis** — new statistical tests, additional datasets, temporal analysis
- **Writing the paper** — LaTeX, two-column format, same style as Paper 1
- **Fixing edge cases** — RDM size mismatches between model and neural data, NaN handling
- **Training pipeline** — setting up Kaggle notebooks for full learning rule training
- **Visualization refinements** — publication-quality figures, consistent styling
- **Code review** — checking statistical validity, ensuring correct implementation of learning rules

## Key references

- Majaj et al. 2015 — "Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates..." (macaque V4/IT data)
- Freeman et al. 2013 — "A functional and perceptual signature of the second visual area in primates" (macaque V1/V2 data)
- Hebart et al. 2023 — THINGS-fMRI dataset
- Yamins & DiCarlo 2016 — layer-to-region mapping conventions
- Lillicrap et al. 2016 — Feedback Alignment
- Whittington & Bogacz 2017 — Predictive Coding
- Schrimpf et al. 2018/2020 — Brain-Score framework

## Contact context

Nils is pursuing arXiv endorsement from Martin Schrimpf (EPFL, Brain-Score founder) using endorsement code 7M8IWV, with his merged Brain-Score PR #2352 as credential. This project is strategically aligned with Schrimpf's ecosystem. Target: conferences (CCN, Cosyne poster), lab internships (ETH Grewe, Sacramento, Indiveri), and quant/ML industry roles.