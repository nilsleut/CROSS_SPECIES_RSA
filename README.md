# Cross-Species RSA: Do Learning Rules Predict Primate and Human Neural Representations Equally?

**Nils Leuthäuser** | Independent Researcher | github.com/nilsleut

## Key Question

Does the learning rule → brain alignment ranking (BP > PC > FA > STDP ≈ Random at IT;
all ≈ equal at V1) hold across species? We test the same 5 learning rules against
both human fMRI (THINGS) and macaque electrophysiology (MajajHong2015, FreemanZiemba2013).

## Datasets

| Dataset | Species | Regions | Stimuli | Neurons/Voxels | Source |
|---------|---------|---------|---------|---------------|--------|
| THINGS-fMRI | Human | V1, V2, V4, LOC, IT | THINGS objects | fMRI voxels | Hebart et al. 2023 |
| MajajHong2015 | Macaque | V4, IT | HVM objects (64 obj, 8 cat) | 88 (V4), 168 (IT) | Majaj et al. 2015 |
| FreemanZiemba2013 | Macaque | V1, V2 | Textures | Multi-unit | Freeman et al. 2013 |

## Method

1. Train ResNet-50 with 5 learning rules: BP, FA, PC, STDP, Random (untrained)
2. Extract layer activations for each stimulus set
3. Build Representational Dissimilarity Matrices (RDMs) per layer
4. Compute RSA (Spearman ρ) between model RDMs and neural RDMs per region
5. Compare learning rule rankings across species using Kendall's τ

## Project Structure

```
cross-species-rsa/
├── src/
│   ├── data_loader.py          # Load Brain-Score + THINGS data
│   ├── models.py               # Learning rule implementations (from paper 1)
│   ├── rsa_engine.py           # RDM construction + RSA computation
│   ├── cross_species_analysis.py  # Cross-species ranking comparison
│   └── visualization.py        # Figures for paper
├── configs/
│   └── experiment_config.yaml  # All hyperparameters
├── scripts/
│   ├── 01_download_data.py     # Fetch all datasets
│   ├── 02_extract_features.py  # Run models on stimuli
│   ├── 03_compute_rsa.py       # RSA per species × region × learning rule
│   ├── 04_cross_species.py     # Ranking comparison + stats
│   └── 05_generate_figures.py  # Publication figures
├── results/                    # Output CSVs + stats
├── figures/                    # Output figures
└── requirements.txt
```

## Installation

```bash
pip install brainscore-vision torch torchvision scipy numpy matplotlib seaborn pyyaml
```

## Quick Start

```bash
python scripts/01_download_data.py   # Downloads ~2GB
python scripts/02_extract_features.py
python scripts/03_compute_rsa.py
python scripts/04_cross_species.py
python scripts/05_generate_figures.py
```
