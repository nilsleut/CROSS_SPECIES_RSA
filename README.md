# Cross-Species RSA: Learning Rule Alignment Across Human fMRI and Macaque Electrophysiology

> ⚠️ **Correction (August 2026).** The predictive-coding and STDP results in this repository are affected by an evaluation-mode defect: both model classes overrode `eval()` with a no-op, so their batch-normalization layers stayed in training mode during feature extraction and normalized each evaluation batch by its own statistics, while the random, backpropagation and feedback-alignment conditions used their stored running statistics. A correction note identifying the affected results accompanies the current arXiv version of this paper; because STDP and PC lead at V1/V2 in the results below, the cross-species claims that rest on those two conditions should be read against it. The repaired re-run, and the resolution analysis that came out of it, are at **[nilsleut/evaluation-resolution-rsa](https://github.com/nilsleut/evaluation-resolution-rsa)** ([arXiv:2608.12408](https://arxiv.org/abs/2608.12408)).

**Nils Leutenegger** | Independent Researcher, Switzerland | [github.com/nilsleut](https://github.com/nilsleut)

**Companion study (Paper 1):** [arXiv:2604.16875](https://arxiv.org/abs/2604.16875) — *Untrained CNNs Match Backpropagation at V1*

---

## Summary

We test whether the relationship between learning rules and brain alignment generalizes across species. Using identical model weights from our human fMRI study, we evaluate five learning rules (BP, FA, PC, STDP, Random) against macaque electrophysiology from two Brain-Score datasets.

**Key findings:**
- Early visual alignment (V1/V2) is qualitatively conserved: STDP and PC consistently outperform BP in both species
- Higher-area alignment (IT) does not transfer, but this is partially confounded by stimulus domain differences and model capacity
- A pretrained ResNet-50 achieves ρ ≈ 0.23 at macaque IT — far above the custom CNN (ρ = 0.07–0.13) — confirming that IT alignment scales with capacity

## Results

### Learning rule profiles across the cortical hierarchy

<img width="1716" height="472" alt="image" src="https://github.com/user-attachments/assets/097669e6-394a-4916-bdab-abcb25e84a3d" />

**Left:** Human fMRI (THINGS, 720 stimuli, 3 subjects). **Right:** Macaque electrophysiology (MajajHong2015 + FreemanZiemba2013). Grey band: noise ceiling (split-half, Spearman–Brown corrected).

Macaque alignment is substantially higher than human (ρ = 0.15–0.30 vs. 0.01–0.08), reflecting the higher SNR of single-neuron recordings. In both species, STDP (ρ = 0.30 macaque, 0.064 human) and PC (ρ = 0.29 macaque, 0.056 human) lead at V1. All conditions show a sharp drop from V2 to V4, with partial recovery at IT.

### Cross-species ranking comparison

| Region | Kendall's τ | p (exhaustive) | Top-1 human | Top-1 macaque | Match? |
|--------|------------|----------------|-------------|---------------|--------|
| V1/V1  | +0.40      | 0.483          | Random      | STDP          | ✗      |
| V2/V2  | −0.20      | 0.817          | Random      | STDP          | ✗      |
| V4/V4  | +0.20      | 0.817          | STDP        | STDP          | ✓      |
| IT/IT  |  0.00      | 1.000          | PC          | STDP          | ✗      |

All p-values from exhaustive permutation over all 5! = 120 orderings. At n = 5 rules, only τ = ±1.0 can reach significance at α = 0.05 (p = 0.0083). Rankings are reported descriptively.

V4 is the only region where the top-ranked rule (STDP) matches across species. At V1, the qualitative pattern is partially conserved (STDP and PC rank high in both), but Random dominates human V1 while ranking third in macaque.

### Architecture comparison: Custom CNN vs. ResNet-50

<img width="1719" height="473" alt="image" src="https://github.com/user-attachments/assets/9f145947-576c-460c-a600-e946e85e79ed" />

| Region | Best CNN rule (macaque ρ) | ResNet-50 (macaque ρ) |
|--------|--------------------------|----------------------|
| V1     | STDP (0.305)             | 0.141                |
| V2     | STDP (0.268)             | 0.235                |
| V4     | STDP (0.062)             | 0.041                |
| IT     | STDP (0.134)             | **0.230**            |

At V1/V2, the custom CNN's best rules (STDP, PC) outperform ResNet-50 — local learning rules on a small architecture produce more V1-like representations than a large pretrained model. At IT, this reverses: ResNet-50 achieves ρ = 0.23, nearly double the best CNN condition. This confirms that the IT convergence observed in Paper 1 reflects model capacity, not a fundamental property of learning rules.

### Stimulus control

| Region | Stimulus type | τ (THINGS vs. macaque stimuli) | Interpretation |
|--------|--------------|-------------------------------|----------------|
| V1     | FZ textures  | +0.40                         | Moderate stability |
| V2     | FZ textures  | −0.20                         | Weak inversion |
| V4     | HVM objects  | +0.20                         | Weak stability |
| IT     | HVM objects  | −0.40                         | Weak inversion |

The IT inversion (τ = −0.40) suggests that cross-species ranking differences at IT may be partially driven by stimulus domain (THINGS vs. HVM objects), not only species. This confound is discussed as a limitation.

## Datasets

| Dataset | Species | Regions | Stimuli | Neurons | Source |
|---------|---------|---------|---------|---------|--------|
| THINGS-fMRI | Human | V1, V2, V4, LOC, IT | 720 objects | fMRI voxels, 3 subjects | Hebart et al. 2023 |
| MajajHong2015 | Macaque | V4, IT | 3,200 HVM objects | 88 (V4), 168 (IT) | Majaj et al. 2015 |
| FreemanZiemba2013 | Macaque | V1, V2 | 135 textures | 102 (V1), 103 (V2) | Freeman et al. 2013 |

## Method

1. Load saved weights from Paper 1 (custom 3-conv CNN trained on CIFAR-10, 5 seeds)
2. Extract layer activations for each stimulus set (224×224 input)
3. Build Representational Dissimilarity Matrices (correlation distance)
4. Compute RSA (Spearman ρ) between model and neural RDMs per region
5. Compare learning rule rankings across species (Kendall's τ, exhaustive permutation test over all 120 orderings)
6. Robustness check: pretrained ResNet-50 (ImageNet) as capacity control

## Architecture

**Custom CNN (Paper 1):** Conv1(32) → Conv2(64) → Conv3(128) → FC1(512) → FC2(10), trained on CIFAR-10.

**Layer mapping:** Conv1 → V1/V2, Conv2 → V4, FC1 → IT

**Capacity control:** ResNet-50 (ImageNet-1K-V2), layer1 → V1/V2, layer2 → V4, layer4 → IT

## Project Structure

```
cross-species-rsa/
├── run_pipeline.py                    # Main 5-step pipeline (download → extract → rsa → analysis → figures)
├── configs/
│   └── experiment_config.yaml
├── src/
│   ├── data_loader.py                 # Brain-Score + THINGS-fMRI loading
│   ├── models.py                      # Custom CNN + weight loading (ported from Paper 1 v8)
│   ├── rsa_engine.py                  # RDM construction, RSA, bootstrap CIs, noise ceilings
│   ├── cross_species_analysis.py      # Kendall's τ (exhaustive), V1 invariance, interaction effects
│   └── visualization.py              # Publication figures (fig1–fig4)
├── scripts/
│   ├── run_per_seed_rsa.py            # Per-seed macaque RSA (5 seeds)
│   ├── run_resnet50_baseline.py       # ResNet-50 capacity control (fig5)
│   ├── stimulus_control_analysis.py   # Stimulus-set stability analysis (fig6)
│   └── train_additional_seeds.py      # Retrain seeds 1–4 if needed
├── results/                           # JSON outputs
├── figures/                           # PDF figures (fig1–fig6)
└── requirements.txt
```

## Figures

| Figure | Description |
|--------|-------------|
| fig1 | RSA profiles across cortical hierarchy, human vs. macaque, with noise ceiling bands |
| fig2 | Cross-species ranking scatter plots (Kendall's τ per region) |
| fig3 | V1 learning-rule invariance comparison across species |
| fig4 | Species × learning rule interaction heatmap |
| fig5 | Architecture comparison: custom CNN vs. ResNet-50 pretrained |
| fig6 | Stimulus control: ranking stability across stimulus sets |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (overnight)
python scripts/run_per_seed_rsa.py && python run_pipeline.py --step all && python scripts/run_resnet50_baseline.py && python scripts/stimulus_control_analysis.py
```

## Requirements

- Python 3.11+
- PyTorch, torchvision
- brainscore-vision
- numpy, scipy, matplotlib, seaborn, pyyaml

Brain-Score data downloads automatically from S3 on first run (~2 GB, cached at `~/.brainio/`).
Paper 1 model weights expected at path configured in `experiment_config.yaml`.

## Noise Ceilings (Macaque)

| Region | Lower bound | Upper bound | Source |
|--------|------------|------------|--------|
| V1     | 0.360      | 0.529      | FreemanZiemba2013, split-half over neurons |
| V2     | 0.327      | 0.493      | FreemanZiemba2013, split-half over neurons |
| V4     | 0.604      | 0.753      | MajajHong2015, split-half over neurons |
| IT     | 0.669      | 0.802      | MajajHong2015, split-half over neurons |

All models fall well below these ceilings, indicating substantial room for improvement with larger architectures.

## Citation

```bibtex
@article{leutenegger2025crossspecies,
  title={Cross-Species RSA Reveals Conserved Early Visual Alignment but Divergent 
         Higher-Area Rankings Across Human fMRI and Macaque Electrophysiology},
  author={Leutenegger, Nils},
  journal={arXiv preprint},
  year={2025}
}
```

## Related

- [Paper 1: Untrained CNNs Match Backpropagation at V1](https://arxiv.org/abs/2604.16875) | [Code](https://github.com/nilsleut/learning-rules-rsa)
- [Brain-Score](https://www.brain-score.org/) — Neural benchmark platform
