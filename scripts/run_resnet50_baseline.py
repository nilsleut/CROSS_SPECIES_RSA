#!/usr/bin/env python3
"""
run_resnet50_baseline.py — ResNet-50 pretrained architecture baseline.

Robustness check: where does a large ImageNet-pretrained model land
relative to the custom 3-conv CNN with different learning rules?

No training. Loads torchvision.models.resnet50(weights='IMAGENET1K_V2').
Layer mapping mirrors the custom CNN:
  layer1 (GAP) -> V1, V2   (custom CNN: Conv1)
  layer2 (GAP) -> V4        (custom CNN: Conv2)
  layer4 (GAP) -> IT        (custom CNN: FC1)

RSA computed identically to the main pipeline (Spearman rho, bootstrap CIs,
max_boot_stimuli=500 for large RDMs).

Outputs:
  results/rsa_results_resnet50.json
  figures/fig5_architecture_comparison.pdf

Run AFTER the main pipeline (reads rsa_results.json for comparison figure).
Usage:
    python scripts/run_resnet50_baseline.py
"""

import sys
import json
import logging
import numpy as np
import yaml
from pathlib import Path

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_majajhong2015, load_freemanziemba2013, load_things_fmri, build_neural_rdm
from rsa_engine import build_rdm, compare_rdms_bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("resnet50")

REPO_ROOT = Path(__file__).parent.parent

# ImageNet normalization (NOT CIFAR-10)
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Layer -> cortical region mapping (mirrors custom CNN)
LAYER_ROI_MAJAJHONG = {"layer2": ["V4"], "layer4": ["IT"]}
LAYER_ROI_FREEMANZIEMBA = {"layer1": ["V1", "V2"]}
LAYER_ROI_THINGS = {"layer1": ["V1", "V2"], "layer2": ["V4"], "layer4": ["IT"]}

RULE_COLORS = {
    "backprop": "#2196F3", "feedback_alignment": "#FF9800",
    "predictive_coding": "#4CAF50", "stdp": "#F44336", "random": "#9E9E9E",
}
RULE_ABBREVS = {
    "backprop": "BP", "feedback_alignment": "FA",
    "predictive_coding": "PC", "stdp": "STDP", "random": "Random",
}
RULE_ORDER = ["backprop", "feedback_alignment", "predictive_coding", "stdp", "random"]
RESNET_COLOR = "#7B1FA2"
REGION_PAIRS = ["V1/V1", "V2/V2", "V4/V4", "IT/IT"]


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

def load_config():
    cfg_path = REPO_ROOT / "configs" / "experiment_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

class _ImgDS(Dataset):
    def __init__(self, paths, t):
        self.paths, self.t = list(paths), t
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.t(Image.open(str(self.paths[i])).convert("RGB")), i


def extract_resnet50_features(image_paths: list, batch_size: int = 64) -> dict:
    """
    Global-average-pool activations from layer1, layer2, layer4 of pretrained ResNet-50.

    Returns {"layer1": (n, 256), "layer2": (n, 512), "layer4": (n, 2048)}.
    """
    model = tv_models.resnet50(weights="IMAGENET1K_V2")
    model.eval()

    collected = {"layer1": [], "layer2": [], "layer4": []}
    current: dict = {}
    handles = []

    for name in ("layer1", "layer2", "layer4"):
        def _hook(module, inp, out, _n=name):
            current[_n] = out.mean(dim=[2, 3]).detach().cpu().numpy()
        handles.append(getattr(model, name).register_forward_hook(_hook))

    loader = DataLoader(_ImgDS(image_paths, IMAGENET_TRANSFORM),
                        batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for imgs, _ in loader:
            current.clear()
            model(imgs)
            for k in collected:
                collected[k].append(current[k])

    for h in handles:
        h.remove()

    return {k: np.concatenate(v) for k, v in collected.items()}


# ──────────────────────────────────────────────
# Stimulus path helpers
# ──────────────────────────────────────────────

def _resolve_things_paths(config: dict) -> list:
    fmri_dir = Path(config["things_fmri_dir"])
    img_dir = Path(config["things_images_dir"])
    order_file = fmri_dir / "stim_order_sub-01.txt"
    with open(order_file, encoding="utf-8") as f:
        filenames = [ln.strip() for ln in f if ln.strip()]
    img_map = {p.name: str(p) for p in img_dir.rglob("*.jpg")}
    paths = [img_map[fn] for fn in filenames if fn in img_map]
    if len(paths) != 720:
        logger.warning(f"  Only {len(paths)}/720 THINGS stimuli resolved")
    return paths


def _resolve_majajhong_paths() -> list:
    import brainscore_vision
    stim_set = brainscore_vision.load_stimulus_set("hvm-public")
    return [stim_set.get_stimulus(img_id) for img_id in stim_set["image_id"]]


def _resolve_freemanziemba_paths() -> list:
    import brainscore_vision
    stim_set = brainscore_vision.load_stimulus_set("FreemanZiemba2013.aperture-public")
    return [stim_set.get_stimulus(img_id) for img_id in stim_set["image_id"]]


# ──────────────────────────────────────────────
# RSA
# ──────────────────────────────────────────────

def _compute_rsa(feats: dict, layer: str, neural_rdm: np.ndarray, n_bootstrap: int) -> dict:
    acts = feats[layer]
    model_rdm = build_rdm(acts)
    n = min(model_rdm.shape[0], neural_rdm.shape[0])
    result = compare_rdms_bootstrap(model_rdm[:n, :n], neural_rdm[:n, :n],
                                    n_bootstrap=n_bootstrap)
    result["layer"] = layer
    return result


def compute_all_rsa(config: dict, neural_data: dict, all_feats: dict) -> dict:
    """Compute RSA for ResNet-50 across all stimulus sets and regions."""
    n_bootstrap = config["rsa"]["n_bootstrap"]
    rsa = {"human": {"resnet50": {}}, "macaque": {"resnet50": {}}}
    human = rsa["human"]["resnet50"]
    macaque = rsa["macaque"]["resnet50"]

    # Macaque V4 + IT: MajajHong
    if "majajhong2015" in all_feats:
        feats = all_feats["majajhong2015"]
        for layer, regions in LAYER_ROI_MAJAJHONG.items():
            if layer not in feats:
                continue
            for region in regions:
                rdm = neural_data.get("majajhong2015", {}).get(region, {}).get("rdm")
                if rdm is None:
                    continue
                result = _compute_rsa(feats, layer, rdm, n_bootstrap)
                macaque[region] = result
                ci = f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
                logger.info(f"  Macaque {region} ({layer}): rho = {result['rho']:.4f} {ci}")

    # Macaque V1 + V2: FreemanZiemba
    if "freemanziemba2013" in all_feats:
        feats = all_feats["freemanziemba2013"]
        for layer, regions in LAYER_ROI_FREEMANZIEMBA.items():
            if layer not in feats:
                continue
            for region in regions:
                rdm = neural_data.get("freemanziemba2013", {}).get(region, {}).get("rdm")
                if rdm is None:
                    continue
                result = _compute_rsa(feats, layer, rdm, n_bootstrap)
                macaque[region] = result
                ci = f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
                logger.info(f"  Macaque {region} ({layer}): rho = {result['rho']:.4f} {ci}")

    # Human V1, V2, V4, IT: THINGS-fMRI
    if "things_fmri" in all_feats:
        feats = all_feats["things_fmri"]
        for layer, regions in LAYER_ROI_THINGS.items():
            if layer not in feats:
                continue
            for region in regions:
                rdm = neural_data.get("things_fmri", {}).get(region, {}).get("rdm")
                if rdm is None:
                    continue
                result = _compute_rsa(feats, layer, rdm, n_bootstrap)
                human[region] = result
                ci = f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
                logger.info(f"  Human {region} ({layer}): rho = {result['rho']:.4f} {ci}")

    return rsa


# ──────────────────────────────────────────────
# Figure 5
# ──────────────────────────────────────────────

def _bar(ax, x, rho, ci, color, alpha, hatch=None, **kw):
    """Plot one bar with error bars; silently skips NaN."""
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return
    yerr = None
    if ci is not None:
        lo, hi = ci
        if not (np.isnan(lo) or np.isnan(hi)):
            yerr = [[max(0.0, rho - lo)], [max(0.0, hi - rho)]]
    ec = color if hatch else "none"
    lw = 0.5 if hatch else 0
    ax.bar(x, rho, color=color, alpha=alpha, hatch=hatch, edgecolor=ec,
           linewidth=lw, yerr=yerr, capsize=3, error_kw={"linewidth": 1}, **kw)


def fig5_architecture_comparison(
    cnn_rsa: dict,
    resnet_rsa: dict,
    per_seed_data: dict | None = None,
    save_path: str = "figures/fig5_architecture_comparison.pdf",
):
    """
    Figure 5: Custom 3-conv CNN (5 learning rules) vs ResNet-50 pretrained.

    One panel per region pair (V1/V1, V2/V2, V4/V4, IT/IT).
    Bars: human = solid, macaque = hatched.
    ResNet-50 is shown as a single pair of bars, separated by a dashed line.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    present_pairs = [p for p in REGION_PAIRS
                     if any(p.split("/")[0] in cnn_rsa.get(sp, {}).get(r, {})
                            for sp, r in [("human", "backprop"), ("macaque", "backprop")]
                            for _ in [None])
                     or p.split("/")[0] in resnet_rsa.get("human", {}).get("resnet50", {})]

    # Fall back to all four if detection is uncertain
    if not present_pairs:
        present_pairs = REGION_PAIRS

    n_panels = len(present_pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    bar_w = 0.35
    GAP_X = 0.9  # extra x-gap before ResNet-50 bar pair

    for ax, region_pair in zip(axes, present_pairs):
        h_region, p_region = region_pair.split("/")

        # Collect CNN data
        x_pos = 0
        xticks, xlabels = [], []

        for rule in RULE_ORDER:
            h_d = cnn_rsa.get("human", {}).get(rule, {}).get(h_region)
            p_d = cnn_rsa.get("macaque", {}).get(rule, {}).get(p_region)
            if h_d is None and p_d is None:
                continue

            color = RULE_COLORS.get(rule, "#555")
            h_rho = h_d["rho"] if h_d else float("nan")
            h_ci = (h_d.get("ci_lower", h_rho), h_d.get("ci_upper", h_rho)) if h_d else None

            # Macaque bar: prefer per-seed mean ± std if available
            if per_seed_data:
                ps = per_seed_data.get(rule, {}).get(p_region, {})
                if "mean" in ps:
                    p_rho = ps["mean"]
                    p_ci = (p_rho - ps.get("std", 0.0), p_rho + ps.get("std", 0.0))
                else:
                    p_rho = p_d["rho"] if p_d else float("nan")
                    p_ci = (p_d.get("ci_lower", p_rho), p_d.get("ci_upper", p_rho)) if p_d else None
            else:
                p_rho = p_d["rho"] if p_d else float("nan")
                p_ci = (p_d.get("ci_lower", p_rho), p_d.get("ci_upper", p_rho)) if p_d else None

            _bar(ax, x_pos - bar_w / 2, h_rho, h_ci, color, alpha=0.85)
            _bar(ax, x_pos + bar_w / 2, p_rho, p_ci, color, alpha=0.45, hatch="///")

            xticks.append(x_pos)
            xlabels.append(RULE_ABBREVS.get(rule, rule))
            x_pos += 1

        # Separator before ResNet-50
        sep_x = x_pos - 1 + GAP_X / 2 + 0.15
        ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.45, linewidth=1)

        # ResNet-50 bars
        rn_x = x_pos - 1 + GAP_X + 0.5
        rn_h = resnet_rsa.get("human", {}).get("resnet50", {}).get(h_region)
        rn_p = resnet_rsa.get("macaque", {}).get("resnet50", {}).get(p_region)

        rnh_rho = rn_h["rho"] if rn_h else float("nan")
        rnp_rho = rn_p["rho"] if rn_p else float("nan")
        rnh_ci = (rn_h.get("ci_lower", rnh_rho), rn_h.get("ci_upper", rnh_rho)) if rn_h else None
        rnp_ci = (rn_p.get("ci_lower", rnp_rho), rn_p.get("ci_upper", rnp_rho)) if rn_p else None

        _bar(ax, rn_x - bar_w / 2, rnh_rho, rnh_ci, RESNET_COLOR, alpha=0.85)
        _bar(ax, rn_x + bar_w / 2, rnp_rho, rnp_ci, RESNET_COLOR, alpha=0.45, hatch="///")

        xticks.append(rn_x)
        xlabels.append("RN-50")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
        ax.set_xlim(-0.7, rn_x + 0.7)
        ax.set_title(region_pair, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)
        ax.grid(axis="y", alpha=0.18)

        if region_pair == present_pairs[0]:
            ax.set_ylabel("RSA (Spearman ρ)")

    # Global legend
    legend_handles = [
        mpatches.Patch(facecolor="#888", alpha=0.85, label="Human (fMRI)"),
        mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///",
                       edgecolor="#888", label="Macaque (ephys)"),
        mpatches.Patch(facecolor=RESNET_COLOR, label="ResNet-50 (ImageNet)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.97), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Architecture Comparison: Custom 3-conv CNN (learning rules) vs ResNet-50 Pretrained",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    config = load_config()
    results_dir = REPO_ROOT / config.get("output", {}).get("results_dir", "results")
    figures_dir = REPO_ROOT / config.get("output", {}).get("figures_dir", "figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load neural data ──────────────────────────────────────
    logger.info("Loading neural data...")

    neural_data = {}

    logger.info("  MajajHong2015 (macaque V4/IT)...")
    try:
        mh = load_majajhong2015(access="public")
        for region in ["V4", "IT"]:
            mh[region]["rdm"] = build_neural_rdm(mh[region]["responses"])
        neural_data["majajhong2015"] = mh
    except Exception as e:
        logger.error(f"  MajajHong failed: {e}")

    logger.info("  FreemanZiemba2013 (macaque V1/V2)...")
    try:
        fz = load_freemanziemba2013(access="public")
        for region in ["V1", "V2"]:
            if region in fz:
                fz[region]["rdm"] = build_neural_rdm(fz[region]["responses"])
        neural_data["freemanziemba2013"] = fz
    except Exception as e:
        logger.error(f"  FreemanZiemba failed: {e}")

    logger.info("  THINGS-fMRI (human)...")
    try:
        things = load_things_fmri(config["things_fmri_dir"])
        neural_data["things_fmri"] = things
    except Exception as e:
        logger.error(f"  THINGS-fMRI failed: {e}")

    # ── 2. Extract ResNet-50 features ───────────────────────────
    logger.info("\nExtracting ResNet-50 features (ImageNet pretrained)...")
    all_feats = {}

    logger.info("  MajajHong2015 stimuli...")
    try:
        mh_paths = _resolve_majajhong_paths()
        all_feats["majajhong2015"] = extract_resnet50_features(mh_paths)
        for k, v in all_feats["majajhong2015"].items():
            logger.info(f"    {k}: {v.shape}")
    except Exception as e:
        logger.error(f"  MajajHong extraction failed: {e}")

    logger.info("  FreemanZiemba2013 stimuli...")
    try:
        fz_paths = _resolve_freemanziemba_paths()
        all_feats["freemanziemba2013"] = extract_resnet50_features(fz_paths)
        for k, v in all_feats["freemanziemba2013"].items():
            logger.info(f"    {k}: {v.shape}")
    except Exception as e:
        logger.error(f"  FreemanZiemba extraction failed: {e}")

    logger.info("  THINGS-720 stimuli...")
    try:
        things_paths = _resolve_things_paths(config)
        all_feats["things_fmri"] = extract_resnet50_features(things_paths)
        for k, v in all_feats["things_fmri"].items():
            logger.info(f"    {k}: {v.shape}")
    except Exception as e:
        logger.error(f"  THINGS-720 extraction failed: {e}")

    # ── 3. Compute RSA ───────────────────────────────────────────
    logger.info("\nComputing RSA...")
    resnet_rsa = compute_all_rsa(config, neural_data, all_feats)

    # ── 4. Save results ─────────────────────────────────────────
    save_path = results_dir / "rsa_results_resnet50.json"
    save_rsa = {}
    for species, rules in resnet_rsa.items():
        save_rsa[species] = {}
        for rule, regions in rules.items():
            save_rsa[species][rule] = {
                region: {k: v for k, v in res.items() if k != "bootstrap_distribution"}
                for region, res in regions.items()
            }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_rsa, f, indent=2, default=str)
    logger.info(f"\nSaved RSA results to {save_path}")

    # ── 5. Load CNN results for comparison ──────────────────────
    cnn_path = results_dir / "rsa_results.json"
    if cnn_path.exists():
        with open(cnn_path, encoding="utf-8") as f:
            cnn_rsa = json.load(f)
        logger.info(f"Loaded CNN results from {cnn_path}")
    else:
        logger.warning(f"rsa_results.json not found at {cnn_path} — "
                       "run the main pipeline first for the comparison figure.")
        cnn_rsa = {"human": {}, "macaque": {}}

    # ── 6. Load per-seed data for macaque error bars ────────────
    per_seed_data = None
    ps_path = results_dir / "rsa_results_macaque_per_seed.json"
    if ps_path.exists():
        with open(ps_path, encoding="utf-8") as f:
            per_seed_data = json.load(f)
        logger.info(f"Loaded per-seed RSA from {ps_path}")

    # ── 7. Generate figure ───────────────────────────────────────
    logger.info("\nGenerating fig5_architecture_comparison.pdf ...")
    fig5_architecture_comparison(
        cnn_rsa, resnet_rsa,
        per_seed_data=per_seed_data,
        save_path=str(figures_dir / "fig5_architecture_comparison.pdf"),
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
