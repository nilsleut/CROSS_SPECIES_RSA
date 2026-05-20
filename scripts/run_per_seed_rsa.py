#!/usr/bin/env python3
"""
run_per_seed_rsa.py — Compute macaque RSA for each available seed.

For each rule, loads weights for seeds 0-4 (skips missing) and computes
Spearman rho against the macaque neural RDMs (no bootstrap — one point
estimate per seed is sufficient; variability comes from comparing seeds).

Output: results/rsa_results_macaque_per_seed.json
  {
    "backprop": {
      "V1": {"seed_0": 0.046, "seed_1": 0.051, ..., "mean": 0.049, "std": 0.003},
      "V2": {...},
      "V4": {...},
      "IT": {...}
    },
    ...
  }

Also writes results/noise_ceilings_macaque.json:
  {"V1": [lo, hi], "V2": [lo, hi], "V4": [lo, hi], "IT": [lo, hi]}

Run after train_additional_seeds.py (or immediately with only seed 0).
Usage:
  python scripts/run_per_seed_rsa.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import yaml

from models import (
    load_model, extract_features, THINGS_TRANSFORM,
    LAYER_ROI_MAJAJHONG, LAYER_ROI_FREEMANZIEMBA,
    _DEFAULT_WEIGHTS_DIR, _WEIGHT_FILES,
    BP_CNN, FA_CNN, PC_CNN, STDP_CNN, Random_CNN,
)
from data_loader import (
    load_majajhong2015, load_freemanziemba2013, build_neural_rdm,
)
from rsa_engine import build_rdm, noise_ceiling as compute_noise_ceiling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("per_seed_rsa")

REPO_ROOT  = Path(__file__).parent.parent
SEEDS      = [42, 123, 456, 789, 1337]   # index 0-4
RULES      = ["backprop", "feedback_alignment", "predictive_coding", "stdp", "random"]
WEIGHT_NAMES = {
    "backprop":           "model_weights_backprop",
    "feedback_alignment": "model_weights_feedback_alignment",
    "predictive_coding":  "model_weights_predictive_coding",
    "stdp":               "model_weights_stdp",
    "random":             "model_weights_random_weights",
}


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config():
    with open(REPO_ROOT / "configs" / "experiment_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Weight loading (seed-aware) ────────────────────────────────────────────────

def load_model_for_seed(rule: str, seed_idx: int, weights_dir: Path):
    """
    Load model for a given rule and seed index.
      seed_idx=0  → model_weights_{rule}_seed0.pt if it exists (full state_dict, retrained),
                    otherwise model_weights_{rule}.pt (Paper 1 original, STDP has only 3 conv keys)
      seed_idx>0  → model_weights_{rule}_seed{N}.pt (trained by train_additional_seeds.py)
    Returns None if no weight file is found.
    """
    import torch

    stem = WEIGHT_NAMES[rule]
    if seed_idx == 0:
        # Prefer _seed0.pt (full state_dict retrain) over .pt (Paper 1 original)
        wf_full = weights_dir / f"{stem}_seed0.pt"
        wf_orig = weights_dir / f"{stem}.pt"
        wf = wf_full if wf_full.exists() else wf_orig
    else:
        wf = weights_dir / f"{stem}_seed{seed_idx}.pt"

    if not wf.exists():
        return None

    if rule == "stdp":
        model = STDP_CNN().to_device(torch.device("cpu"))
        d = torch.load(wf, map_location="cpu", weights_only=True)
        model.L1.conv.weight.data = d["conv1.weight"]
        model.L2.conv.weight.data = d["conv2.weight"]
        model.L3.conv.weight.data = d["conv3.weight"]
        if "fc1.weight" in d:
            # Full checkpoint (retrained with train_additional_seeds.py)
            model.bn1.weight.data = d["bn1.weight"]; model.bn1.bias.data = d["bn1.bias"]
            model.bn1.running_mean.copy_(d["bn1.running_mean"])
            model.bn1.running_var.copy_(d["bn1.running_var"])
            model.bn2.weight.data = d["bn2.weight"]; model.bn2.bias.data = d["bn2.bias"]
            model.bn2.running_mean.copy_(d["bn2.running_mean"])
            model.bn2.running_var.copy_(d["bn2.running_var"])
            model.bn3.weight.data = d["bn3.weight"]; model.bn3.bias.data = d["bn3.bias"]
            model.bn3.running_mean.copy_(d["bn3.running_mean"])
            model.bn3.running_var.copy_(d["bn3.running_var"])
            model.fc1.weight.data = d["fc1.weight"]; model.fc1.bias.data = d["fc1.bias"]
            model.fc2.weight.data = d["fc2.weight"]; model.fc2.bias.data = d["fc2.bias"]
        return model

    _MODEL_CLASSES = {
        "backprop": BP_CNN, "feedback_alignment": FA_CNN,
        "predictive_coding": PC_CNN, "random": Random_CNN,
    }
    cls = _MODEL_CLASSES[rule]
    model = cls()
    state = torch.load(wf, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── RSA point estimate ─────────────────────────────────────────────────────────

def _rho(feats: np.ndarray, neural_rdm: np.ndarray) -> float:
    model_rdm = build_rdm(feats)
    n = min(model_rdm.shape[0], neural_rdm.shape[0])
    mv = model_rdm[:n, :n][np.triu_indices(n, k=1)]
    nv = neural_rdm[:n, :n][np.triu_indices(n, k=1)]
    r, _ = spearmanr(mv, nv)
    return float(r)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config      = load_config()
    results_dir = REPO_ROOT / config.get("output", {}).get("results_dir", "results")
    weights_dir = Path(config.get("paper1_weights_dir", str(_DEFAULT_WEIGHTS_DIR)))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load neural data ────────────────────────────────────────────────────
    logger.info("Loading neural data...")

    logger.info("  MajajHong2015 (macaque V4/IT)...")
    mh = load_majajhong2015(access="public")
    for region in ["V4", "IT"]:
        mh[region]["rdm"] = build_neural_rdm(mh[region]["responses"])
    mh_stim_set = mh["stimulus_set"]

    logger.info("  FreemanZiemba2013 (macaque V1/V2)...")
    fz = load_freemanziemba2013(access="public")
    for region in ["V1", "V2"]:
        if region in fz:
            fz[region]["rdm"] = build_neural_rdm(fz[region]["responses"])

    # ── 2. Compute noise ceilings ──────────────────────────────────────────────
    logger.info("\nComputing noise ceilings...")
    noise_ceilings: dict[str, list] = {}
    for region in ["V4", "IT"]:
        lo, hi = compute_noise_ceiling(mh[region]["responses"])
        noise_ceilings[region] = [round(lo, 4), round(hi, 4)]
        logger.info(f"  Macaque {region}: NC=[{lo:.3f}, {hi:.3f}]")
    for region in ["V1", "V2"]:
        if region in fz:
            lo, hi = compute_noise_ceiling(fz[region]["responses"])
            noise_ceilings[region] = [round(lo, 4), round(hi, 4)]
            logger.info(f"  Macaque {region}: NC=[{lo:.3f}, {hi:.3f}]")

    nc_path = results_dir / "noise_ceilings_macaque.json"
    with open(nc_path, "w", encoding="utf-8") as f:
        json.dump(noise_ceilings, f, indent=2)
    logger.info(f"  Saved to {nc_path}")

    # ── 3. Resolve stimulus paths ──────────────────────────────────────────────
    logger.info("\nResolving stimulus paths...")
    import brainscore_vision
    mh_paths = [mh_stim_set.get_stimulus(img_id)
                for img_id in mh_stim_set["image_id"]]

    fz_stim_set = brainscore_vision.load_stimulus_set("FreemanZiemba2013.aperture-public")
    fz_paths = [fz_stim_set.get_stimulus(img_id)
                for img_id in fz_stim_set["image_id"]]

    logger.info(f"  MajajHong: {len(mh_paths)} images")
    logger.info(f"  FreemanZiemba: {len(fz_paths)} images")

    # ── 4. Per-seed RSA ────────────────────────────────────────────────────────
    logger.info("\nComputing per-seed RSA...")
    # Structure: {rule: {region: {seed_0: rho, seed_1: rho, ...}}}
    per_seed: dict = {rule: {r: {} for r in ["V1", "V2", "V4", "IT"]} for rule in RULES}

    for rule in RULES:
        logger.info(f"\n--- {rule.upper()} ---")
        for seed_idx in range(len(SEEDS)):
            model = load_model_for_seed(rule, seed_idx, weights_dir)
            if model is None:
                logger.info(f"  seed_{seed_idx}: weights not found — skipping")
                continue

            seed_key = f"seed_{seed_idx}"
            logger.info(f"  seed_{seed_idx} (seed={SEEDS[seed_idx]}):")

            # MajajHong → V4 (Conv2), IT (FC1)
            try:
                mh_feats = extract_features(model, mh_paths, THINGS_TRANSFORM)
                for layer, regions in LAYER_ROI_MAJAJHONG.items():
                    for region in regions:
                        neural_rdm = mh[region]["rdm"]
                        rho = _rho(mh_feats[layer], neural_rdm)
                        per_seed[rule][region][seed_key] = round(rho, 5)
                        logger.info(f"    Macaque {region} ({layer}): rho={rho:.4f}")
            except Exception as e:
                logger.error(f"    MajajHong extraction failed: {e}")

            # FreemanZiemba → V1, V2 (Conv1)
            try:
                fz_feats = extract_features(model, fz_paths, THINGS_TRANSFORM)
                for layer, regions in LAYER_ROI_FREEMANZIEMBA.items():
                    for region in regions:
                        if region not in fz:
                            continue
                        neural_rdm = fz[region]["rdm"]
                        rho = _rho(fz_feats[layer], neural_rdm)
                        per_seed[rule][region][seed_key] = round(rho, 5)
                        logger.info(f"    Macaque {region} ({layer}): rho={rho:.4f}")
            except Exception as e:
                logger.error(f"    FreemanZiemba extraction failed: {e}")

    # ── 5. Aggregate mean/std across seeds ────────────────────────────────────
    logger.info("\nAggregating across seeds...")
    for rule in RULES:
        for region in ["V1", "V2", "V4", "IT"]:
            seed_rhos = [v for v in per_seed[rule][region].values()
                         if isinstance(v, float) and not np.isnan(v)]
            if seed_rhos:
                per_seed[rule][region]["mean"] = round(float(np.mean(seed_rhos)), 5)
                per_seed[rule][region]["std"]  = round(float(np.std(seed_rhos, ddof=1)
                                                               if len(seed_rhos) > 1 else 0.0), 5)
                per_seed[rule][region]["n_seeds"] = len(seed_rhos)

    # ── 6. Save ────────────────────────────────────────────────────────────────
    out_path = results_dir / "rsa_results_macaque_per_seed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_seed, f, indent=2)
    logger.info(f"\nSaved per-seed RSA to {out_path}")

    # Print summary
    logger.info("\n--- Summary (mean ± std across seeds) ---")
    for region in ["V1", "V2", "V4", "IT"]:
        logger.info(f"\n  {region}:")
        for rule in RULES:
            d = per_seed[rule][region]
            if "mean" in d:
                n = d["n_seeds"]
                logger.info(f"    {rule:22s}: {d['mean']:.4f} ± {d['std']:.4f}  (n={n})")
            else:
                logger.info(f"    {rule:22s}: no data")


if __name__ == "__main__":
    main()
