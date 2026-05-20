#!/usr/bin/env python3
"""
stimulus_control_analysis.py — Stimulus-set control for cross-species RSA.

Are learning-rule RSA rankings stable across stimulus sets?  If yes, the
cross-species comparison reflects genuine model differences, not an artefact
of showing different images to humans and macaques.

For each region we compute:
  rho_things[rule]   — Spearman ρ of (model RDM on THINGS-720 images) vs
                        (THINGS-fMRI human neural RDM)
  rho_macaque[rule]  — Spearman ρ of (model RDM on macaque stimuli) vs
                        (macaque neural RDM)
                        • V4/IT: HVM object images (MajajHong2015)
                        • V1/V2: texture images (FreemanZiemba2013)

Then Kendall's τ between the two 5-element rho-vectors measures ranking
stability across stimulus sets.

Interpretation:
  V4/IT  — both use object stimuli → high τ expected if cross-species
            comparison is stimulus-independent.
  V1/V2  — texture vs object stimulus mismatch is expected to reduce τ;
            low τ here does NOT invalidate the V1 invariance finding but
            motivates treating V1/V2 with caution.

Output:
  results/stimulus_control.json
  figures/fig6_stimulus_control.pdf

Usage:
  python scripts/stimulus_control_analysis.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import yaml

from models import (
    load_model, extract_features, THINGS_TRANSFORM,
    _DEFAULT_WEIGHTS_DIR,
)
from data_loader import (
    load_majajhong2015, load_freemanziemba2013, load_things_fmri,
    build_neural_rdm,
)
from rsa_engine import build_rdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("stimulus_control")

REPO_ROOT = Path(__file__).parent.parent

RULES = ["backprop", "feedback_alignment", "predictive_coding", "stdp", "random"]
RULE_ABBREVS = {
    "backprop": "BP", "feedback_alignment": "FA",
    "predictive_coding": "PC", "stdp": "STDP", "random": "Random",
}
RULE_COLORS = {
    "backprop": "#2196F3", "feedback_alignment": "#FF9800",
    "predictive_coding": "#4CAF50", "stdp": "#F44336", "random": "#9E9E9E",
}
RULE_MARKERS = {
    "backprop": "o", "feedback_alignment": "s",
    "predictive_coding": "^", "stdp": "D", "random": "x",
}

# (layer_key, human_region, macaque_region, macaque_stimulus_set)
REGION_SPECS: dict[str, tuple] = {
    "V1": ("Conv1", "V1", "V1", "freemanziemba"),
    "V2": ("Conv1", "V2", "V2", "freemanziemba"),
    "V4": ("Conv2", "V4", "V4", "majajhong"),
    "IT": ("FC1",   "IT", "IT", "majajhong"),
}


# ── Config ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "experiment_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── RSA helpers ─────────────────────────────────────────────────────────────

def _rho(feats: np.ndarray, neural_rdm: np.ndarray) -> float:
    """Spearman ρ between model RDM (built from feats) and a neural RDM."""
    model_rdm = build_rdm(feats)
    n = min(model_rdm.shape[0], neural_rdm.shape[0])
    mv = model_rdm[:n, :n][np.triu_indices(n, k=1)]
    nv = neural_rdm[:n, :n][np.triu_indices(n, k=1)]
    r, _ = spearmanr(mv, nv)
    return float(r)


# ── Stimulus path resolution ─────────────────────────────────────────────────

def _resolve_things_paths(config: dict) -> list[str] | None:
    fmri_dir = config.get("things_fmri_dir", "")
    img_dir  = config.get("things_images_dir", "")
    if not fmri_dir or not img_dir:
        return None
    try:
        order_file = Path(fmri_dir) / "stim_order_sub-01.txt"
        with open(order_file, encoding="utf-8") as f:
            filenames = [ln.strip() for ln in f if ln.strip()]
        img_map = {p.name: str(p) for p in Path(img_dir).rglob("*.jpg")}
        resolved = [img_map[fn] for fn in filenames if fn in img_map]
        if len(resolved) != 720:
            logger.warning(f"  Expected 720 THINGS paths, got {len(resolved)}")
            return None
        return resolved
    except Exception as e:
        logger.warning(f"  THINGS path resolution failed: {e}")
        return None


def _resolve_mh_paths(mh: dict) -> list[str]:
    stim_set = mh["stimulus_set"]
    return [stim_set.get_stimulus(img_id) for img_id in stim_set["image_id"]]


def _resolve_fz_paths() -> list[str]:
    import brainscore_vision
    fz_stim = brainscore_vision.load_stimulus_set("FreemanZiemba2013.aperture-public")
    return [fz_stim.get_stimulus(img_id) for img_id in fz_stim["image_id"]]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config      = load_config()
    results_dir = REPO_ROOT / config.get("output", {}).get("results_dir", "results")
    figures_dir = REPO_ROOT / config.get("output", {}).get("figures_dir", "figures")
    weights_dir = Path(config.get("paper1_weights_dir", str(_DEFAULT_WEIGHTS_DIR)))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load neural data ────────────────────────────────────────────────
    logger.info("Loading neural data...")

    logger.info("  MajajHong2015...")
    mh = load_majajhong2015(access="public")
    for region in ["V4", "IT"]:
        mh[region]["rdm"] = build_neural_rdm(mh[region]["responses"])

    logger.info("  FreemanZiemba2013...")
    fz = load_freemanziemba2013(access="public")
    for region in ["V1", "V2"]:
        if region in fz:
            fz[region]["rdm"] = build_neural_rdm(fz[region]["responses"])

    logger.info("  THINGS-fMRI (human)...")
    things_human: dict = {}
    things_fmri_dir = config.get("things_fmri_dir", "")
    if things_fmri_dir:
        try:
            things_human = load_things_fmri(things_fmri_dir)
        except Exception as e:
            logger.error(f"  THINGS-fMRI load failed: {e}")

    if not things_human:
        logger.error("THINGS-fMRI data unavailable — cannot run stimulus control analysis")
        return

    # ── 2. Resolve stimulus paths ──────────────────────────────────────────
    logger.info("\nResolving stimulus paths...")

    things_paths = _resolve_things_paths(config)
    if things_paths is None:
        logger.error("Cannot resolve THINGS-720 paths (check things_images_dir in config)")
        return
    logger.info(f"  THINGS-720: {len(things_paths)} images")

    mh_paths = _resolve_mh_paths(mh)
    logger.info(f"  HVM (MajajHong): {len(mh_paths)} images")

    fz_paths = _resolve_fz_paths()
    logger.info(f"  FreemanZiemba: {len(fz_paths)} images")

    # ── 3. Per-rule feature extraction + RSA ──────────────────────────────
    logger.info("\nExtracting features and computing RSA...")

    # rho_things[rule][region], rho_macaque[rule][region]
    rho_things:  dict = {rule: {} for rule in RULES}
    rho_macaque: dict = {rule: {} for rule in RULES}

    for rule in RULES:
        logger.info(f"\n--- {rule.upper()} ---")
        model = load_model(rule, weights_dir=weights_dir)

        # THINGS-720 → human neural RDMs
        try:
            logger.info("  THINGS-720 features...")
            t_feats = extract_features(model, things_paths, THINGS_TRANSFORM)
            for region, (layer, h_region, _, _) in REGION_SPECS.items():
                if h_region not in things_human:
                    continue
                neural_rdm = things_human[h_region]["rdm"]
                rho = _rho(t_feats[layer], neural_rdm)
                rho_things[rule][region] = round(rho, 5)
                logger.info(f"    THINGS {region} ({layer}): ρ={rho:.4f}")
        except Exception as e:
            logger.error(f"  THINGS extraction failed: {e}")

        # HVM (MajajHong) → macaque V4/IT neural RDMs
        try:
            logger.info("  HVM features...")
            mh_feats = extract_features(model, mh_paths, THINGS_TRANSFORM)
            for region in ["V4", "IT"]:
                layer, _, p_region, _ = REGION_SPECS[region]
                if p_region not in mh:
                    continue
                rho = _rho(mh_feats[layer], mh[p_region]["rdm"])
                rho_macaque[rule][region] = round(rho, 5)
                logger.info(f"    HVM {region} ({layer}): ρ={rho:.4f}")
        except Exception as e:
            logger.error(f"  HVM extraction failed: {e}")

        # FreemanZiemba → macaque V1/V2 neural RDMs
        try:
            logger.info("  FreemanZiemba features...")
            fz_feats = extract_features(model, fz_paths, THINGS_TRANSFORM)
            for region in ["V1", "V2"]:
                layer, _, p_region, _ = REGION_SPECS[region]
                if p_region not in fz:
                    continue
                rho = _rho(fz_feats[layer], fz[p_region]["rdm"])
                rho_macaque[rule][region] = round(rho, 5)
                logger.info(f"    FZ {region} ({layer}): ρ={rho:.4f}")
        except Exception as e:
            logger.error(f"  FreemanZiemba extraction failed: {e}")

    # ── 4. Kendall's τ per region ──────────────────────────────────────────
    logger.info("\nKendall's τ (ranking stability across stimulus sets):")
    results: dict = {}

    for region, (layer, _, _, stim_label) in REGION_SPECS.items():
        pairs = [
            (rho_things[rule].get(region), rho_macaque[rule].get(region), rule)
            for rule in RULES
        ]
        valid = [(t, m, r) for t, m, r in pairs
                 if t is not None and m is not None
                 and not (np.isnan(t) or np.isnan(m))]

        if len(valid) < 3:
            logger.warning(f"  {region}: only {len(valid)} valid rules — skipping τ")
            continue

        t_vals = [v[0] for v in valid]
        m_vals = [v[1] for v in valid]
        tau, p_val = kendalltau(t_vals, m_vals)

        results[region] = {
            "layer":           layer,
            "macaque_stimuli": stim_label,
            "kendall_tau":     round(float(tau),   4),
            "p_value":         round(float(p_val), 4),
            "n_rules":         len(valid),
            "rho_things":      {rule: rho_things[rule].get(region)  for rule in RULES},
            "rho_macaque":     {rule: rho_macaque[rule].get(region) for rule in RULES},
        }

        sig = "  *" if p_val < 0.05 else ("  †" if p_val < 0.10 else "")
        logger.info(f"  {region:3s} ({stim_label:12s}): τ={tau:+.3f}, p={p_val:.3f}{sig}")

    # ── 5. Save results ────────────────────────────────────────────────────
    out_path = results_dir / "stimulus_control.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # ── 6. Generate figure ─────────────────────────────────────────────────
    _generate_fig6(results, str(figures_dir / "fig6_stimulus_control.pdf"))


# ── Figure ───────────────────────────────────────────────────────────────────

def _generate_fig6(results: dict, save_path: str):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    region_order = [r for r in ["V1", "V2", "V4", "IT"] if r in results]
    if not region_order:
        logger.warning("No data for fig6 — skipping")
        return

    n = len(region_order)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, region in zip(axes, region_order):
        d   = results[region]
        tau = d["kendall_tau"]
        p   = d["p_value"]

        xs, ys, colors, markers, abbrevs = [], [], [], [], []
        for rule in RULES:
            t_rho = d["rho_things"].get(rule)
            m_rho = d["rho_macaque"].get(rule)
            if t_rho is None or m_rho is None:
                continue
            xs.append(t_rho)
            ys.append(m_rho)
            colors.append(RULE_COLORS[rule])
            markers.append(RULE_MARKERS[rule])
            abbrevs.append(RULE_ABBREVS[rule])

        if not xs:
            continue

        # Individual scatter points (per-rule marker)
        for x, y, c, m, lbl in zip(xs, ys, colors, markers, abbrevs):
            ax.scatter(x, y, c=c, marker=m, s=90, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(lbl, (x, y), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=c)

        # Identity line (perfect stimulus-set invariance)
        all_vals = xs + ys
        lo = min(all_vals) - abs(min(all_vals)) * 0.25 - 0.005
        hi = max(all_vals) + abs(max(all_vals)) * 0.25 + 0.005
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
                linewidth=1, alpha=0.5, zorder=1)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="#ddd", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#ddd", linewidth=0.6, zorder=0)
        ax.grid(alpha=0.15)

        # Kendall's τ annotation
        sig_str = "*" if p < 0.05 else ("†" if p < 0.10 else "")
        ax.text(0.05, 0.97, f"τ = {tau:+.2f}{sig_str}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left")
        ax.text(0.05, 0.87, f"p = {p:.3f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                color="#555")

        # Note on V1/V2 stimulus mismatch
        if region in ("V1", "V2"):
            ax.text(0.05, 0.77, "⚠ textures vs objects",
                    transform=ax.transAxes, fontsize=7, va="top",
                    ha="left", color="#888", style="italic")

        macaque_label = ("HVM objects" if d["macaque_stimuli"] == "majajhong"
                         else "FZ textures")
        ax.set_title(f"{region}", fontweight="bold")
        ax.set_xlabel("ρ on THINGS-720\n(vs human fMRI)", labelpad=4)
        if region == region_order[0]:
            ax.set_ylabel(f"ρ on macaque stimuli\n(vs macaque ephys)")

        # Small subtitle with stimulus label
        ax.text(0.5, -0.18, macaque_label, transform=ax.transAxes,
                fontsize=8, ha="center", color="#666")

    # Global legend
    legend_handles = [
        mlines.Line2D([0], [0], marker=RULE_MARKERS[r], color="w",
                      markerfacecolor=RULE_COLORS[r],
                      markeredgecolor="white", markersize=8,
                      label=RULE_ABBREVS[r])
        for r in RULES
    ] + [
        mlines.Line2D([0], [0], color="gray", linestyle="--",
                      linewidth=1, label="y = x  (perfect stability)")
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.98), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Stimulus Control: Are Learning-Rule Rankings Stable Across Stimulus Sets?",
        fontweight="bold", fontsize=11, y=1.02,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
