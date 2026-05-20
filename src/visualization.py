"""
visualization.py — Publication figures for cross-species RSA paper.

Key figures:
1. Side-by-side RSA profiles (human vs. macaque) per learning rule
2. Cross-species ranking comparison (scatter + Kendall's τ)
3. V1 invariance comparison across species
4. Interaction effects heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Optional
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color scheme
RULE_COLORS = {
    "backprop": "#2196F3",
    "BP": "#2196F3",
    "feedback_alignment": "#FF9800",
    "FA": "#FF9800",
    "predictive_coding": "#4CAF50",
    "PC": "#4CAF50",
    "stdp": "#F44336",
    "STDP": "#F44336",
    "random": "#9E9E9E",
    "Random": "#9E9E9E",
}

# Distinct line styles so rules are distinguishable even when ρ values overlap
RULE_STYLES = {
    "backprop":           {"linestyle": "-",    "marker": "o"},
    "feedback_alignment": {"linestyle": "--",   "marker": "s"},
    "predictive_coding":  {"linestyle": "-.",   "marker": "^"},
    "stdp":               {"linestyle": ":",    "marker": "D"},
    "random":             {"linestyle": (0,(3,1,1,1)), "marker": "x"},
}

SPECIES_COLORS = {
    "human": "#1976D2",
    "macaque": "#E64A19",
}

RULE_ABBREVS = {
    "backprop": "BP",
    "feedback_alignment": "FA",
    "predictive_coding": "PC",
    "stdp": "STDP",
    "random": "Random",
}

# Canonical anatomical region order (early → late visual hierarchy)
REGION_ORDER_HUMAN   = ["V1", "V2", "V4", "LOC", "IT"]
REGION_ORDER_PRIMATE = ["V1", "V2", "V4", "IT"]
REGION_ORDER_PAIRS   = ["V1/V1", "V2/V2", "V4/V4", "IT/IT"]

# Human noise ceilings from Paper 1 (split-half reliability bounds).
# V4 was not reported in Paper 1; LOC bounds from Paper 1 Fig 3.
# These values do not change — hardcoded here rather than loaded from disk.
HUMAN_NOISE_CEILINGS: dict = {
    "V1":  [0.07, 0.11],
    "V2":  [0.05, 0.09],
    "LOC": [0.03, 0.06],
    "IT":  [0.04, 0.07],
}


def fig1_rsa_profiles(
    human_rsa: Dict,
    primate_rsa: Dict,
    region_order_human: List[str] = None,
    region_order_primate: List[str] = None,
    primate_per_seed: Optional[Dict] = None,
    primate_noise_ceilings: Optional[Dict] = None,
    save_path: str = "figures/fig1_rsa_profiles.pdf",
):
    """
    Figure 1: Side-by-side RSA profiles.

    Left panel:  Human (V1->IT) — bootstrap CI ribbons + Paper 1 noise ceiling band
                 (hardcoded in HUMAN_NOISE_CEILINGS: V1/V2/LOC/IT; V4 not available).
    Right panel: Macaque (V1->IT) — seed mean ± std error bars (if primate_per_seed
                 is provided), otherwise bootstrap CI ribbons; noise ceiling gray
                 band (if primate_noise_ceilings is provided).

    primate_per_seed: {rule: {region: {"mean": float, "std": float, ...}}}
      Produced by scripts/run_per_seed_rsa.py. When present, error bars on the
      macaque panel reflect seed-to-seed variability (training stochasticity)
      rather than within-seed bootstrap CIs.

    primate_noise_ceilings: {region: [lower, upper]}
      Produced by scripts/run_per_seed_rsa.py or step_download. Drawn as a gray
      hatched ribbon connecting noise ceiling bounds across regions (same style as
      Paper 1 Fig 3). Human noise ceilings are always drawn from HUMAN_NOISE_CEILINGS.
    """
    if region_order_human is None:
        region_order_human = REGION_ORDER_HUMAN
    if region_order_primate is None:
        region_order_primate = REGION_ORDER_PRIMATE

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    learning_rules = sorted(human_rsa.keys())

    for ax, species, rsa_data, regions, is_primate in [
        (axes[0], "Human (fMRI)",               human_rsa,   region_order_human,   False),
        (axes[1], "Macaque (electrophysiology)", primate_rsa, region_order_primate, True),
    ]:
        # Noise ceiling band — human panel uses Paper 1 hardcoded values,
        # macaque panel uses values passed in (from noise_ceilings_macaque.json).
        nc_source = HUMAN_NOISE_CEILINGS if not is_primate else (primate_noise_ceilings or {})
        if nc_source:
            nc_xs, nc_lo, nc_hi = [], [], []
            for xi, region in enumerate(regions):
                if region in nc_source:
                    nc_xs.append(xi)
                    nc_lo.append(nc_source[region][0])
                    nc_hi.append(nc_source[region][1])
            if nc_xs:
                ax.fill_between(nc_xs, nc_lo, nc_hi,
                                facecolor="gray", alpha=0.15,
                                hatch="///", edgecolor="gray",
                                linewidth=0, zorder=1,
                                label="Noise Ceiling")

        for rule in learning_rules:
            abbrev = RULE_ABBREVS.get(rule, rule)
            color  = RULE_COLORS.get(rule, RULE_COLORS.get(abbrev, "#000"))
            style  = RULE_STYLES.get(rule, {"linestyle": "-", "marker": "o"})

            xs, rhos, errs_lo, errs_hi = [], [], [], []
            for xi, region in enumerate(regions):
                if region not in rsa_data.get(rule, {}):
                    continue
                d = rsa_data[rule][region]

                # For macaque: prefer per-seed mean/std if available
                if is_primate and primate_per_seed:
                    ps = primate_per_seed.get(rule, {}).get(region, {})
                    if "mean" in ps:
                        rho  = ps["mean"]
                        half = ps.get("std", 0.0)
                        xs.append(xi)
                        rhos.append(rho)
                        errs_lo.append(rho - half)
                        errs_hi.append(rho + half)
                        continue

                # Fallback: bootstrap CI from rsa_results.json
                rho = d["rho"]
                xs.append(xi)
                rhos.append(rho)
                errs_lo.append(d.get("ci_lower", rho - 0.01))
                errs_hi.append(d.get("ci_upper", rho + 0.01))

            if not xs:
                continue

            ax.plot(xs, rhos, color=color, label=abbrev,
                    linewidth=2, markersize=6, zorder=3,
                    linestyle=style["linestyle"], marker=style["marker"])
            ax.fill_between(xs, errs_lo, errs_hi,
                            alpha=0.15, color=color, zorder=2)

        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_xlabel("Brain Region")
        ax.set_title(species, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("RSA (Spearman rho)")
    # Legend on macaque panel — include noise ceiling if drawn
    axes[1].legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Learning Rule x Brain Region x Species", fontweight="bold", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig2_ranking_comparison(
    ranking_results: Dict,
    save_path: str = "figures/fig2_ranking_comparison.pdf",
):
    """
    Figure 2: Cross-species ranking scatter plots.

    Each subplot = one region pair, in anatomical order.
    x = human rho, y = macaque rho, each point = one learning rule.
    Label offsets are adapted per point to avoid overlap.
    """
    # Sort panels anatomically
    ordered_pairs = [p for p in REGION_ORDER_PAIRS if p in ranking_results]
    extra_pairs   = [p for p in ranking_results if p not in ordered_pairs]
    region_pairs  = ordered_pairs + extra_pairs

    n_panels = len(region_pairs)
    if n_panels == 0:
        print(f"Skipped {save_path}: no ranking results")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Candidate label offsets — cycle through them to reduce overlap
    _offset_cycle = [(8, 5), (-45, 5), (8, -14), (-45, -14), (8, 18), (-45, 18)]

    for ax, region_pair in zip(axes, region_pairs):
        data    = ranking_results[region_pair]
        h_rhos  = data["human_rhos"]
        p_rhos  = data["primate_rhos"]
        rules   = sorted(h_rhos.keys())

        # Assign offsets: group rules by (rounded) position to avoid stack
        placed: list[tuple[float, float]] = []

        for idx, rule in enumerate(rules):
            abbrev = RULE_ABBREVS.get(rule, rule)
            color  = RULE_COLORS.get(rule, RULE_COLORS.get(abbrev, "#000"))
            hx, py = h_rhos[rule], p_rhos[rule]

            ax.scatter(hx, py, c=color, s=120,
                       zorder=5, edgecolors="white", linewidth=1.5)

            # Pick the offset that puts the label farthest from already-placed labels
            best_offset = _offset_cycle[0]
            best_dist   = -1.0
            for off in _offset_cycle:
                # convert data coords to display for placed comparison
                candidate = (hx + off[0] * 0.001, py + off[1] * 0.001)
                min_dist  = min(
                    ((candidate[0]-px)**2 + (candidate[1]-py)**2)**0.5
                    for (px, py) in placed
                ) if placed else 999.0
                if min_dist > best_dist:
                    best_dist   = min_dist
                    best_offset = off

            placed.append((hx + best_offset[0]*0.001, py + best_offset[1]*0.001))

            ax.annotate(abbrev, (hx, py),
                        textcoords="offset points", xytext=best_offset,
                        fontsize=9, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.4,
                                        lw=0.8) if best_offset != (8, 5) else None)

        # Identity line
        all_vals = list(h_rhos.values()) + list(p_rhos.values())
        pad = max((max(all_vals) - min(all_vals)) * 0.15, 0.02)
        lo, hi = min(all_vals) - pad, max(all_vals) + pad
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.4, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Human rho")
        ax.set_ylabel("Macaque rho")
        ax.set_title(region_pair, fontweight="bold")
        ax.set_aspect("equal")

        tau = data["kendall_tau"]
        p   = data["p_permutation"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.05, 0.95, f"tau = {tau:.2f} ({sig})",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.grid(alpha=0.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig3_v1_invariance(
    v1_results: Dict,
    save_path: str = "figures/fig3_v1_invariance.pdf",
):
    """
    Figure 3: V1 learning rule invariance across species.

    Grouped bar chart: each group = one learning rule, bars = species.
    Species distinguished by color (human=blue, macaque=orange) AND hatch pattern,
    so the figure is unambiguous in greyscale print.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    species_list = [s for s in ["human", "macaque"] if s in v1_results]
    if not species_list:
        print("No V1 data available for plotting")
        return

    # Use canonical rule order
    rule_order = ["backprop", "feedback_alignment", "predictive_coding", "stdp", "random"]
    rules = [r for r in rule_order if r in v1_results[species_list[0]]["rhos"]]
    n_rules   = len(rules)
    n_species = len(species_list)
    bar_width = 0.8 / n_species
    x = np.arange(n_rules)

    hatches = ["", "///"]  # human = solid, macaque = hatched

    for i, species in enumerate(species_list):
        rhos   = [v1_results[species]["rhos"].get(r, 0) for r in rules]
        offset = (i - (n_species - 1) / 2) * bar_width
        color  = SPECIES_COLORS[species]

        ax.bar(x + offset, rhos, bar_width,
               label=species.capitalize(),
               color=color, alpha=0.75,
               hatch=hatches[i],
               edgecolor="white", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([RULE_ABBREVS.get(r, r) for r in rules])
    ax.set_ylabel("RSA (Spearman rho) at V1")
    ax.set_title("V1 Alignment: Learning Rule Invariance Across Species",
                 fontweight="bold")
    ax.legend(title="Species")
    ax.grid(axis="y", alpha=0.2)

    for species in species_list:
        rho_range = v1_results[species]["rho_range"]
        ax.text(0.98, 0.95 - species_list.index(species) * 0.07,
                f"{species}: delta-rho = {rho_range:.3f}",
                transform=ax.transAxes, ha="right", fontsize=9,
                color=SPECIES_COLORS.get(species, "black"))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig4_interaction_heatmap(
    interaction_results: Dict,
    save_path: str = "figures/fig4_interaction_effects.pdf",
):
    """
    Figure 4: Species x Learning Rule interaction effects.

    Heatmap: rows = learning rules, columns = region pairs in anatomical order.
    Values = interaction effect (delta-rho_human - delta-rho_macaque).
    """
    if not interaction_results:
        print("No interaction data available")
        return

    # Anatomical column order
    ordered_pairs = [p for p in REGION_ORDER_PAIRS if p in interaction_results]
    extra_pairs   = [p for p in interaction_results if p not in ordered_pairs]
    region_pairs  = ordered_pairs + extra_pairs

    all_rules = set()
    for rp in region_pairs:
        all_rules.update(interaction_results[rp].keys())

    rule_order = ["backprop", "feedback_alignment", "predictive_coding", "stdp", "random"]
    rules = [r for r in rule_order if r in all_rules]
    rules += [r for r in sorted(all_rules) if r not in rules]

    matrix = np.zeros((len(rules), len(region_pairs)))
    for j, rp in enumerate(region_pairs):
        for i, rule in enumerate(rules):
            if rule in interaction_results[rp]:
                matrix[i, j] = interaction_results[rp][rule]["interaction"]

    fig, ax = plt.subplots(figsize=(max(6, len(region_pairs) * 2),
                                    max(4, len(rules) * 0.8)))

    vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.1
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(region_pairs)))
    ax.set_xticklabels(region_pairs, rotation=45, ha="right")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([RULE_ABBREVS.get(r, r) for r in rules])

    for i in range(len(rules)):
        for j in range(len(region_pairs)):
            val   = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Interaction (delta-rho_human - delta-rho_macaque)")
    ax.set_title("Species x Learning Rule Interaction Effects", fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_all_figures(
    human_rsa: Dict,
    primate_rsa: Dict,
    ranking_results: Dict,
    v1_results: Dict,
    interaction_results: Dict,
    output_dir: str = "figures",
    primate_per_seed: Optional[Dict] = None,
    primate_noise_ceilings: Optional[Dict] = None,
):
    """Generate all publication figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig1_rsa_profiles(
        human_rsa, primate_rsa,
        primate_per_seed=primate_per_seed,
        primate_noise_ceilings=primate_noise_ceilings,
        save_path=f"{output_dir}/fig1_rsa_profiles.pdf",
    )
    fig2_ranking_comparison(ranking_results,
                            save_path=f"{output_dir}/fig2_ranking_comparison.pdf")
    fig3_v1_invariance(v1_results,
                       save_path=f"{output_dir}/fig3_v1_invariance.pdf")
    fig4_interaction_heatmap(interaction_results,
                             save_path=f"{output_dir}/fig4_interaction_effects.pdf")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    print("Visualization module loaded. Run scripts/05_generate_figures.py")
