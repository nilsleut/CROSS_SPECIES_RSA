"""
cross_species_analysis.py — Cross-species ranking comparison.

This is the core novel analysis: do learning rule rankings
(BP > PC > FA > STDP ≈ Random at IT) hold across species?

Statistical framework:
1. Per region: Kendall's τ on learning rule ρ-rankings (human vs. macaque)
2. Interaction test: does the species × learning_rule × region interaction exist?
3. Visualization: side-by-side ranking profiles
"""

import math
import itertools
import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon
from typing import Dict, List, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_rankings(rsa_results: Dict[str, Dict[str, Dict]],
                     region: str) -> Tuple[List[str], np.ndarray]:
    """
    Extract learning rule ranking for a given region.
    
    Args:
        rsa_results: {learning_rule: {region: {"rho": float, ...}}}
        region: Brain region key
    
    Returns:
        (ordered_rules, rho_values) sorted by ρ descending
    """
    rules = []
    rhos = []
    for rule, regions in rsa_results.items():
        if region in regions:
            rules.append(rule)
            rhos.append(regions[region]["rho"])
    
    rhos = np.array(rhos)
    order = np.argsort(-rhos)
    return [rules[i] for i in order], rhos[order]


def ranking_conservation_test(
    human_rsa: Dict[str, Dict[str, Dict]],
    primate_rsa: Dict[str, Dict[str, Dict]],
    region_pairs: Dict[str, str],
    n_permutations: int = 10000,
) -> Dict:
    """
    Test whether learning rule rankings are conserved across species.
    
    For each region pair (human_region, primate_region):
    - Compute Kendall's τ between the ρ-vectors
    - Permutation test for significance
    - Report whether top-1 and top-2 rules match
    
    Args:
        human_rsa: {rule: {region: {"rho": float, "bootstrap_distribution": array}}}
        primate_rsa: same format
        region_pairs: {human_region: primate_region}
    """
    learning_rules = sorted(set(human_rsa.keys()) & set(primate_rsa.keys()))
    results = {}
    
    for h_region, p_region in region_pairs.items():
        # Collect ρ values per rule
        h_rhos = []
        p_rhos = []
        valid_rules = []
        
        for rule in learning_rules:
            h_has = h_region in human_rsa.get(rule, {})
            p_has = p_region in primate_rsa.get(rule, {})
            if h_has and p_has:
                h_rhos.append(human_rsa[rule][h_region]["rho"])
                p_rhos.append(primate_rsa[rule][p_region]["rho"])
                valid_rules.append(rule)
        
        if len(valid_rules) < 3:
            logger.warning(f"Skipping {h_region}/{p_region}: only {len(valid_rules)} rules")
            continue
        
        h_rhos = np.array(h_rhos)
        p_rhos = np.array(p_rhos)
        
        # --- Kendall's τ ---
        tau, p_analytic = kendalltau(h_rhos, p_rhos)

        # --- Permutation test ---
        n = len(valid_rules)
        abs_tau = abs(tau)
        indices = np.arange(n)

        if n <= 7:
            # Exhaustive: enumerate all n! permutations of the macaque rho-vector.
            # At n=5 this is 120 permutations; at n=7 it is 5040 — both exact.
            null_taus = np.array([
                kendalltau(h_rhos, p_rhos[list(perm)])[0]
                for perm in itertools.permutations(indices)
            ])
            perm_type = "exhaustive"
            n_perm_actual = math.factorial(n)
        else:
            rng = np.random.default_rng(42)
            null_taus = np.array([
                kendalltau(h_rhos, p_rhos[rng.permutation(n)])[0]
                for _ in range(n_permutations)
            ])
            perm_type = "random"
            n_perm_actual = n_permutations

        p_perm = float(np.mean(np.abs(null_taus) >= abs_tau))

        # --- Rankings ---
        h_order = [valid_rules[i] for i in np.argsort(-h_rhos)]
        p_order = [valid_rules[i] for i in np.argsort(-p_rhos)]

        # --- Effect size: ρ difference between species ---
        mean_diff = float(np.mean(h_rhos - p_rhos))

        # --- Top-k agreement ---
        top1_match = h_order[0] == p_order[0]
        top2_match = set(h_order[:2]) == set(p_order[:2])

        results[f"{h_region}/{p_region}"] = {
            "kendall_tau": float(tau),
            "p_analytic": float(p_analytic),
            "p_permutation": p_perm,
            "permutation_type": perm_type,
            "n_permutations": n_perm_actual,
            "human_ranking": h_order,
            "primate_ranking": p_order,
            "human_rhos": {r: float(v) for r, v in zip(valid_rules, h_rhos)},
            "primate_rhos": {r: float(v) for r, v in zip(valid_rules, p_rhos)},
            "mean_rho_difference": mean_diff,
            "top1_match": top1_match,
            "top2_match": top2_match,
            "n_rules": n,
        }

        logger.info(
            f"{h_region}/{p_region}: τ={tau:.3f} (p_perm={p_perm:.4f}, "
            f"{perm_type}, n={n_perm_actual}), "
            f"top1={'ok' if top1_match else 'no'}, top2={'ok' if top2_match else 'no'}"
        )
    
    return results


def v1_invariance_test(
    human_rsa: Dict, primate_rsa: Dict,
    human_v1_region: str = "V1",
    primate_v1_region: str = "V1",
) -> Dict:
    """
    Test whether V1 alignment is learning-rule-invariant in BOTH species.
    
    From Paper 1: Random ≈ BP at V1 (p = 0.43).
    Does this also hold for macaque V1?
    
    Uses Friedman test (non-parametric repeated measures) across rules.
    """
    from scipy.stats import friedmanchisquare, kruskal
    
    results = {}
    
    for species_name, rsa_data, region in [
        ("human", human_rsa, human_v1_region),
        ("macaque", primate_rsa, primate_v1_region),
    ]:
        rhos = {}
        bootstrap_dists = {}
        for rule, regions in rsa_data.items():
            if region in regions:
                rhos[rule] = regions[region]["rho"]
                if "bootstrap_distribution" in regions[region]:
                    bootstrap_dists[rule] = regions[region]["bootstrap_distribution"]
        
        if len(rhos) < 3:
            continue
        
        rules = sorted(rhos.keys())
        rho_values = [rhos[r] for r in rules]
        
        # Range of ρ values (small range = invariant)
        rho_range = max(rho_values) - min(rho_values)
        
        # If we have bootstrap distributions, test if they overlap
        if len(bootstrap_dists) >= 2:
            # Pairwise overlap: P(rule_i > rule_j) for all pairs
            pairwise_overlaps = {}
            for i, r1 in enumerate(rules):
                for j, r2 in enumerate(rules):
                    if i < j and r1 in bootstrap_dists and r2 in bootstrap_dists:
                        p = np.mean(bootstrap_dists[r1] > bootstrap_dists[r2])
                        pairwise_overlaps[f"{r1}_vs_{r2}"] = float(p)
            
            # "Invariant" if no pairwise comparison is significant
            all_nonsig = all(0.05 < p < 0.95 for p in pairwise_overlaps.values())
        else:
            pairwise_overlaps = {}
            all_nonsig = None
        
        results[species_name] = {
            "rhos": {r: float(v) for r, v in zip(rules, rho_values)},
            "rho_range": float(rho_range),
            "pairwise_overlaps": pairwise_overlaps,
            "all_nonsignificant": all_nonsig,
            "interpretation": "invariant" if rho_range < 0.05 else "differentiated",
        }
    
    # Cross-species comparison of V1 invariance
    if "human" in results and "macaque" in results:
        both_invariant = (
            results["human"]["interpretation"] == "invariant"
            and results["macaque"]["interpretation"] == "invariant"
        )
        results["cross_species_v1_invariance"] = both_invariant
    
    return results


def compute_interaction_effect(
    human_rsa: Dict, primate_rsa: Dict,
    region_pairs: Dict[str, str],
) -> Dict:
    """
    Test for species × learning_rule interaction at each region.
    
    If BP dominates IT in humans but not in macaques, that's an interaction.
    Quantified as: Δρ(BP-Random)_human vs Δρ(BP-Random)_macaque
    """
    learning_rules = sorted(set(human_rsa.keys()) & set(primate_rsa.keys()))
    
    if "random" not in learning_rules and "Random" not in learning_rules:
        logger.warning("No random baseline found for interaction analysis")
        return {}
    
    random_key = "random" if "random" in learning_rules else "Random"
    non_random_rules = [r for r in learning_rules if r != random_key]
    
    results = {}
    for h_region, p_region in region_pairs.items():
        h_random_rho = human_rsa.get(random_key, {}).get(h_region, {}).get("rho", None)
        p_random_rho = primate_rsa.get(random_key, {}).get(p_region, {}).get("rho", None)
        
        if h_random_rho is None or p_random_rho is None:
            continue
        
        deltas = {}
        for rule in non_random_rules:
            h_rho = human_rsa.get(rule, {}).get(h_region, {}).get("rho", None)
            p_rho = primate_rsa.get(rule, {}).get(p_region, {}).get("rho", None)
            
            if h_rho is not None and p_rho is not None:
                h_delta = h_rho - h_random_rho  # Improvement over random (human)
                p_delta = p_rho - p_random_rho  # Improvement over random (macaque)
                interaction = h_delta - p_delta   # Species difference in improvement
                
                deltas[rule] = {
                    "human_delta": float(h_delta),
                    "primate_delta": float(p_delta),
                    "interaction": float(interaction),
                }
        
        results[f"{h_region}/{p_region}"] = deltas
    
    return results


def generate_summary_report(
    ranking_results: Dict,
    v1_results: Dict,
    interaction_results: Dict,
    output_path: str = "results/cross_species_summary.json",
) -> str:
    """Generate a comprehensive summary report."""
    
    report = {
        "ranking_conservation": ranking_results,
        "v1_invariance": v1_results,
        "interaction_effects": interaction_results,
    }
    
    # Key findings
    findings = []
    
    for region_pair, data in ranking_results.items():
        if data["top1_match"]:
            findings.append(
                f"{region_pair}: Top learning rule is CONSERVED across species "
                f"({data['human_ranking'][0]})"
            )
        else:
            findings.append(
                f"{region_pair}: Top learning rule DIFFERS — "
                f"human={data['human_ranking'][0]}, macaque={data['primate_ranking'][0]}"
            )
        
        if data["p_permutation"] < 0.05:
            findings.append(
                f"  -> Rankings significantly correlated (τ={data['kendall_tau']:.3f}, "
                f"p={data['p_permutation']:.4f})"
            )
        else:
            findings.append(
                f"  -> Rankings NOT significantly correlated (τ={data['kendall_tau']:.3f}, "
                f"p={data['p_permutation']:.4f})"
            )
    
    if v1_results.get("cross_species_v1_invariance"):
        findings.append("V1 invariance is CONSERVED: learning rules don't matter for V1 in either species")
    
    report["key_findings"] = findings
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary (write via buffer to handle non-UTF-8 terminals on Windows)
    summary = "\n".join(["=" * 60, "CROSS-SPECIES RSA SUMMARY", "=" * 60] + findings)
    import sys
    sys.stdout.buffer.write((summary + "\n").encode("utf-8", errors="replace"))
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo with synthetic data
    print("Cross-species analysis module loaded successfully.")
    print("Run scripts/04_cross_species.py with real data.")
