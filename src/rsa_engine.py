"""
rsa_engine.py — Representational Similarity Analysis engine.

Core pipeline:
1. Extract model activations for stimuli
2. Build model RDMs (layer-wise)
3. Compare model RDMs against neural RDMs (Spearman ρ)
4. Bootstrap confidence intervals + statistical tests
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
# RDM Construction
# ============================================================

def build_rdm(activations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """
    Build RDM from activation matrix.
    
    Args:
        activations: (n_stimuli × n_features) — model or neural responses
        metric: 'correlation' (1 - Pearson r) or 'euclidean'
    
    Returns:
        rdm: (n_stimuli × n_stimuli) dissimilarity matrix
    """
    distances = pdist(activations, metric=metric)
    rdm = squareform(distances)
    rdm = np.nan_to_num(rdm, nan=0.0)
    return rdm


def rdm_upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangle of RDM as 1D vector (excluding diagonal)."""
    n = rdm.shape[0]
    indices = np.triu_indices(n, k=1)
    return rdm[indices]


# ============================================================
# RSA Comparison
# ============================================================

def compare_rdms(rdm_model: np.ndarray, rdm_neural: np.ndarray,
                 method: str = "spearman") -> Tuple[float, float]:
    """
    Compare two RDMs using rank correlation.
    
    Args:
        rdm_model: Model RDM (n × n)
        rdm_neural: Neural RDM (n × n)
        method: 'spearman' or 'kendall'
    
    Returns:
        (correlation, p_value)
    """
    # Ensure same size
    assert rdm_model.shape == rdm_neural.shape, \
        f"RDM shape mismatch: {rdm_model.shape} vs {rdm_neural.shape}"
    
    v_model = rdm_upper_triangle(rdm_model)
    v_neural = rdm_upper_triangle(rdm_neural)
    
    if method == "spearman":
        rho, p = spearmanr(v_model, v_neural)
    elif method == "kendall":
        rho, p = kendalltau(v_model, v_neural)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(rho), float(p)


def compare_rdms_bootstrap(rdm_model: np.ndarray, rdm_neural: np.ndarray,
                           n_bootstrap: int = 10000,
                           method: str = "spearman",
                           ci: float = 0.95,
                           max_boot_stimuli: int = 500) -> Dict:
    """
    RSA comparison with bootstrap confidence intervals.

    Point estimate uses all stimuli. Bootstrap CIs subsample to at most
    max_boot_stimuli per iteration so large RDMs (e.g. 3200×3200) remain
    tractable. Each bootstrap iteration draws max_boot_stimuli indices
    without replacement, giving a valid sampling-variability estimate.
    """
    n = rdm_model.shape[0]
    rho_observed, p_observed = compare_rdms(rdm_model, rdm_neural, method)

    rng = np.random.default_rng(42)
    rho_bootstrap = np.zeros(n_bootstrap)

    boot_n = min(n, max_boot_stimuli)
    replace = boot_n == n  # standard bootstrap only when not subsampling

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=boot_n, replace=replace)
        rdm_m_boot = rdm_model[np.ix_(idx, idx)]
        rdm_n_boot = rdm_neural[np.ix_(idx, idx)]
        rho_bootstrap[i], _ = compare_rdms(rdm_m_boot, rdm_n_boot, method)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(rho_bootstrap, alpha * 100)
    ci_upper = np.percentile(rho_bootstrap, (1 - alpha) * 100)
    se = np.std(rho_bootstrap)

    return {
        "rho": rho_observed,
        "p": p_observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_bootstrap": n_bootstrap,
        "bootstrap_distribution": rho_bootstrap,
    }


# ============================================================
# Pairwise comparisons between learning rules
# ============================================================

def compare_learning_rules(rsa_results: Dict[str, Dict],
                           region: str) -> Dict:
    """
    Statistical comparison of learning rules for a given region.
    
    Args:
        rsa_results: {learning_rule: {region: {rho, p, ci_lower, ci_upper, ...}}}
        region: Which brain region to compare
    
    Returns:
        Dict with pairwise comparisons, effect sizes, ranking
    """
    rules = sorted(rsa_results.keys())
    n_rules = len(rules)
    
    # Get ρ values and bootstrap distributions
    rhos = {r: rsa_results[r][region]["rho"] for r in rules}
    boots = {r: rsa_results[r][region].get("bootstrap_distribution", None) for r in rules}
    
    # Ranking
    ranking = sorted(rules, key=lambda r: rhos[r], reverse=True)
    
    # Pairwise differences (bootstrap)
    pairwise = {}
    for i in range(n_rules):
        for j in range(i + 1, n_rules):
            r1, r2 = rules[i], rules[j]
            diff_rho = rhos[r1] - rhos[r2]
            
            if boots[r1] is not None and boots[r2] is not None:
                diff_boot = boots[r1] - boots[r2]
                p_diff = np.mean(diff_boot <= 0)  # P(r1 <= r2)
                
                # Cohen's d from bootstrap
                d = diff_rho / np.std(diff_boot) if np.std(diff_boot) > 0 else 0
                
                pairwise[f"{r1}_vs_{r2}"] = {
                    "delta_rho": diff_rho,
                    "p": p_diff,
                    "cohens_d": d,
                }
            else:
                pairwise[f"{r1}_vs_{r2}"] = {"delta_rho": diff_rho}
    
    return {
        "rhos": rhos,
        "ranking": ranking,
        "pairwise": pairwise,
    }


# ============================================================
# Cross-species ranking comparison
# ============================================================

def compare_rankings_across_species(
    human_results: Dict[str, Dict],
    primate_results: Dict[str, Dict],
    region_mapping: Dict[str, str],
) -> Dict:
    """
    Compare learning rule rankings between species using Kendall's τ.
    
    Args:
        human_results: {learning_rule: {region: {rho, ...}}}
        primate_results: same format
        region_mapping: {human_region: primate_region}
    
    Returns:
        Per-region Kendall's τ + permutation p-value
    """
    results = {}
    learning_rules = sorted(set(human_results.keys()) & set(primate_results.keys()))
    
    for h_region, p_region in region_mapping.items():
        # Get ρ values per learning rule for each species
        h_rhos = []
        p_rhos = []
        valid_rules = []
        
        for rule in learning_rules:
            if h_region in human_results[rule] and p_region in primate_results[rule]:
                h_rhos.append(human_results[rule][h_region]["rho"])
                p_rhos.append(primate_results[rule][p_region]["rho"])
                valid_rules.append(rule)
        
        if len(valid_rules) < 3:
            logger.warning(f"Not enough rules for {h_region}/{p_region} comparison")
            continue
        
        h_rhos = np.array(h_rhos)
        p_rhos = np.array(p_rhos)
        
        # Kendall's τ on the rankings
        tau, p_tau = kendalltau(h_rhos, p_rhos)
        
        # Rank order
        h_ranking = [valid_rules[i] for i in np.argsort(-h_rhos)]
        p_ranking = [valid_rules[i] for i in np.argsort(-p_rhos)]
        
        # Permutation test
        n_perm = 10000
        rng = np.random.default_rng(42)
        tau_perm = np.zeros(n_perm)
        for i in range(n_perm):
            perm_idx = rng.permutation(len(valid_rules))
            tau_perm[i], _ = kendalltau(h_rhos, p_rhos[perm_idx])
        p_permutation = np.mean(np.abs(tau_perm) >= np.abs(tau))
        
        results[f"{h_region}/{p_region}"] = {
            "tau": float(tau),
            "p_tau": float(p_tau),
            "p_permutation": float(p_permutation),
            "human_ranking": h_ranking,
            "primate_ranking": p_ranking,
            "human_rhos": dict(zip(valid_rules, h_rhos.tolist())),
            "primate_rhos": dict(zip(valid_rules, p_rhos.tolist())),
            "rankings_match": h_ranking == p_ranking,
        }
    
    return results


# ============================================================
# Noise ceiling estimation
# ============================================================

def noise_ceiling(responses: np.ndarray, n_splits: int = 100,
                  metric: str = "correlation",
                  max_stimuli: int = 500) -> Tuple[float, float]:
    """
    Estimate noise ceiling via split-half reliability.

    Split neurons randomly into two halves, build RDMs from each,
    correlate. Repeat n_splits times. Returns (lower, upper) bound.

    Lower bound: mean correlation between splits
    Upper bound: Spearman-Brown corrected

    max_stimuli: subsample stimuli before building RDMs to keep memory
    usage tractable for large datasets (e.g. MajajHong IT, 3200 stimuli).
    NC measures neuron reliability so subsampling stimuli is valid.
    """
    n_stimuli, n_neurons = responses.shape
    rng = np.random.default_rng(42)

    if n_stimuli > max_stimuli:
        idx = rng.choice(n_stimuli, size=max_stimuli, replace=False)
        responses = responses[idx, :]
        n_stimuli = max_stimuli
    
    correlations = []
    for _ in range(n_splits):
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        
        resp_a = responses[:, perm[:half]]
        resp_b = responses[:, perm[half:2*half]]
        
        rdm_a = build_rdm(resp_a, metric=metric)
        rdm_b = build_rdm(resp_b, metric=metric)
        
        rho, _ = compare_rdms(rdm_a, rdm_b)
        correlations.append(rho)
    
    lower = np.mean(correlations)
    # Spearman-Brown prophecy formula
    upper = (2 * lower) / (1 + lower) if lower > 0 else lower
    
    return lower, upper


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n, d = 50, 100
    X = np.random.randn(n, d)
    Y = X + np.random.randn(n, d) * 0.5  # Correlated
    
    rdm_x = build_rdm(X)
    rdm_y = build_rdm(Y)
    
    result = compare_rdms_bootstrap(rdm_x, rdm_y, n_bootstrap=1000)
    print(f"RSA test: ρ = {result['rho']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    
    nc_lower, nc_upper = noise_ceiling(X)
    print(f"Noise ceiling: [{nc_lower:.3f}, {nc_upper:.3f}]")
