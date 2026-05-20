#!/usr/bin/env python3
"""
run_pipeline.py — Complete cross-species RSA pipeline.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --step download    # Just download data
    python run_pipeline.py --step extract     # Just extract features
    python run_pipeline.py --step rsa         # Just compute RSA
    python run_pipeline.py --step analysis    # Just cross-species analysis
    python run_pipeline.py --step figures     # Just generate figures
"""

import argparse
import json
import logging
import sys
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import (
    load_majajhong2015, load_freemanziemba2013, load_things_fmri,
    build_neural_rdm
)
from models import (
    load_model, extract_features, THINGS_TRANSFORM,
    LAYER_ROI_FREEMANZIEMBA, LAYER_ROI_MAJAJHONG,
)
from rsa_engine import build_rdm, compare_rdms_bootstrap, noise_ceiling
from cross_species_analysis import (
    ranking_conservation_test, v1_invariance_test,
    compute_interaction_effect, generate_summary_report
)
from visualization import generate_all_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("pipeline")


def load_config(path: str = "configs/experiment_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# STEP 1: Download / load data
# ============================================================

def step_download(config: dict) -> dict:
    """Download and cache all neural datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading neural data")
    logger.info("=" * 60)
    
    data = {}
    
    # --- Macaque: MajajHong2015 (V4 + IT) ---
    logger.info("Loading MajajHong2015 (macaque V4/IT)...")
    try:
        mh = load_majajhong2015(access="public")
        data["majajhong2015"] = mh
        
        # Build neural RDMs
        for region in ["V4", "IT"]:
            rdm = build_neural_rdm(mh[region]["responses"])
            mh[region]["rdm"] = rdm
            logger.info(f"  {region} RDM: {rdm.shape}, ρ range: [{rdm.min():.3f}, {rdm.max():.3f}]")
            
            # Noise ceiling
            nc_lower, nc_upper = noise_ceiling(mh[region]["responses"])
            mh[region]["noise_ceiling"] = (nc_lower, nc_upper)
            logger.info(f"  {region} noise ceiling: [{nc_lower:.3f}, {nc_upper:.3f}]")
    except Exception as e:
        logger.error(f"Failed to load MajajHong2015: {e}")
        logger.info("Ensure brainscore-vision is installed and S3 access is available")
    
    # --- Macaque: FreemanZiemba2013 (V1 + V2) ---
    logger.info("Loading FreemanZiemba2013 (macaque V1/V2)...")
    try:
        fz = load_freemanziemba2013(access="public")
        data["freemanziemba2013"] = fz

        for region in ["V1", "V2"]:
            if region in fz:
                rdm = build_neural_rdm(fz[region]["responses"])
                fz[region]["rdm"] = rdm
                logger.info(f"  {region} RDM: {rdm.shape}")

                nc_lower, nc_upper = noise_ceiling(fz[region]["responses"])
                fz[region]["noise_ceiling"] = (nc_lower, nc_upper)
                logger.info(f"  {region} noise ceiling: [{nc_lower:.3f}, {nc_upper:.3f}]")
    except Exception as e:
        logger.error(f"Failed to load FreemanZiemba2013: {e}")
    
    # --- Human: THINGS-fMRI ---
    logger.info("Loading THINGS-fMRI (human, from Paper 1)...")
    things_dir = config.get("things_fmri_dir", "data/things_fmri")
    try:
        things = load_things_fmri(things_dir)
        data["things_fmri"] = things
    except Exception as e:
        logger.warning(f"THINGS-fMRI not found at {things_dir}: {e}")
        logger.info("Copy your Paper 1 THINGS data to data/things_fmri/")
    
    # Save metadata
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {}
    for dataset_name, dataset in data.items():
        meta[dataset_name] = {}
        if isinstance(dataset, dict):
            for region, rdata in dataset.items():
                if isinstance(rdata, dict) and "n_neurons" in rdata:
                    meta[dataset_name][region] = {
                        "n_neurons": rdata.get("n_neurons"),
                        "n_stimuli": rdata.get("n_stimuli"),
                        "noise_ceiling": rdata.get("noise_ceiling"),
                    }
    
    with open(results_dir / "data_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save noise ceilings for fig1 (consumed by step_figures even without run_per_seed_rsa.py)
    noise_ceilings: dict = {}
    for region in ["V4", "IT"]:
        nc = data.get("majajhong2015", {}).get(region, {}).get("noise_ceiling")
        if nc is not None:
            noise_ceilings[region] = [round(float(nc[0]), 4), round(float(nc[1]), 4)]
    for region in ["V1", "V2"]:
        nc = data.get("freemanziemba2013", {}).get(region, {}).get("noise_ceiling")
        if nc is not None:
            noise_ceilings[region] = [round(float(nc[0]), 4), round(float(nc[1]), 4)]
    if noise_ceilings:
        nc_path = results_dir / "noise_ceilings_macaque.json"
        with open(nc_path, "w", encoding="utf-8") as f:
            json.dump(noise_ceilings, f, indent=2)
        logger.info(f"  Saved noise ceilings to {nc_path.name}: {list(noise_ceilings.keys())}")

    return data


# ============================================================
# STEP 2: Extract model features
# ============================================================

def step_extract(config: dict, data: dict) -> dict:
    """Extract features from all learning rule models for all stimulus sets."""
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting model features")
    logger.info("=" * 60)

    import torch

    weights_dir = Path(config.get("paper1_weights_dir",
                                   r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs"))
    features = {}  # {rule: {dataset: {"Conv1": ndarray, "Conv2": ..., "Conv3": ..., "FC1": ...}}}

    # Resolve THINGS-720 stimulus paths once (used for human V4 extraction)
    things_image_paths = None
    things_fmri_dir_cfg = config.get("things_fmri_dir", "")
    things_images_dir_cfg = config.get("things_images_dir", "")
    if things_fmri_dir_cfg and things_images_dir_cfg:
        try:
            fmri_path = Path(things_fmri_dir_cfg)
            img_path = Path(things_images_dir_cfg)
            order_file = fmri_path / "stim_order_sub-01.txt"
            with open(order_file, encoding="utf-8") as f:
                filenames = [ln.strip() for ln in f if ln.strip()]
            img_map = {p.name: str(p) for p in img_path.rglob("*.jpg")}
            resolved = [img_map[fn] for fn in filenames if fn in img_map]
            if len(resolved) == 720:
                things_image_paths = resolved
                logger.info(f"  Resolved {len(things_image_paths)} THINGS-720 stimulus paths")
            else:
                logger.warning(f"  Only {len(resolved)}/720 THINGS stimuli found — human V4 extraction skipped")
        except Exception as e:
            logger.warning(f"  Could not resolve THINGS stimulus paths: {e}")

    for rule_config in config["learning_rules"]:
        rule_name = rule_config["name"]
        logger.info(f"\n--- {rule_name.upper()} ---")

        model = load_model(rule_name, weights_dir=weights_dir)
        features[rule_name] = {}

        # --- MajajHong2015 stimuli ---
        if "majajhong2015" in data and "stimulus_set" in data["majajhong2015"]:
            logger.info("  Extracting for MajajHong2015 stimuli...")
            stimulus_set = data["majajhong2015"]["stimulus_set"]
            try:
                image_paths = [stimulus_set.get_stimulus(img_id)
                               for img_id in stimulus_set["image_id"]]
                acts = extract_features(model, image_paths, THINGS_TRANSFORM)
                features[rule_name]["majajhong2015"] = acts
                for layer, act in acts.items():
                    logger.info(f"    {layer}: {act.shape}")
            except Exception as e:
                logger.error(f"  MajajHong feature extraction failed: {e}")

        # --- FreemanZiemba2013 stimuli ---
        if "freemanziemba2013" in data:
            logger.info("  Extracting for FreemanZiemba2013 stimuli...")
            try:
                import brainscore_vision
                stim_set = brainscore_vision.load_stimulus_set("FreemanZiemba2013.aperture-public")
                image_paths = [stim_set.get_stimulus(img_id)
                               for img_id in stim_set["image_id"]]
                acts = extract_features(model, image_paths, THINGS_TRANSFORM)
                features[rule_name]["freemanziemba2013"] = acts
                for layer, act in acts.items():
                    logger.info(f"    {layer}: {act.shape}")
            except Exception as e:
                logger.error(f"  FreemanZiemba feature extraction failed: {e}")

        # --- THINGS-720 stimuli (human V4: Conv2 features) ---
        if things_image_paths is not None:
            logger.info("  Extracting for THINGS-720 stimuli...")
            try:
                acts = extract_features(model, things_image_paths, THINGS_TRANSFORM)
                features[rule_name]["things_fmri"] = acts
                for layer, act in acts.items():
                    logger.info(f"    {layer}: {act.shape}")
            except Exception as e:
                logger.error(f"  THINGS-720 feature extraction failed: {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save features
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    for rule_name, datasets in features.items():
        for dataset_name, layers in datasets.items():
            for layer_name, acts in layers.items():
                save_path = results_dir / "features" / rule_name / dataset_name
                save_path.mkdir(parents=True, exist_ok=True)
                np.save(save_path / f"{layer_name}.npy", acts)

    return features


# ============================================================
# STEP 3: Compute RSA
# ============================================================

def step_rsa(config: dict, data: dict, features: dict) -> dict:
    """Compute RSA between model and neural RDMs using fixed Paper 1 layer mapping."""
    logger.info("=" * 60)
    logger.info("STEP 3: Computing RSA")
    logger.info("=" * 60)

    n_bootstrap = config["rsa"]["n_bootstrap"]
    rsa_results = {"human": {}, "macaque": {}}

    def _rsa_region(rule_name, dataset_key, layer, region, neural_rdm):
        """Compute RSA for one (rule, layer, region) triple."""
        acts = features.get(rule_name, {}).get(dataset_key, {}).get(layer)
        if acts is None:
            return
        model_rdm = build_rdm(acts)
        n = min(model_rdm.shape[0], neural_rdm.shape[0])
        result = compare_rdms_bootstrap(
            model_rdm[:n, :n], neural_rdm[:n, :n], n_bootstrap=n_bootstrap
        )
        result["layer"] = layer
        rsa_results["macaque"][rule_name][region] = result
        ci = f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]" \
             if result.get("ci_lower") is not None else ""
        logger.info(f"  Macaque {region} ({layer}): rho = {result['rho']:.4f} {ci}")

    for rule_config in config["learning_rules"]:
        rule_name = rule_config["name"]
        logger.info(f"\n--- {rule_name.upper()} ---")
        rsa_results["macaque"][rule_name] = {}

        # MajajHong2015: Conv2 -> V4, FC1 -> IT
        for layer, regions in LAYER_ROI_MAJAJHONG.items():
            for region in regions:
                neural_rdm = data.get("majajhong2015", {}).get(region, {}).get("rdm")
                if neural_rdm is not None:
                    _rsa_region(rule_name, "majajhong2015", layer, region, neural_rdm)

        # FreemanZiemba2013: Conv1 -> V1, V2
        for layer, regions in LAYER_ROI_FREEMANZIEMBA.items():
            for region in regions:
                neural_rdm = data.get("freemanziemba2013", {}).get(region, {}).get("rdm")
                if neural_rdm is not None:
                    _rsa_region(rule_name, "freemanziemba2013", layer, region, neural_rdm)

    # Human V4: Conv2 features vs THINGS-fMRI V4 RDM
    things_v4_rdm = data.get("things_fmri", {}).get("V4", {}).get("rdm")
    if things_v4_rdm is not None:
        logger.info("\n--- Human V4 (Conv2 vs THINGS-fMRI V4) ---")
        for rule_config in config["learning_rules"]:
            rule_name = rule_config["name"]
            acts = features.get(rule_name, {}).get("things_fmri", {}).get("Conv2")
            if acts is None:
                continue
            if rule_name not in rsa_results["human"]:
                rsa_results["human"][rule_name] = {}
            model_rdm = build_rdm(acts)
            n = min(model_rdm.shape[0], things_v4_rdm.shape[0])
            result = compare_rdms_bootstrap(
                model_rdm[:n, :n], things_v4_rdm[:n, :n], n_bootstrap=n_bootstrap
            )
            result["layer"] = "Conv2"
            rsa_results["human"][rule_name]["V4"] = result
            ci = (f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
                  if result.get("ci_lower") is not None else "")
            logger.info(f"  {rule_name} V4 (Conv2): rho = {result['rho']:.4f} {ci}")

    # Save (skip bootstrap distributions to keep file small)
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    rsa_save = {}
    for species, rules in rsa_results.items():
        rsa_save[species] = {}
        for rule, regions in rules.items():
            rsa_save[species][rule] = {
                region: {k: v for k, v in res.items() if k != "bootstrap_distribution"}
                for region, res in regions.items()
            }

    with open(results_dir / "rsa_results.json", "w", encoding="utf-8") as f:
        json.dump(rsa_save, f, indent=2, default=str)

    return rsa_results


# ============================================================
# STEP 4: Cross-species analysis
# ============================================================

def step_analysis(config: dict, rsa_results: dict) -> dict:
    """Run cross-species ranking comparison."""
    logger.info("=" * 60)
    logger.info("STEP 4: Cross-species analysis")
    logger.info("=" * 60)
    
    human_rsa = rsa_results.get("human", {})
    primate_rsa = rsa_results.get("macaque", {})
    
    region_mapping = config["cross_species"]["region_mapping"]
    
    # 1. Ranking conservation
    logger.info("\n--- Ranking Conservation Test ---")
    ranking_results = ranking_conservation_test(
        human_rsa, primate_rsa, region_mapping,
        n_permutations=config["cross_species"]["permutation_tests"]
    )
    
    # 2. V1 invariance
    logger.info("\n--- V1 Invariance Test ---")
    v1_results = v1_invariance_test(human_rsa, primate_rsa)
    
    # 3. Interaction effects
    logger.info("\n--- Interaction Effects ---")
    interaction_results = compute_interaction_effect(
        human_rsa, primate_rsa, region_mapping
    )
    
    # 4. Summary report
    logger.info("\n--- Summary Report ---")
    results_dir = config.get("output", {}).get("results_dir", "results")
    generate_summary_report(
        ranking_results, v1_results, interaction_results,
        output_path=f"{results_dir}/cross_species_summary.json"
    )
    
    return {
        "ranking": ranking_results,
        "v1": v1_results,
        "interaction": interaction_results,
    }


# ============================================================
# STEP 5: Generate figures
# ============================================================

def step_figures(config: dict, rsa_results: dict, analysis_results: dict):
    """Generate all publication figures."""
    logger.info("=" * 60)
    logger.info("STEP 5: Generating figures")
    logger.info("=" * 60)

    figures_dir = config.get("output", {}).get("figures_dir", "figures")
    results_dir = Path(config.get("output", {}).get("results_dir", "results"))

    # Load per-seed macaque RSA (from run_per_seed_rsa.py) if available
    primate_per_seed = None
    ps_path = results_dir / "rsa_results_macaque_per_seed.json"
    if ps_path.exists():
        with open(ps_path, encoding="utf-8") as f:
            primate_per_seed = json.load(f)
        logger.info(f"  Loaded per-seed RSA from {ps_path.name}")

    # Load macaque noise ceilings (from run_per_seed_rsa.py or step_download) if available
    primate_noise_ceilings = None
    nc_path = results_dir / "noise_ceilings_macaque.json"
    if nc_path.exists():
        with open(nc_path, encoding="utf-8") as f:
            primate_noise_ceilings = json.load(f)
        logger.info(f"  Loaded noise ceilings from {nc_path.name}")

    generate_all_figures(
        human_rsa=rsa_results.get("human", {}),
        primate_rsa=rsa_results.get("macaque", {}),
        ranking_results=analysis_results.get("ranking", {}),
        v1_results=analysis_results.get("v1", {}),
        interaction_results=analysis_results.get("interaction", {}),
        output_dir=figures_dir,
        primate_per_seed=primate_per_seed,
        primate_noise_ceilings=primate_noise_ceilings,
    )


# ============================================================
# STEP 6: Import Paper 1 results
# ============================================================

def import_paper1_results(paper1_dir: str) -> dict:
    """
    Import human RSA results from Paper 1.
    
    Looks for:
    - results/rsa_results.json or similar
    - Individual ρ values per learning rule × region
    
    Adjust paths to match your Paper 1 output structure.
    """
    p1 = Path(paper1_dir)
    
    # Try loading from JSON results
    for candidate in ["rsa_results.json", "results.json", "phase4_results.json"]:
        fpath = p1 / candidate
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                return json.load(f)
    
    logger.info(f"No JSON found in {paper1_dir} — using hardcoded Paper 1 RSA values (rsa_results_cnn.csv, verified 2026-05-17)")

    # Paper 1 results: learning_rules_v6.py, N=720 THINGS stimuli, 3 subjects, 5 seeds
    # Source: rsa_results_cnn.csv (fixed layer mapping: Conv1→V1/V2, Conv3→LOC, FC1→IT)
    # Verified against actual CSV on 2026-05-17
    #
    # Key findings:
    #   V1: Random(0.075) > STDP(0.064) > PC(0.056) > BP(0.033) > FA(0.012)
    #       → architecture (random init) drives V1 alignment
    #   IT: PC(0.014) ≈ BP(0.013) > STDP(0.012) ≈ FA(0.012) > Random(0.008)
    #       → learning rules matter for IT, BP/PC dominate
    #
    # To update: open rsa_results_cnn.csv, filter by the fixed layer per ROI:
    #   V1 → Conv1, V2 → Conv1, LOC → Conv3, IT → FC1
    # and copy the 'rho' column values here.
    return {
        "backprop":           {"V1": {"rho": 0.03346}, "V2": {"rho": 0.01846}, "LOC": {"rho": 0.01162}, "IT": {"rho": 0.01325}},
        "feedback_alignment": {"V1": {"rho": 0.01173}, "V2": {"rho": 0.00394}, "LOC": {"rho": 0.00564}, "IT": {"rho": 0.01156}},
        "predictive_coding":  {"V1": {"rho": 0.05607}, "V2": {"rho": 0.02790}, "LOC": {"rho": 0.00605}, "IT": {"rho": 0.01364}},
        "stdp":               {"V1": {"rho": 0.06414}, "V2": {"rho": 0.03577}, "LOC": {"rho": 0.00579}, "IT": {"rho": 0.01187}},
        "random":             {"V1": {"rho": 0.07546}, "V2": {"rho": 0.04334}, "LOC": {"rho":-0.00505}, "IT": {"rho": 0.00777}},
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Species RSA Pipeline")
    parser.add_argument("--step", choices=["download", "extract", "rsa", "analysis", "figures", "all"],
                        default="all", help="Which pipeline step to run")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--paper1-dir", default="data/paper1_results",
                        help="Path to Paper 1 results for human RSA values")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.step in ("download", "all"):
        data = step_download(config)
    
    if args.step in ("extract", "all"):
        if "data" not in dir():
            data = step_download(config)
        features = step_extract(config, data)
    
    if args.step in ("rsa", "all"):
        if "features" not in dir():
            logger.error("Run extract step first")
            return
        rsa_results = step_rsa(config, data, features)
        
        # Merge Paper 1 human results — only fill regions not already computed
        # (preserves the V4 values computed above from Conv2 vs THINGS-fMRI V4 RDM)
        paper1 = import_paper1_results(args.paper1_dir)
        if paper1:
            for rule, regions in paper1.items():
                if rule not in rsa_results["human"]:
                    rsa_results["human"][rule] = {}
                for region, val in regions.items():
                    if region not in rsa_results["human"][rule]:
                        rsa_results["human"][rule][region] = val
            results_dir = Path(config.get("output", {}).get("results_dir", "results"))
            with open(results_dir / "rsa_results.json", "w", encoding="utf-8") as f:
                json.dump(rsa_results, f, indent=2, default=str)
    
    if args.step in ("analysis", "all"):
        if "rsa_results" not in dir():
            # Load from saved
            results_path = Path(config["output"]["results_dir"]) / "rsa_results.json"
            if results_path.exists():
                with open(results_path, encoding="utf-8") as f:
                    rsa_results = json.load(f)
            else:
                logger.error("Run rsa step first")
                return
        analysis_results = step_analysis(config, rsa_results)
    
    if args.step in ("figures", "all"):
        if "analysis_results" not in dir():
            # Load from saved files
            results_dir = Path(config.get("output", {}).get("results_dir", "results"))
            summary_path = results_dir / "cross_species_summary.json"
            if not summary_path.exists():
                logger.error("Run analysis step first")
                return
            with open(summary_path, encoding="utf-8") as f:
                summary_data = json.load(f)
            analysis_results = {
                "ranking": summary_data.get("ranking_conservation", {}),
                "v1": summary_data.get("v1_invariance", {}),
                "interaction": summary_data.get("interaction_effects", {}),
            }
            if "rsa_results" not in dir():
                rsa_path = results_dir / "rsa_results.json"
                with open(rsa_path, encoding="utf-8") as f:
                    rsa_results = json.load(f)
        step_figures(config, rsa_results, analysis_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()