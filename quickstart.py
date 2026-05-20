#!/usr/bin/env python3
"""
quickstart.py — Verify data access + run a minimal end-to-end test.

Run this FIRST to check that Brain-Score data downloads work.
If this script succeeds, run_pipeline.py will work.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("quickstart")


def check_brainscore():
    """Verify brainscore-vision is installed and data is accessible."""
    log.info("1) Checking brainscore-vision installation...")
    try:
        import brainscore_vision
        log.info(f"   ✓ brainscore-vision installed (v{brainscore_vision.__version__ if hasattr(brainscore_vision, '__version__') else '?'})")
    except ImportError:
        log.error("   ✗ brainscore-vision not found. Run: pip install brainscore-vision")
        return False
    
    log.info("2) Loading MajajHong2015.public (macaque V4/IT)...")
    try:
        assembly = brainscore_vision.load_dataset("MajajHong2015.public")
        assembly = assembly.squeeze("time_bin")
        
        for region in ["V4", "IT"]:
            region_data = assembly.sel(region=region)
            n_neurons = region_data.sizes.get("neuroid", "?")
            n_presentations = region_data.sizes.get("presentation", "?")
            log.info(f"   ✓ {region}: {n_presentations} presentations × {n_neurons} neurons")
        
        # Check stimulus set
        stim = brainscore_vision.load_stimulus_set("hvm-public")
        log.info(f"   ✓ HVM stimulus set: {len(stim)} images")
        
        # Check first image exists
        first_id = stim["image_id"].iloc[0]
        img_path = stim.get_stimulus(first_id)
        log.info(f"   ✓ Sample image accessible: {img_path}")
        
    except Exception as e:
        log.error(f"   ✗ MajajHong2015 failed: {e}")
        return False
    
    log.info("3) Loading FreemanZiemba2013.public (macaque V1/V2)...")
    try:
        fz = brainscore_vision.load_dataset("FreemanZiemba2013.public")
        log.info(f"   ✓ Shape: {fz.shape}, dims: {fz.dims}")
        if "region" in fz.coords:
            regions = set(fz.coords["region"].values)
            log.info(f"   ✓ Regions: {regions}")
    except Exception as e:
        log.error(f"   ✗ FreemanZiemba2013 failed: {e}")
        log.info("   (This is optional — project works with MajajHong alone)")
    
    return True


def check_torch():
    """Check PyTorch + torchvision."""
    log.info("4) Checking PyTorch...")
    try:
        import torch
        import torchvision.models as models
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"   ✓ PyTorch {torch.__version__}, device: {device}")
        
        model = models.resnet50(weights=None)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        log.info(f"   ✓ ResNet-50 forward pass OK (output: {y.shape})")
        return True
    except Exception as e:
        log.error(f"   ✗ PyTorch error: {e}")
        return False


def mini_rsa_test():
    """Run a tiny RSA to verify the analysis pipeline."""
    log.info("5) Mini RSA test with synthetic data...")
    import numpy as np
    sys.path.insert(0, "src")
    from rsa_engine import build_rdm, compare_rdms_bootstrap
    
    np.random.seed(42)
    X = np.random.randn(30, 64)
    Y = X + np.random.randn(30, 64) * 0.5
    
    rdm_x = build_rdm(X)
    rdm_y = build_rdm(Y)
    result = compare_rdms_bootstrap(rdm_x, rdm_y, n_bootstrap=500)
    
    log.info(f"   ✓ RSA: ρ = {result['rho']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    assert result["rho"] > 0.3, "Sanity check failed: correlated data should yield positive ρ"
    log.info("   ✓ RSA engine working correctly")
    return True


def main():
    log.info("=" * 55)
    log.info("Cross-Species RSA — Quickstart Verification")
    log.info("=" * 55)
    
    torch_ok = check_torch()
    rsa_ok = mini_rsa_test()
    bs_ok = check_brainscore()
    
    log.info("")
    log.info("=" * 55)
    if torch_ok and rsa_ok and bs_ok:
        log.info("ALL CHECKS PASSED — ready to run the full pipeline:")
        log.info("  python run_pipeline.py --step all")
    elif torch_ok and rsa_ok:
        log.info("PARTIAL: PyTorch + RSA OK, Brain-Score data needs fixing")
        log.info("Check your internet connection and try again")
    else:
        log.info("ISSUES FOUND — see errors above")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
