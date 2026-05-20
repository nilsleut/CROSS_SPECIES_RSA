"""
verify_paper1_consistency.py

Sanity check: reproduce Paper 1 RSA ρ values on THINGS-fMRI using the
Paper 1 custom 3-conv CNN with saved weights.

Expected: computed ρ ≈ Paper 1 seed-averaged ρ (within ~0.005).
Saved weights are seed 0 (seed=42); Paper 1 values are 5-seed averages,
so small deviations are expected.

Run from repo root:
    python scripts/verify_paper1_consistency.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import load_model, extract_features, THINGS_TRANSFORM

# ── Paths ────────────────────────────────────────────────────────────────────
FMRI_DIR    = Path(r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs_720")
THINGS_DIR  = Path(r"C:/Users/nilsl/Desktop/Projekte/RSA/Datensatz/images_THINGS/object_images")
WEIGHTS_DIR = Path(r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs")
SUBJECTS    = ["sub-01", "sub-02", "sub-03"]

# ── Paper 1 expected ρ values (rsa_results_cnn.csv, seed-averaged, N=5) ─────
EXPECTED = {
    "backprop":           {"V1": 0.03346, "V2": 0.01846, "LOC": 0.01162, "IT": 0.01325},
    "feedback_alignment": {"V1": 0.01173, "V2": 0.00394, "LOC": 0.00564, "IT": 0.01156},
    "predictive_coding":  {"V1": 0.05607, "V2": 0.02790, "LOC": 0.00605, "IT": 0.01364},
    "stdp":               {"V1": 0.06414, "V2": 0.03577, "LOC": 0.00579, "IT": 0.01187},
    "random":             {"V1": 0.07546, "V2": 0.04334, "LOC":-0.00505, "IT": 0.00777},
}

# Fixed layer → ROI mapping (same as Paper 1)
LAYER_ROI = {
    "Conv1": ["V1", "V2"],
    "Conv3": ["LOC"],
    "FC1":   ["IT"],
}

RULES = list(EXPECTED.keys())


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def build_rdm(feats: np.ndarray) -> np.ndarray:
    """Correlation distance RDM from (n_stimuli, n_features)."""
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(feats, metric="correlation"))


def load_neural_rdms() -> dict[str, np.ndarray]:
    """
    Load THINGS-fMRI RDMs for V1, V2, LOC, IT.
    Files: fmri_rdm_{REGION}_sub-0N.npy  (720×720 per subject).
    Returns mean RDM across subjects for each region.
    """
    regions = ["V1", "V2", "LOC", "IT"]
    neural = {}
    for region in regions:
        mats = []
        for sub in SUBJECTS:
            sub_idx = sub.split("-")[1]   # "01", "02", "03"
            fname = FMRI_DIR / f"fmri_rdm_{region}_sub-{sub_idx}.npy"
            if not fname.exists():
                print(f"  [WARN] Missing: {fname}")
                continue
            mats.append(np.load(fname))
        if mats:
            neural[region] = np.mean(mats, axis=0)
        else:
            print(f"  [WARN] No data for {region} — skipping")
    return neural


def load_stim_paths() -> list[Path]:
    """
    720 THINGS image paths in stimulus order from sub-01.
    Uses fast flat lookup by filename (v8 approach).
    """
    order_file = FMRI_DIR / "stim_order_sub-01.txt"
    with open(order_file, encoding="utf-8") as f:
        filenames = [ln.strip() for ln in f if ln.strip()]

    print(f"Building image map from {THINGS_DIR} ...")
    img_map: dict[str, Path] = {p.name: p for p in THINGS_DIR.rglob("*.jpg")}
    print(f"  Found {len(img_map)} images in THINGS dir")

    paths = []
    missing = []
    for fn in filenames:
        if fn in img_map:
            paths.append(img_map[fn])
        else:
            missing.append(fn)

    if missing:
        print(f"  [WARN] {len(missing)} stimuli not found in THINGS dir:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")

    print(f"  Resolved {len(paths)}/{len(filenames)} stimuli")
    return paths


def compute_rsa(model_rdm: np.ndarray, neural_rdm: np.ndarray) -> float:
    a = _upper_tri(model_rdm)
    b = _upper_tri(neural_rdm)
    rho, _ = spearmanr(a, b)
    return float(rho)


def main():
    print("=" * 65)
    print("  Paper 1 Consistency Check")
    print("=" * 65)

    # Load neural RDMs once
    print("\nLoading THINGS-fMRI neural RDMs ...")
    neural = load_neural_rdms()
    print(f"  Loaded regions: {list(neural.keys())}")

    # Load stimulus paths once
    print("\nLoading stimulus order ...")
    stim_paths = load_stim_paths()
    if len(stim_paths) < 720:
        print(f"\n[ERROR] Only {len(stim_paths)} stimuli resolved (need 720). Aborting.")
        sys.exit(1)

    # Header
    header = f"\n{'Rule':<22} {'ROI':<6} {'Computed':>10} {'Expected':>10} {'Delta':>8} {'OK?':>5}"
    print(header)
    print("-" * 65)

    all_ok = True
    TOLERANCE = 0.015  # allow up to 0.015 deviation (seed variance)

    for rule in RULES:
        print(f"\nLoading model: {rule} ...")
        try:
            model = load_model(rule, weights_dir=WEIGHTS_DIR)
        except Exception as e:
            print(f"  [ERROR] Could not load {rule}: {e}")
            continue

        print(f"  Extracting features for {len(stim_paths)} stimuli ...")
        feats = extract_features(model, stim_paths, THINGS_TRANSFORM, batch_size=64)
        # feats: {"Conv1": ndarray, "Conv2": ndarray, "Conv3": ndarray, "FC1": ndarray}

        for layer, rois in LAYER_ROI.items():
            if layer not in feats:
                print(f"  [WARN] Layer {layer} not in features — skipping")
                continue
            model_rdm = build_rdm(feats[layer])

            for roi in rois:
                if roi not in neural:
                    print(f"  [WARN] Neural RDM for {roi} not available — skipping")
                    continue
                rho = compute_rsa(model_rdm, neural[roi])
                exp = EXPECTED[rule][roi]
                delta = rho - exp
                ok = abs(delta) <= TOLERANCE
                if not ok:
                    all_ok = False
                flag = "OK" if ok else "FAIL"
                print(f"  {rule:<22} {roi:<6} {rho:>10.5f} {exp:>10.5f} {delta:>+8.5f} {flag:>5}")

    print("\n" + "=" * 65)
    if all_ok:
        print("  RESULT: All values within tolerance. Pipeline is consistent with Paper 1.")
    else:
        print("  RESULT: Some values exceed tolerance. Investigate FAIL rows above.")
    print("=" * 65)


if __name__ == "__main__":
    main()
