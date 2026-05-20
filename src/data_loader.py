"""
data_loader.py — Load primate electrophysiology and human fMRI data for cross-species RSA.

Handles:
- MajajHong2015 (Macaque V4/IT, HVM stimuli)
- FreemanZiemba2013 (Macaque V1/V2, texture stimuli)
- THINGS-fMRI (Human, reused from Paper 1)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_majajhong2015(access: str = "public", average_repetitions: bool = True) -> Dict:
    """
    Load MajajHong2015 macaque electrophysiology data via Brain-Score.

    Returns dict with keys 'V4' and 'IT', each containing:
        - 'responses': np.ndarray (n_stimuli × n_neurons), mean firing rates,
          ordered to match stimulus_set["image_id"] so model features and
          neural RDMs are aligned without extra reordering in step_rsa.
        - 'stimulus_ids': list of image_id strings in stimulus-set order
        - 'n_neurons': int
        - 'n_stimuli': int
    """
    import brainscore_vision

    identifier = f"MajajHong2015.{access}"
    logger.info(f"Loading {identifier}...")
    assembly = brainscore_vision.load_dataset(identifier)

    # Load stimulus set early so we can align neural ordering to it
    stimulus_set = brainscore_vision.load_stimulus_set(f"hvm-{access}")
    set_ids = list(stimulus_set["image_id"])  # canonical ordering

    # Squeeze time dimension (mean firing rate across time bins)
    assembly = assembly.squeeze("time_bin")

    result = {"stimulus_set": stimulus_set}
    for region in ["V4", "IT"]:
        region_data = assembly.sel(region=region)

        if average_repetitions:
            # Average over repetitions of same image_id.
            # Single-key groupby keeps "image_id" as a direct coordinate in the result,
            # making it straightforward to read out the post-groupby ordering.
            region_data = region_data.multi_groupby(["image_id"]).mean()

        responses = region_data.values
        # Assembly dims: (n_neurons × n_unique_stimuli) — transpose to (n_stimuli × n_neurons)
        if responses.ndim == 2 and responses.shape[0] < responses.shape[1]:
            responses = responses.T  # → (n_stimuli, n_neurons)

        # Get image_ids in post-groupby order so we can align to stimulus set
        neural_ids = _extract_image_ids(region_data)

        # Reorder responses to match stimulus set's canonical image_id order
        if neural_ids is not None:
            id_to_row = {nid: i for i, nid in enumerate(neural_ids)}
            ordered_ids = [sid for sid in set_ids if sid in id_to_row]
            if ordered_ids:
                row_idx = [id_to_row[sid] for sid in ordered_ids]
                responses = responses[row_idx, :]
            else:
                ordered_ids = neural_ids
        else:
            ordered_ids = set_ids[: responses.shape[0]]

        result[region] = {
            "responses": responses,
            "stimulus_ids": ordered_ids,
            "n_neurons": responses.shape[1] if responses.ndim == 2 else 1,
            "n_stimuli": responses.shape[0],
        }
        logger.info(f"  {region}: {responses.shape[0]} stimuli × {responses.shape[1]} neurons")

    return result


def _extract_image_ids(region_data) -> Optional[list]:
    """Try several methods to get image_id values from a grouped xarray DataArray."""
    # Method 1: direct coord (works after single-key groupby)
    try:
        if "image_id" in region_data.coords:
            return list(region_data.coords["image_id"].values)
    except Exception:
        pass
    # Method 2: MultiIndex level (works after multi-key groupby)
    for dim in region_data.dims:
        try:
            idx = region_data.indexes.get(dim)
            if idx is not None and hasattr(idx, "get_level_values"):
                return list(idx.get_level_values("image_id"))
        except Exception:
            pass
    return None


def load_freemanziemba2013(access: str = "public") -> Dict:
    """
    Load FreemanZiemba2013 macaque V1/V2 electrophysiology.

    Neural responses are reordered to match the stimulus set's canonical
    image_id ordering so that model features (extracted in stim_set order)
    and neural RDMs are aligned.

    Note: These are TEXTURE stimuli, not objects.
    """
    import brainscore_vision

    identifier = f"FreemanZiemba2013.{access}"
    logger.info(f"Loading {identifier}...")
    assembly = brainscore_vision.load_dataset(identifier)

    # Load stimulus set for canonical ordering (same approach as MajajHong)
    stim_id_name = f"FreemanZiemba2013.aperture-{access}"
    stimulus_set = brainscore_vision.load_stimulus_set(stim_id_name)
    set_ids = list(stimulus_set["image_id"])  # canonical ordering

    # Assembly dims: (neuroid, time_bin, presentation)
    # neuroid is a MultiIndex with levels [neuroid_id, region].
    # 2700 presentations = 135 unique stimuli × 20 repetitions.

    result = {}
    for region in ["V1", "V2"]:
        try:
            region_data = assembly.sel(region=region)
        except Exception as exc:
            logger.warning(f"  Could not split {region} from assembly ({exc}); using full assembly")
            region_data = assembly

        # (n_neurons, n_timebins, n_presentations) → average over time_bin
        region_data = region_data.mean("time_bin")  # → (n_neurons, n_presentations)

        # Average over repetitions of the same stimulus
        try:
            region_data = region_data.multi_groupby(["image_id"]).mean()
        except Exception:
            try:
                region_data = region_data.groupby("image_id").mean("presentation")
            except Exception as exc2:
                logger.warning(f"  Could not average repetitions for {region}: {exc2}. Using raw presentations.")

        # Shape after groupby: (n_neurons, n_unique_stimuli) — transpose to (n_stimuli, n_neurons)
        responses = region_data.values
        if responses.ndim == 2 and responses.shape[0] < responses.shape[1]:
            responses = responses.T  # → (n_stimuli, n_neurons)

        # Get post-groupby image_id order and reorder to match stimulus set
        neural_ids = _extract_image_ids(region_data)
        if neural_ids is not None:
            id_to_row = {nid: i for i, nid in enumerate(neural_ids)}
            ordered_ids = [sid for sid in set_ids if sid in id_to_row]
            if ordered_ids:
                row_idx = [id_to_row[sid] for sid in ordered_ids]
                responses = responses[row_idx, :]
            else:
                ordered_ids = neural_ids
        else:
            ordered_ids = set_ids[: responses.shape[0]]

        result[region] = {
            "responses": responses,
            "stimulus_ids": ordered_ids,
            "n_neurons": responses.shape[1] if responses.ndim == 2 else responses.shape[0],
            "n_stimuli": responses.shape[0],
        }
        logger.info(f"  {region}: {responses.shape[0]} stimuli x {responses.shape[1]} neurons")

    return result


def load_things_fmri(data_dir: str, subjects: list = None) -> Dict:
    """
    Load THINGS-fMRI data (reuse from Paper 1 pipeline).

    Accepts two layouts:
      1. Averaged:    rdm_{region}.npy  (e.g. rdm_v1.npy)
      2. Per-subject: fmri_rdm_{REGION}_sub-01.npy  (Paper 1 / outputs_720 format)
         → subjects are averaged automatically.
    """
    data_path = Path(data_dir)
    result = {}
    regions = ["V1", "V2", "V4", "LOC", "IT"]

    for region in regions:
        # Layout 1: averaged file
        rdm_file = data_path / f"rdm_{region.lower()}.npy"
        if rdm_file.exists():
            rdm = np.load(rdm_file)
            result[region] = {"rdm": rdm, "source": "precomputed"}
            logger.info(f"  Loaded {region} RDM: {rdm.shape}")
            continue

        # Layout 2: per-subject files (Paper 1 outputs_720 format)
        sub_files = sorted(data_path.glob(f"fmri_rdm_{region}_sub-*.npy"))
        if sub_files:
            rdms = [np.load(f) for f in sub_files]
            rdm = np.mean(rdms, axis=0)
            result[region] = {
                "rdm": rdm,
                "rdms_per_subject": rdms,
                "source": "paper1_per_subject",
                "n_subjects": len(rdms),
            }
            logger.info(f"  Loaded {region} RDM (avg of {len(rdms)} subjects): {rdm.shape}")
            continue

        logger.warning(f"  {region} RDM not found in {data_path}")

    return result


def build_neural_rdm(responses: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """
    Build a Representational Dissimilarity Matrix from neural responses.
    
    Args:
        responses: (n_stimuli × n_neurons) firing rate matrix
        metric: Distance metric ('correlation' = 1 - Pearson r, 'euclidean')
    
    Returns:
        rdm: (n_stimuli × n_stimuli) symmetric dissimilarity matrix
    """
    from scipy.spatial.distance import pdist, squareform
    
    n = responses.shape[0]
    
    if metric == "correlation":
        # 1 - Pearson correlation between stimulus response patterns
        distances = pdist(responses, metric="correlation")
    elif metric == "euclidean":
        distances = pdist(responses, metric="euclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    rdm = squareform(distances)
    
    # Handle any NaN (e.g., from zero-variance neurons)
    rdm = np.nan_to_num(rdm, nan=0.0)
    
    return rdm


def get_stimulus_images(stimulus_set, image_ids: list = None) -> list:
    """
    Get file paths for stimulus images from a Brain-Score stimulus set.
    Used to feed the same images through our models.
    """
    if image_ids is not None:
        return [stimulus_set.get_stimulus(img_id) for img_id in image_ids]
    else:
        return [stimulus_set.get_stimulus(img_id) for img_id in stimulus_set["image_id"]]


# ============================================================
# Convenience: load everything
# ============================================================

def load_all_primate_data(access: str = "public") -> Dict:
    """Load all available primate electrophysiology datasets."""
    data = {
        "majajhong2015": load_majajhong2015(access=access),
        "freemanziemba2013": load_freemanziemba2013(access=access),
    }
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Cross-Species RSA — Data Loading Test")
    print("=" * 60)
    
    # Test MajajHong2015
    print("\n--- MajajHong2015 ---")
    try:
        mh = load_majajhong2015(access="public")
        for region in ["V4", "IT"]:
            d = mh[region]
            print(f"  {region}: {d['n_stimuli']} stimuli × {d['n_neurons']} neurons")
            rdm = build_neural_rdm(d["responses"])
            print(f"  {region} RDM shape: {rdm.shape}, range: [{rdm.min():.3f}, {rdm.max():.3f}]")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test FreemanZiemba2013
    print("\n--- FreemanZiemba2013 ---")
    try:
        fz = load_freemanziemba2013(access="public")
        for region in ["V1", "V2"]:
            if region in fz:
                d = fz[region]
                print(f"  {region}: {d['n_stimuli']} stimuli × {d['n_neurons']} neurons")
    except Exception as e:
        print(f"  Error: {e}")
