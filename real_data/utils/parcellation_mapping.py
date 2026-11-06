"""Map cc200 (Craddock K=200) time series to Yeo networks (7 or 17).

ABIDE PCP and ADHD-200/Athena use *different* CC200 atlases (different ROI
numberings), so the cc200->Yeo mapping is dataset-specific:

  source='abide' -> uses parcellation/tcorr05_2level_all.nii.gz (PCP),
                    cached at parcellation/cc200_to_yeo{N}_abide.npy
  source='adhd'  -> uses <data>/adhd/cc200/templates/ADHD200_parcellate_200.nii.gz,
                    cached at parcellation/cc200_to_yeo{N}_adhd.npy

The Yeo-7 and Yeo-17 mappings are computed independently, each by resampling
Yeo's own N-network atlas onto the cc200 grid and taking the majority label per
ROI (NOT by collapsing 17 -> 7, which would misassign ROIs that straddle two
17-subnetworks belonging to different 7-groups).

Verified empirically by clusteredness score kappa: with the right atlas,
within-network mean Z-correlation is well above between-network ~0; with the
wrong atlas kappa collapses to ~0. For ADHD cc200: Yeo-17 mu_W~0.36 / kappa~1.06,
Yeo-7 mu_W~0.25 / kappa~0.82 (lower mu_W is the expected coarsening effect).
"""
from pathlib import Path
import numpy as np

from real_data.utils.paths import data_root

PARCELLATION_DIR = Path(__file__).resolve().parents[1] / 'parcellation'

ATLAS_SPECS = {
    'abide': {
        # 4D Craddock atlas; K=200 lives at the 4th-axis index 19.
        'path': PARCELLATION_DIR / 'tcorr05_2level_all.nii.gz',
        'k_index': 19,
    },
    'adhd': {
        # 3D atlas already at K=200 (no time-axis indexing).
        'path': None,  # resolved under the data root, see compute_cc200_to_yeo_mapping
        'k_index': None,
    },
}

# Yeo-2011 FreeSurferConformed 1mm label volumes (integer labels 1..N), both
# shipped in the same nilearn fetch directory.
YEO_ATLAS_FILES = {
    7: 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz',
    17: 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz',
}


def _cache_path(source, n_networks):
    return PARCELLATION_DIR / f'cc200_to_yeo{n_networks}_{source}.npy'


def compute_cc200_to_yeo_mapping(source='abide', n_networks=17, force=False):
    """Compute and cache the cc200 -> Yeo-`n_networks` ROI label mapping.

    Returns
    -------
    mapping : np.ndarray, shape (200,), dtype int8
        mapping[i] is the Yeo network label (1..n_networks) of cc200 ROI (i+1).
        0 means the ROI fell outside Yeo's cortical mask (e.g. subcortical).
    """
    if source not in ATLAS_SPECS:
        raise ValueError(f"Unknown source {source!r}; choose one of {list(ATLAS_SPECS)}.")
    if n_networks not in YEO_ATLAS_FILES:
        raise ValueError(f"n_networks must be one of {list(YEO_ATLAS_FILES)}, got {n_networks}.")
    spec = dict(ATLAS_SPECS[source])
    if spec['path'] is None:
        spec['path'] = data_root() / 'adhd' / 'cc200' / 'templates' / 'ADHD200_parcellate_200.nii.gz'
    cache = _cache_path(source, n_networks)

    if cache.exists() and not force:
        return np.load(cache)

    import nibabel as nib
    from nilearn.datasets import fetch_atlas_yeo_2011
    from nilearn.image import resample_to_img, index_img

    if not spec['path'].exists():
        raise FileNotFoundError(
            f"cc200 atlas for source={source!r} not found at {spec['path']}."
        )
    if spec['k_index'] is None:
        cc_img = nib.load(str(spec['path']))
    else:
        cc_img = index_img(str(spec['path']), spec['k_index'])

    yeo_bunch = fetch_atlas_yeo_2011()
    yeo_path = Path(yeo_bunch['maps']).parent / YEO_ATLAS_FILES[n_networks]
    yeo_img = nib.load(str(yeo_path))

    yeo_on_cc = resample_to_img(yeo_img, cc_img, interpolation='nearest')
    cc = cc_img.get_fdata().astype(int)
    ye = yeo_on_cc.get_fdata().squeeze().astype(int)

    n_cc = int(cc.max())
    mapping = np.zeros(n_cc, dtype=np.int8)
    for roi in range(1, n_cc + 1):
        vals = ye[cc == roi]
        vals = vals[vals > 0]
        mapping[roi - 1] = int(np.bincount(vals).argmax()) if vals.size else 0

    PARCELLATION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, mapping)
    return mapping


def collapse_cc200_to_yeo(ts_cc200, mapping=None, source='abide', n_networks=17):
    """Average columns of a (T, 200) cc200 TS by Yeo network -> (T, n_networks)."""
    if mapping is None:
        mapping = compute_cc200_to_yeo_mapping(source=source, n_networks=n_networks)
    T = ts_cc200.shape[0]
    out = np.zeros((T, n_networks), dtype=ts_cc200.dtype)
    for nid in range(1, n_networks + 1):
        cols = np.where(mapping == nid)[0]
        if len(cols):
            out[:, nid - 1] = ts_cc200[:, cols].mean(axis=1)
    return out


def compute_cc200_to_yeo17_mapping(source='abide', force=False):
    """Backwards-compatible alias for the Yeo-17 mapping."""
    return compute_cc200_to_yeo_mapping(source=source, n_networks=17, force=force)


def collapse_cc200_to_yeo17(ts_cc200, mapping=None, source='abide'):
    """Backwards-compatible alias: average (T, 200) cc200 TS -> (T, 17)."""
    return collapse_cc200_to_yeo(ts_cc200, mapping=mapping, source=source, n_networks=17)
