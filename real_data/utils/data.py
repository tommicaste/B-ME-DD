import numpy as np
import pandas as pd
import os
import glob
import scipy.io
import scipy.stats as stats
import torch
from nilearn.datasets import fetch_abide_pcp
from tqdm.auto import tqdm

from real_data.shared.windowing import (build_entries, windows_for_T, n_mode_inert,
                                       whole_scan_windows)
from real_data.shared.targets import resolve_target
from real_data.shared.splitting import grouped_split
from real_data.utils.paths import data_root


# ------- Low-level Functions -------

def flatten_lower_offdiag(X):
    if isinstance(X, list):
        indices = np.tril_indices(X[0].shape[0], k=-1)
        X_flat = np.array([x[indices] for x in X])
    else:
        if X.ndim == 3:
            n = X.shape[-1]
            indices = np.tril_indices(n, k=-1)
            X_flat = X[:, indices[0], indices[1]]
        elif X.ndim == 2:
            n = X.shape[0]
            indices = np.tril_indices(n, k=-1)
            X_flat = X[indices]
        else:
            raise ValueError(f"Unsupported array shape: {X.shape}. Expected 2D or 3D array.")
    return X_flat


def _window_starts(n, t, stride):
    m = (n - t) // stride + 1
    if m <= 0:
        return np.array([], dtype=np.int64)
    return (np.arange(m) * stride).astype(np.int64)


def is_circle_feasible_at_T(ts_len, T_ref, stride=None):
    """Can a subject of session length ts_len produce non-overlapping
    partners for every one of its windows at T=T_ref?

    Mirrors the exact rule in adjacency.circle_infeasible_subjects.
    Returns True iff every window has at least one other window j with
    se[i] <= ss[j] or se[j] <= ss[i], when windows are placed at stride s.
    """
    if T_ref <= 0 or ts_len < T_ref:
        return False
    s = int(stride) if stride else max(1, int(T_ref * 0.1))
    starts = _window_starts(int(ts_len), int(T_ref), s)
    if len(starts) < 2:
        return False
    ends = starts + int(T_ref)
    for i in range(len(starts)):
        ok = False
        for j in range(len(starts)):
            if j == i:
                continue
            if ends[i] <= starts[j] or ends[j] <= starts[i]:
                ok = True
                break
        if not ok:
            return False
    return True


def window_ts(x, t, stride=1, axis=0, drop_last=True):
    x = np.asarray(x)
    if t <= 0:
        raise ValueError("t must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    axis = axis % x.ndim
    n = x.shape[axis]

    if drop_last:
        m = (n - t) // stride + 1
        if m <= 0:
            return np.empty((0, t) + tuple(d for i, d in enumerate(x.shape) if i != axis), dtype=x.dtype)
        starts = np.arange(m) * stride
    else:
        m = (n + stride - 1) // stride
        starts = np.arange(m) * stride

    x_move = np.moveaxis(x, axis, 0)

    out = []
    for s in starts:
        w = x_move[s:s + t]
        if w.shape[0] < t:
            w = np.pad(w, ((0, t - w.shape[0]),) + ((0, 0),) * (w.ndim - 1))
        out.append(w)

    return np.stack(out, axis=0)


def moving_correlations(x, t, stride=1, axis=0, drop_last=True):
    x = np.asarray(x)

    axis = axis % x.ndim
    x_move = np.moveaxis(x, axis, 0)
    n = x_move.shape[0]

    if drop_last:
        m = (n - t) // stride + 1
        starts = np.arange(m) * stride
    else:
        m = (n + stride - 1) // stride
        starts = np.arange(m) * stride

    p = x_move[0].reshape(-1).shape[0]
    out = np.empty((len(starts), p, p), dtype=float)

    k = 0
    for s in starts:
        w = x_move[s:s + t]
        if w.shape[0] < t:
            w = np.pad(w, ((0, t - w.shape[0]),) + ((0, 0),) * (w.ndim - 1))
        w = w.reshape(t, -1)
        w = w - w.mean(axis=0, keepdims=True)
        std = w.std(axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        z = w / std
        out[k] = (z.T @ z) / (t - 1)
        k += 1

    return out


# ------- Dataset Loaders -------

# Network membership per ROI (1..15) for the Gordon286Cort + 8 subcortical
# 294-ROI atlas used by frankfurt, in NATURAL data ordering.
# Source: real_data/parcellation/Gordon286Cort_ASEG8Subcort_294ROI_15net.mat.
# Built by `mapping[order - 1] = key[:, 1]` where IM_294.order is the natural
# ROI index at each ordered (network-sorted) position. The .mat's `key[:, 1]`
# alone is in ordered ordering, NOT natural ordering.
FRANKFURT_NETWORK_ID = np.array([
    1, 8, 9, 1, 2, 1, 3, 2, 3, 10, 11, 12, 12, 2, 2, 2, 2, 7, 7, 5, 3, 1, 1, 7, 7, 6, 8, 8, 8, 8,
    7, 8, 8, 8, 8, 9, 7, 4, 4, 4, 1, 8, 8, 8, 8, 4, 8, 4, 4, 9, 8, 4, 8, 8, 8, 9, 5, 5, 5, 7,
    10, 10, 10, 10, 10, 10, 10, 7, 7, 4, 5, 7, 10, 3, 5, 5, 7, 7, 6, 7, 5, 5, 4, 4, 11, 2, 4, 4, 11, 1,
    4, 3, 2, 2, 2, 4, 7, 10, 7, 10, 7, 4, 4, 3, 3, 4, 7, 7, 4, 1, 1, 1, 1, 1, 12, 2, 2, 2, 2, 2,
    2, 2, 2, 12, 1, 1, 7, 3, 3, 1, 1, 1, 7, 1, 4, 1, 1, 5, 10, 5, 1, 8, 9, 1, 2, 3, 3, 2, 3, 10,
    11, 12, 2, 2, 2, 7, 7, 3, 6, 1, 7, 1, 7, 7, 4, 8, 8, 7, 8, 8, 8, 7, 9, 7, 4, 1, 8, 8, 4, 8,
    8, 8, 8, 4, 8, 8, 4, 9, 8, 8, 8, 8, 8, 9, 7, 1, 5, 5, 7, 10, 1, 5, 10, 5, 5, 10, 5, 10, 10, 7,
    7, 4, 5, 7, 10, 3, 5, 5, 5, 10, 7, 7, 6, 7, 7, 4, 2, 4, 4, 11, 2, 2, 1, 2, 1, 3, 3, 4, 2, 2,
    2, 4, 2, 10, 10, 8, 4, 3, 3, 7, 4, 3, 3, 1, 1, 1, 2, 12, 12, 2, 2, 2, 2, 2, 2, 2, 12, 1, 1, 7,
    7, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 10, 10, 1, 5, 5, 13, 13, 14, 14, 13, 13, 15, 15,
], dtype=np.int8)
FRANKFURT_NETWORK_NAMES = [
    'Default', 'Visual', 'FrontoParietal', 'DorsalAttention', 'VentralAttn',
    'Salience', 'CinguloOpercular', 'SomatomotorDorsal', 'SomatomotorLateral',
    'Auditory', 'ParietalMemoryNetwork', 'Temporal', 'BasalGanglia',
    'Thalamus', 'Cerebellum',
]


def load_frankfurt_data(
    ts_directory=None,
    hyp_directory=None,
    partecipants=None,
    parcellation="large",
):
    root = data_root() / "frankfurt"
    ts_directory = ts_directory or str(root / "wheelock_ts")
    hyp_directory = hyp_directory or str(root / "wheelock_hypnogram")
    partecipants = partecipants or str(root / "participants.csv")
    exclude_patients = {"20091007DR", "20081125AE"}
    _NETWORK_ID = FRANKFURT_NETWORK_ID
    _NETWORK_NAMES = FRANKFURT_NETWORK_NAMES

    participants_df = pd.read_csv(partecipants)

    ts_files = glob.glob(os.path.join(ts_directory, "*.mat"))
    patients_data = []
    for mat_file_path in ts_files:
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            filename = os.path.basename(mat_file_path)
            patient_id = filename.replace("_ptseries_294.mat", "")
            key = patient_id.replace("wheelock_ts", "")
            participant_info = participants_df[participants_df["key"] == key]

            ts = mat_data["all_rois"]
            if ts.ndim == 2 and ts.shape[0] == 294 and ts.shape[1] != 294:
                ts = ts.T

            patient_dict = {
                "patient_id": patient_id,
                "key": key,
                "ts": ts,
            }

            if not participant_info.empty:
                patient_dict["age"] = participant_info["age"].values[0]
                patient_dict["gender"] = participant_info["sex"].values[0]
                patient_dict["participant_id"] = participant_info["participant_id"].values[0]
                patient_dict["include"] = participant_info["include"].values[0]
            else:
                patient_dict["age"] = None
                patient_dict["gender"] = None
                patient_dict["participant_id"] = None
                patient_dict["include"] = None

            patients_data.append(patient_dict)
        except Exception:
            pass

    hyp_files = glob.glob(os.path.join(hyp_directory, "*AASM_hypnogram.txt"))
    hypnogram_data = {}
    for hyp_file_path in hyp_files:
        try:
            filename = os.path.basename(hyp_file_path)
            patient_id = filename.replace("_AASM_hypnogram.txt", "")
            hypnogram = np.loadtxt(hyp_file_path, dtype=float)
            hypnogram_data[patient_id] = hypnogram
        except Exception:
            pass

    def collapse_to_networks(ts_294):
        n_nets = int(_NETWORK_ID.max())  # 15
        out = np.empty((ts_294.shape[0], n_nets), dtype=ts_294.dtype)
        for nid in range(1, n_nets + 1):
            mask = (_NETWORK_ID == nid)
            out[:, nid - 1] = ts_294[:, mask].mean(axis=1)
        return out

    merged_data = []
    for patient in patients_data:
        patient_id = patient["patient_id"]
        if patient_id in hypnogram_data and patient_id not in exclude_patients:
            if patient["include"] == 1:
                ts = patient["ts"][5:]
                if parcellation == "small":
                    ts = collapse_to_networks(ts)

                merged_patient = {
                    "patient_id": patient_id,
                    "age": patient["age"],
                    "gender": patient["gender"],
                    "ts": ts,
                    "hypnogram": hypnogram_data[patient_id],
                }
                merged_data.append(merged_patient)

    return merged_data


def load_adhd_data(root_dir=None, parcellation="cc200", target="inattentive"):
    root_dir = root_dir or str(data_root() / "adhd")
    pheno_path = os.path.join(root_dir, "adhd200_preprocessed_phenotypics.tsv")

    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Metadata not found at {pheno_path}")

    pheno = pd.read_csv(pheno_path, sep='\t')
    pheno['ScanDir ID'] = pheno['ScanDir ID'].astype(str)
    pheno = pheno.set_index('ScanDir ID')
    inatt = pd.to_numeric(pheno['Inattentive'], errors='coerce')

    # 'yeo7'/'yeo17' load cc200 underneath then collapse to N Yeo networks.
    yeo_n = {'yeo7': 7, 'yeo17': 17}.get(parcellation)
    on_disk = 'cc200' if yeo_n else parcellation
    search_pattern = os.path.join(root_dir, on_disk, "**", f"sfnwmrda*_{on_disk}_TCs.1D")
    all_files = glob.glob(search_pattern, recursive=True)

    if yeo_n:
        from real_data.utils.parcellation_mapping import compute_cc200_to_yeo_mapping
        _yeo_mapping = compute_cc200_to_yeo_mapping(source='adhd', n_networks=yeo_n)
    else:
        _yeo_mapping = None

    dataset = []

    print(f"Scanning {len(all_files)} potential files...")

    for filepath in all_files:
        filename = os.path.basename(filepath)

        if filename.startswith('.'):
            continue

        parent_folder = os.path.dirname(filepath)
        subj_id_raw = os.path.basename(parent_folder)

        if subj_id_raw in pheno.index:
            target_id = subj_id_raw
        elif subj_id_raw.lstrip('0') in pheno.index:
            target_id = subj_id_raw.lstrip('0')
        else:
            continue

        score = inatt.get(target_id, np.nan)
        valid_score = np.isfinite(score) and score not in (-999, -9999) and 0 <= score <= 200
        # Scans without a valid inattention score are dropped only when it is the target;
        # other targets are validated per scan when the windows are built.
        if target == "inattentive" and not valid_score:
            continue

        try:
            with open(filepath, 'r') as f:
                header = f.readline().split()
                total_cols = len(header)

            ts_data = np.loadtxt(filepath, skiprows=1, usecols=range(2, total_cols))

            if _yeo_mapping is not None:
                # Per-file ROI ids parsed from `Mean_N` headers (PCP/Athena
                # drops empty cc200 ROIs, so the column count varies).
                roi_ids = [int(name.split('_', 1)[1]) for name in header[2:]]
                out = np.zeros((ts_data.shape[0], yeo_n), dtype=ts_data.dtype)
                for nid in range(1, yeo_n + 1):
                    cols = [c for c, rid in enumerate(roi_ids)
                            if 1 <= rid <= len(_yeo_mapping)
                            and _yeo_mapping[rid - 1] == nid]
                    if cols:
                        out[:, nid - 1] = ts_data[:, cols].mean(axis=1)
                ts_data = out

            run_id = "rest_1" if "rest_1" in filename else ("rest_2" if "rest_2" in filename else "unknown")

            entry = {
                "subject_id": target_id,
                "run_id": run_id,
                "ts": ts_data,
                "diagnosis": pheno.loc[target_id, "DX"],
                "inattentive": float(score),
                "age": pheno.loc[target_id, "Age"],
                "gender": pheno.loc[target_id, "Gender"],
                "iq": pheno.loc[target_id, "Full4 IQ"],
                "site": pheno.loc[target_id, "Site"]
            }

            dataset.append(entry)

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    scores = np.array([e['inattentive'] for e in dataset])
    scores = scores[np.isfinite(scores)]
    n_subj = len({e['subject_id'] for e in dataset})
    print(f"Loaded {len(dataset)} scans across {n_subj} unique subjects (target {target}).")
    if scores.size:
        print(f"Inattentive: min={scores.min():.2f}  max={scores.max():.2f}  "
              f"mean={scores.mean():.2f}  std={scores.std():.2f}")
    return dataset


def drop_degenerate_rois(dataset, min_frac=0.01):
    """Drop ROIs that are zero-variance in at least `min_frac` of scans, then drop
    the scans still carrying one.

    A constant column makes np.corrcoef emit NaN, and build_entries discards any
    subject whose correlation matrix is not all-finite. On ABIDE rois_ho that costs
    258/1035 scans, concentrated in a few sites (Caltech 37->3, SDSU 36->9), while
    237 of the 450 bad columns are a single ROI. Excluding the few poorly-covered
    ROIs keeps both the cohort and the site balance.
    """
    if not dataset:
        return dataset
    ts = [np.asarray(e['ts']) for e in dataset]
    const = np.stack([t.min(axis=0) == t.max(axis=0) for t in ts])
    keep = const.mean(axis=0) < min_frac
    out = [{**e, 'ts': t[:, keep]}
           for e, t, bad in zip(dataset, ts, const) if not bad[keep].any()]
    print(f"drop_degenerate_rois(min_frac={min_frac}): ROIs {const.shape[1]} -> "
          f"{int(keep.sum())}, scans {len(dataset)} -> {len(out)}")
    return out


def load_abide_data(
    data_dir=None,
    parcellation="rois_ho",
    roi_drop_frac=0.01,
):
    data_dir = data_dir or str(data_root() / "abide")
    # 'yeo7'/'yeo17' are derived parcellations: load cc200 then collapse columns
    # to N Yeo networks via the cached cc200->YeoN mapping.
    yeo_n = {'yeo7': 7, 'yeo17': 17}.get(parcellation)
    derivative = 'rois_cc200' if yeo_n else parcellation

    data = fetch_abide_pcp(
        data_dir=data_dir,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=[derivative],
        n_subjects=None,
        quality_checked=False,
    )

    ts_list = getattr(data, derivative)
    pheno = data.phenotypic.reset_index(drop=True)

    if yeo_n:
        from real_data.utils.parcellation_mapping import (
            compute_cc200_to_yeo_mapping, collapse_cc200_to_yeo,
        )
        mapping = compute_cc200_to_yeo_mapping(source='abide', n_networks=yeo_n)
        ts_list = [collapse_cc200_to_yeo(ts, mapping, n_networks=yeo_n) for ts in ts_list]

    dataset = []
    for i in range(len(ts_list)):
        entry = {"ts": ts_list[i]}
        for col in pheno.columns:
            entry[col] = pheno.iloc[i][col]
        dataset.append(entry)

    if roi_drop_frac is not None:
        dataset = drop_degenerate_rois(dataset, roi_drop_frac)

    return dataset


# ------- High-level Data Functions -------

def load_dataset(args):
    from real_data.utils.config import DATASETS
    config = DATASETS[args.dataset]
    loader_kwargs = {}
    for param in config['loader_params']:
        value = getattr(args, param, None)
        if value is None:
            value = config['defaults'].get(param)
            setattr(args, param, value)
        loader_kwargs[param] = value

    parcellation = loader_kwargs.get('parcellation')
    valid = config.get('valid_parcellations')
    if parcellation is not None and valid is not None and parcellation not in valid:
        raise ValueError(
            f"Parcellation '{parcellation}' is not valid for dataset '{args.dataset}'. "
            f"Valid options: {valid}"
        )

    return config['loader'](**loader_kwargs)


def preprocess_data(raw_data, args, model_config):
    """Window every subject via the shared windowing and build flat features.
    Keeps the per-window correlations, the cyclic circle partner, and the full-
    time-series correlations so the adjacency variants can be built per split."""
    from real_data.utils.config import DATASETS
    subject_key = DATASETS[args.dataset]['subject_key']
    spec = resolve_target(args.dataset, args.target, args.task)
    args.eval_level = spec['eval_level']

    whole_scan = args.time == 'max'
    require_partner = not (whole_scan or getattr(args, 'no_partner', False))
    if not require_partner and getattr(args, 'adjacency', None) == 'circle':
        raise ValueError("--adjacency circle needs a partner window; it cannot be "
                         "combined with --time max or --no_partner.")

    if whole_scan:
        entries = build_entries(raw_data, 1, 1.0, spec, subject_key, require_partner=False)
        if not entries:
            raise ValueError("No subjects with a valid scan and target.")
        C_main, C_partner, y, groups, entry_idx = whole_scan_windows(entries)
    else:
        T = int(args.time)
        T_anchor = int(args.cohort_source_T) if args.cohort_source_T else T
        if n_mode_inert(T, T_anchor):
            print(f"WARNING: --time {T} >= cohort anchor {T_anchor}, so 'fixed' and 'free' select "
                  f"the same windows and --n_mode {args.n_mode} has no effect. Pass "
                  f"--cohort_source_T > {T} if the two modes are meant to differ.")

        entries = build_entries(raw_data, T_anchor, args.stride_frac, spec, subject_key,
                                require_partner=require_partner)
        if not entries:
            raise ValueError("No subjects survived the cohort filter "
                             "(need >= 2 windows at the anchor T).")
        rng = np.random.default_rng(args.seed)
        C_main, C_partner, y, groups, entry_idx = windows_for_T(
            entries, T, args.stride_frac, args.n_mode, rng, require_partner=require_partner)
    C_full = torch.from_numpy(np.stack([m['C_full'] for m in entries]))

    print(f"Total windows: {len(y)}, Unique subjects: {len(np.unique(groups))}")

    data = {
        'C': C_main.numpy().astype(np.float32),
        'C_partner': C_partner.numpy().astype(np.float32),
        'C_full': C_full,
        'entry_idx': entry_idx,
        'y': y.astype(np.float32),
        'groups': groups,
        'X': None,
        'A': None,
    }
    if model_config['needs_flatten']:
        data['X'] = flatten_lower_offdiag(data['C']).astype(np.float32)
    return data


# ------- Train/Val/Test Splitting -------

def split_data(data, args):
    groups = data['groups']

    train_idx, val_idx, test_idx = grouped_split(groups, args.seed)

    def subset(arr, idx):
        return arr[idx] if arr is not None else None

    keys = ('C', 'C_partner', 'X', 'y', 'groups', 'entry_idx')
    splits = {
        name: {k: subset(data.get(k), idx) for k in keys}
        for name, idx in (('train', train_idx), ('val', val_idx), ('test', test_idx))
    }

    for name, split in splits.items():
        n_subj = len(np.unique(split['groups']))
        print(f"{name.capitalize()}: {len(split['y'])} windows, {n_subj} subjects")

    return splits
