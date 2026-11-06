"""Windowing + cohort filtering shared by every real-data experiment.

Cohort filter: an entry must yield K >= 2 windows at T_max (so every window has a
cyclic circle partner). n_mode='fixed' keeps a constant per-entry window count K
(anchored at T_max) across all T; 'free' uses every available window at each T.
The circle partner is the cyclic next window (w+1)%K_sel.

`require_partner=False` drops the partner requirement (an entry then needs a
single window), for runs that never build the `circle` graph; `whole_scan_windows`
turns every entry into one window spanning its whole scan.
"""
import numpy as np
import torch

from real_data.shared.targets import WINDOW_FNS


def stride_at(T, stride_frac):
    return max(1, int(round(stride_frac * T)))


def n_mode_inert(T, T_anchor):
    """True when 'fixed' and 'free' provably select the same windows.

    K is fixed at the anchor, so at T >= T_anchor every entry has n_avail == K
    and both modes take every available window -- `--n_mode` then has no effect
    and the two cells are exact duplicates. The benchmark anchors at T
    itself unless --cohort_source_T is given, so n_mode then does nothing at ANY T. Anchor strictly above every T to keep both modes meaningful.
    """
    return T >= T_anchor


def partner_shift(T, stride):
    """Index offset from a window to its nearest guaranteed non-overlapping partner.

    Windows sit at sorted indices sel[w], so sel[w+shift] - sel[w] >= shift and the
    position gap is >= shift*stride >= T -- this holds even when n_mode='fixed'
    samples an irregular subset. The cyclic wrap gap is (K-shift)*stride, so every
    window has a non-overlapping partner iff K >= 2*shift.

    At stride_frac=1.0 the stride is T, giving shift=1 and K>=2: identical to the
    original cyclic-next rule. Below 1.0 the next window overlaps and shift grows.
    """
    return -(-T // stride)


def build_entries(data, T_max, stride_frac, spec, subject_key, require_partner=True):
    s_max = stride_at(T_max, stride_frac)
    shift_max = partner_shift(T_max, s_max)
    min_windows = 2 * shift_max if require_partner else 1
    sentinels = set(spec.get('sentinels', ()))
    binarize = spec.get('binarize')
    entries = []
    for e in data:
        ts = e.get('ts')
        if ts is None:
            continue
        L = ts.shape[0]
        K = (L - T_max) // s_max + 1 if L >= T_max else 0
        if K < min_windows:
            continue
        C_full = np.corrcoef(ts, rowvar=False)
        if not np.all(np.isfinite(C_full)):
            continue
        m = dict(ts=ts, sid=e[subject_key], C_full=C_full.astype(np.float32),
                 K=K, kind=spec['kind'])
        if spec['kind'] == 'column':
            try:
                yv = float(e.get(spec['key']))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(yv) or yv in sentinels:
                continue
            m['y'] = float(binarize(yv)) if binarize else yv
        else:
            m['hypnogram'] = np.asarray(e['hypnogram'])
            m['win_fn'] = WINDOW_FNS[spec['fn']]
        entries.append(m)
    return entries


def windows_for_T(entries, T, stride_frac, n_mode, rng, require_partner=True):
    s = stride_at(T, stride_frac)
    shift = partner_shift(T, s)
    min_windows = 2 * shift if require_partner else 1
    C_main, C_partner, y, groups, entry_idx = [], [], [], [], []
    for ei, m in enumerate(entries):
        L = m['ts'].shape[0]
        n_avail = (L - T) // s + 1 if L >= T else 0
        if n_mode == 'fixed':
            K = m['K']
            sel = np.arange(K) if n_avail == K else np.sort(rng.choice(n_avail, K, replace=False))
        else:
            sel = np.arange(n_avail)
        if len(sel) < min_windows:
            continue
        Cs, yw = [], []
        ok = True
        for i in sel:
            start = int(i) * s
            C = np.corrcoef(m['ts'][start:start + T], rowvar=False)
            if not np.all(np.isfinite(C)):
                ok = False
                break
            Cs.append(C.astype(np.float32))
            yw.append(m['win_fn'](m, start, T) if m['kind'] == 'window' else m['y'])
        if not ok:
            continue
        K_sel = len(Cs)
        for w in range(K_sel):
            pw = (w + shift) % K_sel if require_partner else w
            C_main.append(Cs[w])
            C_partner.append(Cs[pw])
            y.append(yw[w])
            groups.append(m['sid'])
            entry_idx.append(ei)
    return (torch.from_numpy(np.stack(C_main)),
            torch.from_numpy(np.stack(C_partner)),
            np.asarray(y, dtype=float),
            np.asarray(groups),
            np.asarray(entry_idx))


def whole_scan_windows(entries):
    """One window per entry covering its entire scan (`--time max`).

    No independent partner exists, so C_partner repeats C_main and the `circle`
    graph cannot be built from it.
    """
    C, y = [], []
    for m in entries:
        C.append(m['C_full'])
        y.append(m['win_fn'](m, 0, m['ts'].shape[0]) if m['kind'] == 'window' else m['y'])
    C = torch.from_numpy(np.stack(C))
    return (C, C.clone(), np.asarray(y, dtype=float),
            np.asarray([m['sid'] for m in entries]), np.arange(len(entries)))
