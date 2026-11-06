"""Target resolution shared by every real-data experiment.

A target is either a per-subject column (adhd/abide phenotype -> per-subject eval)
or a per-window function (frankfurt sleep_frac/sleep_median -> per-window eval).
"""
import numpy as np


DEFAULT_TARGET = {
    'regression':     {'adhd': 'inattentive', 'abide': 'AGE_AT_SCAN', 'frankfurt': 'sleep_frac'},
    'classification': {'adhd': 'diagnosis',   'abide': 'DX_GROUP',    'frankfurt': 'sleep_median'},
}
SENTINELS = (-9999.0, -999.0)

# classification binarizers for per-subject column targets (others default to !=0)
CLF_BINARIZE = {
    ('adhd', 'diagnosis'): lambda v: float(v != 0),     # control(0) vs any ADHD
    ('abide', 'DX_GROUP'): lambda v: float(v == 1),     # autism(1) vs control(2)
}


def sleep_frac_window(entry, start, T):
    """Regression target: fraction of the window asleep (mean binarized stage)."""
    hyp = entry['hypnogram']
    return float(np.mean(hyp[start:start + T] != 0))


def sleep_median_window(entry, start, T):
    """Classification target: majority (median) binarized stage in the window (0/1)."""
    hyp = entry['hypnogram']
    return float(np.median((hyp[start:start + T] != 0).astype(float)) >= 0.5)


WINDOW_FNS = {'sleep_frac': sleep_frac_window, 'sleep_median': sleep_median_window}


def resolve_target(dataset, name, task):
    """Map a (dataset, target, task) to a spec {kind, key|fn, eval_level, ...}.

    Frankfurt named window-targets -> per-window kind. Otherwise a per-subject
    column lookup (any numeric column works for adhd/abide); classification adds
    a binarizer for the column.
    """
    if dataset == 'frankfurt' and name in WINDOW_FNS:
        return dict(kind='window', fn=name, eval_level='window')
    spec = dict(kind='column', key=name, eval_level='subject', sentinels=SENTINELS)
    if task == 'classification':
        spec['binarize'] = CLF_BINARIZE.get((dataset, name), lambda v: float(v != 0))
    return spec
