"""Shared real-data prep used by every real-data experiment:
windowing/cohort, target resolution, and the canonical adjacency variants."""
from real_data.shared.windowing import stride_at, build_entries, windows_for_T
from real_data.shared.targets import (
    DEFAULT_TARGET, CLF_BINARIZE, WINDOW_FNS, SENTINELS, resolve_target,
)
from real_data.shared.adjacency import (
    adjacency_matrix, sage_mean, variant_adjacency, VARIANTS,
)

__all__ = [
    'stride_at', 'build_entries', 'windows_for_T',
    'DEFAULT_TARGET', 'CLF_BINARIZE', 'WINDOW_FNS', 'SENTINELS', 'resolve_target',
    'adjacency_matrix', 'sage_mean', 'variant_adjacency', 'VARIANTS',
]
