"""Canonical adjacency construction shared by every real-data experiment.

All variants threshold a correlation matrix to its top-tau most-positive off-diagonal
edges via `adjacency_matrix` (the paper's 1{C > delta}). The variants differ only in
WHICH correlation matrix supplies the graph:

  hat    same window's C            (double-dipping pipeline)
  circle cyclic partner window's C  (independent estimate, no double-dipping)
  note   train-set mean full-TS C   (fixed population graph A_0)
  bar    this subject's full-TS C   (lowest-noise per-subject graph)

NOTE: on real data `bar` uses the full time series (all L samples) as a proxy for the
paper's oracle Abar=1{Sigma>delta}; the TRUE Sigma is unknown, so `bar` here is the
best available estimate, NOT the simulation's true-Sigma oracle.
"""
from utils.utils import adjacency_matrix, sage_mean

VARIANTS = ('hat', 'circle', 'note', 'bar')


def variant_adjacency(variant, tau, *, C_main=None, C_partner=None,
                      C_full=None, entry_idx=None, train_entry_ids=None):
    """Adjacency for one variant. Returns (N,D,D) per-window, except `note` which
    returns a single (D,D) graph (broadcast over windows by `sage_mean`)."""
    if variant == 'hat':
        return adjacency_matrix(C_main, top_pct=tau)
    if variant == 'circle':
        return adjacency_matrix(C_partner, top_pct=tau)
    if variant == 'bar':
        return adjacency_matrix(C_full, top_pct=tau)[entry_idx]
    if variant == 'note':
        C_note_mean = C_full[train_entry_ids].mean(dim=0)
        return adjacency_matrix(C_note_mean, top_pct=tau)
    raise ValueError(f"unknown variant {variant!r}")
