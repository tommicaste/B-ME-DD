import numpy as np
import torch

from utils.utils import adjacency_matrix
from real_data.shared.adjacency import variant_adjacency

# benchmark --adjacency name -> paper/simulation variant. 'none' has no variant: it is the
# empty graph, i.e. the no-convolution control (see apply_adjacency_strategy).
_ADJ_TO_VARIANT = {'window': 'hat', 'bar': 'bar', 'zero': 'note', 'circle': 'circle'}


def _cov_to_corr(s):
    d = np.sqrt(np.diag(s))
    d = np.where(d > 0, d, 1.0)
    return s / np.outer(d, d)


def create_adjacency_matrices(X, tau=0.1, return_thresholds=False):
    """Top-tau most-positive off-diagonal edges, via the shared canonical
    `adjacency_matrix`, the same thresholding the simulations use.
    Reported threshold = the smallest selected edge value."""
    A_list, thresholds = [], []
    for x in X:
        x = np.asarray(x)
        A = adjacency_matrix(torch.as_tensor(x, dtype=torch.float32),
                             top_pct=tau).to(torch.int8).numpy()
        A_list.append(A)
        iu = np.triu_indices(x.shape[0], k=1)
        sel = A[iu] > 0
        thresholds.append(float(np.min(x[iu][sel])) if sel.any() else float('inf'))
    if return_thresholds:
        return A_list, thresholds
    return A_list


def apply_adjacency_strategy(data, splits, args):
    """Build per-split adjacency for the chosen strategy via the shared
    variant_adjacency:
      window=hat (same window), bar=subject full-TS, zero=note (train-mean
      population graph), circle=cyclic partner window.

    `none` is the no-convolution control: the empty graph. Every node keeps only
    its own row (SAGE via its root weight, GCN/GAT via the self-loops they add),
    so the architecture, readout, optimizer and capacity are unchanged and the
    ONLY thing removed is message passing. It is the sole comparison that
    isolates the convolution itself rather than contrasting two graphs. `tau` is
    ignored here, which is why this is an adjacency level and not tau=0.
    """
    if args.adjacency == 'none':
        D = splits['train']['C'].shape[-1]
        for sp in splits.values():
            sp['A'] = np.zeros((len(sp['y']), D, D), dtype=np.float32)
        return splits

    variant = _ADJ_TO_VARIANT[args.adjacency]
    C_full = data['C_full']
    train_entry_ids = np.unique(splits['train']['entry_idx'])
    for sp in splits.values():
        A = variant_adjacency(
            variant, args.tau,
            C_main=torch.as_tensor(sp['C']),
            C_partner=torch.as_tensor(sp['C_partner']),
            C_full=C_full, entry_idx=sp['entry_idx'],
            train_entry_ids=train_entry_ids,
        )
        if A.dim() == 2:                       # note: one population graph -> per window
            A = A.unsqueeze(0).expand(len(sp['y']), -1, -1).contiguous()
        sp['A'] = A.to(torch.float32).cpu().numpy()
    return splits
