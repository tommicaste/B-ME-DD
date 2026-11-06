import math
import torch
import numpy as np
from typing import Optional, Tuple
from torch.distributions import Dirichlet

from utils.utils import is_correlation_matrix, make_pd, flatten_offdiag, adjacency_matrix  # noqa: F401

try:
    from .logistic_convmix_equicorr import convmix_equi
    from .logistic_convmix_banded import convmix_banded
    from .logistic_convmix_block import convmix_block
except ImportError:
    convmix_equi = convmix_banded = convmix_block = None


def generate_logistic_corr(
    n: int,
    input_dim: int,
    matrix_type='convmix_equicorr',
    device=None,
    dtype=torch.float32,
    random_state=None,
    **kwargs,
):
    if matrix_type == 'equicorr':
        n0 = kwargs.get('n0', n // 2)
        n1 = kwargs.get('n1', n - n0)
        rho0 = kwargs.get('rho0', 0.20)
        eps = kwargs.get('eps', 0.25)
        df = kwargs.get('df', max(input_dim, 20))
        A_target = kwargs.get('A_target', 0.75)
        seed = random_state if random_state is not None else 123
        C, y, beta_pop = convmix_equi(
            p=input_dim, n0=n0, n1=n1, rho0=rho0, eps=eps,
            df=int(df), A_target=A_target,
            calib_trials=kwargs.get('calib_trials', 300_000),
            calib_batch=kwargs.get('calib_batch', 4096),
            grid_size=kwargs.get('grid_size', 2001),
            gen_batch=kwargs.get('gen_batch', 2048),
            cov_trials=kwargs.get('cov_trials', 50_000),
            cov_batch=kwargs.get('cov_batch', 2048),
            ridge=kwargs.get('ridge', 1e-6),
            dtype=dtype, device=device, seed=seed,
            shuffle=kwargs.get('shuffle', False),
        )
        if y.dim() > 1:
            y = y.view(-1).to(torch.long)
        true_beta = beta_pop.unsqueeze(0).detach().cpu().numpy()
        return C, y, true_beta

    elif matrix_type == 'block':
        n0 = kwargs.get('n0', n // 2)
        n1 = kwargs.get('n1', n - n0)
        rho0 = kwargs.get('rho0', 0.20)
        eps = kwargs.get('eps', 0.25)
        df = kwargs.get('df', max(input_dim, 20))
        A_target = kwargs.get('A_target', 0.75)
        seed = random_state if random_state is not None else 123
        C, y, beta_pop = convmix_block(
            p=input_dim, n0=n0, n1=n1, rho0=rho0, eps=eps,
            df=int(df), A_target=A_target,
            n_blocks=kwargs.get('n_blocks', 4),
            blocks=kwargs.get('blocks', None),
            calib_trials=kwargs.get('calib_trials', 300_000),
            calib_batch=kwargs.get('calib_batch', 4096),
            grid_size=kwargs.get('grid_size', 2001),
            gen_batch=kwargs.get('gen_batch', 2048),
            cov_trials=kwargs.get('cov_trials', 50_000),
            cov_batch=kwargs.get('cov_batch', 2048),
            ridge=kwargs.get('ridge', 1e-6),
            dtype=dtype, device=device, seed=seed,
            shuffle=kwargs.get('shuffle', False),
        )
        if y.dim() > 1:
            y = y.view(-1).to(torch.long)
        true_beta = beta_pop.unsqueeze(0).detach().cpu().numpy()
        return C, y, true_beta

    elif matrix_type == 'banded':
        n0 = kwargs.get('n0', n // 2)
        n1 = kwargs.get('n1', n - n0)
        rho0 = kwargs.get('rho0', 0.20)
        eps = kwargs.get('eps', 0.25)
        df = kwargs.get('df', max(input_dim, 20))
        A_target = kwargs.get('A_target', 0.75)
        seed = random_state if random_state is not None else 123
        C, y, beta_pop = convmix_banded(
            p=input_dim, n0=n0, n1=n1, rho0=rho0, eps=eps,
            df=int(df), A_target=A_target,
            band_q=kwargs.get('band_q', 3),
            calib_trials=kwargs.get('calib_trials', 300_000),
            calib_batch=kwargs.get('calib_batch', 4096),
            grid_size=kwargs.get('grid_size', 2001),
            gen_batch=kwargs.get('gen_batch', 2048),
            cov_trials=kwargs.get('cov_trials', 50_000),
            cov_batch=kwargs.get('cov_batch', 2048),
            ridge=kwargs.get('ridge', 1e-6),
            dtype=dtype, device=device, seed=seed,
            shuffle=kwargs.get('shuffle', False),
        )
        if y.dim() > 1:
            y = y.view(-1).to(torch.long)
        true_beta = beta_pop.unsqueeze(0).detach().cpu().numpy()
        return C, y, true_beta
    else:
        raise ValueError(f"Unsupported matrix_type: '{matrix_type}'. Supported types: 'equicorr', 'block', 'banded'")


def generate_y(
    X: torch.Tensor,
    snr_ratio: float = 5.0,
    beta0: float = 0.0,
    seed: int | None = None,
    beta_type: str = "random",
    sparsity: float = 0.1,
    labels: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    y = beta0 + X @ beta + eps where Var(X @ beta) / Var(eps) = snr_ratio.

    Returns:
        y          (n,) tensor
        beta_full  (d+1,) tensor [beta0, beta]
        sigma      noise standard deviation used
    """
    if X.dim() != 2:
        raise ValueError("X must have shape (n, d).")
    n, d = X.shape
    if n < 2:
        raise ValueError("need n >= 2.")
    if snr_ratio <= 0:
        raise ValueError("snr_ratio must be positive.")
    device, dtype = X.device, X.dtype
    if seed is not None:
        torch.manual_seed(int(seed))

    if beta_type == "random":
        beta = torch.randn(d, device=device, dtype=dtype)
    elif beta_type in ("community_aligned", "off_community_aligned"):
        p_float = math.sqrt(d)
        if not p_float.is_integer():
            raise ValueError(f"beta_type={beta_type} requires d=p^2; got d={d}")
        p = int(p_float)
        if labels is None:
            raise ValueError(f"beta_type={beta_type} requires labels (p,) to be passed")
        lbl = labels.to(device=device)
        same = (lbl.view(-1, 1) == lbl.view(1, -1))  # p×p
        off_diag = ~torch.eye(p, dtype=torch.bool, device=device)
        if beta_type == "community_aligned":
            mask = same & off_diag
        else:
            mask = (~same) & off_diag
        beta = mask.flatten().to(dtype)
    elif beta_type in ("ones", "constant"):
        beta = torch.ones(d, device=device, dtype=dtype)
    elif beta_type == "single_community_aligned":
        p_float = math.sqrt(d)
        if not p_float.is_integer():
            raise ValueError(f"beta_type={beta_type} requires d=p^2; got d={d}")
        p = int(p_float)
        if labels is None:
            raise ValueError(f"beta_type={beta_type} requires labels (p,) to be passed")
        lbl = labels.to(device=device)
        unique_labels = torch.unique(lbl)
        valid = [int(c.item()) for c in unique_labels
                 if int((lbl == c).sum().item()) >= 2]
        if not valid:
            raise ValueError(f"beta_type={beta_type} requires a community with >= 2 nodes")
        comm_idx = int(torch.randint(len(valid), (1,), device=device).item())
        comm_label = valid[comm_idx]
        in_comm = (lbl == comm_label)
        same = in_comm.view(-1, 1) & in_comm.view(1, -1)
        off_diag = ~torch.eye(p, dtype=torch.bool, device=device)
        mask = same & off_diag
        beta = mask.flatten().to(dtype)
    elif beta_type in ("edge_specific", "within_community_contrast"):
        p_float = math.sqrt(d)
        if not p_float.is_integer():
            raise ValueError(f"beta_type={beta_type} requires d=p^2; got d={d}")
        p = int(p_float)
        if labels is None:
            raise ValueError(f"beta_type={beta_type} requires labels (p,) to be passed")
        lbl = labels.to(device=device)
        unique_labels = torch.unique(lbl)
        valid = [int(c.item()) for c in unique_labels
                 if int((lbl == c).sum().item()) >= 2]
        if not valid:
            raise ValueError(f"beta_type={beta_type} requires a community with >= 2 nodes")
        comm_idx = int(torch.randint(len(valid), (1,), device=device).item())
        comm_label = valid[comm_idx]
        members = torch.nonzero(lbl == comm_label, as_tuple=False).flatten()
        perm = torch.randperm(members.numel(), device=device)
        a = int(members[perm[0]].item())
        b = int(members[perm[1]].item())
        B = torch.zeros((p, p), device=device, dtype=dtype)
        if beta_type == "edge_specific":
            B[a, b] = 1.0
            B[b, a] = 1.0
        else:
            u = torch.zeros(p, device=device, dtype=dtype)
            u[a] = 1.0
            u[b] = -1.0
            v = (lbl == comm_label).to(dtype=dtype)
            if int(v.sum().item()) > 2:
                v[a] = 0.0
                v[b] = 0.0
            B = 0.5 * (u[:, None] * v[None, :] + v[:, None] * u[None, :])
        beta = B.flatten()
    else:
        raise ValueError(f"Unknown beta_type: {beta_type}")

    beta = beta / torch.norm(beta)

    s = X @ beta
    v_s = torch.var(s, unbiased=True)
    if not torch.isfinite(v_s) or v_s <= 0:
        sigma = torch.tensor(1e-6, device=device, dtype=dtype)
    else:
        sigma = torch.sqrt(v_s / torch.as_tensor(snr_ratio, device=device, dtype=dtype))
    eps = sigma * torch.randn(n, device=device, dtype=dtype)
    y = torch.as_tensor(beta0, device=device, dtype=dtype) + s + eps
    beta_full = torch.cat([torch.as_tensor([beta0], device=device, dtype=dtype), beta])
    return y, beta_full, sigma


def make_C_dirichlet(p: int, k: int, alpha_main: float = 5.0, alpha_other: float = 1.0) -> torch.Tensor:
    """Dirichlet mixed-membership latent community matrix; returns (p, p) Pearson correlation of memberships."""
    if p < 1 or k < 1:
        raise ValueError("p and k must be positive")
    if k < 2:
        raise ValueError("k must be at least 2 for signed correlations")
    if k > p:
        raise ValueError("k cannot exceed p")
    L0 = torch.zeros((p, k))
    base_size = p // k
    remainder = p % k
    sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]
    start = 0
    for j, s in enumerate(sizes):
        end = start + s
        alpha = torch.full((k,), alpha_other)
        alpha[j] = alpha_main
        samples = torch.distributions.Dirichlet(alpha).sample((s,))
        L0[start:end] = samples
        start = end
    X = L0 - L0.mean(dim=1, keepdim=True)
    X = X / (L0.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-12))
    C = (X @ X.T) / (k - 1)
    return C


def generate_mmsbm_samples(
    n,
    p,
    K,
    alpha_0=1.0,
    mu_within=0.8,
    community_division="equally",
    community_allocation=None,
    mu_type="equally",
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)

    if community_division == "equally":
        assert p % K == 0, f"p={p} must be divisible by K={K}"
        assigned = torch.arange(K).repeat_interleave(p // K)
    else:
        assert community_allocation is not None
        assert len(community_allocation) == K
        counts = [int(round(f * p)) for f in community_allocation]
        counts[-1] = p - sum(counts[:-1])
        assigned = torch.cat([torch.full((c,), k, dtype=torch.long) for k, c in enumerate(counts)])

    remaining = 1.0 - mu_within
    if mu_type == "equally":
        mu = torch.full((K, K), remaining / (K - 1))
        mu.fill_diagonal_(mu_within)
    elif mu_type == "random":
        mu = torch.zeros(K, K)
        mu.fill_diagonal_(mu_within)
        for k in range(K):
            weights = torch.rand(K - 1)
            weights = weights / weights.sum() * remaining
            mask = torch.arange(K) != k
            mu[k, mask] = weights
    else:
        raise ValueError(f"Unknown mu_type: {mu_type}")

    alpha = alpha_0 * mu[assigned]
    P = Dirichlet(alpha).sample((n,))

    centered = P - 1.0 / K
    norms = centered.norm(dim=2, keepdim=True).clamp(min=1e-8)
    normalized = centered / norms
    C = torch.bmm(normalized, normalized.transpose(1, 2))

    C = C.clamp(-1.0, 1.0)
    C.diagonal(dim1=1, dim2=2).fill_(1.0)

    return C, {"mu": mu, "alpha": alpha, "assigned_communities": assigned}


def make_centroids(K, d, phi=1.0, mode="orthogonal", seed=None):
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    if mode == "random":
        return torch.randn(K, d, generator=gen) * phi

    if mode == "orthogonal":
        assert K <= d, f"Need K <= d for orthogonal centroids, got K={K}, d={d}"
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
        return Q[:K] * (phi * np.sqrt(d))

    raise ValueError(f"Unknown centroid mode: {mode}")


def generate_geometric_random_graph_samples(
    n,
    p,
    d,
    K,
    phi=1.0,
    sigma=0.5,
    psi=0.1,
    community_division="equally",
    community_allocation=None,
    seed=None,
    centroid_mode="random",
):
    """
    Returns:
        covariance   (n, p, p) tensor — unnormalized covariance Sigma
        correlation  (n, p, p) tensor — corresponding correlation C
        meta         dict with global_anchors, assigned_indices, latent_features
    """
    if seed is not None:
        torch.manual_seed(seed)

    if community_division == "equally":
        assert p % K == 0, f"p={p} must be divisible by K={K}"
        assigned_indices = torch.arange(K).repeat_interleave(p // K)
    else:
        assert community_allocation is not None
        assert len(community_allocation) == K
        counts = [int(round(f * p)) for f in community_allocation]
        counts[-1] = p - sum(counts[:-1])
        assigned_indices = torch.cat([torch.full((c,), k, dtype=torch.long) for k, c in enumerate(counts)])

    global_anchors = make_centroids(K, d, phi=phi, mode=centroid_mode, seed=None)
    node_anchors = global_anchors[assigned_indices]
    subject_jitter = torch.randn(n, p, d) * sigma
    latent_features = node_anchors.unsqueeze(0) + subject_jitter

    structure_term = torch.bmm(latent_features, latent_features.transpose(1, 2))
    full_rank_noise = (psi ** 2) * torch.eye(p).unsqueeze(0).expand(n, -1, -1)
    covariance = structure_term + full_rank_noise

    diag_inv_sqrt = torch.diagonal(covariance, dim1=1, dim2=2).pow(-0.5)
    D = torch.diag_embed(diag_inv_sqrt)
    correlation = torch.bmm(torch.bmm(D, covariance), D)
    correlation = correlation.clamp(-1.0, 1.0)
    correlation.diagonal(dim1=1, dim2=2).fill_(1.0)

    meta = {
        "global_anchors": global_anchors,
        "assigned_indices": assigned_indices,
        "latent_features": latent_features,
    }
    return covariance, correlation, meta
