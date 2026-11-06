import math
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from torch.distributions import Wishart
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, log_loss
from sklearn.covariance import graphical_lasso
from joblib import Parallel, delayed
from numpy.linalg import pinv, eigvalsh


def is_correlation_matrix(C: np.ndarray, tol: float = 1e-8) -> np.ndarray | bool:
    if C.ndim == 2:
        symmetric = bool(np.allclose(C, C.T, atol=tol))
        unit_diag = bool(np.allclose(np.diag(C), 1.0, atol=tol))
        min_eig = float(np.min(np.linalg.eigvalsh(C)))
        return symmetric and unit_diag and (min_eig >= -tol)
    
    elif C.ndim == 3:
        N, D, _ = C.shape
        results = np.zeros(N, dtype=bool)
        
        for i in range(N):
            matrix = C[i]
            symmetric = np.allclose(matrix, matrix.T, atol=tol)
            unit_diag = np.allclose(np.diag(matrix), 1.0, atol=tol)
            min_eig = np.min(np.linalg.eigvalsh(matrix))
            
            results[i] = symmetric and unit_diag and (min_eig >= -tol)
        
        return results
    
    else:
        raise ValueError(f"Input must be 2D or 3D array, got {C.ndim}D")

_TRIU_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}

def flatten_offdiag(C):
    N, D, _ = C.shape
    key = (D, C.device)
    if key not in _TRIU_CACHE:
        _TRIU_CACHE[key] = torch.triu_indices(D, D, offset=1, device=C.device)
    idx = _TRIU_CACHE[key]
    return C[:, idx[0], idx[1]]

def adjacency_matrix(C: torch.Tensor, top_pct: float = None, delta: float = None):
    if (top_pct is None) == (delta is None):
        raise ValueError("Exactly one of top_pct or delta must be provided")
    if C.dim() not in (2, 3) or C.size(-1) != C.size(-2):
        raise ValueError("C must be (D,D) or (N,D,D)")
    if top_pct is not None and not (0.0 <= top_pct <= 1.0):
        raise ValueError("top_pct must be in [0, 1]")
    if C.is_floating_point() and torch.isnan(C).any():
        raise ValueError("C contains NaN")

    single = C.dim() == 2
    if single:
        C = C.unsqueeze(0)

    N, D, _ = C.shape
    key = (D, C.device)
    idx = _TRIU_CACHE.get(key)
    if idx is None:
        idx = torch.triu_indices(D, D, 1, device=C.device)
        _TRIU_CACHE[key] = idx
    i0, i1 = idx

    off = C[:, i0, i1]
    M = off.size(1)

    A = C.new_zeros((N, D, D))
    if M == 0:
        return A.squeeze(0) if single else A

    if delta is not None:
        mask = off >= delta
    else:
        if top_pct == 0.0:
            return A.squeeze(0) if single else A
        k = int(math.ceil(top_pct * M))
        k = max(0, min(k, M))
        if k == 0:
            return A.squeeze(0) if single else A
        topk_idx = torch.topk(off, k, dim=1, largest=True, sorted=False).indices  # [N, k]
        mask = torch.zeros_like(off, dtype=torch.bool).scatter(1, topk_idx, True)  # [N, M]

    A[:, i0, i1] = mask.to(A.dtype)
    A[:, i1, i0] = mask.to(A.dtype)

    return A.squeeze(0) if single else A


def adjacency_threshold(C: torch.Tensor, top_pct: float = 0.10) -> torch.Tensor:
    if C.dim() not in (2, 3) or C.size(-1) != C.size(-2):
        raise ValueError("C must be (D,D) or (N,D,D)")
    single = C.dim() == 2
    if single:
        C = C.unsqueeze(0)
    N, D, _ = C.shape
    idx = torch.triu_indices(D, D, 1, device=C.device)
    off = C[:, idx[0], idx[1]]
    M = off.size(1)
    k = max(0, min(int(math.ceil(top_pct * M)), M))
    if k == 0:
        thr = torch.full((N,), float("inf"), device=C.device, dtype=C.dtype)
    else:
        thr = torch.topk(off, k, dim=1, largest=True, sorted=False).values.min(dim=1).values
    return thr.squeeze(0) if single else thr


def global_delta(M: torch.Tensor, tau: float) -> float:
    """(1-tau) quantile of all pooled off-diagonal entries across n samples."""
    single = M.dim() == 2
    if single:
        M = M.unsqueeze(0)
    idx = torch.triu_indices(M.size(1), M.size(1), 1, device=M.device)
    all_off = M[:, idx[0], idx[1]].reshape(-1).cpu().numpy()
    return float(np.percentile(all_off, (1.0 - tau) * 100))


def glasso_alpha(side, T, p, grg_d, grg_phi, grg_sigma, grg_psi, c=0.3):
    """Ravikumar-rate glasso penalty, scaled to the matrix magnitude per side.

    correlation: c*sqrt(ln p / T);  covariance: that times E[Sigma_ii]=d(phi^2+sigma^2)+psi^2.
    """
    base = c * np.sqrt(np.log(p) / T)
    if side == "c":
        return float(base)
    if side == "Sigma":
        scale = grg_d * (grg_phi ** 2 + grg_sigma ** 2) + grg_psi ** 2
        return float(scale * base)
    raise ValueError(f"side must be 'c' or 'Sigma', got {side!r}")


def _glasso_one(A, alpha, ridge, p):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, prec = graphical_lasso(A + ridge * np.eye(p), alpha=alpha, mode="cd",
                                          max_iter=5000, tol=1e-5, enet_tol=1e-5)
            except FloatingPointError:
                return np.zeros((p, p)), 0
    s = np.abs(prec)
    np.fill_diagonal(s, 0.0)
    return s, 1


def glasso_precision(M, alpha, ridge=1e-4, n_jobs=None):
    """Per-matrix graphical lasso; returns |precision| (off-diagonal magnitudes,
    diagonal zeroed) as a tensor shaped like M. Edge selection is done downstream by
    adjacency_matrix(|precision|, delta=...). Parallelized over the leading axis with
    joblib (BLAS pinned to 1 thread per worker). Non-SPD failures yield a zero matrix."""
    single = M.dim() == 2
    Mt = M.unsqueeze(0) if single else M
    Mnp = Mt.detach().cpu().numpy().astype(np.float64)
    N, p, _ = Mnp.shape
    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if N == 1 or n_jobs == 1:
        out = [_glasso_one(Mnp[i], alpha, ridge, p) for i in range(N)]
    else:
        out = Parallel(n_jobs=n_jobs, backend="loky", batch_size=32)(
            delayed(_glasso_one)(Mnp[i], alpha, ridge, p) for i in range(N))
    mats = np.stack([o[0] for o in out], axis=0)
    n_fail = N - sum(o[1] for o in out)
    if n_fail:
        warnings.warn(f"glasso_precision: {n_fail}/{N} fits failed (non-SPD) -> zero matrices")
    res = torch.from_numpy(mats).to(dtype=M.dtype, device=M.device)
    return res.squeeze(0) if single else res


def _lw_intensity(Yi, Mi, target, corr):
    """Ledoit-Wolf analytic shrinkage intensity toward `target` for one subject.
    Yi: (T,p) time series; Mi, target: (p,p)."""
    Y = Yi.detach().cpu().numpy().astype(np.float64)
    W = Y - Y.mean(0, keepdims=True)
    if corr:
        sd = W.std(0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        W = W / sd
    Tn = W.shape[0]
    P = W[:, :, None] * W[:, None, :]
    pi = float((P.var(axis=0, ddof=1) / Tn).sum())
    gamma2 = float(((Mi - target) ** 2).sum().item())
    if gamma2 == 0.0:
        return 1.0
    return float(min(1.0, pi / gamma2))


def shrink_to_mean(M, target, ts=None, corr=False, lam=None):
    """Shrink each per-subject matrix M_i toward `target` (p,p):
    (1-a)*M_i + a*target, a = Ledoit-Wolf analytic intensity from ts[i] (or fixed `lam`)."""
    single = M.dim() == 2
    Mt = M.unsqueeze(0) if single else M
    N = Mt.size(0)
    tgt = target.to(dtype=Mt.dtype, device=Mt.device)
    out = torch.empty_like(Mt)
    for i in range(N):
        if lam is not None:
            a = float(lam)
        elif ts is not None:
            a = _lw_intensity(ts[i], Mt[i], tgt, corr)
        else:
            a = 0.0
        out[i] = (1.0 - a) * Mt[i] + a * tgt
    return out.squeeze(0) if single else out


def poet_shrinkage(M, ts, K, C=0.5, corr=False):
    """POET (Fan-Liao-Mincheva 2013) principal-orthogonal-complement thresholding.

    Splits M = spectral(top-K) + residual R_K, then hard-thresholds R_K's
    off-diagonals with an adaptive entrywise threshold τ_ij = C·√θ_ij·ω_T
    (Cai-Liu form, ω_T = 1/√p + √(log p / T)). Diagonal of R_K is preserved.

    Parameters
    ----------
    M    : (n, p, p) or (p, p) sample cov/corr (batched, from `estimate_c`).
    ts   : (n, T, p) or (T, p) time series matching M.
    K    : int, number of factors (top eigencomponents kept in the spectral part).
    C    : float, Cai-Liu threshold constant (paper default 0.5).
    corr : if True, standardize `ts` per-column (subtract mean, divide by SD)
           before the projection, so u_t is on the correlation scale.

    Denominator convention: T-1 (matches `estimate_c`'s sample cov), so σ̂^u_ij
    == R_K[i,j] exactly. `ω_T` still uses `T` in the log-p term (paper).
    """
    single_M = M.dim() == 2
    single_ts = ts.dim() == 2
    if single_M:
        M = M.unsqueeze(0)
    if single_ts:
        ts = ts.unsqueeze(0)
    n, p, _ = M.shape
    T = ts.shape[1]

    eigvals, eigvecs = torch.linalg.eigh(M)  # ascending; (n, p), (n, p, p)
    top_vals = eigvals[:, -K:]                # (n, K)
    top_vecs = eigvecs[:, :, -K:]             # (n, p, K)
    spectral = top_vecs @ (top_vals.unsqueeze(-1) * top_vecs.transpose(-1, -2))
    R_K = M - spectral                        # (n, p, p)

    Y = ts - ts.mean(dim=1, keepdim=True)     # (n, T, p) centred
    if corr:
        sd = Y.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-12)
        Y = Y / sd
    proj = Y @ top_vecs                       # (n, T, K)
    u = Y - proj @ top_vecs.transpose(-1, -2) # (n, T, p) residual

    u_sq = u * u
    denom = max(T - 1, 1)
    E4 = (u_sq.transpose(1, 2) @ u_sq) / denom      # (n, p, p); mean_t(u_ti² u_tj²)
    theta = (E4 - R_K * R_K).clamp_min(0.0)         # entrywise variance ≥ 0

    omega_T = 1.0 / math.sqrt(p) + math.sqrt(math.log(p) / T)
    tau = C * torch.sqrt(theta) * omega_T           # (n, p, p)

    keep = R_K.abs() >= tau
    diag_mask = torch.eye(p, dtype=torch.bool, device=M.device).unsqueeze(0)
    keep = keep | diag_mask
    R_K_thresh = R_K * keep.to(R_K.dtype)

    Sigma_check = spectral + R_K_thresh
    Sigma_check = 0.5 * (Sigma_check + Sigma_check.transpose(-1, -2))

    if single_M:
        Sigma_check = Sigma_check.squeeze(0)
    return Sigma_check


def _refine_raw(M, **_):
    return M


def _refine_lw(M, *, target, ts, corr=False, **_):
    return shrink_to_mean(M, target, ts=ts, corr=corr)


def _refine_poet(M, *, ts, grg_args, poet_C=0.5, corr=False, **_):
    K = int(grg_args["K"])
    return poet_shrinkage(M, ts, K, C=poet_C, corr=corr)


def _score_identity(M, **_):
    return M


def _score_glasso(M, *, side, alpha_T, grg_args, glasso_c=0.3, **_):
    alpha = glasso_alpha(side, alpha_T, grg_args["p"], grg_args["d"],
                         grg_args["phi"], grg_args["sigma"], grg_args["psi"], c=glasso_c)
    return glasso_precision(M, alpha)


def lct_test_statistic(M, ts, corr=False):
    """Cai-Liu 2016 variance-standardized test statistic |T_ij| per subject.

    T_ij = Σ_k (Z_ki - Z̄_i)(Z_kj - Z̄_j) / sqrt(T · θ̂_ij)
    θ̂_ij = (1/T) Σ_k [(Z_ki - Z̄_i)(Z_kj - Z̄_j) - σ̂_ij]²
         = mean_k(u_ki² u_kj²) - σ̂_ij²   (efficient form; u = Z - Z̄)

    Uses paper's 1/T denominator convention. `M` is passed for interface
    uniformity but not consulted (we recompute σ̂ from ts to match θ̂'s scale).
    Returns |T_ij| shape (n, p, p), symmetric, diagonal zeroed.
    """
    single = ts.dim() == 2
    if single:
        ts = ts.unsqueeze(0)
    n, T, p = ts.shape
    dev = ts.device
    u = ts - ts.mean(dim=1, keepdim=True)
    if corr:
        sd = u.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-12)
        u = u / sd
    sigma_hat = torch.einsum("ntp,ntq->npq", u, u) / T
    u2 = u * u
    E4 = torch.einsum("ntp,ntq->npq", u2, u2) / T
    theta = (E4 - sigma_hat * sigma_hat).clamp_min(1e-30)
    T_stat = math.sqrt(T) * sigma_hat / theta.sqrt()
    T_abs = T_stat.abs()
    diag_idx = torch.arange(p, device=dev)
    T_abs[:, diag_idx, diag_idx] = 0.0
    return T_abs.squeeze(0) if single else T_abs


def lct_bootstrap_null_curve(ts, t_grid, N=50, corr=False, seed=None):
    """Cai-Liu 2016 LCT-B bootstrap null exceedance curve, one row per subject.

    For each t in t_grid, counts how many bootstrap |T*_ij| statistics exceed t
    across N=50 column-wise resamples (with replacement) of ts. Bootstrap statistic
    is centred by subtracting T·σ̂_observed so the signal cancels (paper's Eq. 10
    adapted to one-sample):

        T*_ij  ∝  (Σ_k' (Z*_k'i - Z̄*_i)(Z*_k'j - Z̄*_j) - T·σ̂_ij) / sqrt(T·θ̂*_ij)

    Only upper-triangle pairs are counted (paper's Σ_{i<j}). Returns (n, |t_grid|)
    int64 counts. Memory-efficient running counter — no full bootstrap tensor.
    """
    single = ts.dim() == 2
    if single:
        ts = ts.unsqueeze(0)
    n, T, p = ts.shape
    dev = ts.device
    t_grid_dev = t_grid.to(dev)

    # Observed σ̂ (used for centring the bootstrap statistic)
    u_obs = ts - ts.mean(dim=1, keepdim=True)
    if corr:
        sd_obs = u_obs.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-12)
        u_obs = u_obs / sd_obs
    sigma_obs = torch.einsum("ntp,ntq->npq", u_obs, u_obs) / T

    iu = torch.triu_indices(p, p, offset=1, device=dev)
    n_t = len(t_grid)
    counter = torch.zeros(n, n_t, dtype=torch.int64, device=dev)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

    for k in range(N):
        if gen is not None:
            idx = torch.randint(0, T, (T,), generator=gen)
        else:
            idx = torch.randint(0, T, (T,))
        Z_star = ts[:, idx.to(dev), :]  # column-wise (row-of-ts-wise) resample, shared per replication
        u_star = Z_star - Z_star.mean(dim=1, keepdim=True)
        if corr:
            sd_star = u_star.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-12)
            u_star = u_star / sd_star
        sigma_star = torch.einsum("ntp,ntq->npq", u_star, u_star) / T
        num = T * sigma_star - T * sigma_obs   # signal-cancelling centring
        u_star_sq = u_star * u_star
        E4_star = torch.einsum("ntp,ntq->npq", u_star_sq, u_star_sq) / T
        theta_star = (E4_star - sigma_star * sigma_star).clamp_min(1e-30)
        T_star_abs = (num / (T * theta_star).sqrt()).abs()
        T_star_upper = T_star_abs[:, iu[0], iu[1]]   # (n, P) with P = p(p-1)/2
        # Memory-efficient exceedance count via searchsorted: sort per subject,
        # then # entries ≥ t = P - # entries < t, avoiding the (n, P, n_t) broadcast.
        T_star_sorted, _ = T_star_upper.sort(dim=1)
        idx = torch.searchsorted(T_star_sorted, t_grid_dev.unsqueeze(0).expand(n, -1).contiguous(), right=False)
        exceed = T_star_upper.shape[1] - idx
        counter += exceed.to(torch.int64)

    return counter.squeeze(0) if single else counter


def lct_fdr_threshold(T_abs, boot_counter, t_grid, alpha, p, N):
    """Cai-Liu 2016 LCT-B FDR threshold search over [0, b_p].

    t̂ = inf{ 0 ≤ t ≤ b_p : G*(t) · (p²-p)/2 / max{#{|T_ij|≥t}, 1} ≤ alpha }
    Fallback: t̂ = sqrt(4 log p) when no valid t exists in [0, b_p].

    G*(t) = boot_counter * 2 / (N · p·(p-1))    per paper Eq. 12.

    Returns t̂ shape (n,) (or scalar if T_abs is (p,p)).
    """
    single = T_abs.dim() == 2
    if single:
        T_abs = T_abs.unsqueeze(0)
        boot_counter = boot_counter.unsqueeze(0)
    n = T_abs.shape[0]
    dev = T_abs.device
    t_grid_dev = t_grid.to(dev)

    b_p = math.sqrt(4.0 * math.log(p) - 2.0 * math.log(math.log(p)))
    fallback = math.sqrt(4.0 * math.log(p))

    G_star = boot_counter.to(torch.float64) * 2.0 / (N * p * (p - 1))       # (n, |t_grid|)
    iu = torch.triu_indices(p, p, offset=1, device=dev)
    T_abs_upper = T_abs[:, iu[0], iu[1]]                                     # (n, P)
    # Same searchsorted trick as the bootstrap loop:
    T_abs_sorted, _ = T_abs_upper.sort(dim=1)
    n_denom = T_abs_upper.shape[0]
    idx = torch.searchsorted(T_abs_sorted, t_grid_dev.unsqueeze(0).expand(n_denom, -1), right=False)
    denom_at_t = (T_abs_upper.shape[1] - idx).clamp_min(1).to(torch.float64)
    fdp = G_star * (p * (p - 1) / 2.0) / denom_at_t                          # (n, |t_grid|)

    within_bp = (t_grid_dev <= b_p)                                          # (|t_grid|,)
    valid = (fdp <= alpha) & within_bp                                        # (n, |t_grid|)
    has_valid = valid.any(dim=1)                                              # (n,)
    first_valid_idx = valid.to(torch.int64).argmax(dim=1)                    # (n,); argmax on all-False → 0
    t_hat = torch.where(
        has_valid,
        t_grid_dev[first_valid_idx],
        torch.full((n,), fallback, dtype=t_grid_dev.dtype, device=dev),
    )
    return t_hat[0] if single else t_hat


def lct_adjacency(M, ts, alpha, N=50, corr=False, use_bootstrap=True,
                  t_grid_size=101, seed=None):
    """Cai-Liu 2016 LCT-B top-level: variance-standardized correlation test with
    FDR control at level `alpha`. Returns binary (n, p, p) adjacency; diagonal 0,
    symmetric per subject. Currently `use_bootstrap=False` is not supported."""
    if not use_bootstrap:
        raise NotImplementedError("LCT-N (normal-approx) not implemented; use LCT-B (bootstrap).")
    single = ts.dim() == 2
    if single:
        ts = ts.unsqueeze(0)
        if M.dim() == 2:
            M = M.unsqueeze(0)
    n, T, p = ts.shape
    dev = ts.device
    T_abs = lct_test_statistic(M, ts, corr=corr)
    b_p = math.sqrt(4.0 * math.log(p) - 2.0 * math.log(math.log(p)))
    t_grid = torch.linspace(0.0, b_p, t_grid_size, dtype=torch.float64, device=dev)
    counter = lct_bootstrap_null_curve(ts, t_grid, N=N, corr=corr, seed=seed)
    t_hat = lct_fdr_threshold(T_abs, counter, t_grid, alpha=alpha, p=p, N=N)
    A = (T_abs >= t_hat.view(-1, 1, 1).to(T_abs.dtype)).to(M.dtype)
    diag_idx = torch.arange(p, device=dev)
    A[:, diag_idx, diag_idx] = 0
    return A.squeeze(0) if single else A


def _score_lct(M, *, ts, corr=False, lct_alpha, lct_N=50, lct_bootstrap=True, **_):
    return lct_adjacency(M, ts, alpha=lct_alpha, N=lct_N, corr=corr,
                         use_bootstrap=lct_bootstrap)


SIGMA_REFINERS = {"raw": _refine_raw, "lw": _refine_lw, "poet": _refine_poet}
GRAPH_SCORERS  = {"identity": _score_identity, "glasso": _score_glasso, "lct": _score_lct}


def refine(M, method, **opts):
    """Per-subject Σ-refiner dispatcher. `method` keys SIGMA_REFINERS; returns a
    tensor with the same shape as `M` (per-subject refined covariance/correlation Σ̃).

    Refiner contract — every registered refiner is called as
    ``fn(M, *, target=None, ts=None, corr=False, **_)`` and must accept `**_` so
    it can ignore whichever arguments it does not need. Four styles are supported:

    1. **Analytic on sample cov** (needs M, ignores ts) — e.g. OAS-style
       shrinkage, closed-form LW variants: ``fn(M, target=…) → Σ̃``.
    2. **Sampling-variance shrinkage** (needs M and ts) — our `lw`: uses ts to
       estimate π then blends `M` with `target`.
    3. **Data-driven iterative / MLE** (needs ts, ignores M) — e.g. Tyler M-estimator,
       factor-model MLE: ``fn(ts=…, corr=…) → Σ̃`` computed from ts alone.
    4. **No refinement** — `raw`, returns `M` unchanged.

    Every refiner must handle both `corr=True` (input is a correlation matrix,
    unit diagonal) and `corr=False` (covariance). For style-3 refiners, the
    caller still passes ``M=Σ̂`` for interface uniformity; the refiner ignores it.
    """
    try:
        fn = SIGMA_REFINERS[method]
    except KeyError as e:
        raise KeyError(f"Unknown sigma refiner: {method!r}. "
                       f"Available: {sorted(SIGMA_REFINERS)}") from e
    return fn(M, **opts)


def score(M_tilde, method, **opts):
    """Graph-score dispatcher: refined Σ̃ → score matrix S (same shape) to be
    top-τ thresholded downstream by ``adjacency_matrix(S, top_pct=τ)``.

    Scorer contract — every registered scorer is called as
    ``fn(M_tilde, *, side=…, alpha_T=…, grg_args=…, glasso_c=…, **_)``. Options
    it doesn't need are ignored via `**_`. Currently supported:

    - `identity` — S = M_tilde (threshold Σ̃'s marginals).
    - `glasso` — S = |Θ̂(M_tilde, α(alpha_T))| (threshold precision magnitudes).
      Composes with any Σ-refiner: `score(refine(M, "lw"), "glasso", …)` means
      glasso applied to the LW-shrunk Σ̌, preserving axis orthogonality.

    `side ∈ {"c", "Sigma"}` selects the magnitude scaling of the Ravikumar α.
    """
    try:
        fn = GRAPH_SCORERS[method]
    except KeyError as e:
        raise KeyError(f"Unknown graph scorer: {method!r}. "
                       f"Available: {sorted(GRAPH_SCORERS)}") from e
    return fn(M_tilde, **opts)


def realized_sparsity(A: torch.Tensor) -> float:
    """Fraction of off-diagonal entries equal to 1, averaged over n samples."""
    single = A.dim() == 2
    if single:
        A = A.unsqueeze(0)
    idx = torch.triu_indices(A.size(1), A.size(1), 1, device=A.device)
    return float(A[:, idx[0], idx[1]].float().mean().item())


def graph_diagnostics(A: torch.Tensor, labels: torch.Tensor):
    if A.dim() == 2:
        A = A.unsqueeze(0)
    p = A.shape[-1]
    same = (labels.view(1, p) == labels.view(p, 1))
    triu = torch.ones(p, p, dtype=torch.bool, device=A.device).triu(1)
    W_mask = same & triu
    B_mask = (~same) & triu
    W_count = int(W_mask.sum().item())
    B_count = int(B_mask.sum().item())
    A_bool = A.bool()
    A_W = A_bool[:, W_mask].float().sum(dim=1)
    A_B = A_bool[:, B_mask].float().sum(dim=1)
    pi_W = float((A_W / max(W_count, 1)).mean().item())
    pi_B = float((A_B / max(B_count, 1)).mean().item())
    num = pi_W * W_count
    den = num + pi_B * B_count
    purity = float(num / den) if den > 0 else float("nan")
    return pi_W, pi_B, purity


def flip_rates(A1: torch.Tensor, A2: torch.Tensor, labels: torch.Tensor):
    if A1.dim() == 2: A1 = A1.unsqueeze(0)
    if A2.dim() == 2: A2 = A2.unsqueeze(0)
    p = A1.shape[-1]
    same = (labels.view(1, p) == labels.view(p, 1))
    triu = torch.ones(p, p, dtype=torch.bool, device=A1.device).triu(1)
    W_mask = same & triu
    B_mask = (~same) & triu
    diff = (A1.bool() != A2.bool())
    flip_all = float(diff[:, triu].float().mean().item())
    flip_W = float(diff[:, W_mask].float().mean().item()) if W_mask.any() else float("nan")
    flip_B = float(diff[:, B_mask].float().mean().item()) if B_mask.any() else float("nan")
    return flip_all, flip_W, flip_B


def random_adjacency(C: torch.Tensor, top_pct: float = 0.10):
    src = C if C.dim() == 3 else C.unsqueeze(0)
    N, D, _ = src.shape

    u, v = torch.triu_indices(D, D, 1, device=C.device, dtype=torch.long)

    scores = torch.rand(N, u.shape[0], device=C.device, dtype=C.dtype)
    k = int(u.shape[0] * top_pct)

    top_indices = torch.topk(scores, k, dim=1).indices
    mask = torch.zeros_like(scores).scatter_(1, top_indices, 1.0)

    A = torch.zeros_like(src)
    A[:, u, v] = mask
    A[:, v, u] = mask

    return A if C.dim() == 3 else A.squeeze(0)

def convolution(A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiply with broadcasting. Returns A @ C."""
    if A.dim() not in (2, 3) or C.dim() not in (2, 3):
        raise ValueError("A and C must be 2D or 3D tensors")

    single = A.dim() == 2 and C.dim() == 2
    A3 = A.unsqueeze(0) if A.dim() == 2 else A
    C3 = C.unsqueeze(0) if C.dim() == 2 else C

    if A3.size(-1) != C3.size(-2):
        raise ValueError("inner dimensions do not match for matmul")

    dtype = torch.promote_types(A3.dtype, C3.dtype)
    A3 = A3.to(dtype=dtype)
    C3 = C3.to(device=A3.device, dtype=dtype)

    out = torch.matmul(A3, C3)
    return out.squeeze(0) if single else out

def hadamard(A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Elementwise product with batch broadcasting. Returns A * C."""
    if C.dim() not in (2, 3) or A.dim() not in (2, 3):
        raise ValueError("C and A must be 2D or 3D")

    single = C.dim() == 2 and A.dim() == 2
    C3 = C.unsqueeze(0) if C.dim() == 2 else C
    A3 = A.unsqueeze(0) if A.dim() == 2 else A

    if C3.shape[-2:] != A3.shape[-2:]:
        raise ValueError("matrix dimensions must match")

    dtype = torch.promote_types(C3.dtype, A3.dtype)
    C3 = C3.to(dtype=dtype)
    A3 = A3.to(device=C3.device, dtype=dtype)

    out = A3 * C3
    return out.squeeze(0) if single else out

def gcn(A: torch.Tensor, C: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Compute (D^{-1/2}(A+I)D^{-1/2}) @ C with batch broadcasting."""
    if C.dim() not in (2, 3) or A.dim() not in (2, 3):
        raise ValueError("C must be 2D or 3D; A must be 2D or 3D")

    single = C.dim() == 2
    C3 = C.unsqueeze(0) if single else C
    A3 = A.unsqueeze(0) if A.dim() == 2 else A

    if A3.size(-1) != A3.size(-2):
        raise ValueError("A must be square on the last two dims")
    if C3.size(-2) != A3.size(-1):
        raise ValueError("inner dims must match: C[..., D, F] and A[..., D, D]")

    B, D = C3.size(0), C3.size(-2)
    Ba = A3.size(0)
    if Ba not in (1, B):
        raise ValueError(f"batch sizes are incompatible: C has {B}, A has {Ba}")

    
    A3 = A3.to(device=C3.device, dtype=C3.dtype)
    C32 = C3.to(torch.float32)
    A32 = A3.to(torch.float32)

    I = torch.eye(D, device=C3.device, dtype=torch.float32).unsqueeze(0)  
    A_tilde = A32 + I
    deg = A_tilde.sum(dim=-1).clamp_min(1e-12)
    invsqrt = deg.rsqrt() 

    A_hat = A_tilde * invsqrt.unsqueeze(-1) * invsqrt.unsqueeze(-2)
    Y32 = C32
    for _ in range(k):
        Y32 = torch.matmul(A_hat, Y32)

    out = Y32.to(C.dtype)
    return out.squeeze(0) if single else out

def sage_mean(A: torch.Tensor, C: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Compute (D_inv @ (A + I)) @ C with batch broadcasting."""
    if C.dim() not in (2, 3) or A.dim() not in (2, 3):
        raise ValueError("C must be 2D or 3D; A must be 2D or 3D")

    single = C.dim() == 2
    C3 = C.unsqueeze(0) if single else C
    A3 = A.unsqueeze(0) if A.dim() == 2 else A

    if A3.size(-2) != A3.size(-1):
        raise ValueError("A must be square on the last two dims")
    if C3.size(-2) != A3.size(-1):
        raise ValueError("inner dims must match: C[..., D, F] and A[..., D, D]")

    B, D = C3.size(0), C3.size(-2)
    Ba = A3.size(0)
    if Ba not in (1, B):
        raise ValueError(f"batch sizes are incompatible: C has {B}, A has {Ba}")

    C32 = C3.to(torch.float32)
    A32 = A3.to(device=C3.device, dtype=torch.float32)

    I = torch.eye(D, device=C3.device, dtype=torch.float32).unsqueeze(0)
    A_tilde = A32 + I
    deg = A_tilde.sum(dim=-1).clamp_min(1e-12)
    invdeg = deg.reciprocal() 

    A_hat = invdeg.unsqueeze(-1) * A_tilde
    Y32 = C32
    for _ in range(k):
        Y32 = torch.matmul(A_hat, Y32)

    out = Y32.to(C.dtype)
    return out.squeeze(0) if single else out


def gin(A: torch.Tensor, C: torch.Tensor, epsilon: float = 0.0, k: int = 1) -> torch.Tensor:
    """Compute ((A + epsilon * I))^k @ C with batch broadcasting."""
    if A.dim() not in (2, 3) or C.dim() not in (2, 3):
        raise ValueError("A and C must be 2D or 3D")

    single = C.dim() == 2
    A3 = A.unsqueeze(0) if A.dim() == 2 else A
    C3 = C.unsqueeze(0) if single else C

    if A3.size(-2) != A3.size(-1):
        raise ValueError("A must be square on the last two dims")
    if C3.size(-2) != A3.size(-1):
        raise ValueError("inner dims must match: C[..., D, F] and A[..., D, D]")

    Bc, Ba = C3.size(0), A3.size(0)
    if Ba not in (1, Bc):
        raise ValueError(f"batch sizes are incompatible: C has {Bc}, A has {Ba}")

    comp_dtype = torch.promote_types(A3.dtype, C3.dtype)
    A3c = A3.to(device=C3.device, dtype=comp_dtype)
    C3c = C3.to(dtype=comp_dtype)

    D = A3c.size(-1)
    I = torch.eye(D, device=A3c.device, dtype=comp_dtype).unsqueeze(0)

    A_eps = A3c + epsilon * I
    Y = C3c
    for _ in range(k):
        Y = torch.matmul(A_eps, Y)

    out = Y.to(dtype=C.dtype)
    return out.squeeze(0) if single else out

def gat(A: torch.Tensor, C: torch.Tensor, k: int = 1) -> torch.Tensor:
    if C.dim() not in (2, 3) or A.dim() not in (2, 3):
        raise ValueError("C must be 2D or 3D; A must be 2D or 3D")

    single = C.dim() == 2
    C3 = C.unsqueeze(0) if single else C
    A3 = A.unsqueeze(0) if A.dim() == 2 else A

    if A3.size(-1) != A3.size(-2):
        raise ValueError("A must be square on the last two dims")
    if C3.size(-2) != A3.size(-1):
        raise ValueError("inner dims must match: C[..., D, F] and A[..., D, D]")

    B, D = C3.size(0), C3.size(-2)
    Ba = A3.size(0)
    if Ba not in (1, B):
        raise ValueError(f"batch sizes are incompatible: C has {B}, A has {Ba}")

    C32 = C3.to(torch.float32)
    A32 = A3.to(device=C3.device, dtype=torch.float32)

    I = torch.eye(D, device=C3.device, dtype=torch.float32).unsqueeze(0)
    A_tilde = (A32 + I).clamp(0, 1)

    mask = A_tilde > 0
    Y32 = C32
    for _ in range(k):
        scores = torch.matmul(Y32, Y32.transpose(-1, -2))
        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        Y32 = torch.matmul(attention_weights, Y32)

    out = Y32.to(C.dtype)
    return out.squeeze(0) if single else out

def cv_logistic_regression(
    X_flattened: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    train_scores = []
    val_scores = []
    betas = []
    
    for train_idx, val_idx in skf.split(X_flattened, y):
        X_train, X_val = X_flattened[train_idx], X_flattened[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        betas.append(model.coef_.flatten())
    
    avg_train_accuracy = float(np.mean(train_scores))
    avg_val_accuracy = float(np.mean(val_scores))
    if len(betas) > 0:
        betas_matrix = np.stack(betas, axis=0)
    else:
        d = X_flattened.shape[1] if X_flattened.ndim == 2 else 0
        betas_matrix = np.empty((0, d), dtype=float)
    
    return avg_train_accuracy, avg_val_accuracy, betas_matrix

def split_logistic_regression(
    X_flattened: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    seed: int | None = None,
    random_state: int | None = 42,
) -> tuple[float, float, float, float, float, float, np.ndarray]:

    rs = seed if seed is not None else random_state
    X_train, X_val, y_train, y_val = train_test_split(
        X_flattened, y,
        test_size=test_size,
        stratify=y,
        shuffle=True,
        random_state=rs,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    
    train_acc = float(model.score(X_train, y_train))
    val_acc = float(model.score(X_val, y_val))

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    train_auc = float(roc_auc_score(y_train, y_train_proba))
    val_auc = float(roc_auc_score(y_val, y_val_proba))

    train_logloss = float(log_loss(y_train, y_train_proba))
    val_logloss = float(log_loss(y_val, y_val_proba))

    beta_hat = model.coef_.flatten()

    return train_acc, val_acc, train_auc, val_auc, train_logloss, val_logloss, beta_hat

def split_linear_regression(
    X_flattened: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    seed: int | None = None,
    random_state: int | None = 42,
) -> Dict[str, Any]:
    """
    Linear regression with intercept, NO scaling.
    Returns metrics, beta_full = [intercept, coefficients], and OLS covariance.

    Keys:
      train_mse, val_mse, train_r2, val_r2
      beta_hat        (d+1,) -> [beta0, beta]
      sigma2_hat      unbiased train noise variance RSS / (n_train - (d+1))
      cov_theta       covariance of [beta0, beta] on train design
      eig_XtX, cond_XtX
      train_idx, val_idx
    """
    if y.ndim > 1:
        y = y.ravel()

    rs = seed if seed is not None else random_state
    n_total = X_flattened.shape[0]
    idx_all = np.arange(n_total)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_flattened, y, idx_all,
        test_size=test_size,
        shuffle=True,
        random_state=rs,
    )

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)

    train_mse = float(mean_squared_error(y_train, y_train_pred))
    val_mse   = float(mean_squared_error(y_val,   y_val_pred))
    train_r2  = float(r2_score(y_train, y_train_pred))
    val_r2    = float(r2_score(y_val,   y_val_pred))

    beta0 = float(model.intercept_)
    beta  = model.coef_.astype(float).ravel()
    beta_full = np.concatenate(([beta0], beta))

    n_train, d = X_train.shape
    Z = np.c_[np.ones(n_train), X_train]          
    XtX = Z.T @ Z
    XtX_inv = pinv(XtX)                            
    rss = float(np.sum((y_train - y_train_pred) ** 2))
    dof = max(1, n_train - (d + 1))
    sigma2_hat = rss / dof
    cov_theta = sigma2_hat * XtX_inv               

    eig_XtX = eigvalsh(XtX)
    cond_XtX = float(eig_XtX[-1] / max(eig_XtX[0], 1e-12))

    return dict(
        train_mse=train_mse,
        val_mse=val_mse,
        train_r2=train_r2,
        val_r2=val_r2,
        beta_hat=beta_full,
        sigma2_hat=sigma2_hat,
        cov_theta=cov_theta,
        eig_XtX=eig_XtX,
        cond_XtX=cond_XtX,
        train_idx=idx_train,
        val_idx=idx_val,
    )


def debug_print(*args, debug_mode: bool = True, **kwargs):
    if debug_mode:
        print(*args, **kwargs)

def subsample_balanced(X_full: torch.Tensor,
                       y_full: torch.Tensor,
                       n: int,
                       *,
                       seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    y_all = y_full if isinstance(y_full, torch.Tensor) else torch.as_tensor(y_full, device=X_full.device)
    labels_1d = y_all.reshape(-1)
    idx0 = torch.nonzero(labels_1d == 0, as_tuple=False).squeeze(1)
    idx1 = torch.nonzero(labels_1d == 1, as_tuple=False).squeeze(1)

    n0 = min(idx0.numel(), n // 2)
    n1 = min(idx1.numel(), n - n0)

    if seed is not None:
        g = torch.Generator(device=X_full.device)
        g.manual_seed(int(seed))
        ord0 = torch.randperm(idx0.numel(), generator=g, device=X_full.device)
        ord1 = torch.randperm(idx1.numel(), generator=g, device=X_full.device)
        sel0 = idx0[ord0[:n0]]
        sel1 = idx1[ord1[:n1]]
    else:
        sel0 = idx0[:n0]
        sel1 = idx1[:n1]

    sel = torch.cat((sel0, sel1))
    X = X_full.index_select(0, sel)
    y = y_full.index_select(0, sel) if isinstance(y_full, torch.Tensor) else y_all.index_select(0, sel)
    return X, y


# ---------------------------------------------------------------------------
# Time series, sample covariance/correlation, theoretical m_star thresholds
# ---------------------------------------------------------------------------

def generate_time_series(
    C_true: torch.Tensor,
    T: int = 50,
    temporal: str = "iid",
    rho: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    if T <= 0:
        raise ValueError("T must be positive")

    temporal = temporal.lower()
    if temporal not in {"iid", "ar1"}:
        raise ValueError("temporal must be iid or ar1")

    if seed is not None:
        torch.manual_seed(int(seed))

    is_batched = C_true.dim() == 3
    if not is_batched:
        C_true = C_true.unsqueeze(0)

    B, p, _ = C_true.shape
    device, base_dtype = C_true.device, C_true.dtype

    C = 0.5 * (C_true + C_true.transpose(-1, -2)) + torch.eye(
        p, device=device, dtype=base_dtype
    ) * 1e-6
    L = torch.linalg.cholesky(C)

    if temporal == "iid":
        Z = torch.randn(B, T, p, device=device, dtype=base_dtype)
        Y = torch.bmm(Z, L.transpose(1, 2))
    else:
        rho_t = torch.tensor(float(rho), dtype=base_dtype, device=device)
        scale = torch.sqrt((1.0 - rho_t * rho_t).clamp_min(0.0))
        L_eps = L * scale

        Y = torch.zeros(B, T, p, device=device, dtype=base_dtype)
        z0 = torch.randn(B, p, device=device, dtype=base_dtype)
        y_prev = torch.bmm(L, z0.unsqueeze(2)).squeeze(2)

        E = torch.randn(B, T, p, device=device, dtype=base_dtype)
        E_transformed = torch.bmm(E, L_eps.transpose(1, 2))

        for t in range(T):
            y_t = rho_t * y_prev + E_transformed[:, t, :]
            Y[:, t, :] = y_t
            y_prev = y_t

    return Y if is_batched else Y.squeeze(0)


def estimate_c(
    ts: torch.Tensor,
    corr: bool = True,
    accum_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Sample covariance (corr=False) or correlation (corr=True) from a (B, T, p) time series."""
    is_batched = ts.dim() == 3
    if not is_batched:
        ts = ts.unsqueeze(0)

    B, T, p = ts.shape
    device, base_dtype = ts.device, ts.dtype
    adtype = accum_dtype if accum_dtype is not None else base_dtype

    Za = ts.to(adtype)
    sum_x = Za.sum(dim=1)
    sum_xx = torch.bmm(Za.transpose(1, 2), Za)

    mu = sum_x / T
    Sxx = sum_xx - T * torch.bmm(mu.unsqueeze(2), mu.unsqueeze(1))
    S = Sxx / max(T - 1, 1)
    S = 0.5 * (S + S.transpose(1, 2))

    if corr:
        d2 = torch.diagonal(S, dim1=-2, dim2=-1).clamp(min=1e-12)
        d = torch.sqrt(d2)
        d_outer = d.unsqueeze(2) * d.unsqueeze(1)
        C_hat = (S / d_outer).to(base_dtype)
        C_hat = C_hat.clamp(min=-1.0, max=1.0)
        C_hat.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        C_hat = 0.5 * (C_hat + C_hat.transpose(-1, -2))
    else:
        C_hat = S.to(base_dtype)

    return C_hat if is_batched else C_hat.squeeze(0)


def estimate_correlation(ts: torch.Tensor, accum_dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    """Back-compat alias: sample correlation matrix from a (B, T, p) time series."""
    return estimate_c(ts, corr=True, accum_dtype=accum_dtype)


def generate_and_estimate_corr(
    C_true: torch.Tensor,
    T: int = 50,
    *,
    chunk_size: int = 512,
    accum_dtype: torch.dtype | None = torch.float64,
    temporal: str = "iid",
    rho: float = 0.0,
) -> torch.Tensor:
    """Generate T samples from a Gaussian with covariance C_true (iid or AR1) and return the empirical correlation."""
    if T <= 0:
        raise ValueError("T must be positive")

    temporal = temporal.lower()
    if temporal not in {"iid", "ar1"}:
        raise ValueError('temporal must be "iid" or "ar1"')
    if temporal == "ar1" and not (-1.0 < float(rho) < 1.0):
        raise ValueError("For AR1 you must have abs(rho) < 1")

    is_batched = (C_true.dim() == 3)
    if not is_batched:
        C_true = C_true.unsqueeze(0)

    B, p, _ = C_true.shape
    device, base_dtype = C_true.device, C_true.dtype

    C = 0.5 * (C_true + C_true.transpose(-1, -2)) + torch.eye(p, device=device, dtype=base_dtype) * 1e-6
    L = torch.linalg.cholesky(C)

    adtype = accum_dtype if accum_dtype is not None else base_dtype
    sum_x = torch.zeros(B, p, device=device, dtype=adtype)
    sum_xx = torch.zeros(B, p, p, device=device, dtype=adtype)

    with torch.no_grad():
        if temporal == "iid":
            for start in range(0, T, chunk_size):
                t = min(chunk_size, T - start)
                Z = torch.randn(B, t, p, device=device, dtype=base_dtype)
                Yt = torch.bmm(Z, L.transpose(1, 2))
                Y = Yt.transpose(1, 2)
                sum_x += Y.sum(dim=2, dtype=adtype)
                Ya = Y.to(adtype)
                sum_xx += torch.bmm(Ya, Ya.transpose(1, 2))
        else:
            z0 = torch.randn(B, p, device=device, dtype=base_dtype)
            y_prev = torch.bmm(L, z0.unsqueeze(2)).squeeze(2)

            rho_t = torch.tensor(float(rho), dtype=base_dtype, device=device)
            scale = torch.sqrt((1.0 - rho_t * rho_t).clamp_min(0.0))
            L_eps = L * scale

            for start in range(0, T, chunk_size):
                t = min(chunk_size, T - start)
                Z = torch.randn(B, t, p, device=device, dtype=base_dtype)
                Et = torch.bmm(Z, L_eps.transpose(1, 2))
                for k in range(t):
                    e_k = Et[:, k, :]
                    y = rho_t * y_prev + e_k
                    y_a = y.to(adtype)
                    sum_x += y_a
                    sum_xx += torch.bmm(y_a.unsqueeze(2), y_a.unsqueeze(1))
                    y_prev = y

        mu = sum_x / T
        Sxx = sum_xx - T * torch.bmm(mu.unsqueeze(2), mu.unsqueeze(1))
        S = Sxx / max(T - 1, 1)
        S = 0.5 * (S + S.transpose(1, 2))

        d2 = torch.diagonal(S, dim1=-2, dim2=-1).clamp(min=1e-12)
        d = torch.sqrt(d2)
        d_outer = d.unsqueeze(2) * d.unsqueeze(1)
        C_hat = (S / d_outer).to(base_dtype)
        C_hat = C_hat.clamp(min=-1.0, max=1.0)
        C_hat.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        C_hat = 0.5 * (C_hat + C_hat.transpose(-1, -2))

    if not is_batched:
        C_hat = C_hat.squeeze(0)
    return C_hat


def m_star_covariance(Sigma: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    p = Sigma.shape[-1]
    I = torch.eye(p, device=Sigma.device, dtype=Sigma.dtype)

    A_tilde = A + I
    d_tilde = A_tilde.sum(dim=-1)
    R = A_tilde / d_tilde.unsqueeze(-1)

    L_rw = I - R
    L_rw_Sigma = L_rw @ Sigma
    denom = (L_rw_Sigma * L_rw_Sigma).sum(dim=(-2, -1))

    tr_Sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(dim=-1)
    Sigma_F2 = (Sigma * Sigma).sum(dim=(-2, -1))
    Sigma2 = Sigma @ Sigma

    A_Sigma_At = (A_tilde @ Sigma) @ A_tilde.transpose(-2, -1)
    block_sum_Sigma = torch.diagonal(A_Sigma_At, dim1=-2, dim2=-1)

    A_Sigma2_At = (A_tilde @ Sigma2) @ A_tilde.transpose(-2, -1)
    block_sum_Sigma2 = torch.diagonal(A_Sigma2_At, dim1=-2, dim2=-1)

    inv_d2 = 1.0 / (d_tilde * d_tilde)
    neighborhood_term = (inv_d2 * (
        block_sum_Sigma * tr_Sigma.unsqueeze(-1) + block_sum_Sigma2
    )).sum(dim=-1)

    numer = tr_Sigma ** 2 + Sigma_F2 - neighborhood_term
    return numer / denom


def m_star_correlation(C: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    p = C.shape[-1]
    I = torch.eye(p, device=C.device, dtype=C.dtype)

    A_tilde = A + I
    d_tilde = A_tilde.sum(dim=-1)
    d_inv = 1.0 / d_tilde
    R = A_tilde * d_inv.unsqueeze(-1)

    L_rw_C = (I - R) @ C
    D = (L_rw_C * L_rw_C).sum(dim=(-2, -1))

    C_sq = C * C
    C_cube = C * C_sq
    s = C_sq.sum(dim=-1)
    C2 = C @ C
    C3C = C_cube @ C
    CC3 = C @ C_cube

    G = (p * C
         - C * (s.unsqueeze(-1) + s.unsqueeze(-2))
         - 0.5 * (1 - C_sq) * C2
         + 0.5 * (C3C + CC3))

    AGAT = (A_tilde @ G) @ A_tilde.transpose(-2, -1)
    block_sum_G = torch.diagonal(AGAT, dim1=-2, dim2=-1)
    d_inv2 = d_inv * d_inv
    neighborhood_G = (d_inv2 * block_sum_G).sum(dim=-1)

    V_C = ((1 - C_sq) ** 2).sum(dim=(-2, -1)) - neighborhood_G

    W = C - C_cube
    RW = R @ W
    B_C = (L_rw_C * RW).sum(dim=(-2, -1))

    N1 = V_C - B_C
    N2 = 0.25 * ((W * W).sum(dim=(-2, -1)) - (RW * RW).sum(dim=(-2, -1)))

    discriminant = (N1 * N1 + 4 * D * N2).clamp(min=0)
    m_star = (N1 + torch.sqrt(discriminant)) / (2 * D) + 2

    return m_star


# ---------------------------------------------------------------------------
# Reference correlation/covariance matrix generators
# ---------------------------------------------------------------------------

def make_pd(A: torch.Tensor, min_eig: float = 1e-3) -> torch.Tensor:
    """Project A to the nearest symmetric positive-definite matrix with min eigenvalue >= min_eig."""
    A = A.to(dtype=torch.float64)
    B = 0.5 * (A + A.T)
    eigvals, eigvecs = torch.linalg.eigh(B)
    eigvals.clamp_(min=min_eig)
    A_pd = (eigvecs * eigvals) @ eigvecs.T
    A_pd = 0.5 * (A_pd + A_pd.T)
    return A_pd


def generate_wishart_samples(C_0: torch.Tensor, df: float, n_samples: int, device: str = "cpu") -> torch.Tensor:
    """Sample n correlation matrices from a Wishart distribution around scale matrix C_0."""
    C_0 = C_0.to(device=device, dtype=torch.float32)
    C_0 = make_pd(C_0).to(torch.float32)
    p = C_0.shape[0]
    df = max(df, p + 1)
    wishart_dist = Wishart(df=df, covariance_matrix=C_0)
    cov_samples = wishart_dist.sample((n_samples,))
    diag = torch.diagonal(cov_samples, dim1=-2, dim2=-1)
    diag = torch.clamp(diag, min=1e-12)
    rsqrt_diag = diag.rsqrt()
    rsqrt_outer = rsqrt_diag.unsqueeze(-1) * rsqrt_diag.unsqueeze(-2)
    corr_samples = cov_samples * rsqrt_outer
    corr_samples.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    assert np.all(is_correlation_matrix(corr_samples.detach().cpu().numpy()))
    return corr_samples


def generate_equicorr_matrix(p: int, rho: float, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Equicorrelation matrix of size (p, p) with off-diagonal value rho."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if p > 1:
        lo = -1.0 / (p - 1)
        if not (lo <= rho <= 1.0):
            raise ValueError(f"rho must be in [{lo:.6f}, 1], got {rho}")
    C = torch.full((p, p), rho, device=device, dtype=dtype)
    C.fill_diagonal_(1.0)
    assert np.all(is_correlation_matrix(C.detach().cpu().numpy()))
    return C


def generate_banded_toeplitz_matrix(p: int, band_q: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Banded Toeplitz correlation matrix with Bartlett-style lag weights."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    band_q = int(max(0, min(band_q, max(0, p - 1))))
    dev = torch.device(device)
    t = torch.zeros(p, device=dev, dtype=dtype)
    k = torch.arange(band_q + 1, device=dev, dtype=torch.float32)
    if band_q >= 0:
        t[: band_q + 1] = 1.0 - k / float(band_q + 1)
    idx = torch.arange(p, device=dev)
    diff = (idx[:, None] - idx[None, :]).abs()
    C = t[diff]
    C.fill_diagonal_(1.0)
    assert np.all(is_correlation_matrix(C.detach().cpu().numpy()))
    return C


def generate_block_matrix(
    p: int,
    block_corr: float = 0.5,
    n_blocks: int = 4,
    blocks: Optional[Tuple[int, ...]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-equicorrelation matrix with within-block correlation block_corr."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    dev = torch.device(device)
    if blocks is None:
        n_blocks = max(1, int(n_blocks))
        base = p // n_blocks
        rem = p % n_blocks
        sizes = [base + (1 if i < rem else 0) for i in range(n_blocks)]
        sizes = [s for s in sizes if s > 0]
    else:
        sizes = list(blocks)
        if sum(sizes) != p:
            raise ValueError(f"Block sizes must sum to p={p}, got sum={sum(sizes)}")
        if not all(isinstance(s, int) and s > 0 for s in sizes):
            raise ValueError("All block sizes must be positive integers")
    max_s = max(sizes) if sizes else 1
    if max_s > 1:
        lo = -1.0 / (max_s - 1)
        if not (lo <= block_corr <= 1.0):
            raise ValueError(f"block_corr must be in [{lo:.6f}, 1] for largest block size {max_s}, got {block_corr}")
    C = torch.zeros((p, p), device=dev, dtype=dtype)
    C.fill_diagonal_(1.0)
    start = 0
    for s in sizes:
        end = start + s
        if s > 1:
            C[start:end, start:end] = block_corr
            idx = torch.arange(start, end, device=dev)
            C[idx, idx] = 1.0
        start = end
    assert np.all(is_correlation_matrix(C.detach().cpu().numpy()))
    return C