"""Fast unit tests for the Σ-refiner / graph-scorer dispatchers.
Imports only utils.utils (no sim). Sub-second."""
import numpy as np
import torch
import pytest

from utils.utils import (
    refine, score, SIGMA_REFINERS, GRAPH_SCORERS,
    shrink_to_mean, glasso_precision, glasso_alpha,
    poet_shrinkage,
    lct_test_statistic, lct_bootstrap_null_curve, lct_fdr_threshold, lct_adjacency,
)
import math
import time


def _synth(n=4, T=20, p=8, seed=0):
    torch.manual_seed(seed)
    ts = torch.randn(n, T, p, dtype=torch.float64)
    W = ts - ts.mean(dim=1, keepdim=True)
    M = (W.transpose(1, 2) @ W) / T
    target = M.mean(dim=0)
    return ts, M, target


GRG_ARGS = {"p": 8, "d": 4, "phi": 1.0, "sigma": 1.0, "psi": 0.3}


def test_refine_raw_is_identity():
    _, M, _ = _synth()
    assert torch.equal(refine(M, "raw"), M)


def test_refine_lw_matches_shrink_to_mean():
    ts, M, target = _synth()
    got = refine(M, "lw", target=target, ts=ts, corr=False)
    want = shrink_to_mean(M, target, ts=ts, corr=False)
    assert torch.equal(got, want)


def test_refine_lw_extra_kwargs_are_ignored():
    ts, M, target = _synth()
    got = refine(M, "lw", target=target, ts=ts, corr=False,
                 side="Sigma", alpha_T=20, grg_args=GRG_ARGS, glasso_c=0.3)
    want = shrink_to_mean(M, target, ts=ts, corr=False)
    assert torch.equal(got, want)


def test_refine_raw_extra_kwargs_are_ignored():
    ts, M, target = _synth()
    got = refine(M, "raw", target=target, ts=ts, corr=False)
    assert torch.equal(got, M)


def test_score_identity_is_identity():
    _, M, _ = _synth()
    assert torch.equal(score(M, "identity"), M)


def test_score_glasso_matches_primitives():
    _, M, _ = _synth()
    T_arg = 20
    got = score(M, "glasso", side="Sigma", alpha_T=T_arg,
                grg_args=GRG_ARGS, glasso_c=0.3)
    alpha = glasso_alpha("Sigma", T_arg, GRG_ARGS["p"], GRG_ARGS["d"],
                         GRG_ARGS["phi"], GRG_ARGS["sigma"], GRG_ARGS["psi"], c=0.3)
    want = glasso_precision(M, alpha)
    assert torch.equal(got, want)


def test_registry_keys_stable():
    assert set(SIGMA_REFINERS) == {"raw", "lw", "poet"}
    assert set(GRAPH_SCORERS) == {"identity", "glasso", "lct"}


def test_refine_unknown_method_raises():
    _, M, _ = _synth()
    with pytest.raises(KeyError, match="Unknown sigma refiner"):
        refine(M, "does_not_exist")


def test_score_unknown_method_raises():
    _, M, _ = _synth()
    with pytest.raises(KeyError, match="Unknown graph scorer"):
        score(M, "does_not_exist")


def test_refine_raw_corr_side_is_identity():
    """`raw` on a correlation-like matrix (unit diag) returns it byte-identical."""
    _, M, _ = _synth()
    # standardize to correlation
    sd = torch.sqrt(torch.diagonal(M, dim1=-2, dim2=-1))
    D = 1.0 / sd[:, :, None]
    Ccorr = M * D * D.transpose(-1, -2)
    got = refine(Ccorr, "raw", corr=True)
    assert torch.equal(got, Ccorr)


def test_refine_lw_corr_matches_shrink_to_mean():
    """`lw` with corr=True must delegate to shrink_to_mean(corr=True)."""
    ts, M, _ = _synth()
    sd = torch.sqrt(torch.diagonal(M, dim1=-2, dim2=-1))
    D = 1.0 / sd[:, :, None]
    Ccorr = M * D * D.transpose(-1, -2)
    target_corr = Ccorr.mean(dim=0)
    got  = refine(Ccorr, "lw", target=target_corr, ts=ts, corr=True)
    want = shrink_to_mean(Ccorr, target_corr, ts=ts, corr=True)
    assert torch.equal(got, want)


def test_refine_lw_corr_flag_changes_intensity():
    """Sanity: the corr flag actually threads through and changes LW's α. Use
    a target far from M so α is not saturated at 1 in either mode."""
    from utils.utils import _lw_intensity
    ts, _, _ = _synth(T=500, seed=42)
    subj0_ts = ts[0]
    subj0_M  = (subj0_ts.transpose(0, 1) @ subj0_ts) / subj0_ts.shape[0]
    tgt = torch.zeros_like(subj0_M)  # far from M → gamma^2 large → α not saturated
    a_cov  = _lw_intensity(subj0_ts, subj0_M, tgt, corr=False)
    a_corr = _lw_intensity(subj0_ts, subj0_M, tgt, corr=True)
    assert 0 < a_cov  < 1, f"expected α_cov unsaturated, got {a_cov}"
    assert 0 < a_corr < 1, f"expected α_corr unsaturated, got {a_corr}"
    assert a_cov != a_corr, f"corr flag should change α: cov={a_cov}, corr={a_corr}"


def test_refine_accepts_ts_only_style():
    """A refiner that ignores `M` and computes Σ̃ from ts alone must plug in
    cleanly. Simulates the future Tyler / factor-MLE contract without landing
    such a refiner in production `utils/utils.py`."""
    ts, M, _ = _synth()

    def _refine_from_ts_demo(M, *, ts, corr=False, **_):
        # ignore M; recompute a plain sample covariance from ts (demo only).
        Y = ts - ts.mean(dim=1, keepdim=True)
        if corr:
            sd = Y.std(dim=1, keepdim=True, unbiased=True).clamp_min(1e-12)
            Y = Y / sd
        return (Y.transpose(1, 2) @ Y) / ts.shape[1]

    # Temporarily register a demo refiner (test-local; not persisted).
    SIGMA_REFINERS["demo_from_ts"] = _refine_from_ts_demo
    try:
        got = refine(M, "demo_from_ts", ts=ts, corr=False)
        assert got.shape == M.shape
        assert torch.isfinite(got).all()
        # symmetric
        assert torch.allclose(got, got.transpose(-1, -2), atol=1e-10)
        # also test corr=True: same shape, still finite
        got_corr = refine(M, "demo_from_ts", ts=ts, corr=True)
        assert got_corr.shape == M.shape
        assert torch.isfinite(got_corr).all()
    finally:
        del SIGMA_REFINERS["demo_from_ts"]


def test_refine_poet_shape_symmetric_finite():
    """POET returns an (n, p, p) symmetric, finite tensor via the dispatcher."""
    ts, M, _ = _synth(n=4, T=30, p=8)
    grg = dict(GRG_ARGS, K=2)
    got = refine(M, "poet", ts=ts, grg_args=grg, poet_C=0.5, corr=False)
    assert got.shape == M.shape
    assert torch.isfinite(got).all()
    assert torch.allclose(got, got.transpose(-1, -2), atol=1e-10)


def test_refine_poet_K_equals_p_recovers_input():
    """K=p ⇒ spectral captures all of M ⇒ R_K ≈ 0 ⇒ POET ≈ M."""
    ts, M, _ = _synth(n=3, T=40, p=6)
    grg = dict(GRG_ARGS, p=6, K=6)
    got = refine(M, "poet", ts=ts, grg_args=grg, poet_C=0.5, corr=False)
    assert torch.allclose(got, M, atol=1e-8)


def test_refine_poet_C_zero_recovers_sample_cov():
    """C=0 ⇒ τ=0 ⇒ every |R_K[i,j]| ≥ 0 is kept ⇒ spectral + R_K = M."""
    ts, M, _ = _synth(n=3, T=30, p=8, seed=7)
    grg = dict(GRG_ARGS, K=3)
    got = refine(M, "poet", ts=ts, grg_args=grg, poet_C=0.0, corr=False)
    assert torch.allclose(got, M, atol=1e-8)


def test_refine_poet_C_large_kills_offdiag_residual():
    """Huge C ⇒ off-diag residuals thresholded to 0 ⇒ POET = spectral + diag(R_K)."""
    ts, M, _ = _synth(n=3, T=30, p=8, seed=11)
    K = 3
    grg = dict(GRG_ARGS, K=K)
    got = refine(M, "poet", ts=ts, grg_args=grg, poet_C=1e12, corr=False)

    # Recompute spectral + diag(R_K) directly for comparison.
    eigvals, eigvecs = torch.linalg.eigh(M)
    top_vals = eigvals[:, -K:]
    top_vecs = eigvecs[:, :, -K:]
    spectral = top_vecs @ (top_vals.unsqueeze(-1) * top_vecs.transpose(-1, -2))
    R_K = M - spectral
    p = M.shape[-1]
    diag_R = torch.diag_embed(torch.diagonal(R_K, dim1=-2, dim2=-1))
    want = spectral + diag_R
    want = 0.5 * (want + want.transpose(-1, -2))
    assert torch.allclose(got, want, atol=1e-8)


def test_composition_reduces_to_legacy_paths():
    """The three legacy adjacency_method values map to compositions of
    (refine, score) that reduce to the exact pre-refactor primitives."""
    ts, M, target = _synth()
    T_arg = 20

    # legacy 'threshold' == (raw, identity) == identity
    assert torch.equal(
        score(refine(M, "raw"), "identity"),
        M,
    )
    # legacy 'shrinkage' non-vacuous == (lw, identity) == shrink_to_mean
    assert torch.equal(
        score(refine(M, "lw", target=target, ts=ts, corr=False), "identity"),
        shrink_to_mean(M, target, ts=ts, corr=False),
    )
    # legacy 'glasso' == (raw, glasso) == glasso_precision(M, glasso_alpha(...))
    alpha = glasso_alpha("Sigma", T_arg, GRG_ARGS["p"], GRG_ARGS["d"],
                         GRG_ARGS["phi"], GRG_ARGS["sigma"], GRG_ARGS["psi"], c=0.3)
    assert torch.equal(
        score(refine(M, "raw"), "glasso", side="Sigma", alpha_T=T_arg,
              grg_args=GRG_ARGS, glasso_c=0.3),
        glasso_precision(M, alpha),
    )


# ------- LCT-B tests -------

def _synth_lct(n=4, T=30, p=8, seed=1):
    torch.manual_seed(seed)
    ts = torch.randn(n, T, p, dtype=torch.float64)
    W = ts - ts.mean(dim=1, keepdim=True)
    M = (W.transpose(1, 2) @ W) / (T - 1)
    return ts, M


def test_lct_test_statistic_shape_symmetric():
    ts, M = _synth_lct()
    T_abs = lct_test_statistic(M, ts, corr=False)
    n, p, _ = M.shape
    assert T_abs.shape == (n, p, p)
    assert torch.isfinite(T_abs).all()
    assert (T_abs >= 0).all()
    assert torch.allclose(T_abs, T_abs.transpose(-1, -2), atol=1e-8)
    diag = torch.diagonal(T_abs, dim1=-2, dim2=-1)
    assert (diag == 0).all()


def test_lct_bootstrap_null_exceedance_curve_shape():
    ts, _ = _synth_lct(n=3, T=40, p=6)
    t_grid = torch.linspace(0.0, 3.0, 50, dtype=torch.float64)
    counter = lct_bootstrap_null_curve(ts, t_grid, N=10, corr=False, seed=42)
    assert counter.shape == (3, 50)
    assert (counter >= 0).all()
    # Monotone non-increasing along t axis
    diffs = counter[:, 1:] - counter[:, :-1]
    assert (diffs <= 0).all()


def test_lct_fdr_threshold_monotone():
    ts, M = _synth_lct(n=3, T=50, p=8, seed=3)
    T_abs = lct_test_statistic(M, ts, corr=False)
    t_grid = torch.linspace(0.0, 4.0, 100, dtype=torch.float64)
    counter = lct_bootstrap_null_curve(ts, t_grid, N=20, corr=False, seed=7)
    t_hats = []
    for alpha in [0.20, 0.05, 0.01]:
        t_hat = lct_fdr_threshold(T_abs, counter, t_grid, alpha=alpha, p=8, N=20)
        t_hats.append(t_hat)
    # As alpha decreases, t_hat should be non-decreasing (stricter FDR ⇒ higher t)
    assert (t_hats[1] >= t_hats[0] - 1e-8).all(), f"α=0.05 t_hat < α=0.20: {t_hats[1]} vs {t_hats[0]}"
    assert (t_hats[2] >= t_hats[1] - 1e-8).all(), f"α=0.01 t_hat < α=0.05: {t_hats[2]} vs {t_hats[1]}"


def test_lct_adjacency_binary_symmetric():
    ts, M = _synth_lct(n=4, T=40, p=8, seed=5)
    A = lct_adjacency(M, ts, alpha=0.10, N=25, corr=False, use_bootstrap=True, seed=11)
    n, p, _ = M.shape
    assert A.shape == (n, p, p)
    # 0/1 valued
    assert torch.all((A == 0) | (A == 1))
    # Symmetric
    assert torch.equal(A, A.transpose(-1, -2))
    # Diagonal zero
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    assert (diag == 0).all()


@pytest.mark.slow
def test_lct_perf_25k_p20_T100():
    """LCT-B must complete (n=25000, p=20, T=100, N=50) in ≤60s single-threaded."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        ts = torch.randn(25000, 100, 20, dtype=torch.float32)
        W = ts - ts.mean(dim=1, keepdim=True)
        M = (W.transpose(1, 2) @ W) / (100 - 1)
        t0 = time.perf_counter()
        A = lct_adjacency(M, ts, alpha=0.05, N=50, corr=False, use_bootstrap=True, seed=13)
        elapsed = time.perf_counter() - t0
        assert A.shape == (25000, 20, 20)
        assert elapsed <= 60.0, f"LCT-B took {elapsed:.1f}s (target ≤60s)"
    finally:
        torch.set_num_threads(prev_threads)
