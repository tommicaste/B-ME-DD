import sys, os, json
from datetime import datetime
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import time
import logging

from simulations.simulate_utils import *
from utils.utils import *
from utils.utils import (
    estimate_c, m_star_correlation, m_star_covariance, generate_time_series,
    adjacency_threshold, graph_diagnostics, flip_rates,
    global_delta, realized_sparsity,
    refine, score, SIGMA_REFINERS, GRAPH_SCORERS,
    lct_test_statistic, lct_bootstrap_null_curve, lct_fdr_threshold,
)

torch.set_num_threads(os.cpu_count())
gcn = torch.compile(gcn)
sage_mean = torch.compile(sage_mean)
gin = torch.compile(gin)
gat = torch.compile(gat)


def _build_estimator(kind, *, seed, cv, ridge_alphas, lasso_n_alphas, lasso_max_iter):
    if kind == "ols":
        return LinearRegression()
    if kind == "ridgecv":
        return RidgeCV(alphas=ridge_alphas, cv=None)
    if kind == "lassocv":
        return LassoCV(n_alphas=lasso_n_alphas, cv=cv,
                       random_state=seed, max_iter=lasso_max_iter)
    raise ValueError(f"Unknown estimator: {kind}")


def _safe_name(name):
    return (name.replace("(", "_").replace(")", "").replace(", ", "_").replace(" ", ""))


def fit_and_record(*, X_train, X_test, y_train, y_test,
                   name, feature_dir, results, estimator, cv_cfg):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    p_feat = X_train.shape[1]; n_tr = len(y_train)
    model = _build_estimator(estimator, **cv_cfg)
    model.fit(X_tr_s, y_train)
    y_tr_pred = model.predict(X_tr_s); y_te_pred = model.predict(X_te_s)
    results[f"TrainMSE_{name}"] = float(mean_squared_error(y_train, y_tr_pred))
    results[f"ValMSE_{name}"]   = float(mean_squared_error(y_test,  y_te_pred))
    if estimator == "ols":
        results[f"Sigma2_{name}"] = float(np.sum((y_train - y_tr_pred)**2)
                                          / max(1, n_tr - p_feat - 1))
    elif estimator == "ridgecv":
        results[f"alpha_{name}"] = float(model.alpha_)
    elif estimator == "lassocv":
        results[f"alpha_{name}"] = float(model.alpha_)
        results[f"nnz_{name}"]   = int(np.sum(np.abs(model.coef_) > 0))
    out_dir = os.path.join(feature_dir, estimator)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{_safe_name(name)}.npy"),
            np.concatenate([[float(model.intercept_)], model.coef_]))


def adjacency_score(M, *, sigma_method, graph_method, side, alpha_T, grg_args, glasso_c,
                    shrink_vacuous, target=None, ts=None, corr=False, poet_C=0.5,
                    lct_N=50, lct_bootstrap=True, lct_alpha=None):
    """Return the score matrix to be value-thresholded ONCE (per-tau make_adjacency at
    the call site). Composition: refine(M, sigma_method) → score(M_tilde, graph_method).
    `shrink_vacuous=True` (used for bar/note: oracle or already-a-mean) shortcircuits
    the Σ-refiner to identity regardless of sigma_method.
    LCT-B needs per-subject `ts`; if unavailable (shrink_vacuous variants) OR
    lct_alpha is None, silently falls back to identity so the score-matrix
    contract still holds (returns M_tilde)."""
    eff_sigma = "raw" if shrink_vacuous else sigma_method
    M_tilde = refine(M, eff_sigma, target=target, ts=ts, corr=corr,
                     grg_args=grg_args, poet_C=poet_C)
    eff_graph = graph_method
    if graph_method == "lct" and (ts is None or lct_alpha is None or shrink_vacuous):
        eff_graph = "identity"
    return score(M_tilde, eff_graph, side=side, alpha_T=alpha_T,
                 grg_args=grg_args, glasso_c=glasso_c,
                 ts=ts, corr=corr,
                 lct_alpha=lct_alpha, lct_N=lct_N, lct_bootstrap=lct_bootstrap)


def build_A_dict(M, *, sigma_method, graph_method, side, alpha_T, grg_args, glasso_c,
                 poet_C, lct_N, lct_bootstrap, shrink_vacuous, target, ts, corr,
                 tau_list_p, delta_dict, p):
    """Build the per-τ adjacency dict for one variant.

    - graph_method ≠ 'lct': call adjacency_score once, then top-τ threshold per τ.
    - graph_method == 'lct': compute LCT-B test-stat + bootstrap counter ONCE,
      then FDR-threshold per α (τ_list interpreted as α_list).
    """
    if graph_method != "lct":
        score_mat = adjacency_score(
            M, sigma_method=sigma_method, graph_method=graph_method, side=side,
            alpha_T=alpha_T, grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C,
            shrink_vacuous=shrink_vacuous, target=target, ts=ts, corr=corr,
        )
        return {tau: make_adjacency(score_mat, delta=delta_dict[tau]) for tau in tau_list_p}
    # LCT-B branch: compute T_abs and boot_counter once per (M, ts, corr).
    eff_sigma = "raw" if shrink_vacuous else sigma_method
    M_tilde = refine(M, eff_sigma, target=target, ts=ts, corr=corr,
                     grg_args=grg_args, poet_C=poet_C)
    T_abs = lct_test_statistic(M_tilde, ts, corr=corr)
    import math as _math
    b_p = _math.sqrt(4 * _math.log(p) - 2 * _math.log(_math.log(p)))
    t_grid = torch.linspace(0.0, b_p, 101, dtype=torch.float64, device=ts.device)
    if not lct_bootstrap:
        raise NotImplementedError("LCT-N normal-approx not implemented; set --lct_bootstrap.")
    counter = lct_bootstrap_null_curve(ts, t_grid, N=lct_N, corr=corr)
    out = {}
    for tau in tau_list_p:
        t_hat = lct_fdr_threshold(T_abs, counter, t_grid, alpha=float(tau), p=p, N=lct_N)
        A = (T_abs >= t_hat.view(-1, 1, 1).to(T_abs.dtype)).to(M.dtype)
        diag_idx = torch.arange(p, device=ts.device)
        A[:, diag_idx, diag_idx] = 0
        out[tau] = A
    return out


#----- Parameter definitions -----
n_list = [3000, 15000]
p_list = [80]
T_list = [2, 4, 8, 16, 32, 64, 128, 256]
tau_list = [0.90, 0.70, 0.50, 0.30, 0.10]
seed = 23
matrix_seed = 30
k_list = [1]
conv_list = ["SageMean"]
SNR_list = [1]

log_enabled = True
signal = "Original_C"
adjacency_type = "top_pct"
estimation_type = "both"
variants = ["bar", "hat", "circle", "note"]

simulation_mode = "iid"
rho = 0.2
beta_type = "random"

grg_d = 10
grg_phi = 0.5
grg_sigma = 0.5
centroid_mode = "random"
grg_psi = 1

OUT_DIR = os.environ.get("BMEDD_SIM_OUT_DIR", os.path.join("outputs", "simulations"))

#----- Adjacency function definition -----
if adjacency_type == "top_pct":
    make_adjacency = adjacency_matrix
elif adjacency_type == "random":
    make_adjacency = random_adjacency
else:
    raise ValueError(f"Unsupported adjacency_type: {adjacency_type}")

#----- Optimization parameters -----
dtype = torch.float32
torch.set_grad_enabled(False)
accum_dtype = torch.float64

#----- Script entry and CLI -----
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--seed", type=int, default=seed)
parser.add_argument("--output_dir", type=str, default=OUT_DIR,
                    help="Where the ndjson results, cached coefficients and logs are written.")
parser.add_argument("--matrix_seed", type=int, default=matrix_seed,
                    help="Seed for GRG matrix generation only. If unset, falls back to --seed.")
parser.add_argument("--signal", type=str, default=signal, help="Signal type (Original_C, GCN, Sage)")
parser.add_argument("--p_list", type=int, nargs="+", default=p_list)
parser.add_argument("--n_list", type=int, nargs="+", default=n_list)
parser.add_argument("--SNR_list", type=float, nargs="+", default=SNR_list)
parser.add_argument("--T_list", type=int, nargs="+", default=T_list,
                    help="Direct list of time-series lengths T to evaluate")
parser.add_argument("--tau_list", type=float, nargs="+", default=tau_list)
parser.add_argument("--k_list", type=int, nargs="+", default=k_list)
parser.add_argument("--conv_list", type=str, nargs="+", default=conv_list)
parser.add_argument("--estimation_type", type=str, default=estimation_type,
                    choices=["correlation", "covariance", "both"])
parser.add_argument("--variants", type=str, nargs="+", default=variants,
                    choices=["bar", "hat", "circle", "note"])
parser.add_argument("--grg_K", type=int, default=None,
                    help="Number of GRG communities. Defaults to p // 8 if not set.")
parser.add_argument("--grg_d", type=int, default=grg_d)
parser.add_argument("--grg_phi", type=float, default=grg_phi)
parser.add_argument("--grg_sigma", type=float, default=grg_sigma)
parser.add_argument("--grg_psi", type=float, default=grg_psi)
parser.add_argument("--centroid_mode", type=str, default=centroid_mode,
                    choices=["random", "orthogonal"])
parser.add_argument("--beta_type", type=str, default=beta_type,
                    choices=["random", "ones",
                             "community_aligned", "off_community_aligned",
                             "single_community_aligned",
                             "edge_specific", "within_community_contrast"])
parser.add_argument("--simulation_mode", type=str, default=simulation_mode,
                    choices=["iid", "ar1"])
parser.add_argument("--rho", type=float, default=rho)
parser.add_argument("--estimator", type=str, default="ols",
                    choices=["ols", "ridgecv", "lassocv"])
parser.add_argument("--cv_folds", type=int, default=5)
parser.add_argument("--ridge_alpha_grid", type=float, nargs="+",
                    default=list(np.logspace(-4, 4, 25)))
parser.add_argument("--lasso_n_alphas", type=int, default=50)
parser.add_argument("--lasso_max_iter", type=int, default=5000)
parser.add_argument("--adjacency_method", type=str, default=None,
                    choices=["threshold", "shrinkage", "glasso"],
                    help="DEPRECATED legacy shim. Use --sigma_methods/--graph_methods.")
parser.add_argument("--sigma_methods", type=str, nargs="+", default=None,
                    choices=sorted(SIGMA_REFINERS.keys()),
                    help="Σ-refiners (per row). Default ['raw'].")
parser.add_argument("--graph_methods", type=str, nargs="+", default=None,
                    choices=sorted(GRAPH_SCORERS.keys()),
                    help="Graph-score methods. Default ['identity'].")
parser.add_argument("--feature_methods", type=str, nargs="+", default=None,
                    choices=sorted(SIGMA_REFINERS.keys()) + ["auto"],
                    help="Which refiners to emit as features. Default ['auto'] = row's sigma_method.")
parser.add_argument("--glasso_c", type=float, default=0.3)
parser.add_argument("--poet_C", type=float, default=0.5,
                    help="POET Cai-Liu threshold constant (paper default 0.5).")
parser.add_argument("--lct_N", type=int, default=50,
                    help="LCT-B bootstrap replications (paper default 50).")
parser.add_argument("--lct_bootstrap", action=argparse.BooleanOptionalAction, default=True,
                    help="Use LCT-B bootstrap null (default True); --no-lct_bootstrap for LCT-N (not implemented).")

_args = parser.parse_args() if __name__ == "__main__" else argparse.Namespace(
    seed=seed, output_dir=OUT_DIR, matrix_seed=matrix_seed, signal=signal, p_list=p_list, n_list=n_list, SNR_list=SNR_list,
    T_list=T_list, tau_list=tau_list, k_list=k_list, conv_list=conv_list,
    estimation_type=estimation_type, variants=variants,
    grg_K=None, grg_d=grg_d, grg_phi=grg_phi, grg_sigma=grg_sigma, grg_psi=grg_psi,
    centroid_mode=centroid_mode, beta_type=beta_type,
    simulation_mode=simulation_mode, rho=rho,
    estimator="ols", cv_folds=5,
    ridge_alpha_grid=list(np.logspace(-4, 4, 25)),
    lasso_n_alphas=50, lasso_max_iter=5000,
    adjacency_method=None, sigma_methods=None, graph_methods=None, feature_methods=None,
    glasso_c=0.3, poet_C=0.5, lct_N=50, lct_bootstrap=True,
)
seed = int(_args.seed)
OUT_DIR = _args.output_dir
CACHE_DIR = os.path.join(OUT_DIR, "cache")
matrix_seed = int(_args.matrix_seed) if _args.matrix_seed is not None else seed
signal = _args.signal
p_list = list(_args.p_list)
n_list = list(_args.n_list)
SNR_list = list(_args.SNR_list)
T_list = sorted(set(int(t) for t in _args.T_list))
tau_list = list(_args.tau_list)
k_list = list(_args.k_list)
conv_list = list(_args.conv_list)
estimation_type = _args.estimation_type
variants = list(_args.variants)
grg_d = _args.grg_d
grg_phi = _args.grg_phi
grg_sigma = _args.grg_sigma
grg_psi = _args.grg_psi
centroid_mode = _args.centroid_mode
beta_type = _args.beta_type
temporal_mode = _args.simulation_mode
rho = float(_args.rho)
if temporal_mode == "ar1":
    rho_tag = f"{rho:.3f}".replace(".", "p")
    simulation_mode = f"ar_{rho_tag}"
else:
    simulation_mode = "iid"

estimator = _args.estimator
cv_cfg = dict(
    seed=seed,
    cv=int(_args.cv_folds),
    ridge_alphas=list(_args.ridge_alpha_grid),
    lasso_n_alphas=int(_args.lasso_n_alphas),
    lasso_max_iter=int(_args.lasso_max_iter),
)
_ADJ_METHOD_SHIM = {
    "threshold": (["raw"], ["identity"]),
    "shrinkage": (["lw"],  ["identity"]),
    "glasso":    (["raw"], ["glasso"]),
}
_LEGACY_METHOD_FROM_PAIR = {
    ("raw",  "identity"): "threshold",
    ("lw",   "identity"): "shrinkage",
    ("raw",  "glasso"):   "glasso",
    ("lw",   "glasso"):   "shrinkage_glasso",
    ("poet", "identity"): "poet",
    ("poet", "glasso"):   "poet_glasso",
    ("raw",  "lct"):      "lct",
    ("lw",   "lct"):      "shrinkage_lct",
    ("poet", "lct"):      "poet_lct",
}
if _args.adjacency_method is not None:
    if _args.sigma_methods is not None or _args.graph_methods is not None:
        raise ValueError("Pass either --adjacency_method (legacy) OR "
                         "--sigma_methods/--graph_methods (new), not both.")
    sigma_methods, graph_methods = _ADJ_METHOD_SHIM[_args.adjacency_method]
else:
    sigma_methods = list(_args.sigma_methods) if _args.sigma_methods else ["raw"]
    graph_methods = list(_args.graph_methods) if _args.graph_methods else ["identity"]
if _args.feature_methods is None or _args.feature_methods == ["auto"]:
    feature_methods_arg = "auto"
elif "auto" in _args.feature_methods:
    raise ValueError("--feature_methods 'auto' must be passed alone.")
else:
    feature_methods_arg = list(_args.feature_methods)
if len(sigma_methods) > 1 or len(graph_methods) > 1:
    raise NotImplementedError(
        "Phase-1 refactor only supports a single (sigma_method, graph_method) per invocation. "
        "Phase-2 will add the multi-row outer loop.")
sigma_method = sigma_methods[0]
graph_method = graph_methods[0]
adjacency_method_label = _LEGACY_METHOD_FROM_PAIR[(sigma_method, graph_method)]

if feature_methods_arg == "auto":
    feat_list = [sigma_method]
else:
    feat_list = feature_methods_arg
FEAT_LABEL_C     = {"raw": "C_hat",     "lw": "C_check",     "poet": "C_poet"}
FEAT_LABEL_SIGMA = {"raw": "Sigma_hat", "lw": "Sigma_check", "poet": "Sigma_poet"}

glasso_c = float(_args.glasso_c)
poet_C = float(_args.poet_C)
lct_N = int(_args.lct_N)
lct_bootstrap = bool(_args.lct_bootstrap)
T_max = max(T_list)

do_corr = estimation_type in ("correlation", "both")
do_cov = estimation_type in ("covariance", "both")

run_tag = f"{seed}" if matrix_seed == seed else f"{seed}_m{matrix_seed}"
output_filename = f"regression_{simulation_mode}_grg_{run_tag}.ndjson"
start_time = time.time()
print(f"Starting experiment {simulation_mode}/grg with seed={seed}, matrix_seed={matrix_seed}.")

torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
NDJSON_PATH = os.path.join(OUT_DIR, output_filename)

if log_enabled:
    LOG_DIR = os.path.join(OUT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, f"regression_{simulation_mode}_grg_{run_tag}.log")
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)
    log.info(f"Starting experiment {simulation_mode}/grg with seed={seed}, matrix_seed={matrix_seed}")

#----- Main experiment loop -----
for p in sorted(p_list, reverse=True):
    if log_enabled:
        log.info(f"Starting parameter p={p}")
    grg_K = int(_args.grg_K) if _args.grg_K is not None else p // 8
    tau_note = round((p / grg_K - 1) / (p - 1), 3)
    tau_list_p = sorted(set(round(t, 3) for t in tau_list) | {tau_note}, reverse=True)
    grg_args = {
        "p": p,
        "d": grg_d,
        "K": grg_K,
        "phi": grg_phi,
        "sigma": grg_sigma,
        "psi": grg_psi,
        "centroid_mode": centroid_mode,
    }

    for snr in SNR_list:
        n_desc = sorted(n_list, reverse=True)

        with torch.no_grad():
            n_max = n_desc[0]
            if log_enabled:
                log.info(f"Generating GRG samples for p={p}, snr={snr}, n_max={n_max}")
            Sigma_full, C_full, grg_meta = generate_geometric_random_graph_samples(
                n=n_max, p=p, d=grg_d, K=grg_K,
                phi=grg_phi, sigma=grg_sigma, psi=grg_psi, seed=matrix_seed,
                centroid_mode=centroid_mode,
            )
            torch.manual_seed(seed)
            labels = grg_meta["assigned_indices"].to(torch.long)

            if log_enabled:
                log.info(f"Building oracle adjacencies for GCN/Sage signal (p={p}, snr={snr})")
            # A_c_bar_full / A_Sigma_bar_full needed only when signal is GCN or Sage.
            # For Original_C (default), they are skipped.
            if signal in ("GCN", "Sage"):
                A_c_bar_full = {tau: make_adjacency(C_full, top_pct=tau) for tau in tau_list_p}
                if do_cov:
                    A_Sigma_bar_full = {tau: make_adjacency(Sigma_full, top_pct=tau) for tau in tau_list_p}
                else:
                    A_Sigma_bar_full = {}
            else:
                A_c_bar_full = {}
                A_Sigma_bar_full = {}

            y_full_c = y_full_s = None
            true_beta_c_np = true_beta_s_np = None
            noise_std_c = noise_std_s = None

            if do_corr:
                if signal == "GCN":
                    Xs = torch.flatten(gcn(A=A_c_bar_full[tau_note], C=C_full), start_dim=1)
                elif signal == "Sage":
                    Xs = torch.flatten(sage_mean(A=A_c_bar_full[tau_note], C=C_full), start_dim=1)
                elif signal == "Original_C":
                    Xs = torch.flatten(C_full, start_dim=1)
                else:
                    raise ValueError(f"Unsupported signal type: {signal}")
                y_full_c, tb_c, st_c = generate_y(Xs, snr_ratio=snr, seed=matrix_seed, beta_type=beta_type, sparsity=tau_note, labels=labels)
                true_beta_c_np = tb_c.detach().numpy().flatten()
                noise_std_c = float(st_c.item())

            if do_cov:
                if signal == "GCN":
                    Xs = torch.flatten(gcn(A=A_Sigma_bar_full[tau_note], C=Sigma_full), start_dim=1)
                elif signal == "Sage":
                    Xs = torch.flatten(sage_mean(A=A_Sigma_bar_full[tau_note], C=Sigma_full), start_dim=1)
                elif signal == "Original_C":
                    Xs = torch.flatten(Sigma_full, start_dim=1)
                else:
                    raise ValueError(f"Unsupported signal type: {signal}")
                y_full_s, tb_s, st_s = generate_y(Xs, snr_ratio=snr, seed=matrix_seed + 1, beta_type=beta_type, sparsity=tau_note, labels=labels)
                true_beta_s_np = tb_s.detach().numpy().flatten()
                noise_std_s = float(st_s.item())

            T_max = max(T_list)
            ts_full = generate_time_series(Sigma_full, T=T_max, temporal=temporal_mode, rho=rho, seed=seed)
            if "circle" in variants:
                ts_prime_full = generate_time_series(Sigma_full, T=T_max, temporal=temporal_mode, rho=rho, seed=seed + 1)

            n_perm = torch.randperm(n_max)

        total_runs = len(n_desc) * len(T_list)
        if log_enabled:
            log.info(f"T_list for p={p}, snr={snr}: {T_list}")

        with tqdm(total=total_runs, desc=f"p={p} snr={snr}", file=sys.stdout, ascii=True,
                  dynamic_ncols=True, mininterval=0.1, leave=True) as pbar:
            for n in n_desc:
                idx = n_perm[:n]
                C = C_full[idx]
                ts_n = ts_full[idx]
                if "circle" in variants:
                    ts_prime_n = ts_prime_full[idx]

                # Pre-compute train/test split (shared across all regression calls).
                # Same split as train_test_split(X, y, test_size=0.2, random_state=seed).
                all_idx_np = np.arange(n)
                train_idx_np, test_idx_np = train_test_split(all_idx_np, test_size=0.2, random_state=seed)
                train_idx_t = torch.from_numpy(train_idx_np).long()

                y_np_c = y_full_c[idx].numpy() if do_corr else None
                y_np_s = y_full_s[idx].numpy() if do_cov else None

                # Compute oracle deltas from the n-slice (global quantile across all n samples)
                delta_C = {tau: global_delta(C, tau) for tau in tau_list_p}

                # Bar adjacency: recomputed from n-slice using delta (not sliced from full)
                score_c_bar = adjacency_score(C, sigma_method=sigma_method, graph_method=graph_method, side="c", alpha_T=T_max,
                                              grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                delta_use_c = ({tau: global_delta(score_c_bar, tau) for tau in tau_list_p}
                               if graph_method == "glasso" else delta_C)
                A_c_bar = {tau: make_adjacency(score_c_bar, delta=delta_use_c[tau]) for tau in tau_list_p}
                m_c_bar = {tau: float(m_star_correlation(C, A_c_bar[tau]).mean().item()) for tau in tau_list_p}

                # Note adjacency: train mean with oracle delta (no train/test leakage)
                C_train_mean = C[train_idx_t].mean(0)
                score_c_note = adjacency_score(C_train_mean, sigma_method=sigma_method, graph_method=graph_method, side="c", alpha_T=T_max,
                                               grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                A_c_note_2d_train = {tau: make_adjacency(score_c_note, delta=delta_use_c[tau]) for tau in tau_list_p}
                A_c_note = {tau: A_c_note_2d_train[tau].unsqueeze(0).expand(n, p, p) for tau in tau_list_p}
                m_c_note = {tau: float(m_star_correlation(C, A_c_note[tau]).mean().item()) for tau in tau_list_p}
                if log_enabled:
                    for tau in tau_list_p:
                        A0 = A_c_note_2d_train[tau].cpu().numpy().astype(int)
                        log.info(
                            f"A_0 (note, C, n={n}, p={p}, snr={snr}, tau={tau}):\n"
                            + np.array2string(A0, max_line_width=200, threshold=10**9)
                        )

                if do_cov:
                    Sigma = Sigma_full[idx]
                    delta_Sigma = {tau: global_delta(Sigma, tau) for tau in tau_list_p}
                    score_S_bar = adjacency_score(Sigma, sigma_method=sigma_method, graph_method=graph_method, side="Sigma", alpha_T=T_max,
                                                  grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                    delta_use_S = ({tau: global_delta(score_S_bar, tau) for tau in tau_list_p}
                                   if graph_method == "glasso" else delta_Sigma)
                    A_Sigma_bar = {tau: make_adjacency(score_S_bar, delta=delta_use_S[tau]) for tau in tau_list_p}
                    m_Sigma_bar = {tau: float(m_star_covariance(Sigma, A_Sigma_bar[tau]).mean().item()) for tau in tau_list_p}
                    Sigma_train_mean = Sigma[train_idx_t].mean(0)
                    score_S_note = adjacency_score(Sigma_train_mean, sigma_method=sigma_method, graph_method=graph_method, side="Sigma", alpha_T=T_max,
                                                   grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                    A_Sigma_note_2d_train = {tau: make_adjacency(score_S_note, delta=delta_use_S[tau]) for tau in tau_list_p}
                    A_Sigma_note = {tau: A_Sigma_note_2d_train[tau].unsqueeze(0).expand(n, p, p) for tau in tau_list_p}
                    m_Sigma_note = {tau: float(m_star_covariance(Sigma, A_Sigma_note[tau]).mean().item()) for tau in tau_list_p}
                    if log_enabled:
                        for tau in tau_list_p:
                            A0 = A_Sigma_note_2d_train[tau].cpu().numpy().astype(int)
                            log.info(
                                f"A_0 (note, Sigma, n={n}, p={p}, snr={snr}, tau={tau}):\n"
                                + np.array2string(A0, max_line_width=200, threshold=10**9)
                            )
                else:
                    delta_Sigma = {}
                    m_Sigma_bar = {}
                    m_Sigma_note = {}

                snr_tag = f"{snr:.3f}".replace(".", "p")
                base_dir = os.path.join(CACHE_DIR, simulation_mode, "grg",
                                        f"seed_{seed}", f"p_{p}", f"N_{n}", f"snr_{snr_tag}")
                os.makedirs(base_dir, exist_ok=True)
                beta_tag = (f"ms{matrix_seed}"
                            f"_d{grg_d}_K{grg_K}"
                            f"_phi{grg_phi:g}_sigma{grg_sigma:g}_psi{grg_psi:g}"
                            f"_{centroid_mode}"
                            f"_{signal}"
                            f"_{estimation_type}"
                            f"_{beta_type}")
                true_beta_path_c = os.path.abspath(os.path.join(base_dir, f"BetaPath_True_c_{beta_tag}.npy")) if do_corr else None
                true_beta_path_s = os.path.abspath(os.path.join(base_dir, f"BetaPath_True_s_{beta_tag}.npy")) if do_cov else None
                if do_corr:
                    np.save(true_beta_path_c, true_beta_c_np)
                if do_cov:
                    np.save(true_beta_path_s, true_beta_s_np)

                for T in T_list:
                    try:
                        if log_enabled:
                            log.info(f"Processing T={T} for n={n}, p={p}, snr={snr}")

                        with torch.no_grad():
                            ts = ts_n[:, :T, :]
                            if "circle" in variants:
                                ts_prime = ts_prime_n[:, :T, :]
                            if do_corr:
                                C_hat = estimate_c(ts, corr=True, accum_dtype=accum_dtype)
                                if "circle" in variants:
                                    C_hat_prime = estimate_c(ts_prime, corr=True, accum_dtype=accum_dtype)
                            if do_cov:
                                Sigma_hat = estimate_c(ts, corr=False, accum_dtype=accum_dtype)
                                if "circle" in variants:
                                    Sigma_hat_prime = estimate_c(ts_prime, corr=False, accum_dtype=accum_dtype)

                        # Precompute per-T train-set means (needed for LW target below and for adjacency scores).
                        if do_corr:
                            C_hat_train_mean = C_hat[train_idx_t].mean(0)
                        if do_cov:
                            Sigma_hat_train_mean = Sigma_hat[train_idx_t].mean(0)

                        # Precompute refined per-subject feature matrices for each refiner in feat_list.
                        # feat_refined_C[feat] / feat_refined_S[feat] give the (n, p, p) tensor used as
                        # features in R(A, ·). raw → Σ̂/Ĉ untouched; lw → LW-shrunk toward train mean.
                        feat_refined_C = {}
                        feat_refined_S = {}
                        for _feat in feat_list:
                            if do_corr:
                                feat_refined_C[_feat] = refine(
                                    C_hat, _feat, target=C_hat_train_mean, ts=ts, corr=True,
                                    grg_args=grg_args, poet_C=poet_C)
                            if do_cov:
                                feat_refined_S[_feat] = refine(
                                    Sigma_hat, _feat, target=Sigma_hat_train_mean, ts=ts, corr=False,
                                    grg_args=grg_args, poet_C=poet_C)

                        # Per-T adjacencies and m_star (correlation side) — use oracle delta_C
                        if do_corr:
                            if graph_method == "lct":
                                A_c_hat = build_A_dict(
                                    C_hat, sigma_method=sigma_method, graph_method="lct",
                                    side="c", alpha_T=T, grg_args=grg_args, glasso_c=glasso_c,
                                    poet_C=poet_C, lct_N=lct_N, lct_bootstrap=lct_bootstrap,
                                    shrink_vacuous=False, target=C_hat_train_mean,
                                    ts=ts, corr=True, tau_list_p=tau_list_p,
                                    delta_dict=delta_use_c, p=p,
                                )
                            else:
                                score_c_hat = adjacency_score(C_hat, sigma_method=sigma_method, graph_method=graph_method, side="c", alpha_T=T,
                                                              grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=False,
                                                              target=C_hat_train_mean, ts=ts, corr=True)
                                A_c_hat = {tau: make_adjacency(score_c_hat, delta=delta_use_c[tau]) for tau in tau_list_p}
                            m_c_hat = {tau: float(m_star_correlation(C_hat, A_c_hat[tau]).mean().item()) for tau in tau_list_p}
                            if "circle" in variants:
                                if graph_method == "lct":
                                    A_c_circle = build_A_dict(
                                        C_hat_prime, sigma_method=sigma_method, graph_method="lct",
                                        side="c", alpha_T=T, grg_args=grg_args, glasso_c=glasso_c,
                                        poet_C=poet_C, lct_N=lct_N, lct_bootstrap=lct_bootstrap,
                                        shrink_vacuous=False, target=C_hat_train_mean,
                                        ts=ts_prime, corr=True, tau_list_p=tau_list_p,
                                        delta_dict=delta_use_c, p=p,
                                    )
                                else:
                                    score_c_circle = adjacency_score(C_hat_prime, sigma_method=sigma_method, graph_method=graph_method, side="c", alpha_T=T,
                                                                     grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=False,
                                                                     target=C_hat_train_mean, ts=ts_prime, corr=True)
                                    A_c_circle = {tau: make_adjacency(score_c_circle, delta=delta_use_c[tau]) for tau in tau_list_p}
                                m_c_circle = {tau: float(m_star_correlation(C_hat, A_c_circle[tau]).mean().item()) for tau in tau_list_p}
                            else:
                                A_c_circle = {}
                                m_c_circle = {}
                            if "note_hat" in variants:
                                C_hat_mean = C_hat.mean(0)
                                score_c_note_hat = adjacency_score(C_hat_mean, sigma_method=sigma_method, graph_method=graph_method, side="c", alpha_T=T,
                                                                   grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                                A_c_note_hat_2d = {tau: make_adjacency(score_c_note_hat, delta=delta_use_c[tau]) for tau in tau_list_p}
                                A_c_note_hat = {tau: A_c_note_hat_2d[tau].unsqueeze(0).expand(n, p, p) for tau in tau_list_p}
                                m_c_note_hat = {tau: float(m_star_correlation(C_hat, A_c_note_hat[tau]).mean().item()) for tau in tau_list_p}
                            else:
                                A_c_note_hat = {}
                                m_c_note_hat = {}

                        # Per-T adjacencies and m_star (covariance side) — use oracle delta_Sigma
                        if do_cov:
                            Sigma_hat_train_mean = Sigma_hat[train_idx_t].mean(0)
                            if graph_method == "lct":
                                A_Sigma_hat = build_A_dict(
                                    Sigma_hat, sigma_method=sigma_method, graph_method="lct",
                                    side="Sigma", alpha_T=T, grg_args=grg_args, glasso_c=glasso_c,
                                    poet_C=poet_C, lct_N=lct_N, lct_bootstrap=lct_bootstrap,
                                    shrink_vacuous=False, target=Sigma_hat_train_mean,
                                    ts=ts, corr=False, tau_list_p=tau_list_p,
                                    delta_dict=delta_use_S, p=p,
                                )
                            else:
                                score_S_hat = adjacency_score(Sigma_hat, sigma_method=sigma_method, graph_method=graph_method, side="Sigma", alpha_T=T,
                                                              grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=False,
                                                              target=Sigma_hat_train_mean, ts=ts, corr=False)
                                A_Sigma_hat = {tau: make_adjacency(score_S_hat, delta=delta_use_S[tau]) for tau in tau_list_p}
                            m_Sigma_hat = {tau: float(m_star_covariance(Sigma_hat, A_Sigma_hat[tau]).mean().item()) for tau in tau_list_p}
                            if "circle" in variants:
                                if graph_method == "lct":
                                    A_Sigma_circle = build_A_dict(
                                        Sigma_hat_prime, sigma_method=sigma_method, graph_method="lct",
                                        side="Sigma", alpha_T=T, grg_args=grg_args, glasso_c=glasso_c,
                                        poet_C=poet_C, lct_N=lct_N, lct_bootstrap=lct_bootstrap,
                                        shrink_vacuous=False, target=Sigma_hat_train_mean,
                                        ts=ts_prime, corr=False, tau_list_p=tau_list_p,
                                        delta_dict=delta_use_S, p=p,
                                    )
                                else:
                                    score_S_circle = adjacency_score(Sigma_hat_prime, sigma_method=sigma_method, graph_method=graph_method, side="Sigma", alpha_T=T,
                                                                     grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=False,
                                                                     target=Sigma_hat_train_mean, ts=ts_prime, corr=False)
                                    A_Sigma_circle = {tau: make_adjacency(score_S_circle, delta=delta_use_S[tau]) for tau in tau_list_p}
                                m_Sigma_circle = {tau: float(m_star_covariance(Sigma_hat, A_Sigma_circle[tau]).mean().item()) for tau in tau_list_p}
                            else:
                                A_Sigma_circle = {}
                                m_Sigma_circle = {}
                            if "note_hat" in variants:
                                Sigma_hat_mean = Sigma_hat.mean(0)
                                score_S_note_hat = adjacency_score(Sigma_hat_mean, sigma_method=sigma_method, graph_method=graph_method, side="Sigma", alpha_T=T,
                                                                   grg_args=grg_args, glasso_c=glasso_c, poet_C=poet_C, shrink_vacuous=True)
                                A_Sigma_note_hat_2d = {tau: make_adjacency(score_S_note_hat, delta=delta_use_S[tau]) for tau in tau_list_p}
                                A_Sigma_note_hat = {tau: A_Sigma_note_hat_2d[tau].unsqueeze(0).expand(n, p, p) for tau in tau_list_p}
                                m_Sigma_note_hat = {tau: float(m_star_covariance(Sigma_hat, A_Sigma_note_hat[tau]).mean().item()) for tau in tau_list_p}
                            else:
                                A_Sigma_note_hat = {}
                                m_Sigma_note_hat = {}

                        # Graph diagnostics: realized sparsity, π_W, π_B, purity, flip rates
                        diag_specs = []
                        if do_corr:
                            if "bar"      in variants: diag_specs.append(("c", "bar",      A_c_bar,      delta_C))
                            if "hat"      in variants: diag_specs.append(("c", "hat",      A_c_hat,      delta_C))
                            if "circle"   in variants: diag_specs.append(("c", "circle",   A_c_circle,   delta_C))
                            if "note"     in variants: diag_specs.append(("c", "note",     A_c_note,     delta_C))
                            if "note_hat" in variants: diag_specs.append(("c", "note_hat", A_c_note_hat, delta_C))
                        if do_cov:
                            if "bar"      in variants: diag_specs.append(("Sigma", "bar",      A_Sigma_bar,      delta_Sigma))
                            if "hat"      in variants: diag_specs.append(("Sigma", "hat",      A_Sigma_hat,      delta_Sigma))
                            if "circle"   in variants: diag_specs.append(("Sigma", "circle",   A_Sigma_circle,   delta_Sigma))
                            if "note"     in variants: diag_specs.append(("Sigma", "note",     A_Sigma_note,     delta_Sigma))
                            if "note_hat" in variants: diag_specs.append(("Sigma", "note_hat", A_Sigma_note_hat, delta_Sigma))

                        diag_records = {}
                        for side, variant, A_dict, delta_dict in diag_specs:
                            for tau in tau_list_p:
                                if tau not in A_dict:
                                    continue
                                tp = f"{tau:.3f}".replace(".", "p")
                                pi_w, pi_b, purity = graph_diagnostics(A_dict[tau], labels)
                                diag_records[f"realized_sparsity_{side}_{variant}_{tp}"] = realized_sparsity(A_dict[tau])
                                diag_records[f"piW_{side}_{variant}_{tp}"]    = pi_w
                                diag_records[f"piB_{side}_{variant}_{tp}"]    = pi_b
                                diag_records[f"purity_{side}_{variant}_{tp}"] = purity

                        flip_specs = []
                        if do_corr and ("hat" in variants) and ("circle" in variants):
                            flip_specs.append(("c", A_c_hat, A_c_circle))
                        if do_cov and ("hat" in variants) and ("circle" in variants):
                            flip_specs.append(("Sigma", A_Sigma_hat, A_Sigma_circle))
                        for side, A_hat_dict, A_circle_dict in flip_specs:
                            for tau in tau_list_p:
                                tp = f"{tau:.3f}".replace(".", "p")
                                f_all, f_w, f_b = flip_rates(A_hat_dict[tau], A_circle_dict[tau], labels)
                                diag_records[f"flipAll_{side}_{tp}"] = f_all
                                diag_records[f"flipW_{side}_{tp}"]   = f_w
                                diag_records[f"flipB_{side}_{tp}"]   = f_b

                        signal_tag = signal if signal == "Original_C" else f"{signal}@tau_note={tau_note}"
                        results = {
                            "seed": seed, "matrix_seed": matrix_seed, "n": n, "p": p, "T": T, "snr": snr,
                            "simulation_mode": simulation_mode,
                            "adjacency_type": adjacency_type,
                            "signal": signal_tag,
                            "tau_note": tau_note,
                            "tau_list": tau_list_p,
                            "estimation_type": estimation_type,
                            "variants": variants,
                            "family": "grg",
                            "grg_args": grg_args,
                            "noise_std_c": noise_std_c,
                            "noise_std_s": noise_std_s,
                            "BetaPath_True_c": true_beta_path_c,
                            "BetaPath_True_s": true_beta_path_s,
                            "beta_type": beta_type,
                            "T_list": T_list,
                            "estimator": estimator,
                            "adjacency_method": adjacency_method_label,
                            "sigma_method": sigma_method,
                            "graph_method": graph_method,
                            "run_timestamp": datetime.now().isoformat(),
                        }
                        results.update(diag_records)

                        # Frobenius norms of estimated matrices vs oracle
                        if do_corr:
                            results["frob_C_hat"] = float((C_hat - C).norm(dim=(-2, -1)).mean().item())
                            if "circle" in variants:
                                results["frob_C_hat_prime"] = float((C_hat_prime - C).norm(dim=(-2, -1)).mean().item())
                        if do_cov:
                            results["frob_Sigma_hat"] = float((Sigma_hat - Sigma).norm(dim=(-2, -1)).mean().item())
                            if "circle" in variants:
                                results["frob_Sigma_hat_prime"] = float((Sigma_hat_prime - Sigma).norm(dim=(-2, -1)).mean().item())

                        # Per-tau delta, m_star scalars
                        for tau in tau_list_p:
                            tp_tag = f"{tau:.3f}".replace(".", "p")
                            results[f"delta_c_{tp_tag}"] = delta_C[tau]
                            results[f"m_c_bar_{tp_tag}"] = m_c_bar[tau]
                            results[f"m_c_note_{tp_tag}"] = m_c_note[tau]
                            if do_corr:
                                results[f"m_c_hat_{tp_tag}"] = m_c_hat[tau]
                                if "circle" in variants:
                                    results[f"m_c_circle_{tp_tag}"] = m_c_circle[tau]
                                if "note_hat" in variants:
                                    results[f"m_c_note_hat_{tp_tag}"] = m_c_note_hat[tau]
                            if do_cov:
                                results[f"delta_Sigma_{tp_tag}"] = delta_Sigma[tau]
                                results[f"m_Sigma_bar_{tp_tag}"] = m_Sigma_bar[tau]
                                results[f"m_Sigma_note_{tp_tag}"] = m_Sigma_note[tau]
                                results[f"m_Sigma_hat_{tp_tag}"] = m_Sigma_hat[tau]
                                if "circle" in variants:
                                    results[f"m_Sigma_circle_{tp_tag}"] = m_Sigma_circle[tau]
                                if "note_hat" in variants:
                                    results[f"m_Sigma_note_hat_{tp_tag}"] = m_Sigma_note_hat[tau]

                        feature_dir = os.path.join(base_dir, f"T{T}")
                        os.makedirs(feature_dir, exist_ok=True)

                        #----- Original C -----
                        if do_corr:
                            X = torch.flatten(C, start_dim=1).numpy()
                            X_train, X_test, y_train, y_test = (
                                X[train_idx_np], X[test_idx_np],
                                y_np_c[train_idx_np], y_np_c[test_idx_np],
                            )
                            fit_and_record(
                                X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                name="Original_C",
                                feature_dir=feature_dir, results=results,
                                estimator=estimator, cv_cfg=cv_cfg,
                            )

                        #----- Original Sigma -----
                        if do_cov:
                            X = torch.flatten(Sigma, start_dim=1).numpy()
                            X_train, X_test, y_train, y_test = (
                                X[train_idx_np], X[test_idx_np],
                                y_np_s[train_idx_np], y_np_s[test_idx_np],
                            )
                            fit_and_record(
                                X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                name="Original_Sigma",
                                feature_dir=feature_dir, results=results,
                                estimator=estimator, cv_cfg=cv_cfg,
                            )

                        # ----- Estimated features baselines (no conv) -----
                        # feat=raw → Estimated_C_hat / Estimated_Sigma_hat (byte-identical to legacy).
                        # feat=lw  → Estimated_C_check / Estimated_Sigma_check (new).
                        for _feat in feat_list:
                            if do_corr:
                                X = torch.flatten(feat_refined_C[_feat], start_dim=1).numpy()
                                X_train, X_test, y_train, y_test = (
                                    X[train_idx_np], X[test_idx_np],
                                    y_np_c[train_idx_np], y_np_c[test_idx_np],
                                )
                                fit_and_record(
                                    X_train=X_train, X_test=X_test,
                                    y_train=y_train, y_test=y_test,
                                    name=f"Estimated_{FEAT_LABEL_C[_feat]}",
                                    feature_dir=feature_dir, results=results,
                                    estimator=estimator, cv_cfg=cv_cfg,
                                )
                            if do_cov:
                                X = torch.flatten(feat_refined_S[_feat], start_dim=1).numpy()
                                X_train, X_test, y_train, y_test = (
                                    X[train_idx_np], X[test_idx_np],
                                    y_np_s[train_idx_np], y_np_s[test_idx_np],
                                )
                                fit_and_record(
                                    X_train=X_train, X_test=X_test,
                                    y_train=y_train, y_test=y_test,
                                    name=f"Estimated_{FEAT_LABEL_SIGMA[_feat]}",
                                    feature_dir=feature_dir, results=results,
                                    estimator=estimator, cv_cfg=cv_cfg,
                                )

                        for tau in tau_list_p:
                            tp_tag = f"{tau:.3f}".replace(".", "p")

                            for k_val in k_list:
                                #=========================================================
                                # GCN
                                #=========================================================
                                if "GCN" in conv_list:
                                    if do_corr and "bar" in variants:
                                        #----- GCN(A_bar, C) - Oracle -----
                                        feat = gcn(A=A_c_bar[tau], C=C, k=k_val)
                                        oracle_feat = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_bar_{tp_tag}, C)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        #----- GCN(A_bar, C_hat) - Denoised -----
                                        if do_corr:
                                            feat = gcn(A=A_c_bar[tau], C=C_hat, k=k_val)
                                            results[f"frob_GCN{k_val}(A_bar_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                            X = torch.flatten(feat, start_dim=1).numpy()
                                            X_train, X_test, y_train, y_test = (
                                                X[train_idx_np], X[test_idx_np],
                                                y_np_c[train_idx_np], y_np_c[test_idx_np],
                                            )
                                            name = f"GCN{k_val}(A_bar_{tp_tag}, C_hat)"
                                            fit_and_record(
                                                X_train=X_train, X_test=X_test,
                                                y_train=y_train, y_test=y_test,
                                                name=name,
                                                feature_dir=feature_dir, results=results,
                                                estimator=estimator, cv_cfg=cv_cfg,
                                            )

                                    if do_corr and "hat" in variants:
                                        #----- GCN(A_hat, C_hat) - Fully Estimated -----
                                        feat = gcn(A=A_c_hat[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_hat_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_hat_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "circle" in variants:
                                        #----- GCN(A_circle, C_hat) - Circle -----
                                        feat = gcn(A=A_c_circle[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_circle_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_circle_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "note" in variants:
                                        #----- GCN(A_note, C_hat) - Note -----
                                        feat = gcn(A=A_c_note[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_note_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_note_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "note_hat" in variants:
                                        #----- GCN(A_note_hat, C_hat) - Note_hat -----
                                        feat = gcn(A=A_c_note_hat[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_note_hat_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_note_hat_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "bar" in variants:
                                        #----- GCN(A_Sigma_bar, Sigma) - Oracle (cov) -----
                                        feat = gcn(A=A_Sigma_bar[tau], C=Sigma, k=k_val)
                                        oracle_feat_sig = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_bar_{tp_tag}, Sigma)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        #----- GCN(A_Sigma_bar, Sigma_hat) - Denoised (cov) -----
                                        feat = gcn(A=A_Sigma_bar[tau], C=Sigma_hat, k=k_val)
                                        results[f"frob_GCN{k_val}(A_Sigma_bar_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_bar_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "hat" in variants:
                                        #----- GCN(A_Sigma_hat, Sigma_hat) - Fully Estimated (cov) -----
                                        feat = gcn(A=A_Sigma_hat[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_Sigma_hat_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_hat_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "circle" in variants:
                                        feat = gcn(A=A_Sigma_circle[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_Sigma_circle_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_circle_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "note" in variants:
                                        feat = gcn(A=A_Sigma_note[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_Sigma_note_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_note_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "note_hat" in variants:
                                        feat = gcn(A=A_Sigma_note_hat[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GCN{k_val}(A_Sigma_note_hat_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GCN{k_val}(A_Sigma_note_hat_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                #=========================================================
                                # SageMean
                                #=========================================================
                                if "SageMean" in conv_list:
                                    if do_corr and "bar" in variants:
                                        feat = sage_mean(A=A_c_bar[tau], C=C, k=k_val)
                                        oracle_feat = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"SageMean{k_val}(A_bar_{tp_tag}, C)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        for _feat in feat_list:
                                            _M = feat_refined_C[_feat]
                                            _lbl = FEAT_LABEL_C[_feat]
                                            feat = sage_mean(A=A_c_bar[tau], C=_M, k=k_val)
                                            results[f"frob_SageMean{k_val}(A_bar_{tp_tag}, {_lbl})"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                            results[f"frob_SageMean{k_val}(A_bar_{tp_tag}, {_lbl})_to_C"] = float((feat - C).norm(dim=(-2, -1)).mean().item())
                                            X = torch.flatten(feat, start_dim=1).numpy()
                                            X_train, X_test, y_train, y_test = (
                                                X[train_idx_np], X[test_idx_np],
                                                y_np_c[train_idx_np], y_np_c[test_idx_np],
                                            )
                                            name = f"SageMean{k_val}(A_bar_{tp_tag}, {_lbl})"
                                            fit_and_record(
                                                X_train=X_train, X_test=X_test,
                                                y_train=y_train, y_test=y_test,
                                                name=name,
                                                feature_dir=feature_dir, results=results,
                                                estimator=estimator, cv_cfg=cv_cfg,
                                            )

                                    _c_variant_A = {
                                        "hat":      A_c_hat,
                                        "circle":   A_c_circle,
                                        "note":     A_c_note,
                                        "note_hat": A_c_note_hat,
                                    } if do_corr else {}
                                    for _vname, _A_dict in _c_variant_A.items():
                                        if not (do_corr and _vname in variants):
                                            continue
                                        for _feat in feat_list:
                                            _M = feat_refined_C[_feat]
                                            _lbl = FEAT_LABEL_C[_feat]
                                            feat = sage_mean(A=_A_dict[tau], C=_M, k=k_val)
                                            if "bar" in variants:
                                                results[f"frob_SageMean{k_val}(A_{_vname}_{tp_tag}, {_lbl})"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                            results[f"frob_SageMean{k_val}(A_{_vname}_{tp_tag}, {_lbl})_to_C"] = float((feat - C).norm(dim=(-2, -1)).mean().item())
                                            X = torch.flatten(feat, start_dim=1).numpy()
                                            X_train, X_test, y_train, y_test = (
                                                X[train_idx_np], X[test_idx_np],
                                                y_np_c[train_idx_np], y_np_c[test_idx_np],
                                            )
                                            name = f"SageMean{k_val}(A_{_vname}_{tp_tag}, {_lbl})"
                                            fit_and_record(
                                                X_train=X_train, X_test=X_test,
                                                y_train=y_train, y_test=y_test,
                                                name=name,
                                                feature_dir=feature_dir, results=results,
                                                estimator=estimator, cv_cfg=cv_cfg,
                                            )

                                    if do_cov and "bar" in variants:
                                        feat = sage_mean(A=A_Sigma_bar[tau], C=Sigma, k=k_val)
                                        oracle_feat_sig = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"SageMean{k_val}(A_Sigma_bar_{tp_tag}, Sigma)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        for _feat in feat_list:
                                            _M = feat_refined_S[_feat]
                                            _lbl = FEAT_LABEL_SIGMA[_feat]
                                            feat = sage_mean(A=A_Sigma_bar[tau], C=_M, k=k_val)
                                            results[f"frob_SageMean{k_val}(A_Sigma_bar_{tp_tag}, {_lbl})"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                            results[f"frob_SageMean{k_val}(A_Sigma_bar_{tp_tag}, {_lbl})_to_Sigma"] = float((feat - Sigma).norm(dim=(-2, -1)).mean().item())
                                            X = torch.flatten(feat, start_dim=1).numpy()
                                            X_train, X_test, y_train, y_test = (
                                                X[train_idx_np], X[test_idx_np],
                                                y_np_s[train_idx_np], y_np_s[test_idx_np],
                                            )
                                            name = f"SageMean{k_val}(A_Sigma_bar_{tp_tag}, {_lbl})"
                                            fit_and_record(
                                                X_train=X_train, X_test=X_test,
                                                y_train=y_train, y_test=y_test,
                                                name=name,
                                                feature_dir=feature_dir, results=results,
                                                estimator=estimator, cv_cfg=cv_cfg,
                                            )

                                    _S_variant_A = {
                                        "hat":      A_Sigma_hat,
                                        "circle":   A_Sigma_circle,
                                        "note":     A_Sigma_note,
                                        "note_hat": A_Sigma_note_hat,
                                    } if do_cov else {}
                                    for _vname, _A_dict in _S_variant_A.items():
                                        if not (do_cov and _vname in variants):
                                            continue
                                        for _feat in feat_list:
                                            _M = feat_refined_S[_feat]
                                            _lbl = FEAT_LABEL_SIGMA[_feat]
                                            feat = sage_mean(A=_A_dict[tau], C=_M, k=k_val)
                                            if "bar" in variants:
                                                results[f"frob_SageMean{k_val}(A_Sigma_{_vname}_{tp_tag}, {_lbl})"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                            results[f"frob_SageMean{k_val}(A_Sigma_{_vname}_{tp_tag}, {_lbl})_to_Sigma"] = float((feat - Sigma).norm(dim=(-2, -1)).mean().item())
                                            X = torch.flatten(feat, start_dim=1).numpy()
                                            X_train, X_test, y_train, y_test = (
                                                X[train_idx_np], X[test_idx_np],
                                                y_np_s[train_idx_np], y_np_s[test_idx_np],
                                            )
                                            name = f"SageMean{k_val}(A_Sigma_{_vname}_{tp_tag}, {_lbl})"
                                            fit_and_record(
                                                X_train=X_train, X_test=X_test,
                                                y_train=y_train, y_test=y_test,
                                                name=name,
                                                feature_dir=feature_dir, results=results,
                                                estimator=estimator, cv_cfg=cv_cfg,
                                            )

                                #=========================================================
                                # GAT
                                #=========================================================
                                if "GAT" in conv_list:
                                    if do_corr and "bar" in variants:
                                        feat = gat(A=A_c_bar[tau], C=C, k=k_val)
                                        oracle_feat = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_bar_{tp_tag}, C)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        feat = gat(A=A_c_bar[tau], C=C_hat, k=k_val)
                                        results[f"frob_GAT{k_val}(A_bar_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_bar_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "hat" in variants:
                                        feat = gat(A=A_c_hat[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_hat_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_hat_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "circle" in variants:
                                        feat = gat(A=A_c_circle[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_circle_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_circle_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "note" in variants:
                                        feat = gat(A=A_c_note[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_note_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_note_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_corr and "note_hat" in variants:
                                        feat = gat(A=A_c_note_hat[tau], C=C_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_note_hat_{tp_tag}, C_hat)"] = float((feat - oracle_feat).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_c[train_idx_np], y_np_c[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_note_hat_{tp_tag}, C_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "bar" in variants:
                                        feat = gat(A=A_Sigma_bar[tau], C=Sigma, k=k_val)
                                        oracle_feat_sig = feat
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_bar_{tp_tag}, Sigma)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                        feat = gat(A=A_Sigma_bar[tau], C=Sigma_hat, k=k_val)
                                        results[f"frob_GAT{k_val}(A_Sigma_bar_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_bar_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "hat" in variants:
                                        feat = gat(A=A_Sigma_hat[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_Sigma_hat_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_hat_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "circle" in variants:
                                        feat = gat(A=A_Sigma_circle[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_Sigma_circle_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_circle_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "note" in variants:
                                        feat = gat(A=A_Sigma_note[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_Sigma_note_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_note_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                                    if do_cov and "note_hat" in variants:
                                        feat = gat(A=A_Sigma_note_hat[tau], C=Sigma_hat, k=k_val)
                                        if "bar" in variants:
                                            results[f"frob_GAT{k_val}(A_Sigma_note_hat_{tp_tag}, Sigma_hat)"] = float((feat - oracle_feat_sig).norm(dim=(-2, -1)).mean().item())
                                        X = torch.flatten(feat, start_dim=1).numpy()
                                        X_train, X_test, y_train, y_test = (
                                            X[train_idx_np], X[test_idx_np],
                                            y_np_s[train_idx_np], y_np_s[test_idx_np],
                                        )
                                        name = f"GAT{k_val}(A_Sigma_note_hat_{tp_tag}, Sigma_hat)"
                                        fit_and_record(
                                            X_train=X_train, X_test=X_test,
                                            y_train=y_train, y_test=y_test,
                                            name=name,
                                            feature_dir=feature_dir, results=results,
                                            estimator=estimator, cv_cfg=cv_cfg,
                                        )

                        if log_enabled:
                            log.info(f"Completed T={T} for n={n}, p={p}, snr={snr}")
                        with open(NDJSON_PATH, "a") as f:
                            f.write(json.dumps(results, separators=(",", ":")) + "\n")
                            f.flush()
                            os.fsync(f.fileno())

                    except Exception as e:
                        tqdm.write(f"FAILED on seed={seed}, n={n}, p={p}, T={T}, snr={snr} with error: {e}")
                        failure_log = {"seed": seed, "n": n, "p": p, "T": T, "snr": snr, "error": str(e), "run_timestamp": datetime.now().isoformat()}
                        with open(NDJSON_PATH, "a") as f:
                            f.write(json.dumps(failure_log) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                    finally:
                        pbar.update(1)

elapsed = time.time() - start_time
if log_enabled:
    log.info(f"Experiment completed in {elapsed:.2f} seconds")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Saved results to: {NDJSON_PATH}")
