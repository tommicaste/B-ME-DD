"""Figures 3-9: the simulations under the latent community model.

    python plots/plot_simulations.py [--results outputs/simulations] [--out figures]

Reads the ndjson rows written by `simulations/run_simulations.py` (the commands
from `experiments/make_jobs.py simulations`). Every curve is the test MSE of OLS on
the feature map divided by the test MSE of OLS on the true connectome, mean and
standard deviation over seeds.
"""
import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulations.simulate_utils import generate_geometric_random_graph_samples
from utils.viz_config import (BASE_KWARGS, BETA_PRETTY, ERRBAR_KWARGS, SIG_TEX, VAR_STYLE,
                              VAR_TEX, set_style)

GRG = dict(grg_d=10, grg_K=5, grg_phi=1.0, grg_psi=0.1, grg_centroid_mode="orthogonal")
MATRIX_SEED = 30
VARIANTS = ["bar", "hat", "circle", "note"]
SIGNAL_IN = {"Sigma": "Sigma_hat", "C": "C_hat"}
FEAT_RE = re.compile(
    r"^ValMSE_(?P<conv>SageMean|GCN|GAT)(?P<k>\d+)\("
    r"A_(?P<sig_side>Sigma_)?(?P<variant>bar|hat|circle|note_hat|note)_(?P<tau>\dp\d+),\s*"
    r"(?P<signal_in>[A-Za-z_]+)\)$")
CONFIG = ["matrix_seed", "n", "p", "T", "snr", "simulation_mode", "estimation_type", "beta_type",
          "estimator", "sigma_method", "graph_method", "grg_d", "grg_K", "grg_phi", "grg_sigma",
          "grg_psi", "grg_centroid_mode"]


# ------- loading -------

def load(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "**", "regression_*_grg_*.ndjson"), recursive=True))
    if not paths:
        raise SystemExit(f"no simulation results in {results_dir}; run the simulations first")
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    failed = [r for r in rows if "grg_args" not in r]
    if failed:
        print(f"[load] skipping {len(failed)} rows that record a failed run, e.g.: {failed[0].get('error', '')[-200:]}")
    rows = [r for r in rows if "grg_args" in r]
    if not rows:
        raise SystemExit("every simulation row records a failed run; see the 'error' field in the ndjson files")
    df = pd.DataFrame(rows)
    df = pd.concat([df.drop(columns=["grg_args"]), df["grg_args"].apply(pd.Series).add_prefix("grg_")], axis=1)
    for col, default in (("estimator", "ols"), ("sigma_method", "raw"), ("graph_method", "identity")):
        df[col] = df.get(col, pd.Series(default, index=df.index)).fillna(default)
    # Only the thresholded sample estimate enters the figures (no shrinkage, glasso or LCT graphs).
    if "adjacency_method" in df.columns:
        df = df[df["adjacency_method"].fillna("threshold") == "threshold"]
    df = df[(df["sigma_method"] == "raw") & (df["graph_method"] == "identity")]

    # Rows of the same configuration and seed are merged column by column; the row
    # with the most feature maps wins, then the most recent one.
    keys = [c for c in ["seed", "adjacency_type", "signal"] + CONFIG if c in df.columns]
    feats = [c for c in df.columns if c.startswith(("ValMSE_SageMean", "ValMSE_GCN", "ValMSE_GAT"))]
    df = df.assign(_complete=df[feats].notna().sum(axis=1))
    df = (df.sort_values(["_complete", "run_timestamp"])
            .groupby(keys, as_index=False, sort=False, dropna=False).last().drop(columns="_complete"))

    # Keep the seeds shared by every configuration, so each curve averages the same seeds.
    seeds = df.groupby([k for k in keys if k != "seed"], dropna=False)["seed"].agg(set)
    common = set.intersection(*seeds.tolist())
    if common:
        df = df[df["seed"].isin(common)]
    print(f"[load] {len(df)} rows from {len(paths)} files, seeds {sorted(common) or 'per configuration'}")
    return df.reset_index(drop=True)


def feature_maps(df):
    """Long table: one row per (configuration, seed, feature map), normalised by the oracle."""
    ids = [c for c in ["seed"] + CONFIG if c in df.columns]
    parts = []
    for col in df.columns:
        m = FEAT_RE.match(col)
        if not m:
            continue
        d = m.groupdict()
        side = "Sigma" if d["sig_side"] else "C"
        part = df[ids].copy()
        part["mse_norm"] = df[col] / df[f"ValMSE_Original_{side}"]
        part["conv"], part["k"], part["variant"] = d["conv"], int(d["k"]), d["variant"]
        part["tau"], part["side"], part["signal_in"] = float(d["tau"].replace("p", ".")), side, d["signal_in"]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["mse_norm"])
    group = [c for c in ids if c != "seed"] + ["conv", "k", "variant", "tau", "side", "signal_in"]
    return long.groupby(group, as_index=False, dropna=False)["mse_norm"].agg(mean="mean", std="std")


def baseline(df):
    """No-convolution baseline: OLS on the raw estimate, normalised by the oracle."""
    group = [c for c in CONFIG if c in df.columns]
    out = []
    for side, est in (("Sigma", "Sigma_hat"), ("C", "C_hat")):
        b = df[group].copy()
        b["mse_norm"] = df[f"ValMSE_Estimated_{est}"] / df[f"ValMSE_Original_{side}"]
        b["side"] = side
        out.append(b.groupby(group + ["side"], as_index=False, dropna=False)["mse_norm"].agg(mean="mean", std="std"))
    return pd.concat(out, ignore_index=True)


def select(x, **cfg):
    cfg = {**GRG, "n": 25000, "p": 20, "snr": 1.0, "estimator": "ols", "simulation_mode": "iid",
           "conv": "SageMean", "k": 1, "tau": 0.10, **cfg}
    for k, v in cfg.items():
        if k not in x.columns:
            continue
        x = x[np.isclose(x[k], v)] if k == "tau" else x[x[k] == v]
    return x[x["T"] <= 100]


def delta_of(df, side, tau, **cfg):
    """Absolute threshold delta that the density tau gives on the true matrices."""
    col = f"delta_{'Sigma' if side == 'Sigma' else 'c'}_{tau:.3f}".replace("0.", "0p")
    rows = select(df, **cfg)
    return float(rows[col].dropna().mean()) if col in rows.columns else float("nan")


# ------- drawing -------

def curves(ax, fm, base, side, **cfg):
    """Draw every feature map and the baseline; returns the number of curves drawn."""
    sig = SIG_TEX[SIGNAL_IN[side]]
    drawn = 0
    val = select(fm[(fm["side"] == side) & (fm["signal_in"] == SIGNAL_IN[side])], **cfg)
    for v in VARIANTS:
        s = val[val["variant"] == v].sort_values("T")
        if s.empty:
            continue
        st = VAR_STYLE[v]
        ax.errorbar(s["T"], s["mean"], yerr=s["std"].fillna(0), **ERRBAR_KWARGS, color=st["color"],
                    linestyle=st["ls"], mfc=st["color"], mec=st["color"], label=rf"$R({VAR_TEX[v]},\,{sig})$")
        drawn += 1
    b = select(base[base["side"] == side], **{k: v for k, v in cfg.items() if k not in ("conv", "k", "tau")})
    b = b.sort_values("T")
    if not b.empty:
        ax.errorbar(b["T"], b["mean"], yerr=b["std"].fillna(0), **BASE_KWARGS, label=rf"${sig}$ (no conv)")
    ax.grid(True, ls=":", alpha=0.5, lw=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)
    return drawn


def legend(fig, axes, y):
    """Figure legend from the panel with the most curves (a panel can be empty)."""
    h, l = max((ax.get_legend_handles_labels() for ax in np.ravel(axes)), key=lambda hl: len(hl[0]))
    if h:
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(h), frameon=False)


def panel_row(fm, base, panels, figsize, legend_y, ylabel="ValMSE / Oracle", side="Sigma", sharey=True):
    """One row of panels; `panels` is a list of (title, selection) pairs."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=sharey, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    drawn = []
    for j, (title, cfg) in enumerate(panels):
        drawn.append(curves(axes[j], fm, base, side, **cfg))
        axes[j].set_title(title)
        axes[j].set_xlabel("T")
    axes[0].set_ylabel(ylabel)
    if not all(drawn):
        plt.close(fig)
        return None
    legend(fig, axes, legend_y)
    return fig


def make_B(beta_type, labels, p, community_idx):
    """Support of the coefficient matrix, for the overlays on the sample connectome."""
    target = torch.unique(labels)[community_idx % len(torch.unique(labels))]
    members = torch.nonzero(labels == target, as_tuple=False).flatten()
    B = torch.zeros((p, p))
    if beta_type == "single_community_aligned":
        inside = torch.zeros(p, dtype=torch.bool)
        inside[members] = True
        B = ((inside[:, None] & inside[None, :]) & ~torch.eye(p, dtype=torch.bool)).float()
    elif beta_type == "within_community_contrast":
        a, b = int(members[0]), int(members[1])
        u = torch.zeros(p); u[a], u[b] = 1.0, -1.0
        v = (labels == target).float()
        if int(v.sum()) > 2:
            v[a] = v[b] = 0.0
        B = 0.5 * (u[:, None] * v[None, :] + v[:, None] * u[None, :])
    return B.numpy()


def outline(ax, B, color, lw=0.8):
    nz = np.abs(B) > 1e-12
    H, W = nz.shape
    for i in range(H):
        for j in range(W):
            if not nz[i, j]:
                continue
            if i == 0 or not nz[i - 1, j]: ax.plot([j - .5, j + .5], [i - .5, i - .5], color=color, lw=lw)
            if i == H - 1 or not nz[i + 1, j]: ax.plot([j - .5, j + .5], [i + .5, i + .5], color=color, lw=lw)
            if j == 0 or not nz[i, j - 1]: ax.plot([j - .5, j - .5], [i - .5, i + .5], color=color, lw=lw)
            if j == W - 1 or not nz[i, j + 1]: ax.plot([j + .5, j + .5], [i - .5, i + .5], color=color, lw=lw)


def main_figure(fm, base, side):
    """Figures 3 (side='Sigma') and 5 (side='C')."""
    left, right = "single_community_aligned", "within_community_contrast"
    sigmas = (0.5, 1.0)
    mats, labels = {}, {}
    for s in sigmas:
        Sig, C, meta = generate_geometric_random_graph_samples(
            n=1, p=20, d=GRG["grg_d"], K=GRG["grg_K"], phi=GRG["grg_phi"], sigma=s, psi=GRG["grg_psi"],
            seed=MATRIX_SEED, centroid_mode=GRG["grg_centroid_mode"])
        mats[s] = (Sig if side == "Sigma" else C).squeeze(0).cpu().numpy()
        labels[s] = meta["assigned_indices"].to(torch.long)
    vmin, vmax = min(m.min() for m in mats.values()), max(m.max() for m in mats.values())

    fig, axes = plt.subplots(2, 3, figsize=(5.5, 2.75), gridspec_kw={"width_ratios": [1.4, 0.55, 1.4]},
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.0, wspace=0.10, hspace=0.0)
    drawn = []
    for i, s in enumerate(sigmas):
        drawn.append(curves(axes[i, 0], fm, base, side, beta_type=left, grg_sigma=s))
        axes[i, 0].set_xlabel("T")
        axes[i, 0].set_ylabel(rf"$\sigma={s}$" + "\nValMSE / Oracle")
        im = axes[i, 1].imshow(mats[s], cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        outline(axes[i, 1], make_B(left, labels[s], 20, 0), "red")
        outline(axes[i, 1], make_B(right, labels[s], 20, 3), "blue")
        drawn.append(curves(axes[i, 2], fm, base, side, beta_type=right, grg_sigma=s))
        axes[i, 2].set_xlabel("T")
    axes[0, 0].set_title(BETA_PRETTY[left])
    axes[0, 1].set_title(r"Sample $\Sigma$" if side == "Sigma" else r"sample $C$")
    axes[0, 2].set_title(BETA_PRETTY[right])
    make_axes_locatable(axes[0, 1]).append_axes("bottom", size="6%", pad=0.05).set_visible(False)
    cax = make_axes_locatable(axes[1, 1]).append_axes("bottom", size="6%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="horizontal")
    if not all(drawn):
        plt.close(fig)
        return None
    legend(fig, axes[:, [0, 2]], -0.08)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join("outputs", "simulations"))
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.results)
    fm, base = feature_maps(df), baseline(df)
    set_style()

    def ar_title(rho):
        return rf"AR(1), $\rho={rho:g}$"

    figs = {
        "fig3_main_sigma": main_figure(fm, base, "Sigma"),
        "fig4_threshold": panel_row(
            fm, base,
            [(rf"$\tau={t:.3f},\ \delta={delta_of(df, 'Sigma', t, beta_type='community_aligned', grg_sigma=1.0):.3f}$",
              dict(beta_type="community_aligned", grg_sigma=1.0, tau=t)) for t in (0.10, 0.158, 0.50)],
            (5.5, 1.7), -0.13),
        "fig5_main_correlation": main_figure(fm, base, "C"),
        "fig6_coefficient": panel_row(
            fm, base,
            [(BETA_PRETTY[b], dict(beta_type=b, grg_sigma=1.0))
             for b in ("ones", "community_aligned", "within_community_contrast", "edge_specific")],
            (5.5, 2.2), -0.10),
        "fig7_convolution": panel_row(
            fm, base,
            [(c, dict(beta_type="single_community_aligned", grg_sigma=1.0, conv=c)) for c in ("SageMean", "GCN")],
            (5.5, 2.2), -0.10),
        "fig8_temporal_correlation": panel_row(
            fm, base,
            [(t, dict(beta_type="single_community_aligned", grg_sigma=1.0, simulation_mode=m))
             for t, m in (("iid", "iid"), (ar_title(0.4), "ar_0p400"), (ar_title(0.7), "ar_0p700"))],
            (5.5, 2.2), -0.10),
        "fig9_sample_size": panel_row(
            fm, base,
            [(rf"$n={n}$", dict(beta_type="single_community_aligned", grg_sigma=1.0, n=n)) for n in (400, 800, 25000)],
            (5.5, 1.7), -0.13, sharey=False),
    }
    for name, fig in figs.items():
        if fig is None:
            print(f"skipped {name}: some of its panels have no results yet")
            continue
        path = os.path.join(a.out, f"{name}.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
