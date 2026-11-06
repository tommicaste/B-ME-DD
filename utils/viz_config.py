import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def set_style():
    colors = plt.cm.Paired(np.linspace(0, 1, 12))
    settings = {
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.6,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.prop_cycle": cycler(color=colors),
        "image.cmap": "viridis",
    }
    plt.rcParams.update(settings)


set_style()


VAR_TEX = {
    "bar":      r"A^{\mathrm{oracle}}",
    "hat":      r"\hat A",
    "circle":   r"A^{\perp}",
    "note":     r"A^{\mathrm{pop}}",
    "note_hat": r"\hat A^{\mathrm{pop}}",
}

VAR_STYLE = {
    "bar":      {"color": "#1f77b4", "ls": "-",  "marker": "o"},
    "hat":      {"color": "#d62728", "ls": "--", "marker": "s"},
    "circle":   {"color": "#2ca02c", "ls": "-.", "marker": "^"},
    "note":     {"color": "#ff7f0e", "ls": "-",  "marker": "D"},
    "note_hat": {"color": "#9467bd", "ls": (0, (3, 1, 1, 1)), "marker": "v"},
}

SIG_TEX = {
    "C":         r"C",
    "C_hat":     r"\hat C",
    "Sigma":     r"\Sigma",
    "Sigma_hat": r"\hat \Sigma",
}

BETA_PRETTY = {
    "ones":                       "Ones",
    "constant":                   "Constant",
    "random":                     "Random",
    "community_aligned":          "Community-aligned",
    "off_community_aligned":      "Off-community-aligned",
    "single_community_aligned":   "Single-community-aligned",
    "edge_specific":              "Edge-specific",
    "within_community_contrast":  "Within-community contrast",
}

ERRBAR_KWARGS = dict(
    marker="o",
    ms=1.5,
    lw=0.9,
    elinewidth=1.0,
    capsize=2,
    capthick=0.25,
    alpha=0.75,
)

BASE_KWARGS = dict(
    marker="o",
    ms=1.5,
    lw=0.9,
    ls=(0, (1, 1)),
    color="black",
    mfc="black",
    mec="black",
    elinewidth=1.0,
    capsize=2,
    capthick=0.25,
    alpha=0.75,
)
