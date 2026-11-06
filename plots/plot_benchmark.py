"""Figure 1: graph-agnostic baselines against GNNs on the three datasets.

    python plots/plot_benchmark.py [--results outputs/real_data/benchmark] [--out figures]

Reads the rows written by the commands from `experiments/make_jobs.py benchmark`.
Window-level test MSE (ABIDE age) or accuracy (ADHD diagnosis, Frankfurt sleep),
mean and standard deviation over seeds.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.viz_config import set_style

NAMES = {
    "RandomForest": "Random Forest", "Ridge": "Linear Model", "SVR": "SVM", "SVM": "SVM", "mlp": "MLP",
    "gat_1conv": "GAT 1-Conv", "gat_2conv": "GAT 2-Conv", "gcn_1conv": "GCN 1-Conv", "gcn_2conv": "GCN 2-Conv",
    "sage_1conv": "SAGE 1-Conv", "sage_2conv": "SAGE 2-Conv", "residualgnn": "Residual GNN",
}
ORDER = ["Random Forest", "Linear Model", "SVM", "MLP", "GAT 1-Conv", "GAT 2-Conv",
         "GCN 1-Conv", "GCN 2-Conv", "SAGE 1-Conv", "SAGE 2-Conv", "Residual GNN"]
PANELS = [
    ("abide", "regression", "ABIDE — Regression", "MSE"),
    ("adhd", "classification", "ADHD — Classification", "Accuracy"),
    ("frankfurt", "classification", "Frankfurt — Classification", "Accuracy"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join("outputs", "real_data", "benchmark"))
    p.add_argument("--out", default="figures")
    a = p.parse_args()

    files = glob.glob(os.path.join(a.results, "**", "*.jsonl"), recursive=True)
    if not files:
        raise SystemExit(f"no benchmark results in {a.results}; run the benchmark first")
    df = pd.DataFrame([json.loads(l) for f in files for l in open(f) if l.strip()])
    df["score"] = np.where(df["task"] == "classification", df.get("test_acc"), df.get("test_mse"))
    df["name"] = df["variant"].map(NAMES)
    df = df.dropna(subset=["name", "score"]).drop_duplicates(["dataset", "task", "variant", "seed"], keep="last")

    set_style()
    colors = dict(zip(ORDER, plt.cm.Paired(np.linspace(0, 0.75, len(ORDER)))))
    fig, axes = plt.subplots(3, 1, figsize=(2.25, 2.25))
    fig.subplots_adjust(left=0.18, right=0.97, hspace=0.5)
    for i, (ax, (ds, task, title, ylabel)) in enumerate(zip(axes, PANELS)):
        sub = df[(df["dataset"] == ds) & (df["task"] == task)]
        stats = sub.groupby("name")["score"].agg(["mean", "std"]).reindex(ORDER).dropna(how="all")
        for name, row in stats.iterrows():
            ax.errorbar(name, row["mean"], yerr=row["std"], fmt="_", color=colors[name],
                        capsize=1.5, linewidth=0.6, elinewidth=0.6, markersize=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i < len(PANELS) - 1:
            ax.set_xticks([])
        else:
            ax.tick_params(axis="x", labelrotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        print(f"{title}: {len(stats)} models, seeds per model "
              f"{sub.groupby('name')['seed'].nunique().reindex(stats.index).tolist()}")
    fig.align_ylabels(axes)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "fig1_benchmark.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
