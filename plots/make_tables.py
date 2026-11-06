"""Tables 2 and 5-9: the controlled real-data experiments.

    python plots/make_tables.py [--results outputs/real_data] [--out tables]

Reads every row under --results and matches them on their recorded settings, so
the directory layout does not matter. For each window length T the table gives
the five graphs (mean and standard deviation over seeds) and the four gaps that
isolate the three effects:

    smoothing             A_oracle - C_hat     (bar - none)
    graph estimation      A_perp - A_oracle    (circle - bar)
    double dipping        A_hat - A_perp       (window - circle)
    population graph      A_pop - C_hat        (zero - none)

Stars: paired t-test over seeds against C_hat (scores) or zero (gaps),
* p<0.10, ** p<0.05, *** p<0.01. Writes one Markdown file per table and the
same rows in the LaTeX format of the paper.
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics as st

from scipy import stats

GRAPHS = ["none", "bar", "circle", "window", "zero"]
GAPS = [("bar", "none"), ("circle", "bar"), ("window", "circle"), ("zero", "none")]
HEADER = ("| T | C_hat | A_oracle | A_perp | A_hat | A_pop | A_oracle − C_hat | A_perp − A_oracle "
          "| A_hat − A_perp | A_pop − C_hat |\n|---|---|---|---|---|---|---|---|---|---|")

ABIDE = ("abide", "rois_ho", "classification", "acc_subj")
ADHD = ("adhd", "cc200", "regression", "R2_subj")
FRANKFURT = ("frankfurt", "small", "regression", "R2")
SIM_LIKE = ("identity", "vec", "fixed")
DEPLOYED = ("relu", "mean", "free")

# table -> (title, cell, metric, [(block title, conv, activation, readout, n_mode, tau)])
TABLES = {
    "table2_abide": ("ABIDE autism diagnosis, subject-level accuracy", ABIDE,
                     [("Simulation-like (identity, vectorized readout, n fixed)", "sage", *SIM_LIKE, 0.10)]),
    "table5_relaxation": ("Progressive relaxation on ABIDE, subject-level accuracy", ABIDE,
                          [("ReLU (vectorized readout, n fixed)", "sage", "relu", "vec", "fixed", 0.10),
                           ("ReLU and n free (vectorized readout)", "sage", "relu", "vec", "free", 0.10),
                           ("ReLU, n free and mean pooling (deployed pipeline)", "sage", *DEPLOYED, 0.10)]),
    "table6_gat": ("GAT on ABIDE, subject-level accuracy", ABIDE,
                   [("Simulation-like (identity, vectorized readout, n fixed)", "gat", *SIM_LIKE, 0.10),
                    ("Deployed pipeline (ReLU, mean pooling, n free)", "gat", *DEPLOYED, 0.10)]),
    "table7_adhd": ("ADHD-200 inattention score, subject-level R2", ADHD,
                    [("Simulation-like (identity, vectorized readout, n fixed)", "sage", *SIM_LIKE, 0.10),
                     ("Deployed pipeline (ReLU, mean pooling, n free)", "sage", *DEPLOYED, 0.10)]),
    "table8_frankfurt": ("Frankfurt sleep fraction, window-level R2", FRANKFURT,
                         [("Simulation-like (identity, vectorized readout, n fixed)", "sage", *SIM_LIKE, 0.10),
                          ("Deployed pipeline (ReLU, mean pooling, n free)", "sage", *DEPLOYED, 0.10)]),
    "table9_density": ("Graph density on ABIDE, subject-level accuracy", ABIDE,
                       [("tau = 0.05", "sage", *SIM_LIKE, 0.05), ("tau = 0.20", "sage", *SIM_LIKE, 0.20)]),
}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def paired(A, T, a, b):
    seeds = sorted(set(A[(T, a)]) & set(A[(T, b)]))
    d = [A[(T, a)][s] - A[(T, b)][s] for s in seeds]
    if len(d) < 2:
        return float("nan"), float("nan"), 1.0, len(d)
    m, sd = st.mean(d), st.stdev(d)
    p = 2 * stats.t.sf(abs(m / (sd / math.sqrt(len(d)))), len(d) - 1) if sd > 0 else 0.0
    return m, sd, p, len(d)


def block(rows, cell, conv, act, ro, nm, tau):
    """(T, graph) -> {seed: score}. The no-convolution arm does not depend on tau."""
    ds, parc, task, metric = cell
    A = collections.defaultdict(dict)
    for r in rows:
        if (r["dataset"], r["parcellation"], r["task"], r["conv_type"], r["activation"], r["readout"],
                r["n_mode"]) != (ds, parc, task, conv, act, ro, nm):
            continue
        if r["adjacency"] != "none" and not math.isclose(float(r["tau"]), tau):
            continue
        A[(int(r["time"]), r["adjacency"])][r["seed"]] = r[metric]
    return A


def sd3(x):
    return f"{x:.3f}".lstrip("0")


def render(A):
    md, tex = [], []
    for T in sorted({k[0] for k in A}):
        cm, ct = [], []
        for g in GRAPHS:
            v = list(A[(T, g)].values())
            if len(v) < 2:
                cm.append("n/a"); ct.append("--")
                continue
            s = "" if g == "none" else stars(paired(A, T, g, "none")[2])
            cm.append(f"{st.mean(v):.3f} ± {st.stdev(v):.3f}{s}")
            ct.append(f"\\msd{{{st.mean(v):.3f}}}{{{s}}}{{{sd3(st.stdev(v))}}}")
        for a, b in GAPS:
            m, sd, p, n = paired(A, T, a, b)
            cm.append(f"{m:+.3f} ± {sd:.3f}{stars(p)}" if n > 1 else "n/a")
            ct.append(f"\\gs{{{m:+.3f}}}{{{stars(p)}}}" if n > 1 else "--")
        md.append(f"| {T} | " + " | ".join(cm) + " |")
        tex.append(f"{T} & " + " & ".join(ct) + " \\\\")
    return md, tex


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join("outputs", "real_data"))
    p.add_argument("--out", default="tables")
    a = p.parse_args()

    files = glob.glob(os.path.join(a.results, "**", "*.jsonl"), recursive=True)
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("model") == "gnn" and r.get("readout") is not None]
    if not rows:
        raise SystemExit(f"no controlled-experiment results in {a.results}")
    os.makedirs(a.out, exist_ok=True)

    for name, (title, cell, blocks) in TABLES.items():
        md = [f"# {title}", "",
              "Mean ± SD over seeds; paired t-test over seeds against C_hat (scores) or zero (gaps): "
              "* p<0.10, ** p<0.05, *** p<0.01.", ""]
        tex = []
        for btitle, *spec in blocks:
            A = block(rows, cell, *spec)
            if not A:
                continue
            seeds = min(len(v) for v in A.values())
            bm, bt = render(A)
            md += [f"## {btitle} (min. {seeds} seeds per cell)", "", HEADER, *bm, ""]
            tex += [f"\\multicolumn{{10}}{{l}}{{{btitle}}} \\\\", *bt, "\\midrule"]
        if not tex:
            print(f"{name}: no data")
            continue
        open(os.path.join(a.out, f"{name}.md"), "w").write("\n".join(md) + "\n")
        open(os.path.join(a.out, f"{name}.tex"), "w").write("\n".join(tex[:-1]) + "\n")
        print(f"{name}: written {os.path.join(a.out, name)}.md and .tex")


if __name__ == "__main__":
    main()
