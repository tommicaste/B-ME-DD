"""Write the command list behind each figure and table of the paper.

    python experiments/make_jobs.py abide > jobs_abide.txt
    python experiments/run_jobs.py jobs_abide.txt --workers 8

One command per line. Every real-data command writes to its own output directory
(the result file name only carries dataset, task and seed), so `run_jobs.py` can
skip finished commands and resume an interrupted grid.

Experiments
-----------
quickstart    Figures 3, 5  one seed of the simulations behind the main figure (CPU, minutes)
simulations   Figures 3-9   latent community model, 5 seeds
benchmark     Figure 1      model zoo on ABIDE, ADHD and Frankfurt, 10 seeds
abide         Table 2       ABIDE autism diagnosis, simulation-like setting, 20 seeds
relaxation    Table 5       Table 2 with ReLU, then n free, then mean pooling
gat           Table 6       GAT in place of SageMean
adhd          Table 7       ADHD-200 inattention score
frankfurt     Table 8       Frankfurt sleep fraction
density       Table 9       graph density tau = 0.05 and 0.20
all           every real-data experiment above (not the simulations)
"""
import argparse
import os
import sys

HPARAMS = os.path.join("experiments", "hparams")  # commands run from the repository root
GRAPHS = ("none", "window", "circle", "zero", "bar")

SIM_COMMON = ("--matrix_seed 30 --SNR_list 1 --T_list 5 10 20 30 40 50 60 70 80 90 100 "
              "--k_list 1 --variants bar circle hat note --estimation_type both "
              "--grg_d 10 --grg_K 5 --grg_phi 1.0 --grg_psi 0.1 --centroid_mode orthogonal "
              "--p_list 20 --tau_list 0.10 0.50")
BETAS = ("ones", "community_aligned", "single_community_aligned",
         "within_community_contrast", "edge_specific")

ABIDE = dict(key="abide_rois_ho_classification",
             flags="--dataset abide --task classification --target DX_GROUP --parcellation rois_ho "
                   "--roi_drop_frac 0.01 --cohort_source_T 50",
             hparams="abide_rois_ho_classification.json", Ts=(5, 15, 25, 40))
ADHD = dict(key="adhd_cc200_regression",
            flags="--dataset adhd --task regression --target inattentive --parcellation cc200 "
                  "--cohort_source_T 50",
            hparams="adhd_cc200_regression.json", Ts=(5, 7, 15, 25, 40))
FRANKFURT = dict(key="frankfurt_small_regression",
                 flags="--dataset frankfurt --task regression --target sleep_frac --parcellation small "
                       "--cohort_source_T 60",
                 hparams="frankfurt_small_regression.json", Ts=(5, 8, 14, 25, 45))

SIM_LIKE = ("identity", "vec", "fixed")
DEPLOYED = ("relu", "mean", "free")


def py(module_or_script, args):
    if module_or_script.endswith(".py"):
        return f"python {module_or_script} {args}"
    return f"python -m {module_or_script} {args}"


def controlled(ds, conv, scenarios, out, seeds=range(1, 21), taus=(0.10,), graphs=GRAPHS,
               patience=25, max_epochs=200):
    """The controlled grid: one convolution, frozen hyperparameters, five graphs."""
    for act, ro, nm in scenarios:
        for tau in taus:
            for T in ds["Ts"]:
                for seed in seeds:
                    for adj in graphs:
                        odir = os.path.join(out, ds["key"], f"T{T}_{nm}", f"gnn-{conv}", adj,
                                            f"{ro}_{act}", f"tau{tau:.2f}")
                        yield py("real_data.benchmark.run_experiment",
                                 f"{ds['flags']} --model gnn --conv_type {conv} --num_convs 1 "
                                 f"--stride_frac 1.0 --tau {tau:.2f} --es_monitor val_loss "
                                 f"--patience {patience} --max_epochs {max_epochs} --batch_size 64 "
                                 f"--fixed_params {os.path.join(HPARAMS, ds['hparams'])} "
                                 f"--time {T} --n_mode {nm} --readout {ro} --activation {act} "
                                 f"--seed {seed} --adjacency {adj} --out_dir {odir}")


def sim_dir(out, mode, sigma, beta, seed):
    """One folder per command: runs with the same seed would otherwise append to one file."""
    return os.path.join(out, f"{mode}_sigma{sigma}_{beta}_seed{seed}")


def quickstart(out):
    sim = os.path.join("simulations", "run_simulations.py")
    for sigma in (0.5, 1.0):
        for beta in ("single_community_aligned", "within_community_contrast"):
            yield py(sim, f"--seed 1 {SIM_COMMON} --grg_sigma {sigma} --beta_type {beta} "
                          f"--simulation_mode iid --n_list 25000 --conv_list SageMean "
                          f"--output_dir {sim_dir(out, 'iid', sigma, beta, 1)}")


def simulations(out):
    sim = os.path.join("simulations", "run_simulations.py")
    for seed in range(1, 6):
        for sigma in (0.5, 1.0):
            for beta in BETAS:
                if sigma == 1.0 and beta == "single_community_aligned":
                    extra = "--n_list 400 800 25000 --conv_list SageMean GCN"
                else:
                    extra = "--n_list 25000 --conv_list SageMean"
                yield py(sim, f"--seed {seed} {SIM_COMMON} --grg_sigma {sigma} --beta_type {beta} "
                              f"--simulation_mode iid {extra} --output_dir {sim_dir(out, 'iid', sigma, beta, seed)}")
        for rho in (0.4, 0.7):
            yield py(sim, f"--seed {seed} {SIM_COMMON} --grg_sigma 1.0 "
                          f"--beta_type single_community_aligned --simulation_mode ar1 --rho {rho} "
                          f"--n_list 25000 --conv_list SageMean "
                          f"--output_dir {sim_dir(out, f'ar{rho}', 1.0, 'single_community_aligned', seed)}")


BENCH_PANELS = (
    ("abide_regression", "--dataset abide --task regression --target AGE_AT_SCAN "
                         "--parcellation rois_cc200 --time 150 --stride_frac 0.1 --n_mode free --no_partner"),
    ("adhd_classification", "--dataset adhd --task classification --target diagnosis "
                            "--parcellation cc200 --time max"),
    ("frankfurt_classification", "--dataset frankfurt --task classification --target sleep_median "
                                 "--parcellation large --time 200 --stride_frac 0.1 --n_mode free"),
)
BENCH_MODELS = (
    ("shallow", "--model shallow"),
    ("mlp", "--model mlp"),
    ("gnn", "--model gnn"),
    ("gat", "--model gnn --conv_type gat"),
    ("residualgnn", "--model residualgnn"),
)


def benchmark(out):
    for panel, flags in BENCH_PANELS:
        for seed in range(1, 11):
            for name, model in BENCH_MODELS:
                odir = os.path.join(out, panel, name)
                yield py("real_data.benchmark.run_experiment",
                         f"{flags} {model} --tau 0.10 --adjacency window --seed {seed} --out_dir {odir}")


def real_data(name, out):
    if name == "abide":
        yield from controlled(ABIDE, "sage", [SIM_LIKE], out)
    elif name == "relaxation":
        yield from controlled(ABIDE, "sage", [("relu", "vec", "fixed"), ("relu", "vec", "free"),
                                              ("relu", "mean", "free")], out)
    elif name == "gat":
        yield from controlled(ABIDE, "gat", [SIM_LIKE, DEPLOYED], out)
    elif name == "density":
        # The no-convolution arm does not depend on tau: it comes from the abide grid.
        yield from controlled(ABIDE, "sage", [SIM_LIKE], out, taus=(0.05, 0.20),
                              graphs=("window", "circle", "zero", "bar"), patience=50, max_epochs=300)
    elif name == "adhd":
        yield from controlled(ADHD, "sage", [SIM_LIKE, DEPLOYED], out, patience=50, max_epochs=300)
    elif name == "frankfurt":
        yield from controlled(FRANKFURT, "sage", [SIM_LIKE, DEPLOYED], out, patience=50, max_epochs=300)
    elif name == "benchmark":
        yield from benchmark(out)
    else:
        raise SystemExit(f"unknown experiment {name!r}")


def main():
    names = ("quickstart", "simulations", "benchmark", "abide", "relaxation", "gat", "adhd", "frankfurt", "density", "all")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=names)
    p.add_argument("--out", default=None,
                   help="Results root (default: outputs/simulations or outputs/real_data/<experiment>).")
    a = p.parse_args()

    if a.experiment in ("quickstart", "simulations"):
        jobs = {"quickstart": quickstart, "simulations": simulations}[a.experiment](
            a.out or os.path.join("outputs", "simulations"))
    elif a.experiment == "all":
        root = a.out or os.path.join("outputs", "real_data")
        jobs = (j for n in names[2:-1] for j in real_data(n, os.path.join(root, n)))
    else:
        jobs = real_data(a.experiment, a.out or os.path.join("outputs", "real_data", a.experiment))
    jobs = list(jobs)
    assert len(set(jobs)) == len(jobs), "duplicate commands"
    try:
        sys.stdout.write("\n".join(jobs) + "\n")
    except BrokenPipeError:
        pass
    print(f"{len(jobs)} commands", file=sys.stderr)


if __name__ == "__main__":
    main()
