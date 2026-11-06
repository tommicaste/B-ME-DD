"""Run a command list from `make_jobs.py` in parallel, skipping finished commands.

    python experiments/run_jobs.py jobs_abide.txt --workers 8

A real-data command is finished when its result file
(<out_dir>/<dataset>_<task>_seed<seed>.jsonl) exists and is not empty; a simulation
command is always run, and counts as failed if any of its rows records an error.
Each command logs to <log_dir>/<line number>.log. Run from the repository root.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def result_file(cmd):
    tok = shlex.split(cmd)
    if "real_data.benchmark.run_experiment" not in tok:
        return None
    arg = lambda k: tok[tok.index(k) + 1] if k in tok else None
    return os.path.join(arg("--out_dir"), f"{arg('--dataset')}_{arg('--task')}_seed{arg('--seed')}.jsonl")


def simulation_errors(cmd):
    """Rows in a simulation's output that record a failed fit (the script logs them and goes on)."""
    tok = shlex.split(cmd)
    if not any(t.endswith("run_simulations.py") for t in tok):
        return 0
    out = tok[tok.index("--output_dir") + 1] if "--output_dir" in tok else os.path.join("outputs", "simulations")
    return sum('"error"' in line for f in glob.glob(os.path.join(out, "regression_*.ndjson")) for line in open(f))


def done(cmd):
    f = result_file(cmd)
    return f is not None and os.path.isfile(f) and os.path.getsize(f) > 0


def run(i, cmd, log_dir, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    with open(os.path.join(log_dir, f"{i:05d}.log"), "w") as log:
        log.write(cmd + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        errors = simulation_errors(cmd) if rc == 0 else 0
        if errors:
            log.write(f"\n{errors} result rows record a failed run (see their 'error' field)\n")
            rc = 1
        return i, rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", help="text file with one command per line")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per command")
    p.add_argument("--log_dir", default=os.path.join("outputs", "logs"))
    a = p.parse_args()

    cmds = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    todo = [(i, c) for i, c in enumerate(cmds, 1) if not done(c)]
    print(f"{len(cmds)} commands, {len(cmds) - len(todo)} already finished, {len(todo)} to run", flush=True)
    os.makedirs(a.log_dir, exist_ok=True)

    failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(run, i, c, a.log_dir, a.threads) for i, c in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            i, rc = fut.result()
            if rc:
                failed.append(i)
            print(f"[{n}/{len(todo)}] command {i} {'FAILED' if rc else 'ok'}", flush=True)

    if failed:
        print(f"{len(failed)} failed; see {a.log_dir}/<number>.log for: {sorted(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
