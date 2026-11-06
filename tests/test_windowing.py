"""Windowing on synthetic scans, and the job lists behind the paper. Sub-second."""
import subprocess
import sys

import numpy as np
import pytest

from real_data.shared.windowing import build_entries, whole_scan_windows, windows_for_T

SPEC = dict(kind="column", key="y", sentinels=())


def scans(lengths, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(ts=rng.standard_normal((L, p)), sid=f"s{i}", y=float(i)) for i, L in enumerate(lengths)]


def test_partner_is_a_different_non_overlapping_window():
    entries = build_entries(scans([60, 80]), 20, 1.0, SPEC, "sid")
    C, Cp, y, groups, idx = windows_for_T(entries, 20, 1.0, "free", np.random.default_rng(0))
    assert C.shape[0] == 3 + 4 and not any(np.allclose(C[i], Cp[i]) for i in range(len(C)))


def test_short_scans_need_no_partner_flag():
    data = scans([25, 60])
    assert len(build_entries(data, 20, 1.0, SPEC, "sid")) == 1
    entries = build_entries(data, 20, 1.0, SPEC, "sid", require_partner=False)
    C, Cp, *_ = windows_for_T(entries, 20, 1.0, "free", np.random.default_rng(0), require_partner=False)
    assert len(entries) == 2 and C.shape[0] == 1 + 3 and np.allclose(C, Cp)


def test_whole_scan_is_one_window_per_scan():
    entries = build_entries(scans([30, 50, 70]), 1, 1.0, SPEC, "sid", require_partner=False)
    C, Cp, y, groups, idx = whole_scan_windows(entries)
    assert C.shape == (3, 6, 6) and list(y) == [0.0, 1.0, 2.0] and list(idx) == [0, 1, 2]
    assert np.allclose(C[0], np.corrcoef(entries[0]["ts"], rowvar=False), atol=1e-6)


@pytest.mark.parametrize("experiment,count", [("quickstart", 4), ("simulations", 60), ("benchmark", 150),
                                              ("abide", 400), ("all", 5190)])
def test_job_lists(experiment, count):
    out = subprocess.run([sys.executable, "experiments/make_jobs.py", experiment],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    jobs = [j for j in out if j]
    assert len(jobs) == count == len(set(jobs))
