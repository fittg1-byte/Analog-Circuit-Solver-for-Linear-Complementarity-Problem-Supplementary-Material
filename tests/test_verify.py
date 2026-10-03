"""
Unit tests for lcp/verify.py — diode-model-derived tolerances and the
LCP condition check.

The complementarity element is now a piecewise-linear ideal-diode clamp
(I = DIODE_G * max(V(a,c), 0)), so a conducting port drops Vf = I/DIODE_G.
The "example run" values below are constructed to be consistent with that
linear model: each clamped port sits at z_i = -w_i/DIODE_G. They satisfy the
LCP up to the clamp's finite forward drop and must PASS under the
model-derived tolerances, while a too-tight fixed tolerance rejects them.
"""

import math
import sys
from pathlib import Path

import numpy as np

# Allow imports from repo root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcp.core import DIODE_G, DIODE_IS, DIODE_N
from lcp.verify import (
    V_THERMAL,
    check_lcp,
    compute_settling_time,
    diode_forward_drop,
    diode_forward_drop_shockley,
    lcp_tolerances,
)

# Measured w (clamp currents) and the node voltages z they imply under the
# linear clamp: a clamped port (1 and 3) sits at z = -w/DIODE_G; the free port
# (2) sits at a positive voltage with ~zero current.
W_EXAMPLE = np.array([13.62932601620195, -2.185817260758196e-13, 2.709905128235014])
Z_EXAMPLE = np.array([-W_EXAMPLE[0] / DIODE_G, 0.2175817260758218, -W_EXAMPLE[2] / DIODE_G])


# ---------------------------------------------------------------------------
# diode_forward_drop (piecewise-linear ideal-diode clamp)
# ---------------------------------------------------------------------------

def test_forward_drop_is_linear():
    i = 13.6293
    assert math.isclose(diode_forward_drop(i), i / DIODE_G, rel_tol=1e-12)


def test_forward_drop_zero_and_blocking():
    assert diode_forward_drop(0.0) == 0.0
    assert diode_forward_drop(-5.0) == 0.0  # blocking diode: no forward drop


def test_forward_drop_explains_measured_clamp_voltage():
    """The most-negative measured z is the clamp Vf = I/DIODE_G at that port."""
    vf = diode_forward_drop(W_EXAMPLE[0])
    assert math.isclose(vf, abs(Z_EXAMPLE[0]), rel_tol=1e-9)


def test_forward_drop_shockley_legacy():
    """The legacy Shockley drop is retained for the commented-out diode path."""
    i = 13.6293
    expected = DIODE_N * V_THERMAL * math.log1p(i / DIODE_IS)
    assert math.isclose(diode_forward_drop_shockley(i), expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# lcp_tolerances + check_lcp
# ---------------------------------------------------------------------------

def test_example_run_passes_with_model_tolerances():
    z_tol, w_tol, comp_tol = lcp_tolerances(Z_EXAMPLE, W_EXAMPLE)
    report = check_lcp(Z_EXAMPLE, W_EXAMPLE, z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)
    assert report["z_nonneg"]
    assert report["w_nonneg"]
    assert report["feasible"]


def test_example_run_fails_with_too_tight_tolerances():
    """A fixed tolerance tighter than the clamp drop wrongly rejects a feasible
    solution — the motivation for deriving tolerances from the model instead."""
    # z_tol below the clamp drop |z_min| = W/DIODE_G rejects the clamped ports.
    report = check_lcp(Z_EXAMPLE, W_EXAMPLE, z_tol=1e-9, w_tol=1e-9, comp_tol=1e-12)
    assert not report["feasible"]


def test_genuine_violation_still_fails():
    """Tolerances must not be so loose that a real violation passes."""
    z = np.array([1.0, 2.0])
    w = np.array([5.0, -3.0])   # w_2 genuinely negative
    z_tol, w_tol, comp_tol = lcp_tolerances(z, w)
    report = check_lcp(z, w, z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)
    assert not report["w_nonneg"]
    assert not report["feasible"]


def test_nonzero_complementarity_still_fails():
    z = np.array([1.0, 0.0])
    w = np.array([1.0, 2.0])    # z_1 * w_1 = 1 — gross violation
    z_tol, w_tol, comp_tol = lcp_tolerances(z, w)
    report = check_lcp(z, w, z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)
    assert not report["feasible"]


def test_degenerate_all_zero_solution_passes():
    z = np.zeros(3)
    w = np.zeros(3)
    z_tol, w_tol, comp_tol = lcp_tolerances(z, w)
    assert z_tol > 0 and w_tol > 0 and comp_tol > 0
    assert check_lcp(z, w, z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)["feasible"]


# ---------------------------------------------------------------------------
# compute_settling_time
# ---------------------------------------------------------------------------

def test_settling_time_exponential_decay():
    t = np.linspace(0.0, 10.0, 1001)
    trace = {"1": 1.0 - np.exp(-t)}   # settles to 1 with tau = 1
    t_s = compute_settling_time(t, trace, tol=0.005)
    # |V - 1| <= 0.005 once t >= -ln(0.005) ~ 5.3
    assert 5.0 < t_s < 5.6


def test_settling_time_flags_unsettled_run():
    t = np.linspace(0.0, 1.0, 101)
    trace = {"1": t.copy()}           # ramp: never settles
    assert compute_settling_time(t, trace) == t[-1]
