"""
Tests for the PSD/definiteness expansion and zero-admittance handling:

  * BFS 2-coloring treats zeros as absent edges (fixes the old row-0 anchoring
    that mislabeled 2-colorable graphs whose connectivity ran through zeros).
  * classify_definiteness labels every definiteness class and flags singularity.
  * synthesize records zero-admittance omissions; the netlist comments them.
  * compute_tran_params yields finite, positive timing for negative/zero
    eigenvalues (no divide-by-zero, no negative tstop).
  * validate no longer gates on definiteness; only shape and symmetry are hard.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcp.core import (
    _SETTLE_GAIN,
    _classify_signs,
    classify_definiteness,
    null_space_component,
    is_diagonally_dominant,
    detect_z_type,
    synthesize,
    compute_tran_params,
    validate,
    LCPValidationError,
)
from lcp.netlist import build_netlist


# ---------------------------------------------------------------------------
# BFS 2-coloring: zeros are absent edges, labels propagate through non-zeros
# ---------------------------------------------------------------------------

def test_coloring_propagates_through_zeros():
    """
    Node 1 is isolated from 2 and 3 by zeros; nodes 2-3 have a +coupling and
    must be oppositely labeled. The old row-0 anchoring wrongly returned
    non_k; the BFS coloring finds the valid 2-coloring.
    """
    M = np.array([[5.0, 0.0, 0.0],
                  [0.0, 3.0, 2.0],
                  [0.0, 2.0, 3.0]])
    labels, cls = _classify_signs(M)
    assert cls == "flip_fixable"
    # nodes 2,3 opposite; node 1 free (anchored 0)
    assert labels[1] != labels[2]


def test_coloring_isolated_node_stays_label_zero():
    M = np.array([[1.0, -1.0, 0.0],
                  [-1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])   # node 3 fully isolated
    labels, cls = _classify_signs(M)
    assert cls == "k_type"
    assert labels == [0, 0, 0]


def test_coloring_frustrated_cycle_is_non_k():
    M = np.array([[1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0]])   # all +1: odd frustrated triangle
    labels, cls = _classify_signs(M)
    assert cls == "non_k"
    assert labels == [0, 0, 0]


# ---------------------------------------------------------------------------
# Definiteness classification (flagged, not gated)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diag,label,singular", [
    ([1.0, 2.0, 3.0], "positive_definite", False),
    ([0.0, 1.0, 2.0], "positive_semidefinite", True),
    ([-1.0, -2.0], "negative_definite", False),
    ([0.0, -1.0, -2.0], "negative_semidefinite", True),
    ([-1.0, 2.0], "indefinite", False),
    ([0.0, -1.0, 2.0], "indefinite", True),
])
def test_classify_definiteness(diag, label, singular):
    ev = np.linalg.eigvalsh(np.diag(diag))
    assert classify_definiteness(ev) == (label, singular)


def test_classify_definiteness_scale_relative():
    """A 1e6-scaled PD matrix is not mistaken for singular by an absolute tol."""
    ev = np.linalg.eigvalsh(np.diag([1e6, 2e6]))
    assert classify_definiteness(ev) == ("positive_definite", False)


# ---------------------------------------------------------------------------
# validate: only shape and symmetry are hard; definiteness never raises
# ---------------------------------------------------------------------------

def test_validate_accepts_negative_definite():
    M = np.diag([-1.0, -2.0])
    q = np.array([1.0, 1.0])
    _, _, ev = validate(M, q)          # must NOT raise
    assert classify_definiteness(ev)[0] == "negative_definite"


def test_validate_rejects_asymmetric():
    M = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(LCPValidationError):
        validate(M, np.array([1.0, 1.0]))


def test_validate_shape_is_hard():
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # non-square
    with pytest.raises(LCPValidationError):
        validate(M, np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# Null-space component
# ---------------------------------------------------------------------------

def test_null_space_component_q_in_range_is_zero():
    M = np.array([[2.0, -2.0, 0.0],
                  [-2.0, 2.0, 0.0],
                  [0.0, 0.0, 0.0]])
    q = np.array([1.0, -1.0, 0.0])     # lies in range(M)
    assert null_space_component(M, q) == pytest.approx(0.0, abs=1e-12)


def test_null_space_component_q_excites_null_mode():
    M = np.diag([0.0, 1.0, 2.0])       # null space = e_1
    q = np.array([3.0, 0.0, 0.0])
    assert null_space_component(M, q) == pytest.approx(3.0)


def test_null_space_component_none_for_nonsingular():
    assert null_space_component(np.diag([1.0, 2.0]), np.array([1.0, 1.0])) == 0.0


# ---------------------------------------------------------------------------
# Zero-admittance omission + netlist comments
# ---------------------------------------------------------------------------

def test_synthesize_records_skipped_zero_admittance():
    M = np.array([[2.0, -2.0, 0.0],
                  [-2.0, 2.0, 0.0],
                  [0.0, 0.0, 0.0]])
    q = np.array([1.0, -1.0, 0.0])
    el = synthesize(M, q)
    assert {s.node_i for s in el.skipped_grounded} == {1, 2, 3}   # all row sums 0
    assert {(s.node_i, s.node_j) for s in el.skipped_coupling} == {(1, 3), (2, 3)}


def test_netlist_comments_skipped_resistors():
    M = np.array([[2.0, -2.0, 0.0],
                  [-2.0, 2.0, 0.0],
                  [0.0, 0.0, 0.0]])
    q = np.array([1.0, -1.0, 0.0])
    cir = build_netlist(synthesize(M, q), "zero_adm")
    assert "* R_3 omitted: grounded admittance = 0 (open to ground, no current)" in cir
    assert "* R_1_3 omitted: coupling admittance = 0 (nodes uncoupled, no current)" in cir
    assert "* R_2_3 omitted: coupling admittance = 0 (nodes uncoupled, no current)" in cir


def test_netlist_no_skip_comments_when_all_present():
    """A dense matrix produces no omission comments (guards golden netlists)."""
    M = np.array([[3.0, -1.0], [-1.0, 3.0]])
    cir = build_netlist(synthesize(M, np.array([1.0, 1.0])), "dense")
    assert "omitted" not in cir


# ---------------------------------------------------------------------------
# Transient timing: finite, positive for negative/zero eigenvalues
#
# These pin the eigenvalue selection rule, so they pass settle_gain=1.0 to get
# the bare tstop = settle_factor * (C / lambda_eff). Production runs keep the
# default _SETTLE_GAIN headroom; test_tran_params_applies_settle_gain below is
# the one that covers it.
# ---------------------------------------------------------------------------

def test_tran_params_negative_eig_uses_magnitude():
    tstep, tstop = compute_tran_params(np.array([-1.0, 2.0, 4.0]), C=1.0,
                                       settle_gain=1.0)
    assert tstop > 0 and np.isfinite(tstop)
    assert tstop == pytest.approx(10.0)       # slowest |lambda| = 1
    assert tstep == pytest.approx(tstop / 1000)


def test_tran_params_zero_eig_dropped():
    _, tstop = compute_tran_params(np.array([0.0, 2.0, 4.0]), C=1.0,
                                   settle_gain=1.0)
    assert tstop == pytest.approx(5.0)        # smallest non-zero |lambda| = 2


def test_tran_params_all_zero_falls_back_with_warning():
    with pytest.warns(RuntimeWarning):
        _, tstop = compute_tran_params(np.array([0.0, 0.0]), C=1.0,
                                       fallback_lambda=1.0, settle_gain=1.0)
    assert tstop == pytest.approx(10.0)


def test_tran_params_spd_unchanged():
    """Positive spectrum: behavior identical to the old lambda_min formula."""
    ev = np.array([0.9, 5.0, 5.0])
    _, tstop = compute_tran_params(ev, C=1.0, settle_gain=1.0)
    assert tstop == pytest.approx(10.0 * (1.0 / 0.9))


def test_tran_params_applies_settle_gain():
    """
    settle_gain scales the horizon linearly, and the default is the module's
    _SETTLE_GAIN -- so a default call is exactly _SETTLE_GAIN x the bare formula.
    Guards against the gain being dropped or silently re-tuned.
    """
    ev = np.array([1.0, 2.0])
    _, bare = compute_tran_params(ev, C=1.0, settle_gain=1.0)
    _, scaled = compute_tran_params(ev, C=1.0, settle_gain=7.0)
    _, default = compute_tran_params(ev, C=1.0)
    assert scaled == pytest.approx(7.0 * bare)
    assert default == pytest.approx(_SETTLE_GAIN * bare)


# ---------------------------------------------------------------------------
# Hyperdominance: positive-resistor realizability = 2-colorable AND dominant
# ---------------------------------------------------------------------------

# M_B is a K-type (all off-diagonals <= 0) SPD matrix that is NOT diagonally
# dominant: row 1 sums to -0.5, so synthesizing it gives a negative grounded
# resistor. It is the canonical "sign pattern is fine but dominance is not" case.
M_B = np.array([[1.5, -1.0, -1.0],
                [-1.0, 5.0, 0.0],
                [-1.0, 0.0, 5.0]])


def test_is_diagonally_dominant_true():
    assert is_diagonally_dominant(np.array([[45.0, -20.0, -10.0],
                                            [-20.0, 55.0, -15.0],
                                            [-10.0, -15.0, 40.0]]))


def test_is_diagonally_dominant_false_for_M_B():
    assert not is_diagonally_dominant(M_B)          # row 1 sum = -0.5


def test_is_diagonally_dominant_weak_singular_ok():
    """A zero row sum (open grounded R) still counts as dominant (>=)."""
    L = np.array([[1.0, -1.0], [-1.0, 1.0]])        # graph Laplacian, singular
    assert is_diagonally_dominant(L)


def test_detect_z_type_rejects_2colorable_but_nondominant_inverse():
    """
    The core bug fix: M = inv(M_B) is non-K, so the inverse is tested. Its
    inverse is M_B, which is 2-colorable (K-type sign pattern) but NOT
    diagonally dominant. Accepting it would synthesize a negative resistor, so
    detect_z_type must REJECT it (return None) rather than invert pointlessly.
    """
    M = np.linalg.inv(M_B)
    _, cls, _ = validate(M, np.array([1.0, 1.0, 1.0]))
    assert cls == "non_k"                            # inverse path is taken
    assert detect_z_type(M) is None                  # ... and correctly rejected


def test_detect_z_type_accepts_hyperdominant_inverse():
    """Positive control: Golden A's inverse K_A is dominant, so it stays Z-type."""
    M_A = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    res = detect_z_type(M_A)
    assert res is not None and res.matrix_class == "z_type"
    assert is_diagonally_dominant(res.K)
