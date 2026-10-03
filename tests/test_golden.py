"""
Golden 3x3 test.

M = [[45, -20, -10],
     [-20,  55, -15],
     [-10, -15,  40]]
q = [18, -12, 6]

All expected values are read directly from the validated golden netlist.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from repo root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcp.core import (
    validate,
    detect_z_type,
    synthesize,
    compute_tran_params,
    CircuitElements,
)
from lcp.verify import compute_w
from lcp.netlist import build_netlist, write_netlist
from sim.runner import simulate, _find_ngspice
from sim.parse import extract_node_voltages

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

M = np.array([[45, -20, -10],
              [-20,  55, -15],
              [-10, -15,  40]], dtype=float)

q = np.array([18.0, -12.0, 6.0])

TOL = 1e-9   # relative tolerance for resistance values


@pytest.fixture(scope="module")
def elements() -> CircuitElements:
    return synthesize(M, q)


# ---------------------------------------------------------------------------
# Grounded resistors
# ---------------------------------------------------------------------------

def test_grounded_resistor_count(elements):
    assert len(elements.grounded_resistors) == 3


def test_grounded_resistors(elements):
    grs = {gr.node: gr.value for gr in elements.grounded_resistors}
    assert math.isclose(grs[1], 1/15, rel_tol=TOL), f"R_1 = {grs[1]}"
    assert math.isclose(grs[2], 1/20, rel_tol=TOL), f"R_2 = {grs[2]}"
    assert math.isclose(grs[3], 1/15, rel_tol=TOL), f"R_3 = {grs[3]}"


# ---------------------------------------------------------------------------
# Coupling resistors
# ---------------------------------------------------------------------------

def test_coupling_resistor_count(elements):
    assert len(elements.coupling_resistors) == 3


def test_coupling_resistors(elements):
    crs = {(cr.node_i, cr.node_j): cr.value for cr in elements.coupling_resistors}
    assert math.isclose(crs[(1, 2)], 0.05,          rel_tol=TOL), f"R_1_2 = {crs[(1,2)]}"
    assert math.isclose(crs[(1, 3)], 0.10,          rel_tol=TOL), f"R_1_3 = {crs[(1,3)]}"
    assert math.isclose(crs[(2, 3)], 1/15,          rel_tol=TOL), f"R_2_3 = {crs[(2,3)]}"


def test_coupling_smaller_index_first(elements):
    for cr in elements.coupling_resistors:
        assert cr.node_i < cr.node_j


# ---------------------------------------------------------------------------
# Current sources — sign convention (golden node order)
# ---------------------------------------------------------------------------

def test_current_source_count(elements):
    # q = [18, -12, 6] -> all nonzero -> 3 sources
    assert len(elements.current_sources) == 3


def test_current_source_I1(elements):
    src = next(s for s in elements.current_sources if s.node == 1)
    assert src.pos_node == 1 and src.neg_node == 0    # q_1 > 0 -> (i, 0)
    assert math.isclose(src.magnitude, 18.0)


def test_current_source_I2(elements):
    src = next(s for s in elements.current_sources if s.node == 2)
    assert src.pos_node == 0 and src.neg_node == 2    # q_2 < 0 -> (0, i)
    assert math.isclose(src.magnitude, 12.0)


def test_current_source_I3(elements):
    src = next(s for s in elements.current_sources if s.node == 3)
    assert src.pos_node == 3 and src.neg_node == 0    # q_3 > 0 -> (i, 0)
    assert math.isclose(src.magnitude, 6.0)


# --- DIODE SYNTHESIS (replaced with dependent current sources) ---
# ---------------------------------------------------------------------------
# Diodes
# ---------------------------------------------------------------------------
# def test_diode_count(elements):
#     assert len(elements.diodes) == 3
#
# def test_diode_nodes(elements):
#     nodes = {d.node for d in elements.diodes}
#     assert nodes == {1, 2, 3}

# ---------------------------------------------------------------------------
# Dependent Current Sources (diode behavior)
# ---------------------------------------------------------------------------

def test_dependent_source_count(elements):
    assert len(elements.diode_clamps) == 3


def test_dependent_source_nodes(elements):
    nodes = {ds.node for ds in elements.diode_clamps}
    assert nodes == {1, 2, 3}


# ---------------------------------------------------------------------------
# Netlist text — spot-check critical lines
# ---------------------------------------------------------------------------

def test_netlist_current_source_lines(elements):
    cir = build_netlist(elements, "golden_test")
    assert "I_1" in cir and "1" in cir and "DC 18" in cir
    assert "I_2" in cir and "DC 12" in cir
    assert "I_3" in cir and "DC 6"  in cir


# --- DIODE SYNTHESIS (replaced with dependent current sources) ---
# def test_netlist_diode_lines(elements):
#     cir = build_netlist(elements, "golden_test")
#     # Ideal diodes: D_dN  0  N  DIDEAL
#     assert "D_d1" in cir and "DIDEAL" in cir
#     assert "D_d2" in cir
#     assert "D_d3" in cir
#
# def test_netlist_has_ideal_diode_model(elements):
#     cir = build_netlist(elements, "golden_test")
#     assert ".MODEL DIDEAL D" in cir

def test_netlist_dependent_source_lines(elements):
    cir = build_netlist(elements, "golden_test")
    # Behavioral current sources: B_dN
    assert "B_d1" in cir
    assert "B_d2" in cir
    assert "B_d3" in cir


def test_netlist_has_ideal_diode_clamp(elements):
    cir = build_netlist(elements, "golden_test")
    # Behavioral source implements a piecewise-linear ideal-diode clamp:
    # I = DIODE_G * max(V(anode,cathode), 0).
    assert "max(" in cir
    assert "1e+07" in cir or "10000000" in cir


def test_netlist_op_and_end(elements):
    cir = build_netlist(elements, "golden_test")
    assert ".OP" in cir
    assert ".END" in cir


def test_netlist_i2_node_order(elements):
    """I_2 must have node order '0 2' (q_2 < 0 convention)."""
    cir = build_netlist(elements, "golden_test")
    for line in cir.splitlines():
        if line.strip().startswith("I_2"):
            tokens = line.split()
            # tokens: [I_2, pos, neg, DC, mag]
            assert tokens[1] == "0" and tokens[2] == "2", (
                f"I_2 node order wrong: {line!r}"
            )
            break
    else:
        pytest.fail("I_2 line not found in netlist")


# ---------------------------------------------------------------------------
# Matrix-type golden cases: K-type (Golden B), Z-type (Golden A), P-type
# (Golden P) — SPD synthesis and simulation routing
# ---------------------------------------------------------------------------

try:
    _find_ngspice()
    _NGSPICE_AVAILABLE = True
except FileNotFoundError:
    _NGSPICE_AVAILABLE = False

requires_ngspice = pytest.mark.skipif(not _NGSPICE_AVAILABLE, reason="ngspice not found")


def _resolve_mode(elements: CircuitElements) -> tuple[str, bool]:
    """Mirror main.py's routing: on RESISTOR SIGN, never on matrix_class."""
    all_resistors = elements.grounded_resistors + elements.coupling_resistors
    has_negative_resistors = any(r.value < 0 for r in all_resistors)
    return ("tran" if has_negative_resistors else "op"), has_negative_resistors


def _run_dual_pipeline(res, q, tmp_path, name):
    """
    Mirror main.py's Z-type path: synthesize the K-circuit from M^-1 with
    q' = -M^-1 q, simulate, and un-swap the outputs back to the original LCP.
    Returns (z_original, w_original, mode).
    """
    n = len(q)
    q_prime = -res.K @ q
    elements = synthesize(res.K, q_prime, flip_labels=res.flip_labels)
    mode, _ = _resolve_mode(elements)
    tstep = tstop = None
    if mode == "tran":
        tstep, tstop = compute_tran_params(res.eigenvalues)
    cir_path = write_netlist(elements, name, tmp_path, mode=mode, tstep=tstep, tstop=tstop)
    sim = simulate(cir_path, n, mode=mode)

    z_syn = np.array(extract_node_voltages(sim.voltages, n))
    if any(res.flip_labels):
        signs = np.where(np.array(res.flip_labels, dtype=bool), -1.0, 1.0)
        z_syn = z_syn * signs
    if sim.diode_currents is not None:
        w_syn = np.array([sim.diode_currents[str(i)] for i in range(1, n + 1)])
    else:
        w_syn = compute_w(res.K, z_syn, q_prime)

    # Un-swap: original z = circuit diode currents, original w = node voltages.
    return w_syn, z_syn, mode


# --- Golden A: Z-type — non-K directly, but M^-1 is K-type -----------------
# M = [[2,1,1],[1,2,1],[1,1,2]], q = [-3,-3,1].
# M has all-positive off-diagonals -> non-K directly (a "frustrated" sign
# triangle). Its inverse K = M^-1 = I - J/4 has all-negative off-diagonals,
# i.e. pure K-type, so M is Z-type. The LCP is therefore solved on the
# K-circuit built from M^-1 with q' = -M^-1 q and the outputs are un-swapped
# (circuit diode currents -> original z, node voltages -> original w).
#
# LCP(M_A, Q_A) solution (brute-force complementary search): z=[1,1,0], w=[0,0,3].
# The dual solution is its transpose: z_dual=w=[0,0,3], w_dual=z=[1,1,0];
# q' = -M_A^-1 q = [1.75, 1.75, -2.25]. The dual circuit is all-positive, so it
# solves in op mode where a DIRECT non-K synthesis of M_A would have needed tran.

M_A = np.array([[2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0]])
Q_A = np.array([-3.0, -3.0, 1.0])
Z_EXPECTED_A = np.array([1.0, 1.0, 0.0])
K_A_EXPECTED = np.array([[0.75, -0.25, -0.25],
                         [-0.25, 0.75, -0.25],
                         [-0.25, -0.25, 0.75]])
QPRIME_A_EXPECTED = np.array([1.75, 1.75, -2.25])


def test_golden_a_is_non_k_directly():
    _, matrix_class, _ = validate(M_A, Q_A)
    assert matrix_class == "non_k"


def test_golden_a_detected_as_z_type():
    res = detect_z_type(M_A)
    assert res is not None
    assert res.matrix_class == "z_type"
    assert np.allclose(res.K, K_A_EXPECTED, atol=1e-9), f"K={res.K}"
    assert all(lbl == 0 for lbl in res.flip_labels)


def test_golden_a_q_prime():
    """q' = -M^-1 q is what feeds the dual circuit's current sources."""
    res = detect_z_type(M_A)
    assert np.allclose(-res.K @ Q_A, QPRIME_A_EXPECTED, atol=1e-9)


def test_golden_a_dual_is_all_positive_op():
    """The dual K-circuit has only positive resistors, so it solves in op mode."""
    res = detect_z_type(M_A)
    elements = synthesize(res.K, -res.K @ Q_A, flip_labels=res.flip_labels)
    mode, has_negative_resistors = _resolve_mode(elements)
    assert has_negative_resistors is False
    assert mode == "op"


@requires_ngspice
def test_golden_a_z_type_final_z(tmp_path):
    """Full dual pipeline recovers the original LCP solution z=[1,1,0]."""
    res = detect_z_type(M_A)
    assert res is not None and res.matrix_class == "z_type"
    z_orig, _w_orig, mode = _run_dual_pipeline(res, Q_A, tmp_path, "golden_A_z")
    assert mode == "op"
    assert np.allclose(z_orig, Z_EXPECTED_A, atol=5e-3), f"z={z_orig}"


# --- Golden B: k_type but non-dominant, negative GROUNDED resistor ---------
# M = [[1.5,-1,-1],[-1,5,0],[-1,0,5]], q = [-3,4,4]
# LCP solution (verified by brute-force complementary search): z=[2,0,0]
# This case guards against routing on matrix_class: the off-diagonal signs
# are K-type (no flips needed), yet row 1 is not diagonally dominant, so
# R_1 is negative and the circuit still requires transient simulation.

M_B = np.array([[1.5, -1.0, -1.0],
                [-1.0,  5.0,  0.0],
                [-1.0,  0.0,  5.0]])
Q_B = np.array([-3.0, 4.0, 4.0])
Z_EXPECTED_B = np.array([2.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def golden_b():
    flip_labels, matrix_class, eigenvalues = validate(M_B, Q_B)
    elements = synthesize(M_B, Q_B, flip_labels=flip_labels)
    return flip_labels, matrix_class, eigenvalues, elements


def test_golden_b_is_k_type(golden_b):
    flip_labels, matrix_class, _, _ = golden_b
    assert matrix_class == "k_type"
    assert all(lbl == 0 for lbl in flip_labels)


def test_golden_b_negative_grounded_resistor(golden_b):
    _, _, _, elements = golden_b
    grounded = {gr.node: gr.value for gr in elements.grounded_resistors}
    assert math.isclose(grounded[1], -2.0, rel_tol=1e-9), f"R_1 = {grounded[1]}"
    assert math.isclose(grounded[2], 0.25, rel_tol=1e-9), f"R_2 = {grounded[2]}"
    assert math.isclose(grounded[3], 0.25, rel_tol=1e-9), f"R_3 = {grounded[3]}"


def test_golden_b_auto_selects_tran_despite_k_type_label(golden_b):
    """Guards against routing on matrix_class instead of resistor sign."""
    _, matrix_class, _, elements = golden_b
    assert matrix_class == "k_type"
    mode, has_negative_resistors = _resolve_mode(elements)
    assert has_negative_resistors is True
    assert mode == "tran"


@requires_ngspice
def test_golden_b_tran_final_z(golden_b, tmp_path):
    _, _, eigenvalues, elements = golden_b
    mode, has_negative_resistors = _resolve_mode(elements)
    assert mode == "tran"
    tstep, tstop = compute_tran_params(eigenvalues)
    cir_path = write_netlist(elements, "golden_B_tran", tmp_path,
                              mode=mode, tstep=tstep, tstop=tstop)
    sim = simulate(cir_path, 3, mode=mode)
    z = np.array(extract_node_voltages(sim.voltages, 3))
    assert np.allclose(z, Z_EXPECTED_B, atol=5e-3), f"z={z}"


# --- Golden P: P-type — non-K directly AND M^-1 is non-K -------------------
# M = [[6,-1,-4],[-1,12,5],[-4,5,6]], q = [-4,-4,-4].
# M is non-K (the +5 at (2,3) frustrates the sign triangle) and its inverse is
# non-K too, so it is NOT Z-type: detect_z_type returns None. The circuit is
# synthesized DIRECTLY from M, giving a negative COUPLING resistor
# R_2_3 = -1/5, which routes the run to transient simulation.
#
# LCP(M_P, Q_P) solution (brute-force complementary search): z=[2,0,2], w=[0,4,0].

M_P = np.array([[6.0, -1.0, -4.0],
                [-1.0, 12.0, 5.0],
                [-4.0, 5.0, 6.0]])
Q_P = np.array([-4.0, -4.0, -4.0])
Z_EXPECTED_P = np.array([2.0, 0.0, 2.0])


@pytest.fixture(scope="module")
def golden_p():
    flip_labels, matrix_class, eigenvalues = validate(M_P, Q_P)
    elements = synthesize(M_P, Q_P, flip_labels=flip_labels)
    return flip_labels, matrix_class, eigenvalues, elements


def test_golden_p_is_non_k(golden_p):
    _, matrix_class, _, _ = golden_p
    assert matrix_class == "non_k"


def test_golden_p_is_not_z_type():
    """Inverse is non-K too, so M_P is a genuine P-type, not Z-type."""
    assert detect_z_type(M_P) is None


def test_golden_p_negative_coupling_resistor(golden_p):
    _, _, _, elements = golden_p
    coupling = {(cr.node_i, cr.node_j): cr.value for cr in elements.coupling_resistors}
    assert math.isclose(coupling[(1, 2)], 1.0, rel_tol=1e-9), coupling
    assert math.isclose(coupling[(1, 3)], 0.25, rel_tol=1e-9), coupling
    assert math.isclose(coupling[(2, 3)], -0.2, rel_tol=1e-9), coupling   # negative


def test_golden_p_auto_selects_tran(golden_p):
    _, _, _, elements = golden_p
    mode, has_negative_resistors = _resolve_mode(elements)
    assert has_negative_resistors is True
    assert mode == "tran"


@requires_ngspice
def test_golden_p_tran_final_z(golden_p, tmp_path):
    _, _, eigenvalues, elements = golden_p
    mode, _ = _resolve_mode(elements)
    assert mode == "tran"
    tstep, tstop = compute_tran_params(eigenvalues)
    cir_path = write_netlist(elements, "golden_P_tran", tmp_path,
                             mode=mode, tstep=tstep, tstop=tstop)
    sim = simulate(cir_path, 3, mode=mode)
    z = np.array(extract_node_voltages(sim.voltages, 3))
    assert np.allclose(z, Z_EXPECTED_P, atol=5e-3), f"z={z}"
