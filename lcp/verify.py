"""
lcp/verify.py — Pure LCP condition checks. No I/O, no simulation deps.

z : node voltages (measured directly from the circuit)
w : diode-clamp currents. The B-element clamp exposes no readable branch
    current in ngspice, so w is reconstructed by applying the clamp's own law
    to the measured node voltages (diode_clamp_currents), then cross-checked
    against M*z+q — see main.py.

LCP feasibility requires z >= 0, w >= 0, and z_i * w_i = 0 for all i.

The circuit realizes the complementarity constraint with a piecewise-linear
ideal-diode clamp (DIODE_G from lcp.core): I = DIODE_G * max(V(anode,cathode), 0).
This has a single non-ideality:

  * A conducting (clamped) port carrying current I drops
        Vf(I) = I / DIODE_G
    (Ohm's law across the finite on-conductance), which is O(microvolts) at
    DIODE_G = 1e7 S. A clamped port therefore reads z_i ~ -Vf instead of
    exactly 0, and its complementarity product is |z_i * w_i| ~ Vf * w_i
    rather than exactly 0.

Unlike a real/Shockley diode there is NO reverse leakage: a blocking port
carries exactly zero current, so w picks up no GMIN-style offset. The only
residual on w comes from the clamp drop on z propagating through M into
w = M*z+q (bounded by ||M|| * Vf), which the caller folds into its residual
threshold. lcp_tolerances() derives the pass/fail tolerances from the same
DIODE_G the netlist was built with; raising DIODE_G shrinks Vf proportionally
with no convergence penalty (the clamp is linear).

--- LEGACY (Shockley diode model, retained for the commented-out ideal-diode
    netlist path) ---
  A conducting diode dropped Vf = N * Vt * ln(1 + I/IS) (~1 mV), and a blocking
  diode leaked ~GMIN * V (ngspice GMIN = 1e-12 S). See diode_forward_drop_shockley.
"""

from __future__ import annotations

import numpy as np

from lcp.core import DIODE_G, DIODE_IS, DIODE_N

V_THERMAL    = 0.02585   # thermal voltage kT/q at ~300 K (V), ngspice default
NGSPICE_GMIN = 1e-12     # ngspice default minimum junction conductance (S)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_w(M: np.ndarray, z: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute expected slack w = M z + q from measured node voltages."""
    return np.asarray(M, float) @ np.asarray(z, float) + np.asarray(q, float)


def diode_clamp_currents(z_syn: np.ndarray, g_on: float = DIODE_G) -> np.ndarray:
    """
    Reconstruct each ideal-diode clamp current from the node voltages, i.e. the
    current the B-element source actually injects: w_i = G * max(V(a,c), 0).

    ngspice does not expose a B-source branch current, so this stands in for the
    old measured diode id -- and it is exactly what ngspice computed internally
    for the diode: the device law applied to the junction voltage. z_syn is the
    flip-corrected LCP node variable (z_i = -V(i) on flipped ports), so for every
    port (flipped or not) the forward bias is V(anode,cathode) = -z_i and the
    clamp current is:

        w_i = G * max(-z_i, 0)

    This is >= 0 by construction (a diode cannot conduct in reverse), nonzero
    only on a clamped port (z_i ~ -Vf < 0) where it equals G*Vf, and complementary
    with z. Whether it matches what M demands is the job of the M*z+q residual
    cross-check (see compute_residual / main.py).
    """
    z_syn = np.asarray(z_syn, dtype=float)
    return g_on * np.maximum(-z_syn, 0.0)


def compute_residual(w_measured: np.ndarray, w_expected: np.ndarray) -> float:
    """
    ||w_measured - w_expected||_inf.
    Cross-checks that measured diode currents match M*z+q.
    A large residual suggests a parsing error or a mismatch between
    the synthesized circuit and the M matrix.
    """
    return float(np.max(np.abs(np.asarray(w_measured) - np.asarray(w_expected))))


def compute_settling_time(
    time: np.ndarray,
    voltage_trace: dict[str, np.ndarray],
    tol: float = 0.005,
) -> float:
    """
    Earliest t_s such that for all t >= t_s,
      max_i |V_i(t) - V_i(final)| <= tol * max_i |V_i(final)|.

    Used only for transient (shunt-capacitor) runs. Returns 0.0 if the
    final-value scale is ~0 (nothing to settle relative to).
    """
    time = np.asarray(time, dtype=float)
    nodes = sorted(voltage_trace.keys(), key=int)
    V = np.stack([np.asarray(voltage_trace[node], dtype=float) for node in nodes])  # (n, T)

    final = V[:, -1]
    scale = float(np.max(np.abs(final)))
    if scale <= 1e-12:
        return 0.0

    err = np.max(np.abs(V - final[:, None]), axis=0)  # (T,)
    threshold = tol * scale

    violations = np.nonzero(err > threshold)[0]
    if violations.size == 0:
        return float(time[0])
    last = int(violations[-1])
    if last == len(time) - 1:
        return float(time[-1])  # never actually settles within the run
    return float(time[last + 1])


# ---------------------------------------------------------------------------
# Physically-motivated tolerances (see module docstring for the derivation)
# ---------------------------------------------------------------------------

def diode_forward_drop(
    current: float | np.ndarray,
    g_on: float = DIODE_G,
) -> float | np.ndarray:
    """
    Forward drop Vf = I / DIODE_G of the netlist's piecewise-linear ideal-diode
    clamp, in volts (Ohm's law across the on-conductance). Negative currents
    (blocking diode) carry no drop and are treated as zero.
    """
    i = np.maximum(np.asarray(current, dtype=float), 0.0)
    vf = i / g_on
    return float(vf) if vf.ndim == 0 else vf


def diode_forward_drop_shockley(
    current: float | np.ndarray,
    i_s: float = DIODE_IS,
    n_emission: float = DIODE_N,
) -> float | np.ndarray:
    """
    LEGACY: Shockley forward drop Vf = N * Vt * ln(1 + I/IS) of the old ideal-
    diode model, in volts. Retained for the commented-out diode netlist path.
    """
    i = np.maximum(np.asarray(current, dtype=float), 0.0)
    vf = n_emission * V_THERMAL * np.log1p(i / i_s)
    return float(vf) if vf.ndim == 0 else vf


def lcp_tolerances(
    z: np.ndarray,
    w: np.ndarray,
    safety: float = 3.0,
) -> tuple[float, float, float]:
    """
    Derive (z_tol, w_tol, comp_tol) from the ideal-diode clamp's single
    non-ideality (the finite forward drop Vf = I / DIODE_G), scaled by `safety`
    to allow solver noise on top of the physics:

      z_tol    : largest expected clamp drop, Vf at the largest diode current —
                 a clamped port sits at z ~ -Vf.
      w_tol    : the clamp has NO reverse leakage (a blocking port carries
                 exactly zero current), so the only residual on w is the clamp
                 drop on z propagating through M into w = M*z+q. Bounding ||M||
                 by max|z|/Vf is not available here, so w_tol is set to the same
                 Vf scale (with the floor); main.py additionally guards the
                 M-weighted residual via ||M||_inf * z_tol.
      comp_tol : largest expected complementarity product, Vf * I at the
                 largest diode current.

    Small absolute floors keep the tolerances nonzero for the degenerate
    all-zero solution.

    --- LEGACY (Shockley diode): w_tol was safety * NGSPICE_GMIN * max|z| to
        cover the GMIN reverse leakage of a real diode; the linear clamp has
        none, so that term is gone. ---
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)

    i_max = max(float(np.max(w)), 0.0) if w.size else 0.0
    vf_max = diode_forward_drop(i_max)

    z_tol    = safety * vf_max + 1e-9
    w_tol    = safety * vf_max + 1e-9
    comp_tol = safety * vf_max * i_max + 1e-12
    return z_tol, w_tol, comp_tol


# ---------------------------------------------------------------------------
# LCP condition check
# ---------------------------------------------------------------------------

def check_lcp(
    z: np.ndarray,
    w: np.ndarray,
    z_tol: float = 1e-6,
    w_tol: float = 1e-6,
    comp_tol: float = 1e-6,
) -> dict:
    """
    Evaluate the three LCP feasibility conditions.

    z_tol / w_tol : non-negativity tolerance (default strict 1e-6).
                    Raise to ~1e-3 to accommodate the ~1 mV diode Vf.
    comp_tol      : complementarity tolerance.

    Returns a dict with raw values plus boolean pass/fail per condition.
    """
    z    = np.asarray(z, float)
    w    = np.asarray(w, float)
    comp = np.abs(z * w)
    return {
        "z_min":                float(z.min()),
        "w_min":                float(w.min()),
        "z_nonneg":             bool(np.all(z >= -z_tol)),
        "w_nonneg":             bool(np.all(w >= -w_tol)),
        "complementarity_max":  float(comp.max()),
        "complementarity_vec":  comp.tolist(),
        "feasible": bool(
            np.all(z >= -z_tol)
            and np.all(w >= -w_tol)
            and comp.max() < comp_tol
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    z: np.ndarray,
    w: np.ndarray,
    report: dict,
    w_source: str = "M*z+q",
    residual: float | None = None,
    residual_tol: float = 1e-3,
    z_tol: float = 1e-6,
    w_tol: float = 1e-6,
    comp_tol: float = 1e-6,
) -> None:
    """
    Print a human-readable LCP verification report.

    w_source     : label shown in the header ('measured' or 'M*z+q').
    residual     : if provided, prints ||w_measured - M*z+q||_inf as a
                   cross-check.
    residual_tol : threshold for flagging the residual. The measured z
                   carries the ~Vf clamp offsets, which propagate through M
                   into M*z+q, so the expected residual scale is
                   ||M||_inf * z_tol — pass that, not a fixed constant.
    z_tol / w_tol / comp_tol : tolerances actually used in check_lcp (for display).
    """
    n = len(z)
    print("\n=== LCP Verification ===")
    print(f"  z source : node voltages (measured)")
    print(f"  w source : {w_source}")
    if residual is not None:
        verdict = (f"(< {residual_tol:.2e}: circuit faithfully implements M)"
                   if residual < residual_tol
                   else f"(WARNING: exceeds tol {residual_tol:.2e})")
        print(f"  w residual ||w_meas - Mz+q||_inf = {residual:.4e}  {verdict}")
    print()
    print(f"{'Port':>5}  {'z_i (V)':>12}  {'w_i (A)':>12}  {'|z*w|':>12}")
    print("-" * 47)
    for i in range(n):
        print(f"{i+1:>5}  {z[i]:>12.6f}  {w[i]:>12.6f}  "
              f"{report['complementarity_vec'][i]:>12.2e}")
    print("-" * 47)
    print(f"  z >= 0 : {report['z_nonneg']}  (min = {report['z_min']:.4e})")
    print(f"  w >= 0 : {report['w_nonneg']}  (min = {report['w_min']:.4e})")
    print(f"  max|z*w| = {report['complementarity_max']:.4e}")

    # Report the effective clamp forward drop from the most-negative z
    vf_est = abs(min(report["z_min"], 0.0))
    if vf_est > 1e-9:
        print(f"  (most negative z ~ diode forward drop Vf = {vf_est*1e3:.3f} mV)")

    status = "PASS" if report["feasible"] else "FAIL"
    print(f"  LCP conditions : {status}  "
          f"(diode-model tolerances: z >= -{z_tol:.2e}, "
          f"w >= -{w_tol:.2e}, |z*w| < {comp_tol:.2e})")
    print("========================\n")
