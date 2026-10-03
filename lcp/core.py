"""
lcp/core.py — Pure synthesis functions. No I/O, no PySpice, no matplotlib.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes for circuit elements
# ---------------------------------------------------------------------------

@dataclass
class GroundedResistor:
    node: int
    value: float          # ohms

@dataclass
class CouplingResistor:
    node_i: int           # smaller index
    node_j: int           # larger index
    value: float          # ohms

@dataclass
class CurrentSource:
    """
    SPICE line:  I_i  <pos>  <neg>  DC  <mag>
    q_i > 0  ->  pos=node_i, neg=0  (current from node into ground)
    q_i < 0  ->  pos=0,      neg=node_i  (current from ground into node)
    """
    node: int
    q_value: float        # raw (signed) q_i
    pos_node: int         # SPICE + terminal
    neg_node: int         # SPICE - terminal
    magnitude: float      # |q_i|

@dataclass
class IdealDiodeClamp:
    """
    The complementarity element at one port: a PIECEWISE-LINEAR ideal-diode
    clamp, emitted as a behavioral (B-element) current source so that ngspice's
    imperfect internal diode model is no longer in the loop:

        I = DIODE_G * max(V(anode,cathode), 0)

    i.e. an ideal diode with zero threshold, on-conductance DIODE_G, and exactly
    zero reverse current. The current flows anode -> cathode through the source,
    identical to a diode's id, and the controlling voltage is the source's own
    terminal voltage. Terminal order therefore mirrors the Diode it replaces
    (see the old Diode netlist code):
        normal  (flipped=False): anode=0,    cathode=node   (was  D 0 node)
        flipped (flipped=True):  anode=node,  cathode=0      (was  D node 0)

    Why linear rather than the exact Shockley exp: with N as small as 1e-3 the
    exp is far too stiff for ngspice's transient (it overflows/goes inert at the
    initial timepoint because a behavioral source has no junction limiting, the
    very thing that let the real diode primitive converge). A linear clamp has a
    constant slope, converges everywhere, and is MORE ideal than any real diode:
    the forward drop of a conducting port is just I/DIODE_G, which -> 0 as
    DIODE_G grows (at 1e7 S it is microvolts, vs ~1 mV for the old N=1e-3 diode).

    SPICE line:  B_d<node>  <anode>  <cathode>  I={DIODE_G*max(V(anode,cathode),0)}
    """
    node: int             # port index (1-based)
    flipped: bool = False # True -> anode/cathode reversed (matches Diode.flipped)

# Physical model constants, defined once here and shared by lcp.netlist
# (which writes them into the .cir file) and lcp.verify (which derives
# LCP tolerances from them). Keeping a single source of truth guarantees
# the verification tolerances always match the simulated model.
DIODE_G     = 1e7     # ideal-diode clamp on-conductance (S); forward drop = I/DIODE_G
SHUNT_CAP_F = 1e-14   # shunt capacitance added per node in transient mode (F)

# TODO: replace with an eigenvalue-aware horizon. compute_tran_params sizes tstop
# from the smallest NONZERO eigenvalue, which ignores the slow near-null modes, so
# the bare formula stops the run before those modes settle. This blunt constant
# multiplier buys enough headroom in practice; a settle-then-extend loop would do
# it properly. Exposed as compute_tran_params(settle_gain=...) so the underlying
# formula stays testable at settle_gain=1.0.
_SETTLE_GAIN = 100.0

# --- Legacy Shockley-diode constants (used by the commented-out ideal-diode
#     netlist/verify paths; retained so that reversion needs no edits here) ---
DIODE_IS    = 1e-15   # diode saturation current IS (A)
DIODE_N     = 0.001   # diode emission coefficient N (scales the forward drop)


# --- DIODE SYNTHESIS INFRASTRUCTURE (commented out for use with dependent sources) ---
# @dataclass
# class Diode:
#     node: int             # port index (1-based)
#     flipped: bool = False # True -> node order is reversed in the netlist

@dataclass
class SkippedResistor:
    """
    A resistor that was intentionally NOT placed because its admittance is
    exactly zero (reciprocal would divide by zero). The netlist emits a comment
    in its place. node_j is None for a grounded resistor (open to ground) and
    set for a coupling resistor (the two nodes are uncoupled).
    """
    node_i: int
    node_j: int | None = None  # None -> grounded; set -> coupling (i<j)


@dataclass
class CircuitElements:
    n: int
    grounded_resistors: List[GroundedResistor] = field(default_factory=list)
    coupling_resistors: List[CouplingResistor] = field(default_factory=list)
    current_sources: List[CurrentSource]       = field(default_factory=list)
    # diodes: List[Diode]                        = field(default_factory=list)
    diode_clamps: List[IdealDiodeClamp] = field(default_factory=list)
    skipped_grounded: List[SkippedResistor]    = field(default_factory=list)
    skipped_coupling: List[SkippedResistor]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class LCPValidationError(Exception):
    """Raised when M or q fail LCP pre-conditions."""


def _classify_signs(M: np.ndarray, zero_tol: float = 0.0) -> tuple[list[int], str]:
    """
    2-color the diodes so every non-zero coupling is sign-consistent, by
    propagating labels through the graph of non-zero off-diagonals (BFS per
    connected component), NOT by anchoring on row 0.

    Label semantics: 0 = keep diode as-is, 1 = flip diode polarity.
    Edge/constraint semantics for an off-diagonal M[i,j]:
        M[i,j] < -zero_tol : negative coupling -> i and j want the SAME label
        M[i,j] >  zero_tol : positive coupling -> i and j want DIFFERENT labels
        |M[i,j]| <= zero_tol: structural zero -> NO edge, NO sign constraint

    A zero therefore leaves both diodes undetermined at that edge; a diode's
    group is fixed only when a non-zero edge reaches it, and whole components
    that are never reached stay free (anchored to 0). This is a true bipartite
    2-coloring: it succeeds whenever one exists and only reports "non_k" on a
    genuine odd (frustrated) cycle of sign constraints — unlike the old row-0
    anchoring, which mislabeled 2-colorable graphs whose connectivity ran
    through zeros. Isolated nodes (an all-zero row) simply keep label 0.

    zero_tol > 0 treats near-zeros as structural zeros; needed when classifying
    a floating-point inverse whose "zero" entries carry rounding noise (see
    detect_z_type). The default zero_tol == 0.0 is the exact-arithmetic
    classification used on the input matrix M.

    Returns (flip_labels, matrix_class):
      matrix_class : "k_type"       - already sign-consistent, no flips
                     "flip_fixable" - sign-consistent after flipping some diodes
                     "non_k"        - sign pattern is not 2-colorable
    Never raises; a non-2-colorable pattern yields ([0]*n, "non_k").
    """
    n = M.shape[0]
    if n == 1:
        return [0], "k_type"

    UNDECIDED = -1
    labels = [UNDECIDED] * n

    for start in range(n):
        if labels[start] != UNDECIDED:
            continue
        labels[start] = 0  # anchor each new component at 0
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in range(n):
                if j == i:
                    continue
                m_ij = M[i, j]
                if abs(m_ij) <= zero_tol:
                    continue  # structural zero -> no constraint, stay undecided
                # negative coupling -> same label; positive -> opposite label
                want = labels[i] if m_ij < 0 else 1 - labels[i]
                if labels[j] == UNDECIDED:
                    labels[j] = want
                    queue.append(j)
                elif labels[j] != want:
                    return [0] * n, "non_k"  # frustrated cycle

    return labels, ("flip_fixable" if any(labels) else "k_type")


def validate(
    M: np.ndarray,
    q: np.ndarray,
    spd_tol: float = 1e-9,
) -> tuple[list[int], str, np.ndarray]:
    """
    Validate M and q for LCP circuit synthesis.  Checks run in order and
    abort on the first failure so no compute is wasted downstream.

    Checks:
      1. M is square and len(q) == n            (HARD — structural)
      2. M is symmetric                          (HARD)
      3. Eigenvalues are computed and returned. Definiteness is NO LONGER a
         gate: positive/negative definite, semidefinite (singular), and
         indefinite M are all accepted. The caller flags the definiteness class
         (see classify_definiteness) but the run proceeds regardless.
      4. Upper-triangle sign pattern is classified (informational only —
         non-fatal; describes off-diagonal/coupling signs, NOT diagonal
         dominance, and must not be used to choose the simulation mode)

    The only HARD constraints are shape and symmetry. No I/O happens here.

    Returns (flip_labels, matrix_class, eigenvalues):
      flip_labels : length-n list (0 = keep, 1 = flip diode/source polarity)
                    to be forwarded to synthesis. All zeros if the sign
                    pattern is not 2-colorable (non_k).
      matrix_class: "k_type" | "flip_fixable" | "non_k"
      eigenvalues : ascending eigenvalues of M (from eigvalsh), for reuse
                    by transient timing (compute_tran_params) and by
                    classify_definiteness.

    Raises LCPValidationError with a human-readable explanation on failure of
    the shape or symmetry checks. Definiteness never raises.
    """
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)

    # --- 1. Shape ---
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise LCPValidationError(
            f"M must be a square 2-D matrix; got shape {M.shape}"
        )
    n = M.shape[0]
    if q.ndim != 1 or q.shape[0] != n:
        raise LCPValidationError(
            f"q must be a 1-D vector of length {n} to match M; got shape {q.shape}"
        )

    # --- 2. Symmetry ---
    if not np.allclose(M, M.T):
        diff = np.abs(M - M.T)
        i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise LCPValidationError(
            f"M is not symmetric: M[{i+1},{j+1}]={M[i,j]:.6g} "
            f"but M[{j+1},{i+1}]={M[j,i]:.6g} "
            f"(max discrepancy {diff[i,j]:.3g})"
        )

    # --- 3. Eigenvalues (definiteness is classified by the caller, not gated) ---
    # spd_tol is retained for signature stability but no longer gates the run.
    eigenvalues = np.linalg.eigvalsh(M)

    # --- 4. Upper-triangle sign classification (non-fatal) ---
    flip_labels, matrix_class = _classify_signs(M)

    return flip_labels, matrix_class, eigenvalues


def classify_definiteness(
    eigenvalues: np.ndarray,
    spd_tol: float = 1e-9,
) -> tuple[str, bool]:
    """
    Label M's definiteness from its eigenvalues, and report singularity.

    An eigenvalue counts as zero when |lambda| <= tol, with tol scaled to the
    spectrum (tol = spd_tol * max(1, max|lambda|)) so the test is meaningful
    regardless of how M is scaled.

    Returns (label, is_singular):
      label : "positive_definite" | "negative_definite" |
              "positive_semidefinite" | "negative_semidefinite" | "indefinite"
      is_singular : True if any eigenvalue is ~0. When True the caller MUST
              skip the M^-1 (Z-type) pathway: a singular M has no inverse, and
              a near-singular one inverts to numerical garbage.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tol = spd_tol * max(1.0, scale)

    n_pos = int(np.sum(eigenvalues > tol))
    n_neg = int(np.sum(eigenvalues < -tol))
    n_zero = int(eigenvalues.size) - n_pos - n_neg
    is_singular = n_zero > 0

    if n_neg == 0 and n_zero == 0:
        label = "positive_definite"
    elif n_pos == 0 and n_zero == 0:
        label = "negative_definite"
    elif n_neg == 0:
        label = "positive_semidefinite"
    elif n_pos == 0:
        label = "negative_semidefinite"
    else:
        label = "indefinite"
    return label, is_singular


def null_space_component(
    M: np.ndarray,
    q: np.ndarray,
    spd_tol: float = 1e-9,
) -> float:
    """
    Norm of q's projection onto M's (near-)null eigenspace.

    For a singular symmetric M, a null-space mode has no restoring conductance:
    if q drives that mode (nonzero projection) the corresponding node voltage
    ramps without settling and the LCP has no steady state; if the projection
    is ~0, q lies in range(M) and a (possibly non-unique) solution exists.
    Returns 0.0 when M has no null space. Uses eigh (needs the eigenvectors),
    so the caller only invokes it when classify_definiteness flags singularity.
    """
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)
    vals, vecs = np.linalg.eigh(M)
    scale = float(np.max(np.abs(vals))) if vals.size else 0.0
    tol = spd_tol * max(1.0, scale)
    null_mask = np.abs(vals) <= tol
    if not null_mask.any():
        return 0.0
    return float(np.linalg.norm(vecs[:, null_mask].T @ q))


def is_diagonally_dominant(M: np.ndarray, spd_tol: float = 1e-9) -> bool:
    """
    True if M is (weakly) diagonally dominant in the |.| sense:

        M[i,i] >= sum_{j!=i} |M[i,j]|   for every row i

    within a spectrum-scaled tolerance (spd_tol * max(1, max|M|)), so the test
    is robust for a floating-point inverse whose row sums carry rounding noise.

    Combined with a 2-colorable off-diagonal sign pattern (see _classify_signs)
    this is exactly the condition for synthesize() to emit NO negative
    resistors: flips make every off-diagonal <= 0 (coupling resistors positive),
    and diagonal dominance makes every row sum M[i,i]-sum|off| >= 0 (grounded
    resistors non-negative; a zero row sum simply skips that resistor). This
    dominance is what "hyperdominance" adds on top of the K-type sign pattern,
    and it is INVARIANT under diode flips (a flip never changes the diagonal),
    so it can be tested on M directly regardless of the chosen flip labels.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    n = M.shape[0]
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    tol = spd_tol * max(1.0, scale)
    for i in range(n):
        off = float(np.sum(np.abs(M[i]))) - abs(float(M[i, i]))
        if float(M[i, i]) - off < -tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Z-type detection (inverse is hyperdominant: all-positive-resistor realization)
# ---------------------------------------------------------------------------

@dataclass
class ZTypeResult:
    """
    Outcome of a successful Z-type test on an SPD matrix M.

    A matrix is Z-type when its inverse K = M^-1 is HYPERDOMINANT (or becomes so
    after diode flips): a 2-colorable off-diagonal sign pattern AND diagonal
    dominance. That pair is exactly what guarantees the dual circuit built from
    K has NO negative resistors (see is_diagonally_dominant). A 2-colorable but
    non-dominant inverse is rejected, because inverting only to still produce
    negative resistors buys nothing over direct synthesis of M. The LCP(M, q) is
    then solved on the dual circuit and its outputs transposed back to the
    original problem (see main.py). Every field describes the K = M^-1 system
    that is actually synthesized:

      K            : the (symmetrised) inverse M^-1, the matrix synthesized.
      flip_labels  : diode flips for K's synthesis (0 = keep, 1 = flip).
      matrix_class : "z_type"         - K is hyperdominant, no flips needed.
                     "flip_fixable_z" - K is hyperdominant after flipping diodes.
      eigenvalues  : ascending eigenvalues of K, for transient timing.
    """
    K: np.ndarray
    flip_labels: list[int]
    matrix_class: str
    eigenvalues: np.ndarray


def detect_z_type(
    M: np.ndarray,
    rel_zero_tol: float = 1e-9,
) -> ZTypeResult | None:
    """
    Test whether M is Z-type: its inverse K = M^-1 is hyperdominant (2-colorable
    off-diagonal signs AND diagonally dominant), possibly after diode flips.
    Returns a ZTypeResult on success, else None.

    Both conditions are required. The sign pattern alone is NOT enough: a
    2-colorable but non-dominant K still synthesizes negative (grounded)
    resistors, so the inversion would achieve nothing that direct synthesis of M
    doesn't already do. Requiring dominance ensures the dual circuit is genuinely
    all-positive (op-mode), which is the whole point of taking the inverse.

    The caller must skip this for a singular M (no inverse exists). The inverse
    is computed numerically, so its structurally-zero off-diagonals carry
    rounding noise; entries below rel_zero_tol * max|K_ij| are treated as zero
    when 2-coloring, and dominance is tested with the same relative tolerance.
    K is symmetrised first to drop the small asymmetry np.linalg.inv introduces.
    """
    M = np.asarray(M, dtype=float)
    try:
        K = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    K = 0.5 * (K + K.T)  # M is symmetric => M^-1 is too; discard inversion noise

    scale = float(np.max(np.abs(K))) if K.size else 0.0
    zero_tol = rel_zero_tol * scale
    flip_labels, cls = _classify_signs(K, zero_tol=zero_tol)
    if cls == "non_k":
        return None

    # Sign pattern is 2-colorable; require diagonal dominance too, else the
    # dual circuit would still carry negative resistors (the inversion would be
    # pointless). Reject so the caller falls back to direct synthesis of M.
    if not is_diagonally_dominant(K, spd_tol=rel_zero_tol):
        return None

    matrix_class = "flip_fixable_z" if cls == "flip_fixable" else "z_type"
    eigenvalues = np.linalg.eigvalsh(K)
    return ZTypeResult(
        K=K,
        flip_labels=flip_labels,
        matrix_class=matrix_class,
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    M: np.ndarray,
    q: np.ndarray,
    flip_labels: list[int] | None = None,
) -> CircuitElements:
    """
    Extract circuit elements from M (nxn conductance matrix) and q (length-n).

    Flipping port i negates its row and column in the effective conductance
    matrix (M_eff = D @ M @ D, D[i,i] = -1 if flipped).  All resistor values
    are derived from M_eff so that coupling resistors are always positive.

    Grounded resistor  R_i   = 1 / row_sum(M_eff[i])    (omit if zero)
    Coupling resistor  R_i_j = -1 / M_eff[i,j]          (omit if zero; i<j)
    Current source     I_i:  q>0 -> (i,0); q<0 -> (0,i); flipped -> swap
    Diode clamp        B_di: normal -> (0, i); flipped -> (i, 0)
                             (anode, cathode); see IdealDiodeClamp
    """
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)
    n = M.shape[0]
    flip = flip_labels if flip_labels is not None else [0] * n

    # Effective conductance matrix: negate rows/cols of flipped ports so that
    # all off-diagonal entries become <= 0, making every R value positive.
    signs = np.where(np.array(flip, dtype=bool), -1.0, 1.0)
    M_eff = M * np.outer(signs, signs)

    elements = CircuitElements(n=n)

    # One ideal-diode clamp per port, carrying the complementarity constraint.
    # The clamp is PIECEWISE-LINEAR, not Shockley: see IdealDiodeClamp for why
    # the exponential law was abandoned. `flipped` mirrors the old Diode's
    # polarity flag one-for-one, so the netlist builder derives (anode, cathode)
    # exactly as the diode code did.
    #   --- Legacy ideal-diode synthesis (restore alongside the Diode class) ---
    #   for i in range(1, n + 1):
    #       elements.diodes.append(Diode(node=i, flipped=bool(flip[i - 1])))
    for i in range(1, n + 1):
        elements.diode_clamps.append(
            IdealDiodeClamp(node=i, flipped=bool(flip[i - 1]))
        )

    # Current sources — flip swaps pos/neg terminal order
    for idx in range(n):
        qi = q[idx]
        i = idx + 1
        if qi == 0.0:
            continue
        if qi > 0:
            pos, neg = i, 0
        else:
            pos, neg = 0, i
        if flip[idx]:
            pos, neg = neg, pos
        src = CurrentSource(node=i, q_value=qi,
                            pos_node=pos, neg_node=neg, magnitude=abs(qi))
        elements.current_sources.append(src)

    # Grounded resistors — use M_eff row sums. A negative value (row not
    # diagonally dominant) is legal here; the caller reports it and routes
    # the run to transient simulation. A zero row sum is a zero grounded
    # admittance: R = 1/0 is undefined and no current can flow to ground, so
    # no resistor is placed and the omission is recorded for a netlist comment.
    for idx in range(n):
        row_sum = float(np.sum(M_eff[idx, :]))
        if row_sum == 0.0:
            elements.skipped_grounded.append(SkippedResistor(node_i=idx + 1))
            continue
        elements.grounded_resistors.append(
            GroundedResistor(node=idx + 1, value=1.0 / row_sum)
        )

    # Coupling resistors — use M_eff off-diagonals (upper triangle only).
    # Negative values occur only for non-K matrices (see validate()). A zero
    # off-diagonal is a zero coupling admittance: the two nodes are uncoupled,
    # R = -1/0 is undefined, so no resistor is placed and the omission is
    # recorded for a netlist comment.
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            m_ij = float(M_eff[i_idx, j_idx])
            if m_ij == 0.0:
                elements.skipped_coupling.append(
                    SkippedResistor(node_i=i_idx + 1, node_j=j_idx + 1)
                )
                continue
            elements.coupling_resistors.append(
                CouplingResistor(node_i=i_idx + 1, node_j=j_idx + 1, value=-1.0 / m_ij)
            )

    return elements


# ---------------------------------------------------------------------------
# Transient timing (for circuits with negative resistors)
# ---------------------------------------------------------------------------

def compute_tran_params(
    eigenvalues: np.ndarray,
    C: float = SHUNT_CAP_F,
    settle_factor: float = 10.0,
    n_output_points: int = 1000,
    zero_tol: float = 1e-9,
    fallback_lambda: float = 1.0,
    settle_gain: float = _SETTLE_GAIN,
) -> Tuple[float, float]:
    """
    Derive .tran tstep/tstop from M's eigenvalues and the shunt capacitance C.

    With a capacitor C on every node, the linearized circuit relaxes with time
    constants tau_k = C / lambda_k.  The slowest mode (largest tau, i.e.
    smallest |lambda|) dictates the run length:

        tstop = settle_factor * (C / lambda_eff) * settle_gain

    lambda_eff is chosen so the timing is always finite and positive, even for
    the non-SPD matrices now admitted (they are flagged, not rejected):

      * Negative eigenvalue: an unstable/indefinite mode. Its magnitude still
        sets a timescale, so we use |lambda| (the mode grows on ~C/|lambda|).
      * Zero eigenvalue: a null mode with NO finite time constant (tau -> inf).
        It carries no timescale, so it is dropped and the slowest surviving
        non-zero |lambda| is used instead.
      * All eigenvalues ~0 (e.g. the zero matrix): no timescale exists at all;
        fall back to lambda = fallback_lambda and warn.

    So lambda_eff = min(|lambda|) over eigenvalues with |lambda| > tol, where
    tol scales with the spectrum; if none survive, lambda_eff = fallback_lambda.

    tstep is ngspice's PRINT interval, not its solver step — ngspice picks
    internal timesteps adaptively from local truncation error, so the fast
    modes are resolved automatically.  tstep therefore only controls output
    resolution and file size, and is set to give a fixed number of output
    points regardless of problem size or stiffness:

        tstep = tstop / n_output_points

    settle_gain is a blunt constant multiplier on the whole horizon, defaulting
    to _SETTLE_GAIN. Pass settle_gain=1.0 to get the bare formula above.

    Whether the run actually settled is checked afterwards against the
    simulated trace (see lcp.verify.compute_settling_time); the caller
    warns if it did not and a larger settle_factor is needed.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    abs_eig = np.abs(eigenvalues)
    scale = float(np.max(abs_eig)) if abs_eig.size else 0.0
    tol = zero_tol * max(1.0, scale)

    nonzero = abs_eig[abs_eig > tol]
    if nonzero.size:
        lambda_eff = float(np.min(nonzero))
    else:
        lambda_eff = fallback_lambda
        warnings.warn(
            "compute_tran_params: every eigenvalue is ~0 (no finite time "
            f"constant); falling back to lambda={fallback_lambda:g} for timing.",
            RuntimeWarning,
            stacklevel=2,
        )

    tstop = settle_factor * (C / lambda_eff) * settle_gain

    tstep = tstop / n_output_points
    return tstep, tstop
