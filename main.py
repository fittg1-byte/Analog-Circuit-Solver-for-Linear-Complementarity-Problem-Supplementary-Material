"""
main.py — LCP-to-circuit pipeline entry point.

Usage:
    python main.py --M M.csv --q q.csv --run-name my_run
    python main.py                          # interactive prompts
    python main.py --run-name my_run        # interactive M/q, named run
    python main.py --skip-sim               # write netlist only, no ngspice
    python main.py --no-show                # save plots without opening windows
    python main.py --no-caps                # force .op (no capacitors) even with
                                            # negative resistors  [TEMPORARY]
"""


from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Domain core
from lcp.core import (
    validate,
    classify_definiteness,
    null_space_component,
    is_diagonally_dominant,
    detect_z_type,
    synthesize,
    compute_tran_params,
    LCPValidationError,
)
from lcp.netlist import write_netlist
from lcp.verify import (
    compute_w,
    diode_clamp_currents,
    compute_residual,
    check_lcp,
    lcp_tolerances,
    print_report,
    compute_settling_time,
)

# I/O
from fileio.reader import load_inputs
from fileio.writer import save_results, save_plot

# Simulation
from sim.runner import simulate
from sim.parse import extract_node_voltages

# Plot
from plot.solution import plot_solution, plot_transient, plot_traces


OUTPUT_DIR = Path(__file__).parent / "output"


def _stamped_name(run_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{run_name}"


def _resistor_name(r) -> str:
    """Netlist name: R_<i> for a grounded resistor, R_<i>_<j> for coupling."""
    if hasattr(r, "node"):
        return f"R_{r.node}"
    return f"R_{r.node_i}_{r.node_j}"


def _prompt_no_caps(n_negative: int) -> bool:
    """
    TEMPORARY experiment knob. Ask whether to run a negative-resistor circuit
    in .op mode anyway, i.e. exactly as though no negative resistors were
    present. Since the shunt capacitors are emitted only in tran mode (see
    lcp/netlist.py), choosing .op is what "no capacitors" means here -- no
    synthesis or netlist logic changes, only the mode routing in Step 4.

    Returns False without asking when no console is attached (batch/headless
    runs, tests), so the default auto-routing is unchanged there.
    """
    if not sys.stdin.isatty():
        return False
    print(f"\n[mode] {n_negative} negative resistor(s) present; this run would "
          "normally go to transient mode with a shunt capacitor on every node.")
    while True:
        try:
            answer = input("[mode] Run WITHOUT capacitors (.op, as if there "
                           "were no negative resistors)? [y/N] ").strip().lower()
        except EOFError:
            # stdin closed (piped/batch run): keep the default auto-routing.
            print("[mode] no console input available; keeping the auto-selected mode.")
            return False
        if answer in ("", "n", "no"):
            return False
        if answer in ("y", "yes"):
            return True
        print("  Please answer y or n.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize and simulate an LCP analog circuit."
    )
    parser.add_argument("--M",        metavar="CSV", help="Path to M matrix CSV")
    parser.add_argument("--q",        metavar="CSV", help="Path to q vector CSV")
    parser.add_argument("--run-name", metavar="NAME", default=None,
                        help="Run identifier (date-stamped automatically)")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Write netlist only; skip ngspice simulation")
    parser.add_argument("--out-dir",  metavar="DIR", default=None,
                        help="Output directory (default: ./output)")
    parser.add_argument("--mode", choices=["auto", "op", "tran"], default="auto",
                        help="Simulation mode: auto (default) picks tran when "
                             "any resistor is negative, else op")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open interactive plot windows (useful for "
                             "batch or headless runs; PNGs are always saved)")
    parser.add_argument("--no-caps", action="store_true",
                        help="[TEMPORARY] Force .op mode (no shunt capacitors) "
                             "even when negative resistors are present, i.e. "
                             "run as though there were none. Skips the "
                             "interactive prompt")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Load inputs
    # ------------------------------------------------------------------ #
    try:
        M, q, base_name = load_inputs(
            m_csv=args.M,
            q_csv=args.q,
            run_name=args.run_name,
        )
    except (ValueError, OSError) as exc:
        print(f"[load] INVALID INPUT: {exc}", file=sys.stderr)
        sys.exit(1)
    run_name = _stamped_name(base_name)
    n = M.shape[0]
    print(f"\n[run] {run_name}  (n={n})")

    # ------------------------------------------------------------------ #
    # Step 2 — Validate
    # ------------------------------------------------------------------ #
    try:
        flip_labels, matrix_class, eigenvalues = validate(M, q)
    except LCPValidationError as exc:
        print(f"[validate] INVALID INPUT: {exc}", file=sys.stderr)
        sys.exit(1)

    # Definiteness is flagged, not gated: SPD, negative (semi)definite, singular
    # and indefinite M are all accepted (only shape and symmetry are hard).
    definiteness, is_singular = classify_definiteness(eigenvalues)
    print(f"[validate] eigenvalues: lambda_min={eigenvalues[0]:.6g}, "
          f"lambda_max={eigenvalues[-1]:.6g}")
    if definiteness == "positive_definite":
        print("[validate] definiteness: positive_definite (SPD confirmed)")
    else:
        print(f"[validate] WARNING: M is {definiteness}, not SPD. Proceeding "
              "anyway (flagged, not gated); the LCP may be ill-posed and the "
              "circuit's DC/transient solve may not converge to a solution.")

    # A singular M drives a null-space check: if q excites a null mode there is
    # no steady state (that node voltage ramps); if not, a (possibly non-unique)
    # solution exists. Either way we proceed.
    if is_singular:
        nsc = null_space_component(M, q)
        q_scale = float(np.linalg.norm(q)) or 1.0
        if nsc > 1e-9 * q_scale:
            print(f"[validate] WARNING: M is singular and q has a component in "
                  f"its null space (||proj||={nsc:.3g}); the LCP may have no "
                  "steady state (a null mode will not settle) or be non-unique.")
        else:
            print("[validate] M is singular but q lies in range(M); a solution "
                  "exists though it may be non-unique.")

    print(f"[validate] matrix class (off-diagonal signs only, informational): {matrix_class}")
    if matrix_class == "flip_fixable":
        flipped = [i + 1 for i, lbl in enumerate(flip_labels) if lbl]
        print(f"[validate] diodes flipped to fix coupling signs: {flipped}")

    # ------------------------------------------------------------------ #
    # Step 2b — Hyperdominance / Z-type test. The point of flipping and of
    # inverting is a circuit with ONLY positive resistors, which requires
    # hyperdominance: a 2-colorable off-diagonal sign pattern AND diagonal
    # dominance (the sign pattern alone does not prevent negative grounded
    # resistors). So:
    #   * M hyperdominant           -> synthesize M directly (op mode).
    #   * else M^-1 hyperdominant    -> solve the DUAL LCP(K, q') on the
    #                                   K-circuit (q' = -M^-1 q) and transpose
    #                                   its outputs back (see Step 7).
    #   * else                       -> synthesize M directly; negative
    #                                   resistors remain and route to transient.
    # A singular M has no inverse, so the dual path is skipped for it.
    # ------------------------------------------------------------------ #
    dual = False
    synth_M, synth_q, synth_flip, synth_eig = M, q, flip_labels, eigenvalues
    m_hyperdominant = (matrix_class in ("k_type", "flip_fixable")
                       and is_diagonally_dominant(M))
    if m_hyperdominant:
        print("[validate] M is hyperdominant (2-colorable sign pattern and "
              "diagonally dominant); synthesizing directly, all resistors "
              "positive.")
    elif is_singular:
        print("[validate] M is not hyperdominant and is singular; skipping the "
              "M^-1 (Z-type) pathway (a singular M has no inverse). Synthesizing "
              "directly from M; negative resistors may be present.")
    else:
        why = ("not directly K-type" if matrix_class == "non_k"
               else f"{matrix_class} but not diagonally dominant")
        print(f"[validate] M is {why}; testing M^-1 for hyperdominance (Z-type) ...")
        z_type_result = detect_z_type(M)
        if z_type_result is not None:
            dual = True
            matrix_class = z_type_result.matrix_class          # z_type | flip_fixable_z
            synth_M = z_type_result.K
            synth_flip = z_type_result.flip_labels
            synth_eig = z_type_result.eigenvalues
            synth_q = -z_type_result.K @ q                      # q' = -M^-1 q
            inv_kind = ("hyperdominant after flips" if matrix_class == "flip_fixable_z"
                        else "hyperdominant")
            print(f"[validate] Z-type: M^-1 is {inv_kind}. Building the circuit "
                  "from M^-1 with q' = -M^-1 q; the circuit's z/w are the "
                  "original problem's w/z and are un-swapped after simulation.")
            if any(synth_flip):
                flipped = [i + 1 for i, lbl in enumerate(synth_flip) if lbl]
                print(f"[validate] diodes flipped (on M^-1) to fix coupling signs: {flipped}")
        else:
            print("[validate] M^-1 is not hyperdominant either; synthesizing "
                  "directly from M. Negative resistors will be present "
                  "(transient mode).")

    # ------------------------------------------------------------------ #
    # Step 3 — Synthesize (from M directly, or from M^-1 for a Z-type run)
    # ------------------------------------------------------------------ #
    elements = synthesize(synth_M, synth_q, flip_labels=synth_flip)
    # --- Legacy ideal-diode report line (restored alongside the diode path) ---
    # print(f"[synth] {len(elements.grounded_resistors)} grounded R, "
    #       f"{len(elements.coupling_resistors)} coupling R, "
    #       f"{len(elements.current_sources)} I sources, "
    #       f"{len(elements.diodes)} diodes")
    print(f"[synth] {len(elements.grounded_resistors)} grounded R, "
          f"{len(elements.coupling_resistors)} coupling R, "
          f"{len(elements.current_sources)} I sources, "
          f"{len(elements.diode_clamps)} ideal-diode clamps")

    # ------------------------------------------------------------------ #
    # Step 4 — Choose simulation mode (routes on RESISTOR SIGN, not on
    # matrix_class — a k_type matrix can still carry a negative grounded R
    # if it isn't diagonally dominant).
    # ------------------------------------------------------------------ #
    all_resistors = elements.grounded_resistors + elements.coupling_resistors
    negative_resistors = [r for r in all_resistors if r.value < 0]
    has_negative_resistors = bool(negative_resistors)

    if negative_resistors:
        names = ", ".join(_resistor_name(r) for r in negative_resistors)
        print(f"[synth] {len(negative_resistors)} negative resistor(s) "
              f"(active elements): {names}")

    # TEMPORARY: let the user run a negative-resistor circuit capacitor-free.
    # Capacitors exist only in tran mode, so "no capacitors" == force .op, which
    # is exactly the branch a circuit with no negative resistors would take.
    no_caps = args.no_caps
    if has_negative_resistors and not no_caps and args.mode == "auto":
        no_caps = _prompt_no_caps(len(negative_resistors))

    if no_caps:
        mode = "op"
        reason = ("--no-caps" if args.no_caps else "no-caps prompt") + \
                 ": negative resistors ignored, no capacitors"
        if has_negative_resistors:
            print("[mode] WARNING: running WITHOUT capacitors despite negative "
                  "resistor(s) present; the DC .op solve may fail to converge.")
        if args.mode == "tran":
            print("[mode] WARNING: --mode tran was requested but --no-caps "
                  "overrides it; running .op.")
    elif args.mode == "auto":
        mode = "tran" if has_negative_resistors else "op"
        reason = "auto-selected"
    else:
        mode = args.mode
        reason = "user override"
        if mode == "op" and has_negative_resistors:
            print("[mode] WARNING: --mode op forced despite negative resistor(s) "
                  "present; the DC .op solve may fail to converge.")

    sign_summary = ("negative resistors present" if has_negative_resistors
                    else "no negative resistors")
    print(f"[mode] {sign_summary} -> {mode} mode ({reason})")

    tstep = tstop = None
    if mode == "tran":
        tstep, tstop = compute_tran_params(synth_eig)
        print(f"[tran] tstop={tstop:.6g} s (10 x slowest time constant), "
              f"tstep={tstep:.6g} s (print interval, "
              f"{round(tstop / tstep)} output points)")

    # ------------------------------------------------------------------ #
    # Step 5 — Write netlist (PRIMARY deliverable)
    # ------------------------------------------------------------------ #
    cir_path = write_netlist(elements, run_name, out_dir, mode=mode, tstep=tstep, tstop=tstop)
    print(f"[netlist] Written -> {cir_path}")

    if args.skip_sim:
        print("[skip-sim] Exiting after netlist write.")
        return

    # ------------------------------------------------------------------ #
    # Step 6 — Simulate
    # ------------------------------------------------------------------ #
    print(f"[sim] Running ngspice .{mode} ...")
    try:
        sim = simulate(cir_path, n, mode=mode)
    except Exception as exc:
        print(f"[sim] ERROR: {exc}", file=sys.stderr)
        print("[sim] Netlist is still available at:", cir_path)
        sys.exit(1)

    settling_time = None
    if mode == "tran" and sim.time is not None and sim.voltage_trace is not None:
        settling_time = compute_settling_time(sim.time, sim.voltage_trace)
        print(f"[sim] Settling time t_s = {settling_time:.6g} s "
              f"(tstop={tstop:.6g} s)")
        if settling_time >= float(sim.time[-1]):
            print("[sim] WARNING: voltages had not settled by tstop; the final "
                  "values may not be the steady state. Re-run with a larger "
                  "settle_factor in compute_tran_params.")

    # ------------------------------------------------------------------ #
    # Step 7 — Extract the SYNTHESIZED circuit's solution, then map it back
    # to the original LCP. z_syn / w_syn solve LCP(synth_M, synth_q): for a
    # direct run that IS LCP(M, q); for a Z-type (dual) run it is
    # LCP(M^-1, q'), whose node voltages equal the original slack w and whose
    # diode currents equal the original z (the M -> M^-1 transform transposes
    # the two). We therefore un-swap z_syn/w_syn for a dual run.
    # ------------------------------------------------------------------ #
    z_syn = np.array(extract_node_voltages(sim.voltages, n))
    # Flipped ports enforce V(i) <= 0; the LCP variable is z_i = -V(i)
    if any(synth_flip):
        flip_signs = np.where(np.array(synth_flip, dtype=bool), -1.0, 1.0)
        z_syn = z_syn * flip_signs

    if sim.diode_currents is not None:
        # Legacy path: ngspice measured the ideal-diode id directly.
        w_syn = np.array([sim.diode_currents[str(i)] for i in range(1, n + 1)])
        w_measured = True
    else:
        # B-element ideal-diode clamp exposes no branch current, so reconstruct
        # it from the clamp law applied to the node voltages, w = G*max(-z,0).
        # This is the measured-equivalent diode current (>= 0 by construction);
        # its agreement with M*z+q is checked by the residual below.
        w_syn = diode_clamp_currents(z_syn)
        w_measured = True

    # Un-swap the dual solution back to the original LCP(M, q): the K-circuit's
    # node voltages are the original w, its diode currents are the original z.
    if dual:
        z, w = w_syn, z_syn
    else:
        z, w = z_syn, w_syn
    print(f"[sim] z (LCP solution): {z}")

    # Cross-check w against the ORIGINAL M, q on either path.
    w_expected = compute_w(M, z, q)
    if w_measured:
        if sim.diode_currents is not None:
            src = "diode currents (measured)"
        else:
            src = "ideal-diode clamp current G*max(-z,0) (reconstructed from node voltages)"
        w_source = f"{src}, un-swapped from dual" if dual else src
        residual  = compute_residual(w, w_expected)
        print(f"[sim] w (LCP slack):         {w}")
        print(f"[sim] w (M*z+q cross-check): {w_expected}")
    else:
        # ASCII only: this string is printed to the console, and a cp1252
        # terminal raises UnicodeEncodeError on anything outside Latin-1.
        w_source = "M*z+q (computed; diode currents unavailable)"
        residual = None
        print(f"[sim] w (M*z+q, diode currents not in raw file): {w}")

    # ------------------------------------------------------------------ #
    # Step 8 — Verify LCP conditions
    # ------------------------------------------------------------------ #
    # Tolerances follow from the netlist's ideal-diode clamp, not from arbitrary
    # constants: a conducting clamp drops Vf = I/DIODE_G, so clamped ports read
    # z ~ -Vf and |z*w| ~ Vf*w; the clamp has no reverse leakage. This Vf clamp
    # lives on the CIRCUIT's own quantities (node voltages z_syn, clamp currents
    # w_syn), so derive the tolerances from z_syn/w_syn. For a dual run the
    # original z/w are the circuit's w/z, so the two tolerances swap along with
    # them. See lcp/verify.py for the full derivation.
    tol_node, tol_diode, comp_tol = lcp_tolerances(z_syn, w_syn)
    if dual:
        z_tol, w_tol = tol_diode, tol_node
    else:
        z_tol, w_tol = tol_node, tol_diode
    report = check_lcp(z, w, z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)
    # The clamp offsets in measured z propagate through M into the M*z+q
    # cross-check, so the residual threshold scales with ||M||_inf.
    residual_tol = float(np.linalg.norm(M, np.inf)) * z_tol + w_tol
    print_report(z, w, report, w_source=w_source,
                 residual=residual, residual_tol=residual_tol,
                 z_tol=z_tol, w_tol=w_tol, comp_tol=comp_tol)

    # ------------------------------------------------------------------ #
    # Step 9 — Plot
    # ------------------------------------------------------------------ #
    fig = plot_solution(z, w, run_name)

    fig_tran = None
    fig_v = None
    fig_i = None          # unused while the current-trace figure is disabled
    if mode == "tran" and sim.time is not None and sim.voltage_trace is not None:
        fig_tran = plot_transient(sim.time, sim.voltage_trace, run_name, t_s=settling_time)

        # Ensemble trajectory pair (paper Fig. 12 style): every port's voltage
        # and every port's clamp current, each drawn as a split early/late panel
        # pair over the full run.
        #
        # The clamp current is not readable from the raw file (a B-source exposes
        # no branch current), so it is reconstructed per timepoint with the SAME
        # law the netlist implements, w = G*max(-z,0) — exactly what Step 7 does
        # for the final values, applied to the whole trace. Node order is fixed
        # once here so voltages and currents share a port ordering.
        #
        # NOTE: these are the SYNTHESIZED circuit's own quantities, deliberately
        # NOT un-swapped for a dual (Z-type) run. That keeps the units honest —
        # the voltage panel is always volts measured at nodes, the current panel
        # always clamp amps. On a dual run they are the original problem's w and
        # z respectively, so read the panels as the circuit, not as LCP(M, q).
        trace_nodes = sorted(sim.voltage_trace.keys(), key=int)
        V_trace = np.stack([sim.voltage_trace[k] for k in trace_nodes])   # (n, T)

        # The panel boundary is the point by which the voltage envelope has
        # covered most of its journey: compute_settling_time at a deliberately
        # loose tolerance, i.e. "visually settled" rather than the report's
        # criterion. That is the end of the fan-out, which is exactly what the
        # early panel is for. Falls back to the plot's own default if it
        # degenerates. Shared with the current plot when that is restored, so
        # the two can be read side by side and stacked in a paper figure.
        t_split = compute_settling_time(sim.time, sim.voltage_trace, tol=0.25)
        if not (float(sim.time[0]) < t_split < float(sim.time[-1])):
            t_split = None

        fig_v = plot_traces(
            sim.time, {k: V_trace[i] for i, k in enumerate(trace_nodes)},
            run_name, y_label="node voltage", y_base_unit="V", t_split=t_split,
        )

        # --- Clamp-current traces (<run>_currents.png) - DISABLED for now ---
        # Uncomment this block and its save in Step 10 to restore the figure.
        # symlog: the clamp currents span decades across ports (a few amps-scale
        # ports alongside many milliamp ones), and a blocking port carries
        # exactly zero, so a linear axis shows a few lines and a flat floor.
        #   flip_signs_t = np.where(np.array(synth_flip, dtype=bool), -1.0, 1.0)
        #   z_trace = V_trace * flip_signs_t[:, None]
        #   w_trace = diode_clamp_currents(z_trace)
        #   fig_i = plot_traces(
        #       sim.time, {k: w_trace[i] for i, k in enumerate(trace_nodes)},
        #       run_name, y_label="diode current", y_base_unit="A",
        #       t_split=t_split, yscale="symlog",
        #   )

    # ------------------------------------------------------------------ #
    # Step 10 — Save outputs
    # ------------------------------------------------------------------ #
    results_path = save_results(
        out_dir, run_name, M, q, z, w, report,
        mode=mode,
        matrix_class=matrix_class,
        definiteness=definiteness,
        settling_time=settling_time,
        tolerances={"z_tol": z_tol, "w_tol": w_tol, "comp_tol": comp_tol},
    )
    plot_path = save_plot(fig, out_dir, run_name)
    print(f"[save] Results -> {results_path}")
    print(f"[save] Plot    -> {plot_path}")

    if fig_tran is not None:
        tran_plot_path = save_plot(fig_tran, out_dir, run_name, kind="transient")
        print(f"[save] Transient plot -> {tran_plot_path}")

    if fig_v is not None:
        v_path = save_plot(fig_v, out_dir, run_name, kind="voltages")
        print(f"[save] Voltage traces -> {v_path}")

    # --- Clamp-current traces - DISABLED alongside their figure in Step 9 ---
    #   if fig_i is not None:
    #       i_path = save_plot(fig_i, out_dir, run_name, kind="currents")
    #       print(f"[save] Current traces -> {i_path}")

    if not args.no_show:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
