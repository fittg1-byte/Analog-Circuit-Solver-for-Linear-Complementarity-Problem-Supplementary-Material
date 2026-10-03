"""
lcp/netlist.py — Pure netlist builder. No I/O beyond returning/writing a string.
Produces a SPICE .cir file that matches the golden convention exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from lcp.core import CircuitElements, DIODE_G, DIODE_IS, DIODE_N, SHUNT_CAP_F

_COL = 14            # column width for node fields
_SAVE_PER_LINE = 10  # .save variables per line (multiple .save lines accumulate)


def build_netlist(
    elements: CircuitElements,
    run_name: str,
    mode: str = "op",
    tstep: float | None = None,
    tstop: float | None = None,
) -> str:
    """
    Return the full .cir text as a string.

    mode="op"   : DC operating point.
    mode="tran" : adds a SHUNT_CAP_F shunt capacitor from every node to ground
                  (ic=0) and runs a transient instead of a DC solve — used
                  for circuits containing negative resistors, whose .op
                  solve is fragile. Requires tstep and tstop.

    The capacitance is named, not written out here, so this docstring cannot
    drift from the constant the netlist actually emits (see lcp.core).
    """
    if mode not in ("op", "tran"):
        raise ValueError(f"mode must be 'op' or 'tran'; got {mode!r}")
    if mode == "tran" and (tstep is None or tstop is None):
        raise ValueError("tstep and tstop are required when mode='tran'")

    lines: list[str] = []

    lines.append(f"* LCP circuit - {run_name}")
    lines.append(f"* {elements.n}-port nodal realization")
    lines.append("")

    # --- Independent current sources ---
    lines.append("* Independent Current Sources")
    for src in elements.current_sources:
        lines.append(
            f"I_{src.node:<{_COL-2}} {src.pos_node:<{_COL}} {src.neg_node:<{_COL}} DC {src.magnitude:.12g}"
        )
    lines.append("")

    # --- Resistors ---
    # Grounded and coupling resistors are emitted in node order; where a
    # resistor was intentionally omitted (zero admittance -> reciprocal would
    # divide by zero), an ASCII comment is placed in its stead so the netlist
    # documents the deliberate open circuit rather than silently dropping it.
    lines.append("* Resistors")
    grounded_by_node = {gr.node: gr for gr in elements.grounded_resistors}
    skipped_grounded = {s.node_i for s in elements.skipped_grounded}
    for node in range(1, elements.n + 1):
        gr = grounded_by_node.get(node)
        if gr is not None:
            lines.append(f"R_{gr.node:<{_COL-2}} {gr.node:<{_COL}} 0{'':<{_COL-1}} {gr.value:.12g}")
        elif node in skipped_grounded:
            lines.append(f"* R_{node} omitted: grounded admittance = 0 "
                         "(open to ground, no current)")

    if elements.grounded_resistors and elements.coupling_resistors:
        lines.append("")

    coupling_by_pair = {(cr.node_i, cr.node_j): cr for cr in elements.coupling_resistors}
    skipped_coupling = {(s.node_i, s.node_j) for s in elements.skipped_coupling}
    for i in range(1, elements.n + 1):
        for j in range(i + 1, elements.n + 1):
            cr = coupling_by_pair.get((i, j))
            if cr is not None:
                name = f"R_{cr.node_i}_{cr.node_j}"
                lines.append(f"{name:<{_COL}} {cr.node_i:<{_COL}} {cr.node_j:<{_COL}} {cr.value:.12g}")
            elif (i, j) in skipped_coupling:
                lines.append(f"* R_{i}_{j} omitted: coupling admittance = 0 "
                             "(nodes uncoupled, no current)")
    lines.append("")

    # --- Shunt capacitors (transient mode only) ---
    if mode == "tran":
        lines.append("* Shunt capacitors (stabilizes negative-resistor circuits)")
        for i in range(1, elements.n + 1):
            lines.append(f"C_{i:<{_COL-2}} {i:<{_COL}} 0{'':<{_COL-1}} {SHUNT_CAP_F:g}  ic=0")
        lines.append("")

    # --- Ideal-diode clamps ---
    # Behavioral (B-element) current sources that replace the ngspice diode
    # primitive (which cannot be made perfectly ideal) with an ideal-diode clamp:
    #     I = DIODE_G * max(V(anode,cathode), 0)
    # Zero threshold, on-conductance DIODE_G, exactly zero reverse current. The
    # current flows anode -> cathode through the source, matching a diode's id,
    # and is controlled by the source's own terminal voltage V(anode,cathode).
    # Terminal order mirrors the old diode: normal -> (0, node), flipped ->
    # (node, 0).
    #
    # Why linear and not the exact Shockley exp: with N ~ 1e-3 the exp is far too
    # stiff for ngspice's transient (a behavioral source has no junction limiting,
    # so exp() overflows / goes inert at the initial timepoint and the solve
    # fails). The linear clamp converges everywhere AND is more ideal than any
    # real diode: a conducting port's forward drop is I/DIODE_G, microvolts at
    # DIODE_G=1e7, versus ~1 mV for the old N=1e-3 diode. lcp.verify derives the
    # LCP tolerances from DIODE_G to match (see lcp/verify.py).
    lines.append("* Ideal-diode clamps (piecewise-linear, as behavioral current sources)")
    for clamp in elements.diode_clamps:
        if clamp.flipped:
            anode, cathode = clamp.node, 0
        else:
            anode, cathode = 0, clamp.node
        lines.append(
            f"B_d{clamp.node:<{_COL-3}} {anode:<{_COL}} {cathode:<{_COL}} "
            f"I={{{DIODE_G:g}*max(V({anode},{cathode}),0)}}"
        )
    lines.append("")

    # --- Legacy Shockley B-source (stiff; superseded by the linear clamp above) ---
    #   VT_NETLIST, EXP_ARG_MAX = 0.02585, 700   # exp(700)~1e304, avoids overflow
    #   B_d<node> <anode> <cathode> I={IS*(exp(min(V(anode,cathode)/(N*Vt),700))-1)}
    #
    # --- Ideal diode synthesis (commented out, replaced by B-sources above) ---
    # lines.append("* Ideal diodes")
    # for d in elements.diodes:
    #     if d.flipped:
    #         node_fields = f"{d.node:<{_COL}} 0{'':<{_COL-1}}"
    #     else:
    #         node_fields = f"0{'':<{_COL-1}} {d.node:<{_COL}}"
    #     lines.append(f"D_d{d.node:<{_COL-3}} {node_fields}DIDEAL")
    # lines.append(f".MODEL DIDEAL D (IS={DIODE_IS:g} N={DIODE_N:g} RS=0)")
    lines.append("")

    # --- Save directives: node voltages only ---
    # A B-element current source exposes NO readable branch current in ngspice
    # (i(@b_dN) reads 0), so the diode currents are not saved; the slack w is
    # recovered exactly as w = M z + q from the node voltages (see lcp.verify and
    # main.py). Only node voltages are saved.
    #   --- Legacy diode-current saves (measured id when D_dN diodes were used) ---
    #   save_vars += [f"i(@d_d{c.node}[id])" for c in elements.diode_clamps]
    save_vars = [f"v({clamp.node})" for clamp in elements.diode_clamps]
    lines.append("* Save node voltages")
    for k in range(0, len(save_vars), _SAVE_PER_LINE):
        lines.append(".save " + " ".join(save_vars[k:k + _SAVE_PER_LINE]))
    lines.append("")

    # --- Analysis ---
    if mode == "tran":
        lines.append("* Transient analysis (shunt capacitors stabilize negative resistors)")
        lines.append(".options method=gear")
        lines.append(f".tran {tstep:.6g} {tstop:.6g} uic")
        lines.append(".end")
    else:
        lines.append("* DC operating point")
        lines.append(".OP")
        lines.append(".END")

    return "\n".join(lines) + "\n"


def write_netlist(
    elements: CircuitElements,
    run_name: str,
    out_dir: Union[str, Path],
    mode: str = "op",
    tstep: float | None = None,
    tstop: float | None = None,
) -> Path:
    """Write the .cir file; return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.cir"
    text = build_netlist(elements, run_name, mode=mode, tstep=tstep, tstop=tstop)
    path.write_text(text, encoding="utf-8")
    return path
