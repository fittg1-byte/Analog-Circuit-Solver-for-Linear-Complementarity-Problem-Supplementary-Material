"""
plot/solution.py — Figures for an LCP circuit run. Pure renderers: every
function builds a matplotlib figure and returns it; the caller saves it.

  plot_solution  : complementarity scatter, one point pair per port.
                   Blue circles are z_i (node voltage, V), orange crosses are
                   w_i (diode current, A). For a valid LCP solution exactly one
                   of the two is near zero at each port, which is what the plot
                   is for: a glance says whether complementarity holds.
  plot_transient : V_i(t) for every node on a single axis, with the settling
                   time marked. Used only for tran-mode runs.
  plot_traces    : split early/late panel pair for a quantity over time, in the
                   style of the transient figures in the LCP-circuit literature.
                   Takes voltages or clamp currents; see its own docstring.

Every plot degrades gracefully with problem size: per-point value labels,
one-tick-per-port axes, and per-node legends are drawn only while they remain
readable, and figure width is capped. The saved JSON always holds the exact
values, so the figures stay qualitative overviews.
"""

from __future__ import annotations

import numpy as np

_ANNOTATE_MAX_N = 12   # label individual values only up to this many ports
_TICK_MAX_N     = 30   # one x-tick per port up to this many ports
_LEGEND_MAX_N   = 10   # per-node legend in the transient plot up to this n

# Fraction of the run that goes in the EARLY panel of a split trace plot. The
# transient's fan-out occupies the first few percent of a run whose tstop is
# sized off the slowest mode, so 5% resolves it without starving the late panel.
_DEFAULT_SPLIT_FRAC = 0.05

# SI prefixes for axis auto-scaling, descending. The micro prefix is written as
# mathtext so matplotlib renders a real mu without any non-ASCII source bytes.
_SI_PREFIXES = (
    (1e0,   ""),
    (1e-3,  "m"),
    (1e-6,  r"$\mu$"),
    (1e-9,  "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)


def _si_scale(max_abs: float) -> tuple[float, str]:
    """
    Pick an SI prefix so an axis spanning `max_abs` reads in human units rather
    than "1e-10". Returns (factor, prefix): multiply the data by factor and
    write the prefix in front of the base unit.

        0.003   -> (1e3,  "m")      3 mV
        2e-10   -> (1e12, "p")      200 ps

    Values land in [1, 1000) wherever a prefix exists; anything above 1 keeps the
    bare unit and anything below femto saturates at femto.
    """
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0, ""
    for unit, prefix in _SI_PREFIXES:
        if max_abs >= unit:
            return 1.0 / unit, prefix
    unit, prefix = _SI_PREFIXES[-1]
    return 1.0 / unit, prefix


def _decade_label(value: float, _pos=None) -> str:
    """Tick label for a decade on a symlog scale: 0, $10^{-3}$, $10^{1}$, ..."""
    if value == 0.0:
        return "0"
    exponent = int(round(np.log10(abs(value))))
    sign = "-" if value < 0 else ""
    return rf"${sign}10^{{{exponent}}}$"


def plot_traces(
    time: np.ndarray,
    traces: dict,
    run_name: str = "",
    *,
    y_label: str = "node voltage",
    y_base_unit: str = "V",
    t_split: float | None = None,
    t_max: float | None = None,
    yscale: str = "linear",
    color_cycle: str | None = None,
    linewidth: float = 0.8,
    width_ratios: tuple[float, float] = (1.0, 1.5),
):
    """
    Split-axis ensemble trajectory plot: every port's waveform, drawn as a pair
    of panels sharing one y-axis — a narrow EARLY panel resolving the initial
    transient, and a wider LATE panel carrying the run out to steady state.
    This is the layout of the transient panels in the LCP-circuit literature
    (Fig. 12 of the QPCBLEND paper), and it exists because the two regimes live
    on different timescales: the fan-out happens in the first few percent of the
    run, which a single linear axis squashes into an unreadable sliver.

    time        : time vector (seconds).
    traces      : {node_str: array over time}; any quantity, one entry per port.
    y_label     : physical name of the plotted quantity, e.g. "node voltage" or
                  "diode current". Deliberately NOT an LCP symbol: these panels
                  show what the circuit measures, and which circuit quantity
                  carries the LCP solution depends on the synthesis path (for a
                  Z-type/dual run the roles swap), so naming the physical
                  quantity is the only labeling that is always correct.
    y_base_unit : unquantified unit, e.g. "V" or "A". Both axes auto-pick an SI
                  prefix from the data's own magnitude (see _si_scale), so they
                  read "time [ps]" / "node voltage [mV]" rather than a power of
                  ten. Panels share one prefix so the pair reads as one axis.
    t_split     : seconds; boundary between the two panels. None picks
                  _DEFAULT_SPLIT_FRAC of the plotted span, which puts the
                  initial fan-out in the early panel for a run whose tstop is
                  sized off the slowest mode.
    t_max       : right edge of the LATE panel, in seconds. None plots to the
                  end of the run.
    yscale      : "linear", or "symlog" for a quantity whose ports span decades.
                  The clamp currents do: a handful of ports carry tens of amps
                  while most carry milliamps, so on a linear axis every small
                  port collapses onto zero and the switching is invisible.
                  symlog (not plain log) because a BLOCKING port carries exactly
                  zero current — a real value a log axis cannot draw. The linear
                  band around zero is sized to the smallest nonzero value on
                  screen, so "sitting at zero" and "conducting faintly" stay
                  visually distinct and every decade above is readable. The
                  y-axis is left unscaled in this mode (decade ticks already
                  read cleanly; an SI prefix on top of them would mislead).
    color_cycle : None uses the axes' default property cycle, so each port gets
                  its own color and the colors repeat every ~10 ports exactly
                  as in plot_transient. Color here encodes port IDENTITY, not
                  magnitude: with this many traces no encoding makes an
                  individual port recoverable, and cycled hues keep neighboring
                  trajectories separable where a single ramp would blend them
                  into one band. Pass a qualitative colormap name ("tab20",
                  "hsv") for more hues before the cycle repeats.

    Returns the figure; the caller saves it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    time = np.asarray(time, dtype=float)
    nodes = sorted(traces.keys(), key=int)
    Y = np.stack([np.asarray(traces[k], dtype=float) for k in nodes])  # (n, T)
    n = len(nodes)

    # --- Visible span: everything up to t_max (default, the whole run) ---
    if t_max is not None and np.isfinite(t_max) and t_max > time[0]:
        keep = time <= t_max
        if keep.sum() < 2:          # degenerate window; show everything instead
            keep = np.ones_like(time, dtype=bool)
    else:
        keep = np.ones_like(time, dtype=bool)
    t_vis, Y_vis = time[keep], Y[:, keep]

    # --- Panel boundary ---
    t_lo, t_hi = float(t_vis[0]), float(t_vis[-1])
    if t_split is None or not np.isfinite(t_split) or not (t_lo < t_split < t_hi):
        t_split = t_lo + _DEFAULT_SPLIT_FRAC * (t_hi - t_lo)

    # One SI prefix for BOTH panels (a pair with different units is unreadable),
    # and one for y, taken from the data that is actually on screen.
    t_factor, t_prefix = _si_scale(t_hi)
    if yscale == "symlog":
        y_factor, y_prefix = 1.0, ""      # decade ticks carry their own scale
    else:
        y_factor, y_prefix = _si_scale(float(np.max(np.abs(Y_vis))))

    # Constrained (not tight) layout: tight_layout cannot account for a figure
    # level supxlabel/suptitle and drops them on top of the tick labels.
    fig, (ax_early, ax_late) = plt.subplots(
        1, 2, sharey=True,
        figsize=(8.0, 3.6),
        gridspec_kw={"width_ratios": list(width_ratios)},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.03)

    # Color by port index, cycling — identity, not magnitude.
    if color_cycle is None:
        prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        colors = [prop_colors[k % len(prop_colors)] for k in range(n)]
    else:
        colormap = plt.get_cmap(color_cycle)
        count = getattr(colormap, "N", 256)
        if count >= 256:            # continuous map: spread hues across n ports
            colors = [colormap(k / max(n - 1, 1)) for k in range(n)]
        else:                       # qualitative map: cycle its discrete entries
            colors = [colormap(k % count) for k in range(n)]

    early = t_vis <= t_split
    late = t_vis >= t_split         # overlap by one point so the panels join up
    for ax, mask in ((ax_early, early), (ax_late, late)):
        if mask.sum() < 2:
            continue
        for k in range(n):
            ax.plot(t_vis[mask] * t_factor, Y_vis[k][mask] * y_factor,
                    color=colors[k], linewidth=linewidth,
                    solid_joinstyle="round", zorder=2)

    if yscale == "symlog":
        # Linear band sized so an exactly-zero (blocking) port sits on the floor
        # instead of vanishing. Size it from the smallest value a port actually
        # HOLDS at the end, not the smallest value anywhere: a port switching off
        # sweeps through arbitrarily small currents on its way to zero, and
        # letting those set the floor spends half the axis on empty decades that
        # contain nothing but the near-vertical transitions themselves.
        held = np.abs(Y_vis[:, -1])
        held = held[held > 0.0]
        if held.size == 0:                       # nothing conducting at the end
            held = np.abs(Y_vis[np.abs(Y_vis) > 0.0])
        linthresh = float(held.min()) if held.size else 1.0
        # linscale=1 gives the zero floor a full decade of height; at 0.5 the
        # floor tick and the first decade tick overprint each other.
        ax_early.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
        ax_early.yaxis.set_major_formatter(FuncFormatter(_decade_label))

    for ax in (ax_early, ax_late):
        if yscale != "symlog":
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--",
                       alpha=0.45, zorder=1)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=10)

    # Anchor the early panel at 0 (the paper's convention) — the first simulated
    # point sits a few femtoseconds in, which would otherwise crop the origin.
    ax_early.set_xlim(min(0.0, t_lo * t_factor), t_split * t_factor)
    ax_late.set_xlim(t_split * t_factor, t_hi * t_factor)
    ax_early.set_ylabel(f"{y_label} [{y_prefix}{y_base_unit}]", fontsize=11)
    if yscale != "symlog":
        ax_early.margins(y=0.05)

    # A single centered x-label under both panels, as in the reference figure.
    fig.supxlabel(f"time [{t_prefix}s]", fontsize=11)
    title = f"{y_label} — {run_name}" if run_name else y_label
    fig.suptitle(f"{title}  ({n} ports)", fontsize=11)
    return fig


def plot_solution(
    z: np.ndarray,
    w: np.ndarray,
    run_name: str = "",
    w_label: str = r"$w_i$ (diode current, A)",
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(z)
    ports = np.arange(1, n + 1)

    width = min(12.0, max(5.0, 0.9 * n + 2.0))
    fig, ax = plt.subplots(figsize=(width, 4))

    ax.scatter(
        ports, z,
        color="#4C72B0", marker="o", s=90, zorder=3,
        label=r"$z_i$ (node voltage, V)",
    )
    ax.scatter(
        ports, w,
        color="#DD8452", marker="x", s=90, linewidths=2, zorder=3,
        label=w_label,
    )

    # Annotate each point with its value — only while the labels stay legible
    if n <= _ANNOTATE_MAX_N:
        for i, (zi, wi) in enumerate(zip(z, w), start=1):
            ax.annotate(f"{zi:.4f}", (i, zi),
                        textcoords="offset points", xytext=(6, 2),
                        fontsize=8, color="#4C72B0")
            ax.annotate(f"{wi:.4f}", (i, wi),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color="#DD8452")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Diode index $i$", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    if n <= _TICK_MAX_N:
        ax.set_xticks(ports)
        ax.set_xticklabels([str(p) for p in ports])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    title = f"LCP Solution — {run_name}" if run_name else "LCP Solution"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig


def plot_transient(
    time: np.ndarray,
    voltage_trace: dict,
    run_name: str = "",
    t_s: float | None = None,
):
    """
    Transient waveform plot: V_i(t) for every node, with a dashed vertical
    line at the settling time t_s (if it falls within the plotted range).
    Additive to plot_solution — used only for tran-mode runs.
    """
    import matplotlib.pyplot as plt

    time = np.asarray(time, dtype=float)
    nodes = sorted(voltage_trace.keys(), key=int)
    n = len(nodes)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Beyond a handful of nodes, individual labels/opacity stop being useful
    line_alpha = 1.0 if n <= _LEGEND_MAX_N else 0.6
    for node in nodes:
        ax.plot(time, np.asarray(voltage_trace[node], dtype=float),
                alpha=line_alpha,
                label=f"$V_{{{node}}}(t)$" if n <= _LEGEND_MAX_N else None)

    if t_s is not None and time[0] <= t_s <= time[-1]:
        ax.axvline(t_s, color="black", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"$t_s$ = {t_s:.4g} s")
        # Simulation length is set by the slowest mode's settle_factor and is
        # typically much longer than t_s, which squashes the settling curve
        # into an unreadable sliver at the left edge. Zoom the x-axis to a
        # window that puts t_s at its center instead of showing the full run.
        xlim_hi = min(float(time[-1]), 2.0 * t_s - float(time[0]))
        if xlim_hi > float(time[0]):
            ax.set_xlim(float(time[0]), xlim_hi)

    title = f"Transient response — {run_name}" if run_name else "Transient response"
    if n > _LEGEND_MAX_N:
        title += f"  ({n} nodes)"
    if t_s is not None:
        title += f"  (settled at $t_s$ = {t_s:.4g} s)"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Node voltage (V)", fontsize=11)
    if n <= _LEGEND_MAX_N or t_s is not None:
        ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig
