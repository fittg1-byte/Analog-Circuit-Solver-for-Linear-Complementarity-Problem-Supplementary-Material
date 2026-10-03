"""
sim/runner.py — Simulate a .cir netlist via ngspice.

Primary path  : subprocess (ngspice CLI) writing a raw file, then parsed.
Fallback path : PySpice NgSpiceShared — used when PYSPICE_BACKEND=shared is set.

Returns SimResult(voltages, diode_currents); caller unpacks. diode_currents is
None on the current B-element clamp path (ngspice exposes no branch current for
a behavioral source); the caller reconstructs the clamp currents from the node
voltages instead. See sim/parse.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from sim.parse import (
    parse_raw_ascii_trace,
    parse_stdout,
    extract_voltages,
    extract_voltage_trace,
    extract_diode_currents,
    available_variables,
)


@dataclass
class SimResult:
    voltages: Dict[str, float]                  # {node_str: voltage} at the final timepoint
    diode_currents: Optional[Dict[str, float]]  # {node_str: current}, None if unavailable
    n: int                                      # number of ports
    mode: str = "op"                            # "op" or "tran"
    time: Optional[np.ndarray] = None                      # tran only: time vector
    voltage_trace: Optional[Dict[str, np.ndarray]] = None  # tran only: {node_str: V(t)}


# ---------------------------------------------------------------------------
# Executable discovery
# ---------------------------------------------------------------------------

_NGSPICE_NAMES = ["ngspice_con.exe", "ngspice_con", "ngspice.exe", "ngspice"]

_NGSPICE_CANDIDATES = [
    r"C:\ngspice-46_64\Spice64\bin\ngspice_con.exe",
    r"C:\ngspice-46_64\Spice64\bin\ngspice.exe",
    r"C:\Program Files\Spice64\bin\ngspice_con.exe",
    r"C:\Program Files\Spice64\bin\ngspice.exe",
    r"C:\Program Files (x86)\ngspice\bin\ngspice_con.exe",
    r"C:\Program Files (x86)\ngspice\bin\ngspice.exe",
    r"C:\ngspice\bin\ngspice_con.exe",
    r"C:\ngspice\bin\ngspice.exe",
]


def _find_ngspice() -> str:
    for name in _NGSPICE_NAMES:
        exe = shutil.which(name)
        if exe:
            return exe
    for candidate in _NGSPICE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "ngspice not found on PATH or in common install locations.\n"
        "Searched names: " + ", ".join(_NGSPICE_NAMES) + "\n"
        "Searched paths:\n" + "\n".join(f"  {c}" for c in _NGSPICE_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Subprocess backend
# ---------------------------------------------------------------------------

def _simulate_subprocess(
    cir_path: Path, n: int, mode: str = "op", timeout: float = 600.0
) -> SimResult:
    """
    Run ngspice -b -r <raw> on cir_path.
    Parses the ASCII raw file (every timepoint) for node voltages and diode
    currents; voltages/diode_currents are taken from the FINAL timepoint
    (identical to the old behavior for .op, which has exactly one point).
    Falls back to stdout parsing for voltages if the raw file fails.
    """
    ngspice  = _find_ngspice()
    raw_path = cir_path.with_suffix(".raw")

    result = subprocess.run(
        [ngspice, "-b", "-r", str(raw_path), str(cir_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    raw_trace: Dict[str, np.ndarray] = {}
    if raw_path.exists():
        try:
            raw_trace = parse_raw_ascii_trace(raw_path)
        except Exception as exc:
            print(f"[runner] Raw file parse failed ({exc}); trying stdout fallback.")

    final_dict: Dict[str, float] = {name: float(arr[-1]) for name, arr in raw_trace.items()}

    # --- Extract node voltages (final timepoint) ---
    voltages = extract_voltages(final_dict)
    if not voltages:
        voltages = parse_stdout(result.stdout + "\n" + result.stderr)
    if not voltages:
        print("=== ngspice stdout ===")
        print(result.stdout)
        print("=== ngspice stderr ===")
        print(result.stderr)
        raise RuntimeError(
            "ngspice produced no parseable node voltages. "
            "See output above and netlist at: " + str(cir_path)
        )

    # --- Extract diode currents (final timepoint) ---
    # The complementarity elements are now behavioral (B-element) current sources,
    # whose branch current ngspice does not expose (i(@b_dN) reads 0), so no diode
    # currents are saved and this returns None by design. The slack w is instead
    # recovered exactly as w = M*z+q from the node voltages (see main.py). The
    # legacy ideal-diode path DID expose @d_dN[id]; if it is restored, this will
    # pick the measured currents back up automatically.
    diode_currents = extract_diode_currents(final_dict, n)

    if diode_currents is None:
        # Expected with B-element sources; w falls back to M*z+q in main.py.
        # (When diodes are restored but currents are still missing, uncomment the
        # diagnostic below to see which variables ngspice actually wrote.)
        # print(f"[runner] Diode currents unavailable; variables present: "
        #       f"{available_variables(final_dict)}")
        pass

    time = None
    voltage_trace = None
    if mode == "tran" and raw_trace:
        time = raw_trace.get("time")
        voltage_trace = extract_voltage_trace(raw_trace)

    return SimResult(
        voltages=voltages,
        diode_currents=diode_currents,
        n=n,
        mode=mode,
        time=time,
        voltage_trace=voltage_trace,
    )


# ---------------------------------------------------------------------------
# PySpice shared-lib backend
# ---------------------------------------------------------------------------

def _simulate_pyspice(cir_path: Path, n: int) -> SimResult:
    import PySpice.Logging.Logging as Logging
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    Logging.setup_logging(logging_level="WARNING")
    netlist_text = cir_path.read_text(encoding="utf-8")
    ngspice = NgSpiceShared.new_instance()
    ngspice.load_circuit(netlist_text)
    ngspice.run()

    analysis = ngspice.last_simulation
    voltages: Dict[str, float] = {
        node_name: float(node_data)
        for node_name, node_data in analysis.nodes.items()
    }
    return SimResult(voltages=voltages, diode_currents=None, n=n, mode="op")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate(
    cir_path: Path | str, n: int, mode: str = "op", timeout: float = 600.0
) -> SimResult:
    """
    Simulate cir_path in ngspice (.op or .tran); return SimResult with
    final-timepoint node voltages and (when available) measured diode
    currents. For mode="tran", also returns the full time/voltage trace.

    n       : number of ports (needed to look up D_1 .. D_n currents).
    mode    : "op" or "tran" — must match the analysis card in cir_path.
    timeout : seconds before the ngspice subprocess is killed.
    Set PYSPICE_BACKEND=shared to use NgSpiceShared instead of subprocess
    (op mode only — tran always uses the subprocess backend).
    """
    cir_path = Path(cir_path)
    backend  = os.environ.get("PYSPICE_BACKEND", "subprocess").lower()

    if backend == "shared" and mode == "op":
        try:
            return _simulate_pyspice(cir_path, n)
        except Exception as exc:
            print(f"[runner] PySpice shared backend failed ({exc}); falling back to subprocess.")

    return _simulate_subprocess(cir_path, n, mode=mode, timeout=timeout)
