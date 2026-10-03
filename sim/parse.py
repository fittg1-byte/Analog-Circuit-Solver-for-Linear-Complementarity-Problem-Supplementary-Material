"""
sim/parse.py — Parse ngspice ASCII raw files into dicts of all variables.

Primary path : ASCII raw file written by ngspice -b -r <raw>.
Fallback     : stdout regex (kept for node voltages only).

ngspice variable naming in raw files:
  Node voltages  : v(1), v(2), ...
  Ideal diodes   : i(@d_d1[id]), i(@d_d2[id]), ...  (from .save @D_dN[id])
  B-elem fallback: i(@b_d1[i]),  i(@b_d2[i]),  ...  (from .save @b_dN[i])
ngspice wraps all .save @device[param] entries as i(@device[param]) in the raw file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# ASCII raw file parser — returns ALL variables
# ---------------------------------------------------------------------------

def parse_raw_ascii_trace(path: Path) -> Dict[str, np.ndarray]:
    """
    Parse an ngspice ASCII raw file, returning EVERY data point per variable.
    For .op there is exactly one point; for .tran this returns the full time
    series, including the "time" variable itself.

    Returns {variable_name_lowercase: np.ndarray of values, one per point}.
    """
    text = path.read_text(errors="replace")

    for marker in ("Values:\n", "Values:\r\n"):
        idx = text.find(marker)
        if idx >= 0:
            break
    else:
        raise ValueError(f"'Values:' marker not found in {path}")

    header_text = text[:idx]
    data_text   = text[idx + len(marker):]

    n_vars: int = 0
    var_names: List[str] = []
    in_vars = False

    for line in header_text.splitlines():
        stripped = line.strip()
        low      = stripped.lower()

        if low.startswith("no. variables:"):
            n_vars = int(stripped.split(":")[1].strip())
        elif low == "variables:":
            in_vars = True
        elif in_vars and stripped:
            parts = stripped.split()
            if len(parts) >= 2:
                var_names.append(parts[1].lower())

    if n_vars == 0 or not var_names:
        raise ValueError(f"No variables found in raw file header: {path}")

    # Each point: first line is "{index} {value0}" (2 tokens), the remaining
    # n_vars-1 lines are "{valueN}" (1 token each).
    values_per_var: List[List[float]] = [[] for _ in var_names]
    current: List[float] = []

    def _flush() -> None:
        if not current:
            return
        for k, v in enumerate(current):
            values_per_var[k].append(v)

    for line in data_text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            _flush()
            current = [float(parts[1])]
        elif len(parts) == 1:
            try:
                current.append(float(parts[0]))
            except ValueError:
                pass
    _flush()

    return {name: np.array(vals, dtype=float) for name, vals in zip(var_names, values_per_var)}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_voltages(raw_dict: Dict[str, float]) -> Dict[str, float]:
    """Return {node_name: voltage} for all v(node) variables."""
    result: Dict[str, float] = {}
    for name, val in raw_dict.items():
        if name.startswith("v(") and name.endswith(")"):
            result[name[2:-1]] = val
    return result


def extract_voltage_trace(raw_trace: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return {node_name: array of voltages over time} for all v(node) variables."""
    result: Dict[str, np.ndarray] = {}
    for name, arr in raw_trace.items():
        if name.startswith("v(") and name.endswith(")"):
            result[name[2:-1]] = arr
    return result


def extract_diode_currents(
    raw_dict: Dict[str, float],
    n: int,
) -> Optional[Dict[str, float]]:
    """
    Return {node_str: current} for the complementarity elements, or None.

    The elements are now behavioral (B-element) current sources, whose branch
    current ngspice does NOT expose (i(@b_dN) reads 0), so the netlist saves no
    diode currents and this returns None by design — the caller then recovers the
    slack exactly as w = M*z+q. The candidate list is retained (and matches the
    legacy ideal-diode @d_dN[id] form) so that restoring the diode synthesis
    automatically re-enables measured currents.

    ngspice lowercases all element names; it wraps .save @device[param] entries
    as i(@device[param]) in the raw file.
    """
    result: Dict[str, float] = {}
    for i in range(1, n + 1):
        candidates = [
            # --- Legacy ideal-diode patterns (re-enabled if diodes are restored) ---
            f"i(@d_d{i}[id])",    # ideal diode D_d{i} — preferred
            f"@d_d{i}[id]",       # ideal diode, unwrapped (older ngspice)
            # f"id(d_d{i})",      # printed form in some builds
            # f"d_d{i}#id",       # alternate raw-file form
            # Note: B-element sources (B_d{i}) expose no readable branch current,
            # so there is deliberately no candidate for them here.
        ]
        for key in candidates:
            if key in raw_dict:
                result[str(i)] = raw_dict[key]
                break

    if len(result) == n:
        return result

    # Partial or missing (the expected case for B-element sources) — signal the
    # caller to fall back to w = M*z+q.
    return None


def available_variables(raw_dict: Dict[str, float]) -> List[str]:
    """Return sorted list of all variable names in the raw dict."""
    return sorted(raw_dict.keys())


# ---------------------------------------------------------------------------
# Stdout fallback (node voltages only)
# ---------------------------------------------------------------------------

_OP_LINE = re.compile(
    r"^v\((\w+)\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_stdout(text: str) -> Dict[str, float]:
    """Extract node voltages from ngspice stdout/stderr text."""
    voltages: Dict[str, float] = {}
    for line in text.splitlines():
        m = _OP_LINE.match(line.strip())
        if m:
            voltages[m.group(1)] = float(m.group(2))
    return voltages


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def extract_node_voltages(voltages: Dict[str, float], n: int) -> list[float]:
    """Return [V_1, ..., V_n] from a voltage dict keyed by node name string."""
    missing = [str(i) for i in range(1, n + 1) if str(i) not in voltages]
    if missing:
        raise KeyError(
            f"Node voltage(s) {missing} not found in simulation output; "
            f"available nodes: {sorted(voltages)}"
        )
    return [voltages[str(i)] for i in range(1, n + 1)]
