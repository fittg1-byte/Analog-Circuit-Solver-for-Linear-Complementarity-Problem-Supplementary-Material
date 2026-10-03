"""
fileio/writer.py — Save results JSON and plot PNG to the output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_results(
    out_dir: Path,
    run_name: str,
    M: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    lcp_report: dict,
    *,
    mode: str | None = None,
    matrix_class: str | None = None,
    definiteness: str | None = None,
    settling_time: float | None = None,
    tolerances: dict | None = None,
) -> Path:
    """
    Write a JSON results file; return its path.

    The keyword arguments record run metadata (simulation mode, matrix class,
    definiteness class, transient settling time, verification tolerances) so a
    results file is self-describing and reproducible on its own.
    """
    data = {
        "run_name": run_name,
        "M": M,
        "q": q,
        "z_circuit": z,
        "w_computed": w,
        "sim_mode": mode,
        "matrix_class": matrix_class,
        "definiteness": definiteness,
        "settling_time_s": settling_time,
        "tolerances": tolerances,
        "lcp_report": lcp_report,
    }
    path = out_dir / f"{run_name}_results.json"
    path.write_text(json.dumps(data, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    return path


def save_plot(fig: Any, out_dir: Path, run_name: str, kind: str = "solution") -> Path:
    """Save a matplotlib figure as <run_name>_<kind>.png; return its path."""
    path = out_dir / f"{run_name}_{kind}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
