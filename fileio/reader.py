"""
fileio/reader.py — Load M and q from CSV files; fall back to interactive prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _load_matrix_csv(path: Path) -> np.ndarray:
    # ndmin=2 keeps a 1x1 CSV two-dimensional (loadtxt would return a scalar)
    return np.loadtxt(path, delimiter=",", dtype=float, encoding="utf-8-sig", ndmin=2)


def _load_vector_csv(path: Path) -> np.ndarray:
    # flatten() accepts q written as either a row or a column
    return np.loadtxt(path, delimiter=",", dtype=float, encoding="utf-8-sig", ndmin=1).flatten()


def _prompt_row(prompt: str) -> Optional[list[float]]:
    """Read one comma-separated row; None on blank line. Re-prompts on typos."""
    while True:
        line = input(prompt).strip()
        if not line:
            return None
        try:
            return [float(x) for x in line.split(",")]
        except ValueError:
            print("  Could not parse; enter comma-separated numbers (e.g. 1, -2.5, 0).")


def _prompt_matrix() -> Tuple[np.ndarray, np.ndarray]:
    print("Enter M row by row (comma-separated). Blank line to finish:")
    rows: list[list[float]] = []
    while True:
        row = _prompt_row("  row> ")
        if row is None:
            if rows:
                break
            print("  M needs at least one row.")
            continue
        if rows and len(row) != len(rows[0]):
            print(f"  Row has {len(row)} entries but earlier rows have "
                  f"{len(rows[0])}; please re-enter.")
            continue
        rows.append(row)
    M = np.array(rows, dtype=float)

    print(f"Enter q ({len(rows)} comma-separated values):")
    while True:
        q_vals = _prompt_row("  q> ")
        if q_vals is not None and len(q_vals) == len(rows):
            break
        print(f"  q must have exactly {len(rows)} values.")
    q = np.array(q_vals, dtype=float)
    return M, q


def load_inputs(
    m_csv: Optional[str | Path] = None,
    q_csv: Optional[str | Path] = None,
    run_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (M, q, run_name).
    Falls back to interactive prompts when both CSV paths are None;
    supplying only one of the two paths is an error.
    """
    if (m_csv is None) != (q_csv is None):
        raise ValueError("Provide both --M and --q, or neither (interactive mode).")

    if m_csv is not None and q_csv is not None:
        M = _load_matrix_csv(Path(m_csv))
        q = _load_vector_csv(Path(q_csv))
    else:
        print("No CSV paths supplied - switching to interactive input.")
        M, q = _prompt_matrix()

    if run_name is None:
        run_name = input("Run name (no spaces): ").strip() or "lcp_run"

    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError(f"M must be square; got shape {M.shape}")
    if q.shape != (n,):
        raise ValueError(f"q must have length {n}; got shape {q.shape}")

    return M, q, run_name
