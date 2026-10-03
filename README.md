# LCP-to-Circuit Synthesizer

> **Disclaimer.** This tool is still undergoing modifications, and is not in its
> final form.

Supplementary software for the paper "Analog Circuit Solver for Linear
Complementarity Problem" by Gabriel Fitt and A. A. Adegbege. This tool takes a
Linear Complementarity Problem (LCP) defined by a symmetric matrix `M` and a
vector `q`, synthesizes an equivalent analog circuit (resistors, current
sources, and one ideal-diode clamp per port), simulates it in ngspice, and
verifies that the circuit's steady state reproduces the LCP solution.

The complementarity constraint `z_i * w_i = 0` is realized physically at each
port: the clamp is either conducting (port voltage held at ~0, so `z_i = 0`) or
blocking (no current, so `w_i = 0`). Solving the LCP is therefore reduced to
letting the circuit find its own operating point.


## What it does

Given `M` (n x n, symmetric) and `q` (length n), the pipeline:

1. Loads `M` and `q` from CSV files (or interactive prompts).
2. Validates shape and symmetry — the only hard requirements. Eigenvalues are
   computed and the definiteness class is reported, but it is **flagged, not
   gated**: positive definite, positive/negative semi-definite (singular), and
   indefinite `M` are all accepted and run. The off-diagonal sign pattern is
   classified alongside it (K-type, flip-fixable, or non-K), which is
   informational and never selects the analysis mode.
3. For a matrix that is not hyperdominant, tests whether the inverse `M^-1` is
   hyperdominant (a Z-type problem) and, if so, solves the dual LCP on the
   inverse circuit. This path is skipped for a singular `M`, which has no
   inverse.
4. Synthesizes the circuit elements from `M` and `q`.
5. Chooses a DC operating-point (`.op`) analysis, or a transient (`.tran`)
   analysis with per-node shunt capacitors when negative resistors are present.
   The choice routes on **resistor sign**, not on the matrix class.
6. Writes a SPICE netlist (`.cir`), the primary deliverable.
7. Simulates it in ngspice and extracts node voltages (`z`). A behavioral
   source exposes no branch current, so the slack `w` is reconstructed by
   applying the clamp's own law to those voltages, then cross-checked
   against `M*z+q`.
8. Verifies the three LCP conditions (`z >= 0`, `w >= 0`, `z_i * w_i = 0`) with
   tolerances derived from the clamp model's own non-ideality — the finite
   forward drop `Vf = I / DIODE_G` — rather than arbitrary constants.
9. Saves a JSON results file and PNG plots.

### On the matrix class

Definiteness and the sign pattern are diagnostics, not gates. A singular `M`
additionally triggers a null-space check on `q`: if `q` excites a null mode the
LCP may have no steady state (that mode never settles); if `q` lies in
`range(M)` a solution exists, though it may be non-unique. Either way the run
proceeds and the finding is printed.

The point of diode flips and of the `M^-1` detour is a circuit with **only
positive resistors**, which needs hyperdominance: a 2-colorable off-diagonal
sign pattern *and* diagonal dominance. The sign pattern alone does not prevent
negative grounded resistors, so a K-type matrix can still route to transient.


## Requirements

- **Python 3.10 or newer** (the code uses PEP 604 `X | None` type syntax).
- **ngspice 46** (or a compatible recent release), installed separately and
  available on `PATH` or at a standard Windows install location. This is an
  external native program, not a Python package; see Installation below.
- Python packages listed in `requirements.txt`:
  - numpy >= 1.26
  - matplotlib >= 3.8
  - pytest >= 8.0 (tests only)

PySpice is **not** a dependency and is not installed by `requirements.txt`. See
[License](#license) for the optional backend that can use it.


## Installation

Install the Python dependencies:

    pip install -r requirements.txt

Install ngspice separately. On Windows, download the binary distribution from
https://ngspice.sourceforge.io/download.html and either add its `bin` folder to
your `PATH` or install it to one of the locations the runner searches (for
example `C:\ngspice-46_64\Spice64\bin`). The exact search list is in
`sim/runner.py`. This project was developed and tested against **ngspice
version 46** on Windows; report the version you use when reproducing results.


## Usage

Run with CSV inputs:

    python main.py --M path/to/M.csv --q path/to/q.csv --run-name demo

Run interactively, entering M row by row and then q at the prompts — the
quickest way to try the tool without preparing files:

    python main.py

Useful flags:

- `--run-name NAME`  run identifier (a date stamp is prepended automatically)
- `--skip-sim`       write the netlist only; do not run ngspice
- `--out-dir DIR`    output directory (default: `./output`)
- `--mode {auto,op,tran}`  analysis mode (default `auto`: `tran` when any
  resistor is negative, else `op`)
- `--no-show`        save PNGs without opening interactive plot windows
- `--no-caps`        **[temporary]** force `.op` (no shunt capacitors) even when
  negative resistors are present, i.e. run as though there were none. Without
  this flag an interactive run with negative resistors offers the same choice
  at a prompt; answering no, or running with no console attached, keeps the
  normal auto-routing. Expect the DC solve to struggle — that is the point of
  the experiment.

Example input matrices are not yet included in this repository; they are being
prepared for release alongside the paper. In the meantime, see **Input and
output formats** below for the CSV layout, or use interactive mode. The test
suite is self-contained and needs no input files.


## Input and output formats

- **Input**: `M` as a comma-separated square matrix (one row per line); `q` as a
  comma-separated row or column vector of matching length. A UTF-8 BOM is
  tolerated.
- **Output** (written to `output/`):
  - `<run>.cir`            the SPICE netlist (primary deliverable)
  - `<run>.raw`            the ngspice raw output
  - `<run>_results.json`   solution `z`, slack `w`, run metadata, tolerances,
                           and the full LCP verification report
  - `<run>_solution.png`   complementarity scatter of `z` and `w` per port
  - `<run>_transient.png`  node voltages vs time, settling time marked
                           (transient runs only)
  - `<run>_voltages.png`   split early/late panel pair of every node's voltage
                           (transient runs only)

The companion `<run>_currents.png` (the same split for every port's clamp
current, on a symlog axis) is **currently disabled**; see the commented block in
Step 9 of `main.py` to restore it.

Every file is prefixed with a `YYYYMMDD_HHMMSS_` stamp, so repeated runs never
overwrite each other.


## Tests

    pytest

- `tests/test_verify.py`    the pure LCP logic and the clamp-derived tolerances.
- `tests/test_golden.py`    generated netlists and end-to-end solutions against
                            known-good references, across K-type, Z-type and
                            P-type matrices.
- `tests/test_psd_zeros.py` the definiteness expansion (semi-definite and
                            indefinite `M`), sign 2-coloring through structural
                            zeros, and zero-admittance handling.


## License

This software is released under the **MIT License**. See [LICENSE](LICENSE) for
the full text.

    Copyright (c) 2026 Gabriel Fitt and A. A. Adegbege

### Third-party software

This project is built on the following open-source software. None of it is
vendored or redistributed here; each is installed separately by the user and
used under its own license.

| Software    | Role                                        | License        |
|-------------|---------------------------------------------|----------------|
| ngspice 46  | Circuit simulation engine (`.op` / `.tran`) | BSD-3-Clause (with some GPL-licensed components) |
| NumPy       | Linear algebra, arrays                      | BSD-3-Clause   |
| Matplotlib  | Plotting                                    | Matplotlib (BSD-compatible) |
| pytest      | Test framework (development only)           | MIT            |
| PySpice     | **Optional** shared-library ngspice backend | GPL-3.0        |

ngspice is invoked as a separate executable via `subprocess`, never linked into
this program, so it imposes no license condition on this code.

**On PySpice and the MIT license.** PySpice is GPL-3.0, this project is MIT, and
the two are kept apart deliberately:

- PySpice is **not** in `requirements.txt` (the line is commented out) and is
  never installed as a dependency of this project.
- Nothing is imported at module scope. The two `import PySpice...` statements
  live inside `_simulate_pyspice()` in `sim/runner.py` and execute only if you
  opt in.
- Opting in requires setting `PYSPICE_BACKEND=shared` *and* installing PySpice
  yourself. The default path shells out to the ngspice executable.
- No PySpice code is bundled, vendored or distributed in this repository.

So this repository contains no GPL code and distributes none. If you install
PySpice and enable that backend, the resulting combination on your machine is
yours to use under the GPL's terms; the MIT grant above covers only the code in
this repository.

### Citation

The LCP-to-circuit synthesis method and verification approach implemented here
are the contribution of the accompanying MUSE26 paper; please cite the paper
when using this software. BibTeX entries for the dependencies above are in
`references.bib`.


## References

ngspice: H. Vogt, M. Hendrix, P. Nenzi, and D. Warning, "ngspice, the open
source Spice circuit simulator," version 46, 2024.
https://ngspice.sourceforge.io/

NumPy: C. R. Harris et al., "Array programming with NumPy," Nature, vol. 585,
pp. 357-362, 2020. doi:10.1038/s41586-020-2649-2

Matplotlib: J. D. Hunter, "Matplotlib: A 2D graphics environment," Computing in
Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007. doi:10.1109/MCSE.2007.55

PySpice: F. Salvaire, "PySpice: Simulate electronic circuit using Python and the
ngspice / Xyce simulators." https://pyspice.fabrice-salvaire.fr/

pytest: H. Krekel et al., "pytest." https://pytest.org/
