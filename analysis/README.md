# Analysis Artifacts

This directory contains deterministic, reproducibility-grade analysis scripts used to generate the verification and visualization results reported in the accompanying paper(s).

All scripts operate exclusively on released experimental artifacts (OpenQASM programs and shot-count CSVs) and produce fixed, audit-ready outputs.

No quantum hardware execution occurs within this directory.

---

## Scope and Intent

The contents of this directory are intentionally constrained to:

- Logical circuit equivalence certification
- Empirical probability-mass analysis
- Execution-dependent geometry analysis
- Deterministic visualization reproduction

These scripts:

- Operate only on released artifacts
- Enforce strict schema and qubit-length validation
- Use fixed ordering, seeded randomness (where applicable), and controlled rendering parameters
- Produce outputs corresponding to figures reported in the paper

They do not:

- Generate circuits
- Modify QASM programs
- Execute circuits on hardware
- Perform adaptive optimization
- Encode proprietary construction logic

All methodological context is described in `/docs` and the associated publication.

---

## Directory Structure

analysis/
    README.md
    esc_analyze.py
    logical_unitary_equivalence.py
    mass_composition.py
    ghz_geometry.py
        10Q-CBP/
        15Q-MAIN/
        15Q-MCE/
        20Q-SBP/

Each test directory contains:

- Baseline QASM (if applicable)
- Conditioned QASM (if applicable)
- Baseline shot-count CSV
- Conditioned shot-count CSV

Naming conventions match those used in `/qasm`, `/data`, and `/reports`.

Consistency across directories is required for traceability.

---

# Script Overview

## 1. Deterministic Statistical Verification (ESC Analysis)

**File:** `esc_analyze.py`

### Purpose

Reproduces the full statistical analysis reported in the paper using only:
    Shot-order bitstring logs (CSV)
    Released OpenQASM files (for burden metrics only)

This script verifies empirical probability structure, divergence metrics, bootstrap confidence intervals, shell-space geometry, and execution-structure conditioning (ESC) metrics.

It does not certify circuit equivalence (handled by Script 2)

### Script Output

- esc_full_analysis.csv
- esc_full_analysis.json

Outputs include:

- Point estimates
- Bootstrap CIs
- Bootstrap standard errors
- PCA summaries
- Δv eigenspectrum fractions
- SHA256 hashes of inputs and key outputs

All outputs are deterministic for fixed seed and B.

### Execution

python analysis/esc_analyze.py --analyze 15Q-MAIN
python analysis/esc_analyze.py --analyze 20Q-SBP --B 50000 --seed 2026
python analysis/esc_analyze.py --all --hash

## 2. Logical Unitary Equivalence Verification

**File:** `logical_unitary_equivalence.py`

### Purpose

Certifies that baseline and conditioned circuits are logically equivalent up to global phase.

### Verification Method

For each test case:

- Remove measurement operations
- Construct the miter circuit  
  `M = U_cond† U_base`
- Convert to ZX representation (PyZX)
- Apply full ZX reduction
- Extract scalar α

Equivalence is certified if:

`M = α I`  with  `α ≠ 0`

### Script Output

- Gate counts (baseline vs conditioned)
- Reduced graph size
- Extracted scalar α
- PASS / FAIL certification

### Execution

python logical_unitary_equivalence.py --analyze 15Q-MAIN
python logical_unitary_equivalence.py --all
python logical_unitary_equivalence.py --all --hash

This script is purely symbolic and deterministic.

---

## 3. Distance-Binned Probability Mass Composition

**File:** `mass_composition.py`

### Purpose

Reproduces the probability mass flow visualization comparing baseline and conditioned executions.

### Core Quantity

For bitstring `x`:

`d_H(x) = min(wt(x), n − wt(x))`

Outcomes are grouped into GHZ-manifold distance shells and ordered deterministically by:

- Shell index
- Baseline probability (descending)
- Lexicographic bitstring tie-break

### Rendered Output

- Two stacked probability columns (Baseline vs Conditioned)
- Flow bands connecting identical outcomes
- Shell-encoded coloring
- Explicit conservation cue: `Σₓ P(x) = 1`

### Execution

python mass_composition.py --analyze 15Q-MAIN


### Output File

[TESTNAME] - Distance-Binned Probability Mass Composition.png


All layout, typography, and ordering rules are fixed to ensure reproducibility.

---

## 4. GHZ Manifold Distance Geometry

**File:** `ghz_geometry.py`

### Purpose

Reproduces the execution-dependent geometry surfaces reported in the paper.

### Core Quantity

For bitstring `x`:

`d(x) = min(wt(x), n − wt(x))`

### Procedure

The script:

- Expands counts into per-shot distance samples
- Applies deterministic resampling:
  - Partition mode (shuffle + block), or
  - Bootstrap mode (sampling with replacement)
- Estimates `P(d)` across resampled batches
- Renders paired 3D surfaces:
  - Baseline
  - Conditioned

### Determinism Controls

- Fixed RNG seed
- Fixed resampling strategy
- Fixed canvas size and DPI
- Shared color normalization across surfaces

### Execution

python ghz_geometry.py --analyze 15Q-MAIN


### Output File

[TESTNAME] - Execution-Dependent Geometry of Near-Manifold Outcome Resolution.png

---

# Reproducibility Controls

All scripts enforce determinism via:

- Strict schema validation
- Fixed qubit counts per test
- Deterministic ordering rules
- Seeded RNG (where applicable)
- Fixed rendering parameters
- No hardware execution
- No adaptive logic

Outputs are reproducible across compliant Python environments with matching dependency versions.

---

# Third-Party Dependencies

The scripts rely on:

- PyZX (logical equivalence verification)
- NumPy
- Pandas
- Matplotlib

These libraries are used under their respective licenses. No third-party source code is redistributed.

---

# Relationship to Other Directories

- `/qasm` contains the fixed circuit execution artifacts.
- `/data` contains the recorded empirical outputs.
- `/reports` contains derived analytical summaries.
- `/docs` describes the experimental and methodological framework.

This directory contains the deterministic analytical layer connecting released artifacts to published figures and certification claims.

---

# Licensing

All analysis scripts in this directory are licensed under the Apache License, Version 2.0.

See the repository root for full license terms.

