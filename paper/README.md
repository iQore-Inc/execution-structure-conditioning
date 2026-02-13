# Paper Sources

This directory contains the LaTeX source files and associated materials
for the manuscript:

"Empirical Characterization of Deterministic Execution-Structure Variants in Superconducting Quantum Processors"

The contents of this directory represent the authoritative source used
to generate the published manuscript and any corresponding archival releases
(e.g., arXiv submissions).

No experimental execution or data generation occurs within this directory.

---

## Contents

This directory may include:

- `main.tex` (primary manuscript source)
- Bibliography files (`.bib`)
- Figures used in the manuscript
- Custom style files (`.sty`, if applicable)
- arXiv submission bundle (if applicable)
- Supplementary appendices (if included)

All figures included here correspond to outputs reproducible from
the `/analysis` directory using artifacts in `/qasm` and `/data`.

---

## Relationship to Repository Structure

- `/analysis` contains deterministic scripts used to generate reported figures.
- `/qasm` contains the fixed OpenQASM circuit artifacts.
- `/data` contains experimental shot-count outputs and reference distributions.
- `/reports` contains derived analytical artifacts.
- `/docs` contains methodological and protocol documentation.

This directory provides the formal narrative and interpretive layer
describing the experimental design, analysis methods, and conclusions.

---

## Build Instructions

To compile the manuscript locally:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex


If custom style files are included, ensure they are present in this directory
or installed in your LaTeX environment.

For arXiv submission, a clean source bundle should be prepared containing:

- `main.tex`
- All referenced figures
- Bibliography files
- Any custom `.sty` files

Generated auxiliary files (`.aux`, `.log`, `.out`, etc.) should not be included.

---

## Versioning and Release Discipline

The manuscript sources correspond to a specific repository release.

When submitting to arXiv or creating a public release:

1. Tag the repository commit corresponding to the submission.
2. Ensure that figures and numerical values match those reproducible from `/analysis`.
3. Do not modify artifacts in `/qasm` or `/data` after release.

This directory should reflect a **frozen, reproducible manuscript state** tied to
the repository’s checksum manifest.

---

## Reproducibility Alignment

All quantitative claims in the manuscript are supported by:

- Released OpenQASM artifacts
- Released experimental shot-count data
- Deterministic analysis scripts
- Cryptographic integrity verification (`/checksums`)

The paper should be read in conjunction with the full repository.

---

## Licensing

The manuscript text and associated expressive materials in this directory
are licensed under the Creative Commons Attribution 4.0 International
License (CC BY 4.0), unless otherwise noted.

See the repository root for full license information.
