# Execution-Structure Conditioning

This repository accompanies the paper:

> **Empirical Characterization of Deterministic Execution-Structure Variants in Superconducting Quantum Processors**

It provides the **experimental circuits, empirical data, analytical scripts, derived artifacts, and integrity materials** used to support the results and conclusions reported in the paper.

The repository is structured to support **transparent scientific review**, **independent reproduction of results**, and **long-term archival of experimental evidence**, while maintaining clear separation between executable artifacts, raw data, derived analyses, and documentation.

---

## Repository Overview

The repository is organized into clearly scoped directories, each serving a distinct role in the experimental workflow:

  execution-structure-conditioning/
      analysis/ # Analytical scripts and experiment-specific derived artifacts
      checksums/ # SHA256 integrity verification files
      data/ # Experimental and reference measurement outputs
      docs/ # Documentation and changelog
      paper/ # Paper-related repository documentation
      qasm/ # OpenQASM circuit specifications
      reports/ # Derived analyses, figures, and report artifacts


Additional top-level files:

      CITATION.cff
      LICENSE-CODE.md
      LICENSE-DATA.md
      LICENSE-PAPER.md
      pyproject.toml
      requirements.txt
      README.md
      .gitignore


Each top-level directory listed above contains its own `README.md` describing scope, contents, and usage.

---

## Scientific Scope

This repository supports the following activities:

- Verification of reported experimental results  
- Reproduction of figures and quantitative metrics  
- Audit of experimental provenance and data lineage  
- Comparative analysis of baseline and conditioned executions  

It does **not** provide:

- General-purpose circuit synthesis tools  
- Hardware control software or backend-specific tooling  
- Production deployment infrastructure  

The repository should be read in conjunction with the accompanying paper.

---

## Data and Reproducibility

Reproduction of the reported results is supported using:

- OpenQASM circuit specifications in `/qasm`
- Recorded experimental outputs and metadata in `/data`
- Analytical scripts and experiment-specific artifacts in `/analysis`
- Derived analytical reports and figures in `/reports`
- Documentation and update history in `/docs`
- Integrity verification hashes in `/checksums`

The repository also includes:

- `pyproject.toml`
- `requirements.txt`

to define the Python environment used for analytical processing.

All transformations from raw execution outputs to reported figures are analytical and non-destructive.

---

## Licensing

This repository uses multiple licenses, scoped by content type:

- Paper text and documentation materials are licensed under  
  **Creative Commons Attribution 4.0 (CC BY 4.0)**  
  (see `LICENSE-PAPER.md`)

- Raw experimental data and reference results are released under  
  **CC0 1.0 (Public Domain Dedication)**  
  (see `LICENSE-DATA.md`)

- QASM circuit specifications and analytical code are licensed under the  
  **Apache License, Version 2.0**  
  (see `LICENSE-CODE.md`)

Refer to the corresponding license files in the repository root for full details.

---

## Patent Notice

Portions of the execution methodology described in this repository are the subject of one or more pending patent applications.

Publication of code, data, and documentation is intended to support scientific reproducibility and peer review and does **not** grant rights beyond those explicitly provided under the applicable licenses.

---

## Trademark Notice

**iQore** and **Engineered Coherence** are trademarks or pending trademarks of iQore Inc.

Use of these names does not imply endorsement, affiliation, or licensing beyond what is explicitly stated.

---

## Citation

If you use this repository in academic work, please cite the accompanying paper and follow the citation instructions provided in `CITATION.cff`.

---

## Contact and Attribution

This repository is maintained to support open scientific review of the associated research. Questions regarding methodology should be directed to the corresponding author as listed in the paper.

---

## Status

This repository represents a **frozen experimental snapshot** associated with the published work.

Updates, if any, are documented in:

    docs/changelog.md