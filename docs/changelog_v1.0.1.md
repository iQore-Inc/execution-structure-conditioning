# Changelog

All notable changes to this repository are documented in this file.

This repository represents a frozen experimental snapshot associated with:

> Empirical Characterization of Deterministic Execution-Structure Variants in Superconducting Quantum Processors

Versioning is archival and audit-oriented. Changes are limited to structural corrections,
documentation alignment, and integrity verification artifacts. No experimental data
is modified after publication.

---
## [v1.0.1] — 2026-02-22

### Documentation & Analysis Integrity Alignment Update

- This update preserves all experimental artifacts and extends the deterministic analysis layer with an additional reproducibility-grade script.
- No experimental data, QASM artifacts, or published reports were modified.

### Added

- New deterministic statistical analysis script:
  - analysis/esc_analyze.py

- This script reproduces the full statistical verification layer, including:

  - GHZ-manifold shell metrics
  - TVD / JSD divergence calculations
  - Multinomial bootstrap uncertainty quantification
  - PCA shell-space diagnostics
  - ESC (Execution-Structure Conditioning) derived metrics
  - Deterministic QASM burden summaries

- The script operates exclusively on released shot-order CSV and QASM artifacts. (No hardware execution or provider API calls occur.)

### Updated

- analysis/README.md
  - Expanded to include:
      - esc_analyze.py
      - Deterministic statistical verification description
      - Bootstrap protocol documentation
      - ESC metric definitions
      - Output file descriptions
    - checksums/CHECKSUMS.sha256
      - Added SHA-256 entry for:
        - analysis/esc_analyze.py
      - Added updated SHA-256 entry for:
        - analysis/README.md
      - Regenerated manifest to maintain repository-wide integrity coverage.

### Integrity Statement

  - No experimental CSV files were modified.
  - No QASM files were modified.
  - No reports were modified.
  - No previously published integrity hashes were altered.
  - Only new artifacts were added and documented.

- All prior hashes remain valid.
- The repository remains a frozen experimental snapshot with extended deterministic analysis tooling.

## [v1.0.0] — 2026-02-12

### Initial Archival Snapshot

- Established repository as the canonical archival record of experimental artifacts.
- Added full SHA-256 integrity manifest (`checksums/CHECKSUMS.sha256`) covering:
  - Raw experimental outputs (baseline, conditioned, ideal)
  - Metadata files
  - QASM definitions
  - Analysis scripts
  - Reports
  - Documentation files

### Structural Alignment

- Updated top-level `README.md` to reflect actual repository structure:
  - Added `analysis/`
  - Added `checksums/`
  - Removed non-existent `.github/`
  - Documented environment files (`pyproject.toml`, `requirements.txt`)

### Filename Correction

- Corrected:
  - `20Q-SBP - Shot Count - Ideal (lAER).csv`
- Renamed to:
  - `20Q-SBP - Shot Count - Ideal (AER).csv`
- No data content changes; filename normalization only.

### Documentation

- Added `docs/changelog.md` to track structural and integrity updates.
- Confirmed repository status as frozen experimental snapshot.

---

## Change Policy

After v1.0.0:

- Experimental data will not be modified.
- QASM artifacts will not be modified.
- Reports will not be altered.
- Any future updates must:
  - Preserve prior integrity hashes
  - Be documented here
  - Increment version number
  - Regenerate full checksum manifest

