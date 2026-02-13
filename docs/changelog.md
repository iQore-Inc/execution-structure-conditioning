# Changelog

All notable changes to this repository are documented in this file.

This repository represents a frozen experimental snapshot associated with:

> Empirical Characterization of Deterministic Execution-Structure Variants in Superconducting Quantum Processors

Versioning is archival and audit-oriented. Changes are limited to structural corrections,
documentation alignment, and integrity verification artifacts. No experimental data
is modified after publication.

---

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

