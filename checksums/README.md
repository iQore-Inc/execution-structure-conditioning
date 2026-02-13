# Checksum Manifest

This directory contains the cryptographic checksum manifest used to verify the integrity of released artifacts in this repository.

The checksum file provides deterministic, tamper-evident verification of:

- OpenQASM circuit artifacts
- Experimental data files (CSV)
- Analysis scripts
- Report artifacts
- Documentation files (`/docs`)

The checksums are generated at release time and serve as the authoritative integrity reference for the public distribution.

No artifact generation or analysis occurs within this directory.

---

## Purpose and Scope

The contents of this directory exist solely to support:

- Integrity verification
- Independent auditability
- Reproducibility assurance
- Tamper detection

The checksum manifest:

- Covers released repository artifacts
- Uses SHA-256 cryptographic hashing
- Is fixed at release time
- Should be treated as immutable

It does not:

- Generate hashes dynamically
- Modify repository contents
- Replace version control history
- Substitute for signed releases (if applicable)

---

## File Structure

checksums/
    CHECKSUM.sha256
    README.md


### CHECKSUM.sha256

This file contains SHA-256 digests for released artifacts.

Each entry corresponds to a specific file and includes:

- The SHA-256 hash
- The relative file path

The format follows standard SHA-256 checksum conventions:

<64-hex-character SHA256 hash> <relative file path>


Example:

3b7f3a0d4e0b7e9b2c2e1a8f... qasm/test01-15Q-MAIN/baseline.qasm


---

## Verification Instructions

Verification should be performed from the repository root to ensure relative paths resolve correctly.

### macOS / Linux

sha256sum -c checksums/CHECKSUM.sha256


If the command completes without errors, all files match their recorded digests.

### Windows (PowerShell)

Get-FileHash -Algorithm SHA256 <filename>


Manually compare the resulting hash to the value listed in `CHECKSUM.sha256`.

---

## Integrity Model

Integrity guarantees are based on:

- SHA-256 cryptographic hashing
- Fixed release-time manifest generation
- Public artifact transparency

If any file is modified—even by a single byte—its SHA-256 hash will no longer match the manifest.

A mismatch indicates one of the following:

- File corruption
- Accidental modification
- Incomplete download
- Unauthorized alteration

Any mismatch should be treated as a verification failure.

---

## Relationship to Other Directories

- `/qasm` contains released circuit artifacts covered by this manifest.
- `/data` contains experimental shot-count data covered by this manifest.
- `/analysis` contains reproducibility scripts covered by this manifest.
- `/reports` contains derived analytical outputs covered by this manifest.
- `/docs` contains methodological and protocol documentation covered by this manifest.

The checksum manifest binds these components into a single verifiable release unit.

---

## Reproducibility Context

The checksum file enables independent researchers to:

- Confirm artifact integrity prior to analysis
- Validate that results were produced from unmodified inputs
- Ensure alignment with published experimental claims
- Verify that documentation matches the released experimental artifacts

It serves as the cryptographic anchor for the repository’s reproducibility guarantees.

---

## Licensing

The checksum manifest is distributed under the same licensing terms as the repository.

See the repository root for full license information.
