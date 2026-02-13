# Reports

This directory contains **analysis artifacts and interpretive materials** derived from experimental
and simulated data stored in `/data`.  
All contents in this directory are **post-processed outputs** and **do not include raw experimental
results**.

The purpose of this directory is to provide transparent, reviewable, and reproducible analytical
outputs that support the scientific claims, figures, and conclusions presented in the associated
paper and documentation.

---

## Intended Use

Materials in this directory are intended for:

- Independent result validation
- Peer review and technical auditing
- Executive and stakeholder interpretation
- Figure reproduction and verification
- Long-term archival of analytical context

This directory explicitly separates **measurement** from **interpretation** to preserve scientific
integrity and traceability.

---

## Directory Structure

Each subdirectory corresponds to a single experimental test case and mirrors the naming conventions
used in `/qasm` (methodological code) and `/data` (experimental outputs).

reports/
README.md
    testXX-<QUBITS>-<LABEL>/
        analysis_report.pdf
        figures/
            *.png
        notes.md


### Example

reports/
    test01-15Q-MAIN/
        analysis_report.pdf
        figures/
            *.png
        notes.md


Consistency across `/qasm`, `/data`, and `/reports` is intentional and required for reproducibility.

---

## Contents per Test Case

### `analysis_report.pdf`

A formal, static analysis document summarizing the results of the corresponding test case.

Typically includes:
- Experimental objective and test description
- Comparison between baseline, conditioned, and ideal executions
- Statistical summaries and performance metrics
- Error characterization and uncertainty discussion
- Interpretation of observed effects
- Conclusions relevant to the associated paper

These reports are considered **derived scholarly artifacts**, not primary data.

---

### `figures/`

Contains all figures referenced in the analysis report.

Figures may include:
- Measurement outcome histograms
- Fidelity, divergence, or distance metrics
- Comparative plots across execution conditions
- Supporting visualizations for interpretive claims

All figures are reproducible from the data in `/data` using documented analysis procedures.

---

### `notes.md`

Optional supplementary notes intended for technical traceability.

May include:
- Analyst observations
- Assumptions or simplifications
- Deviations from planned protocol
- Known limitations or anomalies
- Context unsuitable for inclusion in formal reports

This file is recommended for internal transparency but is not required for publication.

---

## Data Lineage and Provenance

All materials in this directory are derived exclusively from:

- `/data/<test-case>/` — recorded experimental and simulated outputs
- `/qasm/<test-case>/` — circuit specifications and methodological inputs

No raw experimental data is modified, duplicated, or overwritten during report generation.
All transformations are analytical, non-destructive, and reproducible.

---

## Reproducibility

Independent reproduction of the results in this directory is possible using:

- The corresponding QASM files in `/qasm`
- The recorded outputs and metadata in `/data`
- The experimental and analytical methodology described in `/docs/experimental_protocol.md`

This directory exists to support transparent scientific verification.

---

## Versioning and Stability

Reports are static artifacts tied to specific data snapshots.
Updates to reports reflect:
- Corrections
- Methodological clarifications
- Improved analysis procedures

Such changes should be documented in `/docs/changelog.md`.

---

## Licensing

Analytical reports, figures, and interpretive materials are governed by:

- `LICENSE-PAPER.md`

Raw experimental data remains governed by:

- `LICENSE-DATA.md`

---

## Scope and Intent

This directory intentionally contains **interpretation**, not **measurement**.
It exists to support peer review, long-term archival, and scientific accountability,
and should be read in conjunction with the primary paper and data directories.

