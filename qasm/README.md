# QASM Circuit Artifacts

This directory contains **OpenQASM circuit specifications** used in the
experiments reported in the accompanying paper.

The QASM files in this directory represent **fixed execution artifacts**
that were submitted to specific quantum hardware execution backends and form
the methodological basis for the reported results.

They are provided to support **transparency, reproducibility, and
independent verification** of experimental outcomes.

---

## Scope and Intent

The contents of this directory are intentionally limited in scope.

These QASM files:
- Represent concrete, fully specified circuits as executed
- Correspond to named experimental test cases
- Are not parameterized or dynamically generated
- Should be treated as immutable execution artifacts

They do **not** encode:
- General circuit construction logic
- Automated synthesis or optimization procedures
- Circuit selection or adaptation rules
- Execution-structure conditioning algorithms

High-level descriptions of the conditioning methodology are provided in
the accompanying paper. Detailed construction logic is outside the scope
of this repository and is the subject of pending patent applications.

---

## Directory Structure

Each subdirectory corresponds to a single experimental test case and
mirrors the naming conventions used in `/data` and `/reports`.

qasm/
    README.md
    testXX-<QUBITS>-<LABEL>/
        baseline.qasm
        conditioned.qasm

### Example

qasm/
    test01-15Q-MAIN/
        baseline.qasm
        conditioned.qasm


Consistency across `/qasm`, `/data`, and `/reports` is intentional and
required for traceability.

---

## Circuit Variants

For each test case, two circuit variants are provided:

### `baseline.qasm`

The unmodified reference circuit.

This circuit reflects the nominal algorithmic structure prior to any
execution-structure conditioning and serves as the control condition
for comparative analysis.

---

### `conditioned.qasm`

A structurally augmented circuit used in execution-structure conditioning
experiments.

Conditioned circuits preserve the logical intent of the baseline while
introducing structural modifications designed to influence execution
behavior under real hardware constraints.

---

## Execution Semantics

All QASM files:
- Are valid OpenQASM programs
- Were executed without runtime modification
- Are tied to specific qubit counts and register layouts
- Should be interpreted exactly as written

No transpilation, gate synthesis, or routing steps are encoded here.
Such steps were performed by the target execution environment and are
documented via metadata in `/data`.

---

## Reproducibility Notes

These QASM files are sufficient to:
- Re-run the exact circuits used in the experiments
- Compare baseline versus conditioned behavior
- Validate reported figures and metrics

They are not sufficient to regenerate the full experimental pipeline
without the accompanying data, metadata, and methodological context
provided elsewhere in the repository.

---

## Relationship to Other Directories

- `/data` contains the recorded outputs and metadata resulting from
  executing these circuits.
- `/reports` contains derived analytical artifacts interpreting those
  outputs.
- `/docs` describes experimental protocols and analytical methodology.

This directory represents the **methodological anchor point** linking
theoretical intent to measured outcomes.

---

## Licensing

All QASM files in this directory are licensed under the **Apache License,
Version 2.0**.

See the repository root for full license terms.
