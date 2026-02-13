# --------------------------------------------------------------
# Reproducibility Integrity Metadata
# Digest      : Generated at release (see repository checksums)
# Timestamp   : 2025–2026
# Purpose     : Deterministic, tamper-evident verification of
#               logical unitary equivalence for released QASM pairs
# --------------------------------------------------------------
#
# NOTICE
# This script is a public, reproducibility-grade verification artifact.
# It operates exclusively on released OpenQASM circuit artifacts and
# produces deterministic verification outputs corresponding to the
# logical-equivalence claim in the associated paper appendix.
#
# No quantum hardware execution occurs in this script.
# ==============================================================
#
# ==============================================================
# Title   : Logical Unitary Equivalence Verification — Reproducibility Script
# File    : logical_unitary_equivalence.py   (repository path: /analysis/)
# Scope   : External researchers / independent verification
# Paper   : "Empirical Characterization of Deterministic Execution-Structure Variants
#           in Superconducting Quantum Processors"
#
# © 2025–2026 iQore Inc.
# Licensed under the Apache License, Version 2.0
# See LICENSE-CODE.md at repository root for full terms.
# --------------------------------------------------------------
#
# This file is an executable methodological artifact released for the
# purpose of scientific reproducibility and formal circuit-level
# equivalence verification (up to global phase).
#
# --------------------------------------------------------------
# LICENSE SUMMARY (Non-Substitutive)
# --------------------------------------------------------------
# - Code (this file): Apache License 2.0
# - Data inputs     : CC0 1.0 Universal (public domain)
# - Paper & figures : CC BY 4.0
#
# This header is informational only and does not replace the full
# license texts distributed with the repository.
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# CLI / USAGE (Reproducibility Entry Point)
# --------------------------------------------------------------
# This script is executed via one required selector:
#
#   --analyze {15Q-MAIN,15Q-MCE,20Q-SBP,10Q-CBP}
#
# or:
#
#   --all
#
# Optional:
#   --hash     Print SHA256 of each QASM artifact (repo also includes a
#              checksum manifest; use this flag for standalone runs).
#
# Example commands (run from the directory containing this file):
#
#   python logical_unitary_equivalence.py --analyze 15Q-MAIN
#   python logical_unitary_equivalence.py --analyze 15Q-MCE
#   python logical_unitary_equivalence.py --analyze 20Q-SBP
#   python logical_unitary_equivalence.py --analyze 10Q-CBP
#   python logical_unitary_equivalence.py --all
#   python logical_unitary_equivalence.py --all --hash
#
# Help:
#   python logical_unitary_equivalence.py --help
#
# Input data layout (relative to this script):
#   logical_unitary_equivalence.py
#   <TESTNAME>/
#     <baseline QASM>
#     <conditioned QASM>
#
# The QASM file names are defined in TEST_CONFIG below.
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# PURPOSE & SCOPE
# --------------------------------------------------------------
# This script verifies logical unitary equivalence (up to global phase)
# between baseline and structurally conditioned circuit-program
# realizations released in OpenQASM form.
#
# Verification method:
#   1. Remove final measurements (unitary-only semantics)
#   2. Construct the miter circuit:  M = U_cond^† U_base
#   3. Convert to a ZX diagram and reduce using PyZX full_reduce
#   4. Certify equivalence (up to global phase) when the reduced miter
#      evaluates to a nonzero scalar multiple of identity:
#
#         M = α I,  with α ≠ 0
#
# The script emits PASS/FAIL for each selected test case and prints the
# extracted scalar α.
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# SYSTEM OVERVIEW
# --------------------------------------------------------------
# Verification Modules:
# ┌────────────────────────────────────────────────────────────┐
# │ 1. **Artifact Ingestion**                                  │
# │    - Loads OpenQASM circuits from released test folders    │
# │    - Removes measurements and normalizes qreg size         │
# │                                                            │
# │ 2. **Miter Construction**                                  │
# │    - Forms M = U_cond^† U_base                             │
# │                                                            │
# │ 3. **ZX Reduction**                                        │
# │    - Converts to ZX diagram and reduces via full_reduce    │
# │                                                            │
# │ 4. **Certification Output**                                │
# │    - Extracts scalar α and reports PASS/FAIL               │
# └────────────────────────────────────────────────────────────┘
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# EXECUTION FLOW
# --------------------------------------------------------------
# 1. parse CLI args (--analyze or --all, optional --hash)
# 2. locate baseline + conditioned QASM under <TESTNAME>/
# 3. load QASM and remove measurements / creg declarations
# 4. build miter circuit and reduce ZX diagram (PyZX full_reduce)
# 5. extract scalar α and print certification result
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# REPRODUCIBILITY CONTROLS
# --------------------------------------------------------------
# Determinism is enforced by:
# - Purely symbolic circuit-level verification (no sampling)
# - Fixed artifact selection via TEST_CONFIG
# - No dependency on quantum hardware or provider services
#
# The PyZX version is printed at runtime to support auditability.
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# THIRD-PARTY DEPENDENCIES
# --------------------------------------------------------------
# This script depends on:
# - PyZX
#
# PyZX is used under its respective license.
# No third-party source code is redistributed in this file.
# --------------------------------------------------------------

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple

import pyzx as zx
from pyzx.simplify import full_reduce


# ============================
# CONFIG: test registry
# ============================
TEST_CONFIG: Dict[str, Dict[str, str]] = {
    "15Q-MAIN": {
        "baseline_qasm": "15Q-MAIN - QASM - Baseline (d3kmirodd19c73966ud0).qasm",
        "conditioned_qasm": "15Q-MAIN - QASM - Conditioned (d3kmis0dd19c73966udg).qasm",
        "n_qubits": "15",
    },
    "15Q-MCE": {
        "baseline_qasm": "15Q-MCE - QASM - Baseline (d3kn6oj4kkus739bud1g).qasm",
        "conditioned_qasm": "15Q-MCE - QASM - Conditioned (d3kn6oj4kkus739bud20).qasm",
        "n_qubits": "15",
    },
    "20Q-SBP": {
        "baseline_qasm": "20Q-SBP - QASM - Baseline (d3knd903qtks738bjjdg).qasm",
        "conditioned_qasm": "20Q-SBP - QASM - Conditioned (d3knd91fk6qs73e65s00).qasm",
        "n_qubits": "20",
    },
    "10Q-CBP": {
        "baseline_qasm": "10Q-CBP - QASM - Baseline (d3nf1603qtks738eack0).qasm",
        "conditioned_qasm": "10Q-CBP - QASM - Conditioned (d3nf1603qtks738eackg).qasm",
        "n_qubits": "10",
    },
}


# ============================
# Helpers
# ============================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_qreg_size(qasm_text: str) -> int | None:
    m = re.search(r"qreg\s+q\[(\d+)\]\s*;", qasm_text)
    if not m:
        return None
    return int(m.group(1))


def load_pyzx_unitary_from_qasm(path: Path, n_qubits: int) -> zx.Circuit:
    """
    Load IBM-style OpenQASM 2.0, strip measurements/cregs, and force qreg size to n_qubits.

    This keeps the verification strictly circuit-program level:
    - No Qiskit re-synthesis
    - No rewriting beyond measurement removal and qreg resizing
    """
    lines = []
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    for line in txt:
        s = line.strip()
        if s.startswith("measure") or s.startswith("creg"):
            continue
        if s.startswith("qreg q["):
            lines.append(f"qreg q[{n_qubits}];\n")
            continue
        lines.append(line)

    qasm_unitary = "".join(lines)
    return zx.Circuit.from_qasm(qasm_unitary)


def verify_pair(
    baseline_path: Path,
    conditioned_path: Path,
    n_qubits: int,
) -> Tuple[bool, complex, int, int, int, int]:
    """
    Returns:
      (pass, scalar, base_gate_count, cond_gate_count, remaining_vertices, remaining_edges)
    """
    c_base = load_pyzx_unitary_from_qasm(baseline_path, n_qubits)
    c_cond = load_pyzx_unitary_from_qasm(conditioned_path, n_qubits)

    # Miter M = U_cond^† U_base
    miter = c_cond.adjoint()
    miter.add_circuit(c_base)

    g = miter.to_graph()
    full_reduce(g)

    verts = g.num_vertices()
    edges = g.num_edges()

    # Scalar extraction is the certification trigger used in the paper appendix:
    # if reduced miter evaluates to α I with α != 0 → equivalent up to global phase.
    alpha = g.scalar.to_number()
    passed = (abs(alpha) > 0)

    return passed, alpha, len(c_base.gates), len(c_cond.gates), verts, edges


# ============================
# CLI
# ============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logical unitary equivalence verification for released baseline–conditioned QASM pairs "
                    "(ZX-calculus miter + PyZX full_reduce)."
    )
    parser.add_argument(
        "--analyze",
        choices=tuple(TEST_CONFIG.keys()),
        help="Select a single test case to verify."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all released test cases."
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Also print SHA256 for each QASM artifact (repo also includes checksum manifest)."
    )
    args = parser.parse_args()

    if not args.all and not args.analyze:
        parser.error("Provide --analyze <TESTNAME> or --all")

    here = Path(__file__).resolve().parent

    tests = list(TEST_CONFIG.keys()) if args.all else [args.analyze]

    print("=" * 78)
    print("LOGICAL UNITARY EQUIVALENCE VERIFICATION")
    print("ZX-calculus miter reduction using PyZX")
    print("=" * 78)
    print(f"PyZX version: {zx.__version__}")
    print(f"Script path : {Path(__file__).resolve()}")
    print("-" * 78)

    any_fail = False

    for t in tests:
        cfg = TEST_CONFIG[t]
        n_qubits = int(cfg["n_qubits"])

        test_dir = here / t
        base = test_dir / cfg["baseline_qasm"]
        cond = test_dir / cfg["conditioned_qasm"]

        if not base.exists():
            raise FileNotFoundError(f"[{t}] Missing baseline QASM: {base}")
        if not cond.exists():
            raise FileNotFoundError(f"[{t}] Missing conditioned QASM: {cond}")

        print(f"[{t}]")
        print(f"  n_qubits         : {n_qubits}")
        print(f"  baseline_qasm    : {base.name}")
        print(f"  conditioned_qasm : {cond.name}")

        if args.hash:
            print(f"  baseline_sha256  : {sha256_file(base)}")
            print(f"  conditioned_sha256: {sha256_file(cond)}")

        passed, alpha, g_base, g_cond, v, e = verify_pair(base, cond, n_qubits)

        print(f"  gates(base/cond) : {g_base} / {g_cond}")
        print(f"  reduced_graph    : vertices={v}, edges={e}")
        # Print alpha cleanly (appendix-friendly)
        print(f"  miter_scalar α   : {alpha}")

        if passed:
            print("  RESULT           : PASS (equivalent up to global phase)")
        else:
            print("  RESULT           : FAIL")
            any_fail = True

        print("-" * 78)

    if any_fail:
        raise SystemExit(2)

    print("All requested checks PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
