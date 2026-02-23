# --------------------------------------------------------------
# Reproducibility Integrity Metadata
# Digest      : Generated at release (see repository checksums / manifests)
# Timestamp   : 2025–2026
# Purpose     : Deterministic, tamper-evident reproduction of the
#               *statistical* analysis outputs reported for released
#               baseline vs conditioned GHZ-manifold experiments.
# --------------------------------------------------------------
#
# NOTICE
# This script is a public, reproducibility-grade analysis artifact.
# It operates exclusively on released, static input artifacts:
#   (1) Shot-order CSV logs (bitstring outcomes, one row per shot)
#   (2) OpenQASM circuit files (baseline + conditioned)
#
# It produces deterministic analysis outputs (CSV/JSON/NPZ) corresponding
# to the appendix / supplemental analysis tables and metrics.
#
# No quantum hardware execution occurs in this script.
# No provider APIs are contacted. No sampling from devices is performed.
# ==============================================================
#
# ==============================================================
# Title   : GHZ Manifold Conditioning — Deterministic Statistical Verification Script
# File    : esc_analyze.py
# Repo    : iQore-Inc/exection-structure-conditioning
# Path    : /analysis/esc_analyze.py
# Scope   : External researchers / independent reproduction
# Paper   : "Empirical Characterization of Deterministic Execution-Structure Variants
#           in Superconducting Quantum Processors"
#
# © 2025–2026 iQore Inc.
# Licensed under the Apache License, Version 2.0
# See LICENSE-CODE.md at repository root for full terms.
# --------------------------------------------------------------
#
# This file is an executable methodological artifact released for the
# purpose of scientific reproducibility. It reproduces the paper’s
# *reported statistics* from the released experimental artifacts and
# emits audit-friendly, content-addressable hashes for all inputs and
# key outputs.
#
# --------------------------------------------------------------
# LICENSE SUMMARY (Non-Substitutive)
# --------------------------------------------------------------
# - Code (this file) : Apache License 2.0
# - Data inputs      : CC0 1.0 Universal (public domain)  [as distributed in repo]
# - Paper & figures  : CC BY 4.0                          [as distributed in repo]
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
# Optional controls:
#   --seed <int>     Base deterministic seed (default: 2026). Per-test
#                    effective seeds are derived via CRC32-tag mixing
#                    (stable_seed) and split into independent baseline
#                    vs conditioned bootstrap RNG streams.
#   --B <int>        Number of bootstrap replicates (default: 10,000).
#   --outdir <path>  Output directory root (default: RESULTS/).
#   --hash           Print SHA256 of each input artifact to stdout.
#
# Example commands (run from the repository root):
#
#   python analysis/esc_analyze.py --analyze 15Q-MAIN
#   python analysis/esc_analyze.py --analyze 20Q-SBP --B 50000 --seed 2026
#   python analysis/esc_analyze.py --all --hash
#
# Help:
#   python analysis/esc_analyze.py --help
#
# Input data layout (relative to analysis/esc_analyze.py):
#   analysis/esc_analyze.py
#   analysis/15Q-MAIN/
#     <baseline_shot_order_file>.csv
#     <conditioned_shot_order_file>.csv
#     <baseline_qasm>.qasm
#     <conditioned_qasm>.qasm
#   analysis/15Q-MCE/
#     ...
#   analysis/20Q-SBP/
#     ...
#   analysis/10Q-CBP/
#     ...
#
# Exact filenames are defined in TEST_CONFIG (constant, release-locked).
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# PURPOSE & SCOPE
# --------------------------------------------------------------
# This script compares *baseline* vs *conditioned* realizations for each
# released test case using only the released artifacts:
#
#   - Shot-order bitstring outcomes (CSV; one bitstring per shot)
#   - OpenQASM circuits (QASM; used only for burden metrics)
#
# It verifies and reproduces the following *analysis outputs*:
#
#   (A) GHZ-manifold mass and leakage structure:
#       - P_GHZ = p(0^n) + p(1^n) from empirical outcome probabilities
#       - Shell distribution v[d] where d(x) = min(wt(x), n - wt(x))
#       - Near-manifold probability P(d <= k) and conditional P_GHZ | d<=k
#       - Leakage masses (e.g., v[d=1], v[d=2]) and rank concentration P_top-k
#
#   (B) Distribution divergence (paired baseline vs conditioned):
#       - Total variation distance TVD(P,Q) computed over union support
#       - Jensen–Shannon divergence JSD(P,Q) in bits over union support
#
#   (C) Bootstrap uncertainty with protocol-correct percentile intervals:
#       - Multinomial bootstrap on empirical distributions, preserving total N
#       - Independent baseline and conditioned resamples per replicate
#       - Percentile 95% CIs for Δ metrics and bias-corrected divergence
#       - Bootstrap standard errors (empirical SD with ddof=1)
#
#   (D) Shell-space geometry diagnostics:
#       - Pooled PCA on shell vectors (center-only; no whitening)
#       - PC1/PC2 variance explained and deterministic sign convention
#       - PC1 separation summary (means, SDs, SNR-style separation)
#       - Low-rank structure of Δv via covariance eigenspectrum fractions
#
#   (E) ESC add-on metrics (derived only; does not alter any prior sums):
#       - UO-A set-membership objective with S = {0^n, 1^n}
#       - Event cost C_event = 1/p for baseline and conditioned
#       - Ratio R_event = pB/pA and fractional savings Δ = 1 - (pA/pB)
#       - Waste W10k = 10,000*(1-p) and W10k savings
#       - CIs computed directly from bootstrap draws of the metric
#
# IMPORTANT: This script does NOT prove circuit-level unitary equivalence.
# QASM files are parsed only to compute deterministic burden summaries
# (total gates, 2-qubit gate count heuristic, and greedy ASAP depth).
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# SYSTEM OVERVIEW
# --------------------------------------------------------------
# Modules / Stages:
# ┌────────────────────────────────────────────────────────────┐
# │ 1. **Artifact Ingestion**                                  │
# │    - Load shot-order CSV and extract bitstrings            │
# │    - Clean quotes; filter to [01]+; left-pad to n_qubits   │
# │    - Build sparse empirical distribution (support, probs)  │
# │                                                            │
# │ 2. **Deterministic Metrics (Point Estimates)**             │
# │    - P_GHZ, shell vector v[d], entropies, top-k mass       │
# │    - TVD/JSD over union support with sorted accumulation   │
# │                                                            │
# │ 3. **Deterministic QASM Burden Summary**                   │
# │    - Parse OpenQASM text; ignore measure/barrier/directive │
# │    - Count total gates, 2q gates (name or >=2 touched)     │
# │    - Greedy ASAP depth estimate on serialized gate list    │
# │                                                            │
# │ 4. **Bootstrap Protocol (Multinomial)**                    │
# │    - c* ~ Multinomial(N, p_hat) per mode                   │
# │    - Independent baseline vs conditioned RNG streams       │
# │    - Percentile 95% CIs + bootstrap SE                     │
# │    - Bias-correct TVD/JSD by subtracting bootstrap mean    │
# │                                                            │
# │ 5. **PCA + Low-Rank Diagnostics**                          │
# │    - Pooled PCA (SVD) on shell vectors                     │
# │    - Sign-fixed loadings (max-abs entry positive)          │
# │    - Δv covariance eigenspectrum fractions                 │
# │                                                            │
# │ 6. **Outputs + Auditability**                              │
# │    - RESULTS/<TEST>/esc_full_analysis.csv                  │
# │    - RESULTS/<TEST>/esc_full_analysis.json                 │
# │    - RESULTS/<TEST>/esc_bootstrap_draws.npz                │
# │    - SHA256 hashes for inputs and key outputs (JSON audit) │
# └────────────────────────────────────────────────────────────┘
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# REPRODUCIBILITY CONTROLS
# --------------------------------------------------------------
# Determinism is enforced by:
# - Fixed artifact selection via TEST_CONFIG (release-locked filenames)
# - Stable seed derivation via CRC32(tag) mixing (no Python hash())
# - numpy.default_rng with SeedSequence for bootstrap streams
# - Deterministic support ordering:
#     • empirical support sorted lexicographically
#     • union-support iteration via sorted(set(...))
# - Percentile CIs computed via np.quantile(..., method="linear")
# - PCA sign convention fixed by max-absolute loading per component
#
# Note on numeric determinism:
# - Linear algebra and reductions may vary at the last bit across BLAS
#   implementations and thread scheduling. For maximal repeatability,
#   set BLAS thread env vars to 1 (OMP/OPENBLAS/MKL/VECLIB/NUMEXPR).
# - All reported statistics are stable at publication precision under
#   typical deterministic compute environments.
# --------------------------------------------------------------
#
# --------------------------------------------------------------
# THIRD-PARTY DEPENDENCIES
# --------------------------------------------------------------
# Required:
# - NumPy
# - pandas
#
# No third-party source code is redistributed in this file.
# Dependencies are used under their respective licenses.
# --------------------------------------------------------------

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

TEST_CONFIG: dict[str, dict[str, object]] = {
    "15Q-MAIN": {
        "shots_expected": 10_000,
        "n_qubits": 15,

        # Shot Order CSVs
        "baseline_shot_order_file": "15Q-MAIN - Shot Order - Baseline (d3kmirodd19c73966ud0).csv",
        "conditioned_shot_order_file": "15Q-MAIN - Shot Order - Conditioned (d3kmis0dd19c73966udg).csv",

        # Shot Count CSVs
        "baseline_shot_count_file": "15Q-MAIN - Shot Count - Baseline (d3kmirodd19c73966ud0).csv",
        "conditioned_shot_count_file": "15Q-MAIN - Shot Count - Conditioned (d3kmis0dd19c73966udg).csv",

        # QASM
        "baseline_qasm": "15Q-MAIN - QASM - Baseline (d3kmirodd19c73966ud0).qasm",
        "conditioned_qasm": "15Q-MAIN - QASM - Conditioned (d3kmis0dd19c73966udg).qasm",

        # protocol: k=2 in Table I / Appendix B
        "near_k": 2,

        # rank concentration in Table III
        "topk_list": [4, 8, 16],
    },

    "15Q-MCE": {
        "shots_expected": 5_000,
        "n_qubits": 15,

        # Shot Order CSVs
        "baseline_shot_order_file": "15Q-MCE - Shot Order - Baseline (d3kn6oj4kkus739bud1g).csv",
        "conditioned_shot_order_file": "15Q-MCE - Shot Order - Conditioned (d3kn6oj4kkus739bud20).csv",

        # Shot Count CSVs
        "baseline_shot_count_file": "15Q-MCE - Shot Count - Baseline (d3kn6oj4kkus739bud1g).csv",
        "conditioned_shot_count_file": "15Q-MCE - Shot Count - Conditioned (d3kn6oj4kkus739bud20).csv",

        # QASM
        "baseline_qasm": "15Q-MCE - QASM - Baseline (d3kn6oj4kkus739bud1g).qasm",
        "conditioned_qasm": "15Q-MCE - QASM - Conditioned (d3kn6oj4kkus739bud20).qasm",

        "near_k": 2,
        "topk_list": [4, 8, 16],
    },

    "20Q-SBP": {
        "shots_expected": 5_000,
        "n_qubits": 20,

        # Shot Order CSVs
        "baseline_shot_order_file": "20Q-SBP - Shot Order - Baseline (d3knd903qtks738bjjdg).csv",
        "conditioned_shot_order_file": "20Q-SBP - Shot Order - Conditioned (d3knd91fk6qs73e65s00).csv",

        # Shot Count CSVs
        "baseline_shot_count_file": "20Q-SBP - Shot Count - Baseline (d3knd903qtks738bjjdg).csv",
        "conditioned_shot_count_file": "20Q-SBP - Shot Count - Conditioned (d3knd91fk6qs73e65s00).csv",

        # QASM
        "baseline_qasm": "20Q-SBP - QASM - Baseline (d3knd903qtks738bjjdg).qasm",
        "conditioned_qasm": "20Q-SBP - QASM - Conditioned (d3knd91fk6qs73e65s00).qasm",

        "near_k": 2,
        "topk_list": [4, 8, 16],
    },

    "10Q-CBP": {
        "shots_expected": 2_000,
        "n_qubits": 10,

        # Shot Order CSVs
        "baseline_shot_order_file": "10Q-CBP - Shot Order - Baseline (d3nf1603qtks738eack0).csv",
        "conditioned_shot_order_file": "10Q-CBP - Shot Order - Conditioned (d3nf1603qtks738eackg).csv",

        # Shot Count CSVs
        "baseline_shot_count_file": "10Q-CBP - Shot Count - Baseline (d3nf1603qtks738eack0).csv",
        "conditioned_shot_count_file": "10Q-CBP - Shot Count - Conditioned (d3nf1603qtks738eackg).csv",

        # QASM
        "baseline_qasm": "10Q-CBP - QASM - Baseline (d3nf1603qtks738eack0).qasm",
        "conditioned_qasm": "10Q-CBP - QASM - Conditioned (d3nf1603qtks738eackg).qasm",

        "near_k": 2,
        "topk_list": [4, 8, 16],
    },
}


# ============================================================
# DETERMINISM HELPERS
# ============================================================
def stable_seed(base_seed: int, tag: str) -> int:
    """Stable 32-bit seed derived from base_seed and tag (no Python hash())."""
    tag_crc = zlib.crc32(tag.encode("utf-8")) & 0xFFFFFFFF
    return (int(base_seed) ^ tag_crc) & 0xFFFFFFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ci95_percentile(x: np.ndarray) -> Tuple[float, float]:
    # Protocol: percentile bootstrap interval (no normal approximation).
    lo = float(np.quantile(x, 0.025, method="linear"))
    hi = float(np.quantile(x, 0.975, method="linear"))
    return lo, hi


def se_sample(x: np.ndarray) -> float:
    # Protocol: empirical SD of bootstrap samples (ddof=1).
    return float(np.std(x, ddof=1))

# ============================================================
# BITSTRING UTILS
# ============================================================
def clean_bitstring(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
    elif s.startswith("'"):
        s = s[1:]
    elif s.endswith("'"):
        s = s[:-1]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    elif s.startswith('"'):
        s = s[1:]
    elif s.endswith('"'):
        s = s[:-1]
    return s.strip()


def wt(x: str) -> int:
    return x.count("1")


def dist_to_ghz(x: str) -> int:
    # Protocol: d(x) = min(wt(x), n-wt(x))
    w = wt(x)
    n = len(x)
    return min(w, n - w)

# ============================================================
# LOAD SHOT ORDER
# ============================================================
def load_shot_order(csv_path: Path, n_qubits: int) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path.name}: CSV is empty.")

    col = None
    for candidate in ("bitstring", "Bitstring", "Outcome", "outcome", "Result", "result"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(
            f"{csv_path.name}: expected a bitstring column named one of "
            f"bitstring/Bitstring/Outcome/Result. Got {df.columns.tolist()}"
        )

    shots = df[col].map(clean_bitstring)
    mask = shots.str.fullmatch(r"[01]+", na=False)
    shots = shots[mask]
    if shots.empty:
        raise ValueError(f"{csv_path.name}: no valid bitstrings after cleaning/filtering.")

    # Left-pad missing leading zeros (required to interpret as n-bit strings)
    shots = shots.map(lambda s: s.zfill(n_qubits) if len(s) < n_qubits else s)

    lengths = shots.str.len()
    if not (lengths == n_qubits).all():
        bad = shots[lengths != n_qubits].iloc[0]
        raise ValueError(f"{csv_path.name}: bad bitstring after padding: {bad}")

    arr = shots.to_numpy(dtype=str)
    print(f"[load] {str(csv_path.resolve())}: shots={arr.size} (column '{col}', padded-to-{n_qubits})")
    return arr

# ============================================================
# SPARSE DIST
# ============================================================
@dataclass(frozen=True)
class SparseDist:
    support: np.ndarray  # (K,) sorted lexicographically
    probs: np.ndarray    # (K,) float64
    counts: np.ndarray   # (K,) int64
    N: int
    n_qubits: int

def build_sparse_dist(shots: np.ndarray, n_qubits: int) -> SparseDist:
    vc = pd.Series(shots).value_counts()
    support = np.array(sorted(vc.index.tolist()), dtype=str)  # lock deterministic order
    counts = np.array([int(vc[x]) for x in support], dtype=np.int64)
    N = int(counts.sum())
    probs = counts.astype(np.float64) / float(N)
    return SparseDist(support=support, probs=probs, counts=counts, N=N, n_qubits=n_qubits)

def p_bitstring(dist: SparseDist, bitstring: str) -> float:
    idx = {s: i for i, s in enumerate(dist.support)}
    return float(dist.probs[idx[bitstring]]) if bitstring in idx else 0.0

def pghz(dist: SparseDist) -> float:
    """
    Protocol: P_GHZ = p(0^n) + p(1^n)
    Computed directly from empirical probabilities; absent outcomes contribute 0.
    """
    z = "0" * dist.n_qubits
    o = "1" * dist.n_qubits
    return p_bitstring(dist, z) + p_bitstring(dist, o)

def shell_vec(dist: SparseDist) -> np.ndarray:
    # Protocol: v = [P(d=0),...,P(d=floor(n/2))]
    n_shells = dist.n_qubits // 2 + 1
    v = np.zeros(n_shells, dtype=np.float64)
    for s, p in zip(dist.support, dist.probs):
        v[dist_to_ghz(str(s))] += float(p)
    return v

def p_d_leq_k(v_shell: np.ndarray, k: int) -> float:
    return float(np.sum(v_shell[: k + 1]))

def conditional_pghz_given_near(v_shell: np.ndarray, k: int) -> float:
    # Protocol: P_GHZ|d<=k = P(d=0)/P(d<=k). Here P(d=0)=P_GHZ.
    denom = p_d_leq_k(v_shell, k)
    return float(v_shell[0] / denom) if denom > 0 else 0.0

def shannon_entropy_bits_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def shannon_entropy_bits_dist(dist: SparseDist) -> float:
    return shannon_entropy_bits_probs(dist.probs)

def conditional_entropy_near_far(dist: SparseDist, k: int) -> tuple[float, float]:
    # Hnear: entropy of P(x | d<=k); Hfar: entropy of P(x | d>k)
    near_mask = np.array([dist_to_ghz(str(s)) <= k for s in dist.support], dtype=bool)
    far_mask = ~near_mask

    p_near = dist.probs[near_mask]
    p_far = dist.probs[far_mask]

    m_near = float(np.sum(p_near))
    m_far = float(np.sum(p_far))

    Hnear = shannon_entropy_bits_probs(p_near / m_near) if m_near > 0 else 0.0
    Hfar = shannon_entropy_bits_probs(p_far / m_far) if m_far > 0 else 0.0
    return Hnear, Hfar

def coarse_shell_entropy_near(v_shell: np.ndarray, k: int) -> float:
    # H_near^(d): entropy over distance-binned P(d) restricted to d<=k, renormalized
    v = np.asarray(v_shell[: k + 1], dtype=np.float64)
    m = float(np.sum(v))
    return shannon_entropy_bits_probs(v / m) if m > 0 else 0.0

def ptop_k(dist: SparseDist, k: int) -> float:
    # Protocol: sum of top-k probabilities; for GHZ, Ptop-2 = PGHZ.
    p_sorted = np.sort(dist.probs)[::-1]
    k = int(min(k, p_sorted.size))
    return float(np.sum(p_sorted[:k]))

def tvd_union(sup_p, p, sup_q, q) -> float:
    # Protocol: TVD(P,Q) = 0.5 * sum_x |P(x)-Q(x)| over union support
    dp = {s: float(v) for s, v in zip(sup_p, p)}
    dq = {s: float(v) for s, v in zip(sup_q, q)}
    keys = sorted(set(dp) | set(dq))  # lock deterministic accumulation order
    return 0.5 * sum(abs(dp.get(k, 0.0) - dq.get(k, 0.0)) for k in keys)

def jsd_union_bits(sup_p, p, sup_q, q) -> float:
    # Protocol: JSD in bits (log base 2), KL terms with P(x)=0 omitted naturally
    dp = {s: float(v) for s, v in zip(sup_p, p)}
    dq = {s: float(v) for s, v in zip(sup_q, q)}
    keys = sorted(set(dp) | set(dq))  # lock deterministic accumulation order
    jsd = 0.0
    for k in keys:
        pk = dp.get(k, 0.0)
        qk = dq.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if pk > 0.0:
            jsd += 0.5 * pk * (np.log2(pk) - np.log2(mk))
        if qk > 0.0:
            jsd += 0.5 * qk * (np.log2(qk) - np.log2(mk))
    return float(jsd)

# ============================================================
# QASM PARSING (EXECUTION BURDEN)
# ============================================================
_TWO_Q_GATES = {
    "cx", "cz", "swap", "iswap", "ecr", "rzz", "rxx", "ryy", "crx", "cry", "crz", "cu1", "cu3", "cp"
}

def _strip_qasm_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()

def parse_qasm_burden(qasm_path: Path) -> dict:
    """
    Publication-grade, deterministic QASM burden summary.

    - "Total gates": counts non-measurement, non-barrier, non-directive operations.
    - "Two-qubit gates": subset of total gates acting on >=2 qubits (name heuristic + arg count).
    - "Circuit depth": greedy layer depth on qubit-occupancy (excluding measure/barrier).
      This is the standard "as-soon-as-possible" depth estimate from the serialized gate list.
    """
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM not found: {qasm_path}")

    lines = qasm_path.read_text(encoding="utf-8", errors="replace").splitlines()

    qregs: dict[str, int] = {}
    qreg_offsets: dict[str, int] = {}
    total_qubits = 0

    def ensure_offsets():
        nonlocal total_qubits, qreg_offsets
        if qreg_offsets:
            return
        off = 0
        for name, size in qregs.items():  # deterministic insertion order
            qreg_offsets[name] = off
            off += size
        total_qubits = off

    def qubit_id(token: str) -> int:
        name, rest = token.split("[", 1)
        idx = int(rest.split("]", 1)[0])
        ensure_offsets()
        return qreg_offsets[name] + idx

    depth = None
    total_gates = 0
    twoq_gates = 0

    for raw in lines:
        line = _strip_qasm_comment(raw)
        if not line:
            continue

        low = line.lower()
        if low.startswith("openqasm") or low.startswith("include") or low.startswith("creg"):
            continue

        if low.startswith("qreg"):
            inside = line.split("qreg", 1)[1].strip().rstrip(";").strip()
            name = inside.split("[", 1)[0].strip()
            size = int(inside.split("[", 1)[1].split("]", 1)[0])
            qregs[name] = size
            continue

        if low.startswith("barrier") or low.startswith("measure"):
            continue

        stmt = line.rstrip(";").strip()
        if not stmt:
            continue

        name_part = stmt.split(None, 1)[0]
        gate_name = name_part.split("(", 1)[0].strip().lower()

        rest = stmt[len(name_part):].strip()
        if not rest:
            continue

        arg_tokens = [t.strip() for t in rest.split(",")]
        q_tokens = [t for t in arg_tokens if "[" in t and "]" in t]
        if not q_tokens:
            continue

        if depth is None:
            ensure_offsets()
            depth = np.zeros(total_qubits, dtype=np.int64)

        total_gates += 1
        touches = [qubit_id(t) for t in q_tokens]
        if (gate_name in _TWO_Q_GATES) or (len(set(touches)) >= 2):
            twoq_gates += 1

        layer = int(np.max(depth[touches])) if touches else 0
        new_layer = layer + 1
        for qid in touches:
            depth[qid] = new_layer

    if depth is None:
        ensure_offsets()
        depth = np.zeros(total_qubits, dtype=np.int64)

    return {
        "path": str(qasm_path.resolve()),
        "total_qubits": int(total_qubits),
        "circuit_depth": int(np.max(depth)) if depth.size else 0,
        "total_gates": int(total_gates),
        "two_qubit_gates": int(twoq_gates),
    }

# ============================================================
# BOOTSTRAP
# ============================================================
@dataclass
class BootOut:
    delta_pghz: np.ndarray
    delta_leak1: np.ndarray
    delta_leak2: np.ndarray
    tvd: np.ndarray
    jsd: np.ndarray
    v_base: np.ndarray
    v_cond: np.ndarray
    pghz_base: np.ndarray   
    pghz_cond: np.ndarray   


def multinomial_bootstrap(dist: SparseDist, B: int, rng: np.random.Generator) -> np.ndarray:
    # Protocol: c^(b) ~ Multinomial(N, p_hat), preserving total N and discrete support.
    return rng.multinomial(dist.N, dist.probs, size=B).astype(np.int64)

def bias_correct(boot_vals: np.ndarray, theta_hat: float) -> np.ndarray:
    # Protocol: theta_adj^(b) = theta^(b) - (mean(theta^*) - theta_hat)
    return boot_vals - (float(np.mean(boot_vals)) - float(theta_hat))

def compute_bootstrap_all(
    base: SparseDist,
    cond: SparseDist,
    B: int,
    rngB: np.random.Generator,
    rngC: np.random.Generator,
) -> BootOut:
    # Protocol: baseline and conditioned modes resampled independently each iteration.
    cB = multinomial_bootstrap(base, B, rngB)
    cC = multinomial_bootstrap(cond, B, rngC)

    pB = cB.astype(np.float64) / float(base.N)
    pC = cC.astype(np.float64) / float(cond.N)

    z = "0" * base.n_qubits
    o = "1" * base.n_qubits
    idxB = {s: i for i, s in enumerate(base.support)}
    idxC = {s: i for i, s in enumerate(cond.support)}

    def pghz_row(idx: Dict[str, int], probs_row: np.ndarray) -> float:
        pz = float(probs_row[idx[z]]) if z in idx else 0.0
        po = float(probs_row[idx[o]]) if o in idx else 0.0
        return pz + po

    n_shells = base.n_qubits // 2 + 1
    shell_map_B = np.array([dist_to_ghz(str(s)) for s in base.support], dtype=np.int64)
    shell_map_C = np.array([dist_to_ghz(str(s)) for s in cond.support], dtype=np.int64)

    vB = np.zeros((B, n_shells), dtype=np.float64)
    vC = np.zeros((B, n_shells), dtype=np.float64)
    for b in range(B):
        np.add.at(vB[b], shell_map_B, pB[b])
        np.add.at(vC[b], shell_map_C, pC[b])

    pghzB = np.array([pghz_row(idxB, pB[b]) for b in range(B)], dtype=np.float64)
    pghzC = np.array([pghz_row(idxC, pC[b]) for b in range(B)], dtype=np.float64)

    delta_pghz = pghzC - pghzB
    delta_leak1 = vC[:, 1] - vB[:, 1]
    delta_leak2 = vC[:, 2] - vB[:, 2]

    tvd = np.zeros(B, dtype=np.float64)
    jsd = np.zeros(B, dtype=np.float64)
    for b in range(B):
        tvd[b] = tvd_union(base.support, pB[b], cond.support, pC[b])
        jsd[b] = jsd_union_bits(base.support, pB[b], cond.support, pC[b])

    return BootOut(delta_pghz, delta_leak1, delta_leak2, tvd, jsd, vB, vC, pghzB, pghzC)

# ============================================================
# PCA + LOW-RANK
# ============================================================
def pca_svd_pooled(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # Protocol: pooled PCA on shell vectors; center columns only; no scaling/whitening.
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T

    n_rows = X.shape[0]
    evals = (S**2) / float(n_rows - 1)

    for k in range(min(2, V.shape[1])):
        j = int(np.argmax(np.abs(V[:, k])))
        if V[j, k] < 0:
            V[:, k] *= -1.0

    scores2 = Xc @ V[:, :2]
    tot = float(np.sum(evals))
    pc1_pct = 100.0 * float(evals[0]) / tot if tot > 0 else 0.0
    pc2_pct = 100.0 * float(evals[1]) / tot if tot > 0 and evals.size > 1 else 0.0
    return scores2, V, evals, pc1_pct, pc2_pct

def low_rank_fracs(delta_v: np.ndarray) -> tuple[float, float, np.ndarray]:
    # Protocol: Δv covariance eigenspectrum on bootstrap Δv samples
    X = np.asarray(delta_v, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / float(X.shape[0] - 1)
    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)[::-1]
    tot = float(np.sum(evals))
    f1 = float(evals[0] / tot) if tot > 0 else 0.0
    f12 = float((evals[0] + evals[1]) / tot) if tot > 0 and evals.size > 1 else f1
    return f1, f12, evals

def pc1_separation_stats(scores_b: np.ndarray, scores_c: np.ndarray) -> dict[str, float]:
    """
    PC1 separation summary matching your plotting script:
      mu_b, mu_c, dmu, sig_b, sig_c, snr = |dmu| / sqrt(sig_b^2 + sig_c^2)
    """
    pc1_b = np.asarray(scores_b[:, 0], dtype=np.float64)
    pc1_c = np.asarray(scores_c[:, 0], dtype=np.float64)

    mu_b = float(np.mean(pc1_b))
    mu_c = float(np.mean(pc1_c))
    dmu = float(mu_c - mu_b)

    sig_b = float(np.std(pc1_b, ddof=1))
    sig_c = float(np.std(pc1_c, ddof=1))

    denom = float(np.sqrt(sig_b**2 + sig_c**2))
    snr = float(np.abs(dmu) / denom) if denom > 0 else float("inf")

    return {
        "mu_pc1_baseline": mu_b,
        "mu_pc1_conditioned": mu_c,
        "delta_mu": dmu,
        "sigma_pc1_baseline": sig_b,
        "sigma_pc1_conditioned": sig_c,
        "snr_pc1": snr,
    }

# ============================================================
# ESC HELPERS (ADDED — does not change any existing sums)
# ============================================================
def safe_reciprocal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.inf, dtype=np.float64)
    mask = x > 0
    out[mask] = 1.0 / x[mask]
    return out

def compute_esc_from_p(
    *,
    pA_hat: float,
    pB_hat: float,
    pA_boot: np.ndarray,
    pB_boot: np.ndarray,
    shots_scale: int = 10_000,
) -> dict:
    """
    ESC protocol metrics (UO-A set-membership) computed *directly* from bootstrap draws.
    Here p is the success probability for the predeclared useful set S.
    """
    # Per-mode event cost
    C_A_hat = (1.0 / pA_hat) if pA_hat > 0 else float("inf")
    C_B_hat = (1.0 / pB_hat) if pB_hat > 0 else float("inf")

    C_A_boot = safe_reciprocal(pA_boot)
    C_B_boot = safe_reciprocal(pB_boot)

    # Comparators
    with np.errstate(divide="ignore", invalid="ignore"):
        R_boot = np.where((pA_boot > 0) & np.isfinite(pA_boot), pB_boot / pA_boot, np.inf)
        D_boot = np.where((pB_boot > 0) & np.isfinite(pB_boot), 1.0 - (pA_boot / pB_boot), np.nan)

    R_hat = (pB_hat / pA_hat) if pA_hat > 0 else float("inf")
    D_hat = (1.0 - (pA_hat / pB_hat)) if pB_hat > 0 else float("nan")

    # Absolute waste per shots_scale
    W_A_hat = shots_scale * (1.0 - pA_hat)
    W_B_hat = shots_scale * (1.0 - pB_hat)
    W_A_boot = shots_scale * (1.0 - pA_boot)
    W_B_boot = shots_scale * (1.0 - pB_boot)

    # CIs (percentile, protocol-correct) from the bootstrap distribution of the metric itself
    esc = {
        "point": {
            "pA": pA_hat,
            "pB": pB_hat,
            "C_event_A": C_A_hat,
            "C_event_B": C_B_hat,
            "R_event": R_hat,
            "Delta_frac": D_hat,
            "Delta_pct": 100.0 * D_hat if np.isfinite(D_hat) else float("nan"),
            "W10k_A": W_A_hat,
            "W10k_B": W_B_hat,
            "W10k_savings": (W_A_hat - W_B_hat),
        },
        "ci95": {
            "pA": ci95_percentile(pA_boot),
            "pB": ci95_percentile(pB_boot),
            "C_event_A": ci95_percentile(C_A_boot),
            "C_event_B": ci95_percentile(C_B_boot),
            "R_event": ci95_percentile(R_boot[np.isfinite(R_boot)]),
            "Delta_frac": ci95_percentile(D_boot[np.isfinite(D_boot)]),
            "Delta_pct": tuple(100.0 * v for v in ci95_percentile(D_boot[np.isfinite(D_boot)])),
            "W10k_A": ci95_percentile(W_A_boot),
            "W10k_B": ci95_percentile(W_B_boot),
            "W10k_savings": ci95_percentile((W_A_boot - W_B_boot)),
        },
        "se": {
            "pA": se_sample(pA_boot),
            "pB": se_sample(pB_boot),
            "C_event_A": se_sample(C_A_boot[np.isfinite(C_A_boot)]),
            "C_event_B": se_sample(C_B_boot[np.isfinite(C_B_boot)]),
            "R_event": se_sample(R_boot[np.isfinite(R_boot)]),
            "Delta_frac": se_sample(D_boot[np.isfinite(D_boot)]),
            "W10k_A": se_sample(W_A_boot),
            "W10k_B": se_sample(W_B_boot),
            "W10k_savings": se_sample((W_A_boot - W_B_boot)),
        },
        "boot_draws": {
            "pA": pA_boot,
            "pB": pB_boot,
            "C_event_A": C_A_boot,
            "C_event_B": C_B_boot,
            "R_event": R_boot,
            "Delta_frac": D_boot,
            "W10k_A": W_A_boot,
            "W10k_B": W_B_boot,
            "W10k_savings": (W_A_boot - W_B_boot),
        }
    }
    return esc

# ============================================================
# TERMINAL TABLES
# ============================================================
def _hr(width: int) -> str:
    return "-" * width

def print_table(title: str, headers: list[str], rows: list[list[str]], pad: int = 2) -> None:
    col_w = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return (" " * pad).join(cells[i].ljust(col_w[i]) for i in range(len(headers)))

    total_width = len(fmt_row(headers))
    print(f"\n=== {title} ===")
    print(fmt_row(headers))
    print(_hr(total_width))
    for r in rows:
        print(fmt_row(r))
    print(_hr(total_width))

def pct_delta(b: float, c: float) -> str:
    if b == 0:
        return "—"
    return f"{100.0*(c-b)/b:+.2f}"

def print_esc_terminal_tables(*, test: str, esc: dict) -> None:
    """
    ADDED: ESC tables (UO-A on GHZ manifold in this script).
    This prints additional info only; does not alter any existing calculations.
    """
    pA = esc["point"]["pA"]
    pB = esc["point"]["pB"]
    CA = esc["point"]["C_event_A"]
    CB = esc["point"]["C_event_B"]
    R = esc["point"]["R_event"]
    Dpct = esc["point"]["Delta_pct"]
    W10A = esc["point"]["W10k_A"]
    W10B = esc["point"]["W10k_B"]
    W10S = esc["point"]["W10k_savings"]

    ci = esc["ci95"]
    rows = [
        ["p_A (success)", f"{pA:.6f}", f"[{ci['pA'][0]:.6f}, {ci['pA'][1]:.6f}]", f"{esc['se']['pA']:.6f}"],
        ["p_B (success)", f"{pB:.6f}", f"[{ci['pB'][0]:.6f}, {ci['pB'][1]:.6f}]", f"{esc['se']['pB']:.6f}"],
        ["C_event(A)=1/pA", f"{CA:.6f}", f"[{ci['C_event_A'][0]:.6f}, {ci['C_event_A'][1]:.6f}]", f"{esc['se']['C_event_A']:.6f}"],
        ["C_event(B)=1/pB", f"{CB:.6f}", f"[{ci['C_event_B'][0]:.6f}, {ci['C_event_B'][1]:.6f}]", f"{esc['se']['C_event_B']:.6f}"],
        ["R_event = pB/pA", f"{R:.6f}", f"[{ci['R_event'][0]:.6f}, {ci['R_event'][1]:.6f}]", f"{esc['se']['R_event']:.6f}"],
        ["Delta_% = 100*(1-pA/pB)", f"{Dpct:.3f}%", f"[{ci['Delta_pct'][0]:.3f}%, {ci['Delta_pct'][1]:.3f}%]", f"{100.0*esc['se']['Delta_frac']:.3f}%"],
        ["W10k(A)=10k*(1-pA)", f"{W10A:.1f}", f"[{ci['W10k_A'][0]:.1f}, {ci['W10k_A'][1]:.1f}]", f"{esc['se']['W10k_A']:.3f}"],
        ["W10k(B)=10k*(1-pB)", f"{W10B:.1f}", f"[{ci['W10k_B'][0]:.1f}, {ci['W10k_B'][1]:.1f}]", f"{esc['se']['W10k_B']:.3f}"],
        ["W10k savings (A-B)", f"{W10S:.1f}", f"[{ci['W10k_savings'][0]:.1f}, {ci['W10k_savings'][1]:.1f}]", f"{esc['se']['W10k_savings']:.3f}"],
    ]
    print_table(
        title=f"ESC (UO-A: GHZ MANIFOLD MEMBERSHIP) — BOOTSTRAP-PERCENTILE CIs ({test})",
        headers=["Metric", "Point", "95% CI", "Boot SE"],
        rows=rows
    )

def print_pdf_tables_point_estimates(
    *, test: str, n: int, k_near: int, topk_list: list[int],
    dist_b: SparseDist, dist_c: SparseDist,
    v_b: np.ndarray, v_c: np.ndarray,
    tvd_hat: float, jsd_hat: float,
    burden_b: dict, burden_c: dict,
) -> None:
    z = "0" * n
    o = "1" * n

    P_le_k_b = p_d_leq_k(v_b, k_near)
    P_le_k_c = p_d_leq_k(v_c, k_near)
    P_gt_k_b = 1.0 - P_le_k_b
    P_gt_k_c = 1.0 - P_le_k_c

    PGHZ_near_b = conditional_pghz_given_near(v_b, k_near)
    PGHZ_near_c = conditional_pghz_given_near(v_c, k_near)

    near_leak_b = 1.0 - PGHZ_near_b
    near_leak_c = 1.0 - PGHZ_near_c

    H_d_near_b = coarse_shell_entropy_near(v_b, k_near)
    H_d_near_c = coarse_shell_entropy_near(v_c, k_near)

    rows_I = [
        [f"P(d ≤ {k_near})", f"{P_le_k_b:.4f}", f"{P_le_k_c:.4f}", pct_delta(P_le_k_b, P_le_k_c)],
        [f"P(d > {k_near})", f"{P_gt_k_b:.4f}", f"{P_gt_k_c:.4f}", pct_delta(P_gt_k_b, P_gt_k_c)],
        [f"P_GHZ | d≤{k_near}", f"{PGHZ_near_b:.4f}", f"{PGHZ_near_c:.4f}", pct_delta(PGHZ_near_b, PGHZ_near_c)],
        ["Near-leakage frac", f"{near_leak_b:.4f}", f"{near_leak_c:.4f}", pct_delta(near_leak_b, near_leak_c)],
        [r"H_near^(d) [bits]", f"{H_d_near_b:.4f}", f"{H_d_near_c:.4f}", pct_delta(H_d_near_b, H_d_near_c)],
    ]
    print_table(
        title=f"NEAR-MANIFOLD CONDITIONING SUMMARY (k={k_near}, {test})",
        headers=["Metric", "Baseline", "Conditioned", "Δ(%)"],
        rows=rows_I
    )

    P0_b = p_bitstring(dist_b, z)
    P0_c = p_bitstring(dist_c, z)
    P1_b = p_bitstring(dist_b, o)
    P1_c = p_bitstring(dist_c, o)
    PGHZ_b = pghz(dist_b)
    PGHZ_c = pghz(dist_c)

    H_b = shannon_entropy_bits_dist(dist_b)
    H_c = shannon_entropy_bits_dist(dist_c)

    Hnear_b, Hfar_b = conditional_entropy_near_far(dist_b, k_near)
    Hnear_c, Hfar_c = conditional_entropy_near_far(dist_c, k_near)

    rows_II = [
        [f"P(0^{n})", f"{P0_b:.4f}", f"{P0_c:.4f}", pct_delta(P0_b, P0_c)],
        [f"P(1^{n})", f"{P1_b:.4f}", f"{P1_c:.4f}", pct_delta(P1_b, P1_c)],
        ["P_GHZ", f"{PGHZ_b:.4f}", f"{PGHZ_c:.4f}", pct_delta(PGHZ_b, PGHZ_c)],
        ["H(P) [bits]", f"{H_b:.4f}", f"{H_c:.4f}", pct_delta(H_b, H_c)],
        [f"H_near [bits] (d≤{k_near})", f"{Hnear_b:.4f}", f"{Hnear_c:.4f}", pct_delta(Hnear_b, Hnear_c)],
        [f"H_far  [bits] (d>{k_near})", f"{Hfar_b:.4f}", f"{Hfar_c:.4f}", pct_delta(Hfar_b, Hfar_c)],
    ]
    print_table(
        title=f"LOGICAL MANIFOLD PROBABILITIES & ENTROPY DIAGNOSTICS ({test})",
        headers=["Metric", "Baseline", "Conditioned", "Δ(%)"],
        rows=rows_II
    )

    rows_III: list[list[str]] = []
    for kk in topk_list:
        pb = ptop_k(dist_b, kk)
        pc = ptop_k(dist_c, kk)
        rows_III.append([f"P_top-{kk}", f"{pb:.4f}", f"{pc:.4f}", pct_delta(pb, pc)])

    rows_III.append(["Leakage mass d=1", f"{v_b[1]:.4f}", f"{v_c[1]:.4f}", pct_delta(v_b[1], v_c[1])])
    rows_III.append(["Leakage mass d=2", f"{v_b[2]:.4f}", f"{v_c[2]:.4f}", pct_delta(v_b[2], v_c[2])])

    Db = burden_b["circuit_depth"]
    Dc = burden_c["circuit_depth"]
    Ng_b = burden_b["total_gates"]
    Ng_c = burden_c["total_gates"]
    N2_b = burden_b["two_qubit_gates"]
    N2_c = burden_c["two_qubit_gates"]

    rows_III.append(["Circuit depth", f"{Db:d}", f"{Dc:d}", pct_delta(float(Db), float(Dc))])
    rows_III.append(["Total gates", f"{Ng_b:d}", f"{Ng_c:d}", pct_delta(float(Ng_b), float(Ng_c))])
    rows_III.append(["Two-qubit gates", f"{N2_b:d}", f"{N2_c:d}", pct_delta(float(N2_b), float(N2_c))])

    rows_III.append(["P_GHZ / depth", f"{(PGHZ_b/Db):.8f}" if Db else "—", f"{(PGHZ_c/Dc):.8f}" if Dc else "—",
                     pct_delta(PGHZ_b/Db if Db else 0.0, PGHZ_c/Dc if Dc else 0.0)])
    rows_III.append(["P_GHZ / total gates", f"{(PGHZ_b/Ng_b):.8f}" if Ng_b else "—", f"{(PGHZ_c/Ng_c):.8f}" if Ng_c else "—",
                     pct_delta(PGHZ_b/Ng_b if Ng_b else 0.0, PGHZ_c/Ng_c if Ng_c else 0.0)])
    rows_III.append(["P_GHZ / 2q gates", f"{(PGHZ_b/N2_b):.8f}" if N2_b else "—", f"{(PGHZ_c/N2_c):.8f}" if N2_c else "—",
                     pct_delta(PGHZ_b/N2_b if N2_b else 0.0, PGHZ_c/N2_c if N2_c else 0.0)])

    rows_III.append(["TVD(P,Q)", "—", f"{tvd_hat:.4f}", "—"])
    rows_III.append(["JSD(P,Q) [bits]", "—", f"{jsd_hat:.4f}", "—"])

    print_table(
        title=f"CONCENTRATION, LEAKAGE, CIRCUIT BURDEN, AND DISTRIBUTION DIVERGENCE ({test})",
        headers=["Metric", "Baseline", "Conditioned", "Δ(%)"],
        rows=rows_III
    )

def print_all_terminal_tables(
    *,
    test: str,
    # bootstrap
    delta_pghz_hat: float, ci_pghz: tuple[float, float], se_pghz: float, snr_pghz: float,
    delta_leak1_hat: float, ci_l1: tuple[float, float], se_l1: float,
    delta_leak2_hat: float, ci_l2: tuple[float, float], se_l2: float,
    tvd_hat: float, ci_tvd: tuple[float, float], se_tvd: float,
    jsd_hat: float, ci_jsd: tuple[float, float], se_jsd: float,
    # pca separation snr
    snr_pc1: float, pc1_mu_b: float, pc1_mu_c: float, pc1_sig_b: float, pc1_sig_c: float, pc1_dmu: float,
    # low-rank
    frac1: float, frac12: float,
    # pca
    pc1_var: float, pc2_var: float, pc12: float,
    pc1_loadings: np.ndarray,
    # shells
    v_b_hat: np.ndarray, v_c_hat: np.ndarray,
) -> None:
    def ci_signed(ci): return f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
    def ci_unsigned(ci): return f"[{ci[0]:.6f}, {ci[1]:.6f}]"

    rows_boot = [
        ["ΔP_GHZ",          f"{delta_pghz_hat:+.6f}", ci_signed(ci_pghz),   f"{se_pghz:.6f}", f"SNR_metric={snr_pghz:.2f}"],
        ["ΔLeak d=1",       f"{delta_leak1_hat:+.6f}", ci_signed(ci_l1),    f"{se_l1:.6f}",  ""],
        ["ΔLeak d=2",       f"{delta_leak2_hat:+.6f}", ci_signed(ci_l2),    f"{se_l2:.6f}",  ""],
        ["TVD",             f"{tvd_hat:.6f}",          ci_unsigned(ci_tvd), f"{se_tvd:.6f}", "(bias-corr)"],
        ["JSD (bits)",      f"{jsd_hat:.6f}",          ci_unsigned(ci_jsd), f"{se_jsd:.6f}", "(bias-corr)"],
    ]
    print_table(
        title=f"BOOTSTRAP UNCERTAINTY SUMMARY (PERCENTILE 95%; {test})",
        headers=["Metric", "Value/Δ", "95% CI", "Boot SE", "Notes"],
        rows=rows_boot
    )

    rows_pc1 = [
        ["mu_b", f"{pc1_mu_b:+.7f}"],
        ["mu_c", f"{pc1_mu_c:+.7f}"],
        ["dmu",  f"{pc1_dmu:+.7f}"],
        ["sig_b", f"{pc1_sig_b:.7f}"],
        ["sig_c", f"{pc1_sig_c:.7f}"],
        ["SNR_pc1", f"{snr_pc1:.4f}"],
    ]
    print_table(
        title=f"POOLED PCA (SHELL SPACE): PC1 SEPARATION STATS ({test})",
        headers=["Quantity", "Value"],
        rows=rows_pc1
    )

    rows_lr = [
        ["λ1/Σ",           f"{frac1:.6f}",  f"{100.0*frac1:.3f}%"],
        ["(λ1+λ2)/Σ",      f"{frac12:.6f}", f"{100.0*frac12:.3f}%"],
    ]
    print_table(
        title=f"Δv LOW-RANK STRUCTURE (COVARIANCE EIGNSPECTRUM FRACTIONS; {test})",
        headers=["Quantity", "Value", "Percent"],
        rows=rows_lr
    )

    rows_pca_var = [
        ["PC1",      f"{pc1_var:.6f}", f"{100.0*pc1_var:.4f}%"],
        ["PC2",      f"{pc2_var:.6f}", f"{100.0*pc2_var:.4f}%"],
        ["PC1+PC2",  f"{pc12:.6f}",    f"{100.0*pc12:.4f}%"],
    ]
    print_table(
        title=f"POOLED PCA (SHELL SPACE): VARIANCE EXPLAINED ({test})",
        headers=["Component", "Fraction", "Percent"],
        rows=rows_pca_var
    )

    rows_load = [[f"d={d}", f"{float(val):+.6f}"] for d, val in enumerate(pc1_loadings.tolist())]
    print_table(
        title=f"POOLED PCA (SHELL SPACE): PC1 LOADINGS BY SHELL DISTANCE ({test})",
        headers=["Shell", "Loading"],
        rows=rows_load
    )

    rows_shell = []
    for d in range(len(v_b_hat)):
        rows_shell.append([f"d={d}", f"{float(v_b_hat[d]):.6f}", f"{float(v_c_hat[d]):.6f}", f"{float(v_c_hat[d]-v_b_hat[d]):+.6f}"])
    print_table(
        title=f"SHELL MASS DISTRIBUTION BY GHZ DISTANCE (POINT ESTIMATES, {test})",
        headers=["Shell", "v_base", "v_cond", "Δv"],
        rows=rows_shell
    )

# ============================================================
# UNIFIED OUTPUT FILES (CSV + JSON)
# ============================================================
def _add_row(rows: list[dict], section: str, metric: str, variant: str, value: float | int | str,
             units: str = "", notes: str = "") -> None:
    rows.append({
        "test": "",
        "section": section,
        "metric": metric,
        "variant": variant,
        "value": value,
        "units": units,
        "notes": notes,
    })

def write_unified_csv(
    out_csv: Path,
    *,
    test: str,
    input_hashes: dict,
    burden_b: dict, burden_c: dict,
    n: int, k_near: int, topk_list: list[int],
    dist_b: SparseDist, dist_c: SparseDist,
    v_b_hat: np.ndarray, v_c_hat: np.ndarray,
    tvd_hat: float, jsd_hat: float,
    delta_pghz_hat: float, ci_pghz: tuple[float, float], se_pghz: float, snr_pghz: float,
    delta_leak1_hat: float, ci_l1: tuple[float, float], se_l1: float,
    delta_leak2_hat: float, ci_l2: tuple[float, float], se_l2: float,
    ci_tvd: tuple[float, float], se_tvd: float,
    ci_jsd: tuple[float, float], se_jsd: float,
    frac1: float, frac12: float,
    pc1_var: float, pc2_var: float, pc12: float,
    pc1_loadings: np.ndarray,
    pc1_stats: dict[str, float],  # <-- NEW
    esc_summary: Optional[dict] = None,  # <-- ADDED (ESC protocol support)
) -> None:
    rows: list[dict] = []

    def add(section, metric, variant, value, units="", notes=""):
        _add_row(rows, section, metric, variant, value, units, notes)

    for k, v in input_hashes.items():
        add("inputs", f"{k}.path", "meta", v["path"])
        add("inputs", f"{k}.sha256", "meta", v["sha256"])

    for side, b in (("baseline", burden_b), ("conditioned", burden_c)):
        add("qasm_burden", "total_qubits", side, b["total_qubits"], "qubits")
        add("qasm_burden", "circuit_depth", side, b["circuit_depth"], "layers", "greedy ASAP on serialized gates")
        add("qasm_burden", "total_gates", side, b["total_gates"], "gates", "excludes measure/barrier/directives")
        add("qasm_burden", "two_qubit_gates", side, b["two_qubit_gates"], "gates", "heuristic by name or >=2 touched qubits")

    z = "0" * n
    o = "1" * n

    P0_b = p_bitstring(dist_b, z)
    P0_c = p_bitstring(dist_c, z)
    P1_b = p_bitstring(dist_b, o)
    P1_c = p_bitstring(dist_c, o)
    PGHZ_b = pghz(dist_b)
    PGHZ_c = pghz(dist_c)

    add("point_estimates", f"P(0^{n})", "baseline", P0_b)
    add("point_estimates", f"P(0^{n})", "conditioned", P0_c)
    add("point_estimates", f"P(1^{n})", "baseline", P1_b)
    add("point_estimates", f"P(1^{n})", "conditioned", P1_c)
    add("point_estimates", "P_GHZ", "baseline", PGHZ_b)
    add("point_estimates", "P_GHZ", "conditioned", PGHZ_c)

    add("divergence", "TVD(P,Q)", "paired", tvd_hat)
    add("divergence", "JSD(P,Q)", "paired", jsd_hat, "bits")

    P_le_k_b = p_d_leq_k(v_b_hat, k_near)
    P_le_k_c = p_d_leq_k(v_c_hat, k_near)
    add("near_manifold", f"P(d<= {k_near})", "baseline", P_le_k_b)
    add("near_manifold", f"P(d<= {k_near})", "conditioned", P_le_k_c)

    add("near_manifold", f"P_GHZ | d<= {k_near}", "baseline", conditional_pghz_given_near(v_b_hat, k_near))
    add("near_manifold", f"P_GHZ | d<= {k_near}", "conditioned", conditional_pghz_given_near(v_c_hat, k_near))

    add("near_manifold", "H_near^(d)", "baseline", coarse_shell_entropy_near(v_b_hat, k_near), "bits")
    add("near_manifold", "H_near^(d)", "conditioned", coarse_shell_entropy_near(v_c_hat, k_near), "bits")

    H_b = shannon_entropy_bits_dist(dist_b)
    H_c = shannon_entropy_bits_dist(dist_c)
    Hnear_b, Hfar_b = conditional_entropy_near_far(dist_b, k_near)
    Hnear_c, Hfar_c = conditional_entropy_near_far(dist_c, k_near)

    add("entropy", "H(P)", "baseline", H_b, "bits")
    add("entropy", "H(P)", "conditioned", H_c, "bits")
    add("entropy", f"H_near (d<= {k_near})", "baseline", Hnear_b, "bits")
    add("entropy", f"H_near (d<= {k_near})", "conditioned", Hnear_c, "bits")
    add("entropy", f"H_far (d> {k_near})", "baseline", Hfar_b, "bits")
    add("entropy", f"H_far (d> {k_near})", "conditioned", Hfar_c, "bits")

    for kk in topk_list:
        add("rank_concentration", f"P_top-{kk}", "baseline", ptop_k(dist_b, kk))
        add("rank_concentration", f"P_top-{kk}", "conditioned", ptop_k(dist_c, kk))

    for d in range(len(v_b_hat)):
        add("shells", f"v[d={d}]", "baseline", float(v_b_hat[d]))
        add("shells", f"v[d={d}]", "conditioned", float(v_c_hat[d]))
        add("shells", f"Δv[d={d}]", "delta", float(v_c_hat[d] - v_b_hat[d]))

    add("bootstrap", "ΔP_GHZ", "delta", delta_pghz_hat)
    add("bootstrap", "ΔP_GHZ.CI95_low", "delta", ci_pghz[0])
    add("bootstrap", "ΔP_GHZ.CI95_high", "delta", ci_pghz[1])
    add("bootstrap", "ΔP_GHZ.SE", "delta", se_pghz)
    add("bootstrap", "ΔP_GHZ.SNR_metric", "delta", snr_pghz)

    add("bootstrap", "ΔLeak(d=1)", "delta", delta_leak1_hat)
    add("bootstrap", "ΔLeak(d=1).CI95_low", "delta", ci_l1[0])
    add("bootstrap", "ΔLeak(d=1).CI95_high", "delta", ci_l1[1])
    add("bootstrap", "ΔLeak(d=1).SE", "delta", se_l1)

    add("bootstrap", "ΔLeak(d=2)", "delta", delta_leak2_hat)
    add("bootstrap", "ΔLeak(d=2).CI95_low", "delta", ci_l2[0])
    add("bootstrap", "ΔLeak(d=2).CI95_high", "delta", ci_l2[1])
    add("bootstrap", "ΔLeak(d=2).SE", "delta", se_l2)

    add("bootstrap", "TVD.bias_corrected.CI95_low", "paired", ci_tvd[0])
    add("bootstrap", "TVD.bias_corrected.CI95_high", "paired", ci_tvd[1])
    add("bootstrap", "TVD.bias_corrected.SE", "paired", se_tvd)

    add("bootstrap", "JSD_bits.bias_corrected.CI95_low", "paired", ci_jsd[0])
    add("bootstrap", "JSD_bits.bias_corrected.CI95_high", "paired", ci_jsd[1])
    add("bootstrap", "JSD_bits.bias_corrected.SE", "paired", se_jsd)

    add("pca_separation", "mu_pc1", "baseline", pc1_stats["mu_pc1_baseline"])
    add("pca_separation", "mu_pc1", "conditioned", pc1_stats["mu_pc1_conditioned"])
    add("pca_separation", "delta_mu_pc1", "paired", pc1_stats["delta_mu"])
    add("pca_separation", "sigma_pc1", "baseline", pc1_stats["sigma_pc1_baseline"])
    add("pca_separation", "sigma_pc1", "conditioned", pc1_stats["sigma_pc1_conditioned"])
    add("pca_separation", "snr_pc1", "paired", pc1_stats["snr_pc1"])

    add("lowrank", "λ1/Σ", "scalar", frac1)
    add("lowrank", "(λ1+λ2)/Σ", "scalar", frac12)

    add("pca", "PC1 variance explained", "scalar", pc1_var)
    add("pca", "PC2 variance explained", "scalar", pc2_var)
    add("pca", "PC1+PC2 cumulative", "scalar", pc12)
    for d, val in enumerate(pc1_loadings.tolist()):
        add("pca", f"PC1 loading d={d}", "scalar", float(val))

    if esc_summary is not None:

        add("esc", "objective", "meta", "UO-A: GHZ manifold membership", "", "S={0^n,1^n}")
        add("esc", "shots_scale_for_waste", "meta", 10_000, "shots", "W10k uses this scale")

        add("esc.point", "p_A", "baseline", float(esc_summary["point"]["pA"]))
        add("esc.point", "p_B", "conditioned", float(esc_summary["point"]["pB"]))
        add("esc.point", "C_event", "baseline", float(esc_summary["point"]["C_event_A"]), "shots/event")
        add("esc.point", "C_event", "conditioned", float(esc_summary["point"]["C_event_B"]), "shots/event")
        add("esc.point", "R_event", "paired", float(esc_summary["point"]["R_event"]), "", "pB/pA")
        add("esc.point", "Delta_pct", "paired", float(esc_summary["point"]["Delta_pct"]), "%", "100*(1-pA/pB)")
        add("esc.point", "W10k", "baseline", float(esc_summary["point"]["W10k_A"]), "shots", "10k*(1-pA)")
        add("esc.point", "W10k", "conditioned", float(esc_summary["point"]["W10k_B"]), "shots", "10k*(1-pB)")
        add("esc.point", "W10k_savings", "paired", float(esc_summary["point"]["W10k_savings"]), "shots", "W10k(A)-W10k(B)")

        add("esc.ci95", "p_A.low", "baseline", float(esc_summary["ci95"]["pA"][0]))
        add("esc.ci95", "p_A.high", "baseline", float(esc_summary["ci95"]["pA"][1]))
        add("esc.ci95", "p_B.low", "conditioned", float(esc_summary["ci95"]["pB"][0]))
        add("esc.ci95", "p_B.high", "conditioned", float(esc_summary["ci95"]["pB"][1]))

        add("esc.ci95", "C_event.low", "baseline", float(esc_summary["ci95"]["C_event_A"][0]))
        add("esc.ci95", "C_event.high", "baseline", float(esc_summary["ci95"]["C_event_A"][1]))
        add("esc.ci95", "C_event.low", "conditioned", float(esc_summary["ci95"]["C_event_B"][0]))
        add("esc.ci95", "C_event.high", "conditioned", float(esc_summary["ci95"]["C_event_B"][1]))

        add("esc.ci95", "R_event.low", "paired", float(esc_summary["ci95"]["R_event"][0]))
        add("esc.ci95", "R_event.high", "paired", float(esc_summary["ci95"]["R_event"][1]))
        add("esc.ci95", "Delta_pct.low", "paired", float(esc_summary["ci95"]["Delta_pct"][0]))
        add("esc.ci95", "Delta_pct.high", "paired", float(esc_summary["ci95"]["Delta_pct"][1]))

        add("esc.ci95", "W10k.low", "baseline", float(esc_summary["ci95"]["W10k_A"][0]))
        add("esc.ci95", "W10k.high", "baseline", float(esc_summary["ci95"]["W10k_A"][1]))
        add("esc.ci95", "W10k.low", "conditioned", float(esc_summary["ci95"]["W10k_B"][0]))
        add("esc.ci95", "W10k.high", "conditioned", float(esc_summary["ci95"]["W10k_B"][1]))
        add("esc.ci95", "W10k_savings.low", "paired", float(esc_summary["ci95"]["W10k_savings"][0]))
        add("esc.ci95", "W10k_savings.high", "paired", float(esc_summary["ci95"]["W10k_savings"][1]))

        add("esc.se", "p_A", "baseline", float(esc_summary["se"]["pA"]))
        add("esc.se", "p_B", "conditioned", float(esc_summary["se"]["pB"]))
        add("esc.se", "C_event", "baseline", float(esc_summary["se"]["C_event_A"]))
        add("esc.se", "C_event", "conditioned", float(esc_summary["se"]["C_event_B"]))
        add("esc.se", "R_event", "paired", float(esc_summary["se"]["R_event"]))
        add("esc.se", "Delta_frac", "paired", float(esc_summary["se"]["Delta_frac"]))
        add("esc.se", "W10k", "baseline", float(esc_summary["se"]["W10k_A"]))
        add("esc.se", "W10k", "conditioned", float(esc_summary["se"]["W10k_B"]))
        add("esc.se", "W10k_savings", "paired", float(esc_summary["se"]["W10k_savings"]))

    for r in rows:
        r["test"] = test

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "section", "metric", "variant", "value", "units", "notes"])
        w.writeheader()
        w.writerows(rows)

def run_one_test(test: str, args) -> None:
    cfg = TEST_CONFIG[test]
    n = int(cfg["n_qubits"])
    expected = int(cfg["shots_expected"])
    k_near = int(cfg["near_k"])
    topk_list = list(cfg["topk_list"])

    here = Path(__file__).resolve().parent
    test_dir = here / test

    base_csv = (test_dir / str(cfg["baseline_shot_order_file"])).resolve()
    cond_csv = (test_dir / str(cfg["conditioned_shot_order_file"])).resolve()

    base_qasm = (test_dir / str(cfg["baseline_qasm"])).resolve()
    cond_qasm = (test_dir / str(cfg["conditioned_qasm"])).resolve()

    if args.hash:
        print(f"[{test}] SHA256 inputs")
        print(f"  baseline_csv     : {sha256_file(base_csv)}")
        print(f"  conditioned_csv  : {sha256_file(cond_csv)}")
        print(f"  baseline_qasm    : {sha256_file(base_qasm)}")
        print(f"  conditioned_qasm : {sha256_file(cond_qasm)}")

    outdir = (here / args.outdir / test).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "baseline_csv": {"path": str(base_csv), "sha256": sha256_file(base_csv)},
        "conditioned_csv": {"path": str(cond_csv), "sha256": sha256_file(cond_csv)},
        "baseline_qasm": {"path": str(base_qasm), "sha256": sha256_file(base_qasm)},
        "conditioned_qasm": {"path": str(cond_qasm), "sha256": sha256_file(cond_qasm)},
    }

    shots_b = load_shot_order(base_csv, n)
    shots_c = load_shot_order(cond_csv, n)

    if shots_b.size != expected:
        print(f"[warn] baseline shots={shots_b.size} != expected {expected}")
    if shots_c.size != expected:
        print(f"[warn] conditioned shots={shots_c.size} != expected {expected}")

    print(f"[load] {str(base_qasm)}")
    burden_b = parse_qasm_burden(base_qasm)
    print(f"[load] {str(cond_qasm)}")
    burden_c = parse_qasm_burden(cond_qasm)

    dist_b = build_sparse_dist(shots_b, n)
    dist_c = build_sparse_dist(shots_c, n)

    v_b_hat = shell_vec(dist_b)
    v_c_hat = shell_vec(dist_c)

    pghz_b = pghz(dist_b)
    pghz_c = pghz(dist_c)
    delta_pghz_hat = pghz_c - pghz_b

    delta_leak1_hat = float(v_c_hat[1] - v_b_hat[1])
    delta_leak2_hat = float(v_c_hat[2] - v_b_hat[2])

    tvd_hat = tvd_union(dist_b.support, dist_b.probs, dist_c.support, dist_c.probs)
    jsd_hat = jsd_union_bits(dist_b.support, dist_b.probs, dist_c.support, dist_c.probs)

    eff_seed = stable_seed(args.seed, test)
    seedB = (eff_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
    seedC = (eff_seed ^ 0x5A5A5A5A) & 0xFFFFFFFF
    rngB = np.random.default_rng(np.random.SeedSequence(seedB))
    rngC = np.random.default_rng(np.random.SeedSequence(seedC))

    B = int(args.B)
    boot = compute_bootstrap_all(dist_b, dist_c, B, rngB, rngC)

    tvd_adj = bias_correct(boot.tvd, tvd_hat)
    jsd_adj = bias_correct(boot.jsd, jsd_hat)

    ci_pghz = ci95_percentile(boot.delta_pghz); se_pghz = se_sample(boot.delta_pghz)
    ci_l1 = ci95_percentile(boot.delta_leak1);  se_l1 = se_sample(boot.delta_leak1)
    ci_l2 = ci95_percentile(boot.delta_leak2);  se_l2 = se_sample(boot.delta_leak2)
    ci_tvd = ci95_percentile(tvd_adj);          se_tvd = se_sample(tvd_adj)
    ci_jsd = ci95_percentile(jsd_adj);          se_jsd = se_sample(jsd_adj)

    snr_pghz = abs(delta_pghz_hat) / se_pghz if se_pghz > 0 else float("inf")

    delta_v = boot.v_cond - boot.v_base
    frac1, frac12, evals_dv = low_rank_fracs(delta_v)

    X = np.vstack([boot.v_base, boot.v_cond])
    scores2, V, evals_pca, pc1_pct, pc2_pct = pca_svd_pooled(X)

    sb = scores2[:B]
    sc = scores2[B:]
    pc1_stats = pc1_separation_stats(sb, sc)

    tot_pca = float(np.sum(evals_pca))
    pc1_var = float(evals_pca[0] / tot_pca) if tot_pca > 0 else 0.0
    pc2_var = float(evals_pca[1] / tot_pca) if tot_pca > 0 else 0.0
    pc12 = pc1_var + pc2_var
    pc1_loadings = V[:, 0].copy()

    # ========================================================
    # ADDED: ESC computation (protocol-correct CIs from draws)
    # (Does not change any existing sums; only derives more outputs)
    # ========================================================
    esc = compute_esc_from_p(
        pA_hat=pghz_b,
        pB_hat=pghz_c,
        pA_boot=boot.pghz_base,
        pB_boot=boot.pghz_cond,
        shots_scale=10_000,
    )

    out_npz = (outdir / "esc_bootstrap_draws.npz").resolve()
    np.savez_compressed(
        out_npz,
        pA=esc["boot_draws"]["pA"],
        pB=esc["boot_draws"]["pB"],
        C_event_A=esc["boot_draws"]["C_event_A"],
        C_event_B=esc["boot_draws"]["C_event_B"],
        R_event=esc["boot_draws"]["R_event"],
        Delta_frac=esc["boot_draws"]["Delta_frac"],
        W10k_A=esc["boot_draws"]["W10k_A"],
        W10k_B=esc["boot_draws"]["W10k_B"],
        W10k_savings=esc["boot_draws"]["W10k_savings"],
        meta=np.array([f"test={test}", f"B={B}", "objective=UO-A:GHZ"], dtype=object),
    )
    print(f"[save] {str(out_npz)}")

    out_csv = (outdir / "esc_full_analysis.csv").resolve()
    out_json = (outdir / "esc_full_analysis.json").resolve()

    write_unified_csv(
        out_csv,
        test=test,
        input_hashes=input_hashes,
        burden_b=burden_b, burden_c=burden_c,
        n=n, k_near=k_near, topk_list=topk_list,
        dist_b=dist_b, dist_c=dist_c,
        v_b_hat=v_b_hat, v_c_hat=v_c_hat,
        tvd_hat=tvd_hat, jsd_hat=jsd_hat,
        delta_pghz_hat=delta_pghz_hat, ci_pghz=ci_pghz, se_pghz=se_pghz, snr_pghz=snr_pghz,
        delta_leak1_hat=delta_leak1_hat, ci_l1=ci_l1, se_l1=se_l1,
        delta_leak2_hat=delta_leak2_hat, ci_l2=ci_l2, se_l2=se_l2,
        ci_tvd=ci_tvd, se_tvd=se_tvd,
        ci_jsd=ci_jsd, se_jsd=se_jsd,
        frac1=frac1, frac12=frac12,
        pc1_var=pc1_var, pc2_var=pc2_var, pc12=pc12,
        pc1_loadings=pc1_loadings,
        pc1_stats=pc1_stats, 
        esc_summary=esc,    
    )
    print(f"[save] {str(out_csv)}")

    results = {
        "test": test,
        "n_qubits": n,
        "near_k": k_near,
        "N_base": dist_b.N,
        "N_cond": dist_c.N,
        "B": B,
        "seed_base": args.seed,
        "seed_effective": eff_seed,
        "seed_baseline_stream": seedB,
        "seed_conditioned_stream": seedC,
        "inputs": input_hashes,
        "qasm_burden": {"baseline": burden_b, "conditioned": burden_c},
        "point_estimates": {
            "p0_base": p_bitstring(dist_b, "0"*n),
            "p0_cond": p_bitstring(dist_c, "0"*n),
            "p1_base": p_bitstring(dist_b, "1"*n),
            "p1_cond": p_bitstring(dist_c, "1"*n),
            "pghz_base": pghz_b,
            "pghz_cond": pghz_c,
            "delta_pghz": delta_pghz_hat,
            "shell_v_base": v_b_hat.tolist(),
            "shell_v_cond": v_c_hat.tolist(),
            "delta_leak_d1": delta_leak1_hat,
            "delta_leak_d2": delta_leak2_hat,
            "tvd": tvd_hat,
            "jsd_bits": jsd_hat,
            "entropy": {
                "H_base": shannon_entropy_bits_dist(dist_b),
                "H_cond": shannon_entropy_bits_dist(dist_c),
                "Hnear_base": conditional_entropy_near_far(dist_b, k_near)[0],
                "Hnear_cond": conditional_entropy_near_far(dist_c, k_near)[0],
                "Hfar_base": conditional_entropy_near_far(dist_b, k_near)[1],
                "Hfar_cond": conditional_entropy_near_far(dist_c, k_near)[1],
                "H_d_near_base": coarse_shell_entropy_near(v_b_hat, k_near),
                "H_d_near_cond": coarse_shell_entropy_near(v_c_hat, k_near),
            },
            "near_manifold": {
                "P_d_leq_k_base": p_d_leq_k(v_b_hat, k_near),
                "P_d_leq_k_cond": p_d_leq_k(v_c_hat, k_near),
                "Pghz_given_near_base": conditional_pghz_given_near(v_b_hat, k_near),
                "Pghz_given_near_cond": conditional_pghz_given_near(v_c_hat, k_near),
            },
            "topk": ({
                f"Ptop_{k}_base": ptop_k(dist_b, k) for k in topk_list
            } | {
                f"Ptop_{k}_cond": ptop_k(dist_c, k) for k in topk_list
            }),
        },
        "bootstrap": {
            "delta_pghz": {"ci95": ci_pghz, "se": se_pghz, "snr_metric": snr_pghz},
            "delta_leak_d1": {"ci95": ci_l1, "se": se_l1},
            "delta_leak_d2": {"ci95": ci_l2, "se": se_l2},
            "tvd_bias_corrected": {"ci95": ci_tvd, "se": se_tvd},
            "jsd_bits_bias_corrected": {"ci95": ci_jsd, "se": se_jsd},
        },
        "lowrank_deltav": {
            "lambda1_over_sum": frac1,
            "lambda12_over_sum": frac12,
            "eigenvalues": evals_dv.tolist(),
        },
        "pca": {
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "variance_explained": {"pc1": pc1_var, "pc2": pc2_var, "pc1_plus_pc2": pc12},
            "pc1_loadings": pc1_loadings.tolist(),
            "eigenvalues": evals_pca.tolist(),
            "pc1_separation": pc1_stats,
        },

        "esc": {
            "objective": "UO-A: GHZ manifold membership (S={0^n,1^n})",
            "point": esc["point"],
            "ci95": esc["ci95"],
            "se": esc["se"],
            "bootstrap_draws_file": {"path": str(out_npz), "sha256": sha256_file(out_npz)},
            "notes": "ESC CIs are percentile intervals computed from bootstrap metric distributions (no analytic propagation).",
        },
        "audit": {
            "output_hashes": {
                "unified_csv": {"path": str(out_csv), "sha256": sha256_file(out_csv)},
                "unified_json": {"path": str(out_json), "sha256": ""},
                "esc_bootstrap_draws_npz": {"path": str(out_npz), "sha256": sha256_file(out_npz)},
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "env_vars_relevant": {k: os.environ.get(k) for k in [
                    "PYTHONHASHSEED",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]},
            },
            "determinism": {
                "rng": "numpy.default_rng(SeedSequence(seedB/seedC)) using PCG64; baseline/conditioned independent",
                "quantile_method": "linear",
                "union_support_iteration": "sorted(set(support_base) | set(support_cond))",
                "pca": "explicit SVD numpy.linalg.svd; sign-fixed by max-abs entry",
                "qasm_burden": "excludes measure/barrier; greedy ASAP depth on serialized gate list",
                "note": "For best determinism set BLAS threads to 1 in CMD env vars.",
            },
        },
    }

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["audit"]["output_hashes"]["unified_json"]["sha256"] = sha256_file(out_json)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {str(out_json)}")

    # ------------------------------------------------------------
    # TERMINAL PRINT (PDF tables + your add-ons)
    # ------------------------------------------------------------
    print_pdf_tables_point_estimates(
        test=test, n=n, k_near=k_near, topk_list=topk_list,
        dist_b=dist_b, dist_c=dist_c,
        v_b=v_b_hat, v_c=v_c_hat,
        tvd_hat=tvd_hat, jsd_hat=jsd_hat,
        burden_b=burden_b, burden_c=burden_c,
    )

    print_all_terminal_tables(
        test=test,
        delta_pghz_hat=delta_pghz_hat, ci_pghz=ci_pghz, se_pghz=se_pghz, snr_pghz=snr_pghz,
        delta_leak1_hat=delta_leak1_hat, ci_l1=ci_l1, se_l1=se_l1,
        delta_leak2_hat=delta_leak2_hat, ci_l2=ci_l2, se_l2=se_l2,
        tvd_hat=tvd_hat, ci_tvd=ci_tvd, se_tvd=se_tvd,
        jsd_hat=jsd_hat, ci_jsd=ci_jsd, se_jsd=se_jsd,
        snr_pc1=pc1_stats["snr_pc1"],
        pc1_mu_b=pc1_stats["mu_pc1_baseline"],
        pc1_mu_c=pc1_stats["mu_pc1_conditioned"],
        pc1_sig_b=pc1_stats["sigma_pc1_baseline"],
        pc1_sig_c=pc1_stats["sigma_pc1_conditioned"],
        pc1_dmu=pc1_stats["delta_mu"],
        frac1=frac1, frac12=frac12,
        pc1_var=pc1_var, pc2_var=pc2_var, pc12=pc12,
        pc1_loadings=pc1_loadings,
        v_b_hat=v_b_hat, v_c_hat=v_c_hat,
    )

    print_esc_terminal_tables(test=test, esc=esc)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--outdir", type=str, default="RESULTS")
    ap.add_argument(
        "--analyze",
        choices=tuple(TEST_CONFIG.keys()),
        help="Select a single test case to run."
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all released test cases."
    )
    ap.add_argument(
        "--hash",
        action="store_true",
        help="Also print SHA256 of each input artifact."
    )

    args = ap.parse_args()
    
    if not args.all and not args.analyze:
        ap.error("Provide --analyze <TESTNAME> or --all")

    tests = list(TEST_CONFIG.keys()) if args.all else [args.analyze]

    for test in tests:
        print("=" * 78)
        print(f"RUNNING TEST: {test}")
        print("=" * 78)
        run_one_test(test, args)

if __name__ == "__main__":
    main()
