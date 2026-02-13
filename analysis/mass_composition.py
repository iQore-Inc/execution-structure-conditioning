# --------------------------------------------------------------
# Reproducibility Integrity Metadata
# Digest      : Generated at release (see repository checksums)
# Timestamp   : 2026-02-09
# Purpose     : Deterministic, tamper-evident reproduction of
#               published probability-mass-flow analysis
# --------------------------------------------------------------

# NOTICE
# This script is a public, reproducibility-grade analysis artifact.
# It operates exclusively on released experimental data and produces
# deterministic visualizations corresponding to the figures reported
# in the associated paper.
#
# No quantum hardware execution occurs in this script.
# ==============================================================

# ==============================================================
# Title   : Distance-Binned Probability Mass Composition — Reproducibility Script
# File    : mass_composition.py
# Scope   : External researchers / independent verification
# Paper   : "Execution-Dependent Geometry of Near-Manifold Outcome Resolution"
#
# © 2025–2026 iQore Inc.
# Licensed under the Apache License, Version 2.0
# See LICENSE-CODE.md at repository root for full terms.
# --------------------------------------------------------------
#
# This file is an executable methodological artifact released for
# the purpose of scientific reproducibility.
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

# --------------------------------------------------------------
# CLI / USAGE (Reproducibility Entry Point)
# --------------------------------------------------------------
# This script is executed via a single required CLI flag:
#
#   --analyze {15Q-MAIN,15Q-MCE,20Q-SBP,10Q-CBP}
#
# Example commands (run from the directory containing this file):
#
#   python mass_composition.py --analyze 10Q-CBP
#   python mass_composition.py --analyze 15Q-MAIN
#   python mass_composition.py --analyze 15Q-MCE
#   python mass_composition.py --analyze 20Q-SBP
#
# Help:
#   python mass_composition.py --help
#
# Expected output (written next to this script):
#   "[<TESTNAME>] - Distance-Binned Probability Mass Composition.png"
#
# Input data layout (relative to this script):
#   mass_composition.py
#   <TESTNAME>/
#     <baseline CSV>
#     <conditioned CSV>
#
# Each CSV must contain columns:
#   - Bitstring : 0/1 string of length n_qubits
#   - Count     : integer >= 0
# --------------------------------------------------------------

# --------------------------------------------------------------
# PURPOSE & SCOPE
# --------------------------------------------------------------
# This script reproduces the "probability mass flow" visualization
# used in the paper to show how probability redistributes under
# execution-structure conditioning, while total mass is conserved.
#
# Given two empirical shot-count distributions over n-qubit
# bitstrings:
#   - Baseline distribution P_B(x)
#   - Conditioned distribution P_C(x)
#
# the script:
#   1. Loads the released CSVs (baseline + conditioned)
#   2. Validates bitstrings (0/1-only) and fixed qubit length n
#   3. Normalizes shot counts into probabilities P(x)
#   4. Orders outcomes by GHZ-manifold distance shell and
#      within-shell baseline probability
#   5. Renders a two-column stacked-mass diagram (Baseline vs
#      Conditioned) with connecting flow bands per bitstring
#
# The resulting figure visually encodes:
#   - How much probability remains near the GHZ endpoints
#   - How probability in each distance shell shifts between
#     Baseline and Conditioned executions
#   - Conservation of total probability mass: sum_x P(x) = 1
# --------------------------------------------------------------

# --------------------------------------------------------------
# DEFINITIONS
# --------------------------------------------------------------
# Let wt(x) be the Hamming weight of bitstring x (number of 1s),
# and let n be the number of qubits (bitstring length).
#
# GHZ-manifold distance (shell index) is defined as:
#
#     d_H(x) = min(wt(x), n - wt(x))
#
# The GHZ endpoints are:
#     |0...0>  (wt=0, d_H=0)
#     |1...1>  (wt=n, d_H=0)
#
# Shells are grouped by d_H(x), then outcomes within each shell are
# ordered by baseline probability P_B(x) (descending), with a stable
# bitstring tiebreaker to ensure deterministic output.
# --------------------------------------------------------------

# --------------------------------------------------------------
# SYSTEM OVERVIEW
# --------------------------------------------------------------
# Analysis Modules:
# ┌────────────────────────────────────────────────────────────┐
# │ 1. **Data Ingestion**                                      │
# │    - Loads empirical shot-count CSVs                       │
# │    - Validates bitstrings and qubit counts                 │
# │    - Normalizes counts to probabilities                    │
# │                                                            │
# │ 2. **Outcome Ordering**                                    │
# │    - Computes GHZ distance shells d_H(x)                   │
# │    - Orders outcomes by shell then by P_B(x)              │
# │    - Locks endpoints |0...0>, |1...1> to the top of d=0    │
# │                                                            │
# │ 3. **Stack Construction**                                  │
# │    - Converts ordered probabilities into cumulative         │
# │      vertical intervals for stacked rendering              │
# │                                                            │
# │ 4. **Color + Shell Encoding**                              │
# │    - Explicit colors for GHZ endpoints                     │
# │    - Distinct base hue per shell (d_H)                     │
# │    - Within-shell washout/alpha encodes relative magnitude │
# │                                                            │
# │ 5. **Visualization Layer**                                 │
# │    - Two stacked columns (Baseline vs Conditioned)         │
# │    - Flow bands connecting matched outcomes                │
# │    - Fixed canvas/typography and deterministic save        │
# └────────────────────────────────────────────────────────────┘
# --------------------------------------------------------------

# --------------------------------------------------------------
# EXECUTION FLOW
# --------------------------------------------------------------
# 1. parse CLI args (--analyze)
# 2. resolve baseline + conditioned CSV paths under <TESTNAME>/
# 3. load_counts() for baseline + conditioned:
#      - schema + bitstring validation
#      - strict length check == n_qubits
#      - normalize counts -> probabilities
# 4. compute_order_and_probs():
#      - compute shells d_H(x)
#      - order outcomes deterministically by (shell, P_B, bitstring)
# 5. stack_from_bottom() to form cumulative intervals
# 6. assign_colors() to encode shells and within-shell intensity
# 7. plot_mass_flow() writes final PNG next to this script
# --------------------------------------------------------------

# --------------------------------------------------------------
# REPRODUCIBILITY CONTROLS
# --------------------------------------------------------------
# Determinism is enforced via:
# - Fixed input data files selected only by --analyze
# - Strict validation of qubit length (n_qubits) per test
# - Deterministic ordering:
#     * shell distance d_H(x)
#     * within-shell sort by baseline probability (descending)
#     * stable tiebreak by lexicographic bitstring
# - Fixed figure geometry, margins, typography, and DPI
# - Headless-safe Matplotlib backend ("Agg")
#
# All adjustable visualization parameters are defined in the
# constants section near the top of this file.
# --------------------------------------------------------------

# --------------------------------------------------------------
# THIRD-PARTY DEPENDENCIES
# --------------------------------------------------------------
# This script depends on:
# - NumPy
# - Pandas
# - Matplotlib
#
# These packages are used under their respective licenses.
# No third-party source code is redistributed in this file.
# --------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import argparse
import colorsys

import numpy as np
import pandas as pd

# Headless-safe backend for servers/CI. Must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# ============================================================
# QE / TQE Single-Column Figure Standard (BAKED IN)
# ============================================================
FIG_W_IN = 7.03
FIG_H_IN = 4.03
SAVE_DPI = 300

TITLE_PT = 12
LABEL_PT = 10
TICK_PT = 8
ANNOTATION_PT = 8

MARGINS = dict(left=0.11, right=0.98, top=0.90, bottom=0.16)

# ============================================================
# TEST CONFIG (must match released data layout)
# ============================================================
TEST_CONFIG: dict[str, dict[str, object]] = {
    "15Q-MAIN": {
        "shots_expected": 10_000,
        "n_qubits": 15,
        "baseline_file": "15Q-MAIN - Shot Count - Baseline (d3kmirodd19c73966ud0).csv",
        "conditioned_file": "15Q-MAIN - Shot Count - Conditioned (d3kmis0dd19c73966udg).csv",
    },
    "15Q-MCE": {
        "shots_expected": 5_000,
        "n_qubits": 15,
        "baseline_file": "15Q-MCE - Shot Count - Baseline (d3kn6oj4kkus739bud1g).csv",
        "conditioned_file": "15Q-MCE - Shot Count - Conditioned (d3kn6oj4kkus739bud20).csv",
    },
    "20Q-SBP": {
        "shots_expected": 5_000,
        "n_qubits": 20,
        "baseline_file": "20Q-SBP - Shot Count - Baseline (d3knd903qtks738bjjdg).csv",
        "conditioned_file": "20Q-SBP - Shot Count - Conditioned (d3knd91fk6qs73e65s00).csv",
    },
    "10Q-CBP": {
        "shots_expected": 2_000,
        "n_qubits": 10,
        "baseline_file": "10Q-CBP - Shot Count - Baseline (d3nf1603qtks738eack0).csv",
        "conditioned_file": "10Q-CBP - Shot Count - Conditioned (d3nf1603qtks738eackg).csv",
    },
}

# ----------------------------
# Helper: robust bitstring cleanup (matches geometry script behavior)
# ----------------------------
def clean_bitstring(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
    elif s.startswith("'"):
        s = s[1:]
    elif s.endswith("'"):
        s = s[:-1]
    # also strip double-quotes if present
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    elif s.startswith('"'):
        s = s[1:]
    elif s.endswith('"'):
        s = s[:-1]
    return s.strip()

# ----------------------------
# Helper: make Matplotlib accept RGBA reliably
# ----------------------------
def as_rgba_list(rgba: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """
    Some Matplotlib versions treat an (r,g,b,a) tuple as an iterable of colors.
    Wrapping it in a list-of-one prevents it from iterating the floats.
    Also clamps channels to [0,1] to avoid tiny floating overshoots.
    """
    r, g, b, a = rgba
    r = min(max(float(r), 0.0), 1.0)
    g = min(max(float(g), 0.0), 1.0)
    b = min(max(float(b), 0.0), 1.0)
    a = min(max(float(a), 0.0), 1.0)
    return [(r, g, b, a)]

# ----------------------------
# Core GHZ distance (for shell assignment)
# ----------------------------
def wt(x: str) -> int:
    return x.count("1")

def dist_to_ghz(x: str) -> int:
    w = wt(x)
    n = len(x)
    return min(w, n - w)

# ----------------------------
# Load + validate + normalize (production-grade)
# ----------------------------
def load_counts(csv_path: Path, n_qubits: int) -> tuple[pd.DataFrame, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Bitstring", "Count"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path.name}: expected columns {sorted(required)}; got {df.columns.tolist()}"
        )

    df = df.copy()
    df["Bitstring"] = df["Bitstring"].map(clean_bitstring)

    # Only keep valid 0/1 strings
    df = df[df["Bitstring"].str.fullmatch(r"[01]+", na=False)].copy()
    if df.empty:
        raise ValueError(f"{csv_path.name}: no valid bitstrings after cleaning/filtering.")

    lengths = df["Bitstring"].str.len()
    if not (lengths == n_qubits).all():
        bad = df[lengths != n_qubits].iloc[0]["Bitstring"]
        raise ValueError(
            f"{csv_path.name}: bitstring length mismatch. Example: {bad} (len={len(bad)})"
        )

    counts = pd.to_numeric(df["Count"], errors="raise").astype(int).to_numpy()
    if np.any(counts < 0):
        raise ValueError(f"{csv_path.name}: negative counts found.")

    total = int(counts.sum())
    if total <= 0:
        raise ValueError(f"{csv_path.name}: total counts is {total}")

    df["Count"] = counts
    df["Prob"] = df["Count"] / float(total)

    # Normalize by bitstring in case the CSV has duplicates (production robustness)
    df = df.groupby("Bitstring", as_index=False)["Prob"].sum()

    print(f"[load] {csv_path.name}: unique_states={len(df)}, total_shots={total}")
    return df[["Bitstring", "Prob"]], total

# ----------------------------
# Ordering + stacking
# ----------------------------
def compute_order_and_probs(
    b: pd.DataFrame,
    c: pd.DataFrame,
    n_qubits: int,
) -> tuple[list[str], pd.Series, pd.Series, str, str]:
    states = sorted(set(b["Bitstring"]).union(set(c["Bitstring"])))
    zeros = "0" * n_qubits
    ones = "1" * n_qubits

    pb = b.set_index("Bitstring")["Prob"].reindex(states).fillna(0.0)
    pc = c.set_index("Bitstring")["Prob"].reindex(states).fillna(0.0)

    by_d: dict[int, list[str]] = {}
    for s in states:
        by_d.setdefault(dist_to_ghz(s), []).append(s)

    order: list[str] = []
    for d in sorted(by_d.keys()):
        group = by_d[d]
        # Deterministic: sort by baseline prob desc, then by bitstring as stable tiebreak
        group_sorted = sorted(
            group,
            key=lambda s: (float(pb.get(s, 0.0)), s),
            reverse=True
        )

        if d == 0:
            # Force endpoints first if present
            group_sorted = [x for x in (zeros, ones) if x in group_sorted] + [
                s for s in group_sorted if s not in {zeros, ones}
            ]

        order.extend(group_sorted)

    pb = pb.reindex(order)
    pc = pc.reindex(order)
    return order, pb, pc, zeros, ones

def stack_from_bottom(p: pd.Series) -> dict[str, tuple[float, float]]:
    y = 0.0
    out: dict[str, tuple[float, float]] = {}
    for k, v in p.items():
        out[k] = (y, y + float(v))
        y += float(v)
    return out

# ----------------------------
# Coloring
# ----------------------------
def shade_family(base_rgb, t: float, alpha: float) -> tuple[float, float, float, float]:
    r, g, b = base_rgb[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l_min, l_max = 0.16, 0.96
    l2 = l_min + t * (l_max - l_min)

    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)

    r2 = min(max(r2, 0.0), 1.0)
    g2 = min(max(g2, 0.0), 1.0)
    b2 = min(max(b2, 0.0), 1.0)
    a2 = min(max(float(alpha), 0.0), 1.0)

    return (r2, g2, b2, a2)

def assign_colors(
    order: list[str],
    pb: pd.Series,
    zeros: str,
    ones: str,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[int, tuple[float, float, float]]]:
    colors: dict[str, tuple[float, float, float, float]] = {
        zeros: to_rgba("#2ca02c"),  # green
        ones:  to_rgba("#d62728"),  # red
    }

    unique_ds = sorted({dist_to_ghz(s) for s in order if s not in {zeros, ones}})

    tab10 = plt.get_cmap("tab10").colors
    SHELL_IDXS = [0, 1, 4, 5, 6, 7, 9]  # blue, orange, purple, brown, pink, gray, cyan
    available_shell_colors = [tab10[i] for i in SHELL_IDXS]
    d_to_color = {d: available_shell_colors[i % len(available_shell_colors)] for i, d in enumerate(unique_ds)}

    for d in unique_ds:
        members = [s for s in order if (s not in {zeros, ones} and dist_to_ghz(s) == d)]
        if not members:
            continue

        base = d_to_color[d]
        members = sorted(members, key=lambda s: (float(pb.get(s, 0.0)), s), reverse=True)
        probs = np.array([float(pb.get(s, 0.0)) for s in members], dtype=float)

        pmin, pmax = probs.min(), probs.max()
        if pmax > pmin:
            u = (probs - pmin) / (pmax - pmin)
        else:
            u = np.zeros_like(probs)

        alpha_min = 0.10
        alpha_max_default = 0.78
        alpha_max_d1 = 0.60
        alpha_max = alpha_max_d1 if d == 1 else alpha_max_default

        for s, ui in zip(members, u):
            t_light = 1.0 - ui
            alpha = alpha_min + (alpha_max - alpha_min) * ui
            colors[s] = shade_family(base, t=t_light, alpha=alpha)

    return colors, d_to_color

# ----------------------------
# Plot
# ----------------------------
def plot_mass_flow(
    order: list[str],
    pb: pd.Series,
    pc: pd.Series,
    int_b: dict[str, tuple[float, float]],
    int_c: dict[str, tuple[float, float]],
    colors: dict[str, tuple[float, float, float, float]],
    d_to_color: dict[int, tuple[float, float, float]],
    zeros: str,
    ones: str,
    out_png: Path,
    title: str,
) -> None:
    plt.rcParams.update({
        "figure.figsize": (FIG_W_IN, FIG_H_IN),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",

        "axes.titlesize": TITLE_PT,
        "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,

        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.8,
        "grid.color": "#B0B0B0",

        "legend.frameon": False,
        "savefig.dpi": SAVE_DPI,
    })

    fig, ax = plt.subplots()
    ax.grid(False)

    xL0, xL1 = 0.018, 0.10
    xR0, xR1 = 0.70, 0.85

    bar_w = xL1 - xL0
    xR1 = xR0 + bar_w

    ax.axvspan(xL0 - 0.01, xL0, ymin=0.0, ymax=0.89, color="#cfe3f7", alpha=0.9, linewidth=0)
    ax.axvspan(xR1, xR1 + 0.01, ymin=0.0, ymax=0.89, color="#fde2c7", alpha=0.9, linewidth=0)

    for s in order:
        y0, y1 = int_b[s]
        ax.fill_between([xL0, xL1], [y0, y0], [y1, y1],
                        color=as_rgba_list(colors[s]), alpha=0.70, linewidth=0)

    for s in order:
        y0, y1 = int_c[s]
        ax.fill_between([xR0, xR1], [y0, y0], [y1, y1],
                        color=as_rgba_list(colors[s]), alpha=0.70, linewidth=0)

    xs = np.linspace(xL1, xR0, 60)
    for s in order:
        yLb0, yLb1 = int_b[s]
        yRc0, yRc1 = int_c[s]
        flow_alpha = float(colors[s][3])
        ax.fill_between(xs,
                        np.linspace(yLb0, yRc0, xs.size),
                        np.linspace(yLb1, yRc1, xs.size),
                        color=as_rgba_list(colors[s]), alpha=flow_alpha, linewidth=0)

    # Ultra-subtle top-shell fade cue (no text)
    y_fade0 = 0.72
    y_fade1 = 1.00
    H = 256
    W = 2
    alpha_top = 0.12
    alpha = np.linspace(0.0, alpha_top, H).reshape(H, 1)
    alpha = np.repeat(alpha, W, axis=1)

    overlay = np.ones((H, W, 4), dtype=float)  # white RGBA
    overlay[..., 3] = alpha

    ax.imshow(
        overlay,
        extent=(0.0, 1.0, y_fade0, y_fade1),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=2.0,
        clip_on=True
    )

    fig.text(
        0.5, 0.955,
        "Distance-Binned Probability Mass Composition",
        ha="center", va="top",
        fontsize=TITLE_PT
    )

    ax.set_ylabel("Probability mass", fontsize=LABEL_PT, labelpad=10)

    ax.text((xL0 + xL1) / 2, -0.06, "Baseline",
            transform=ax.transAxes, ha="center", va="top", fontsize=LABEL_PT)
    ax.text((xR0 + xR1) / 2, -0.06, "Conditioned",
            transform=ax.transAxes, ha="center", va="top", fontsize=LABEL_PT)

    ax.set_xlim(0, 1)
    Y_TOP = 1.12
    ax.set_ylim(0, Y_TOP)

    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])

    x_orange_right = xR1 + 0.01
    ax.hlines(y=1.0, xmin=0.0, xmax=x_orange_right,
              linestyles="--", linewidth=1.0, color="#777777", alpha=0.8)

    # Total probability mass label only (no arrows)
    y_arrow = 1.065
    x_center = 0.5 * (xL0 + xR1)

    ax.text(x_center, y_arrow, r"Σ_x P(x) = 1 (each execution class)",
            ha="center", va="center", fontsize=ANNOTATION_PT,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


    # Bracket
    x_orange_axes = ax.transAxes.inverted().transform(ax.transData.transform((x_orange_right, 0.0)))[0]
    gap_br = 0.020
    cap = 0.010
    x_br = x_orange_axes + gap_br

    tr = blended_transform_factory(ax.transAxes, ax.transData)
    br_color = "#555555"
    br_lw = 1.0

    ax.plot([x_br, x_br], [0.0, 1.0], transform=tr, color=br_color, lw=br_lw, clip_on=False)
    ax.plot([x_br, x_br - cap], [1.0, 1.0], transform=tr, color=br_color, lw=br_lw, clip_on=False)
    ax.plot([x_br, x_br - cap], [0.0, 0.0], transform=tr, color=br_color, lw=br_lw, clip_on=False)

    # Legend
    unique_ds = sorted({dist_to_ghz(s) for s in order if s not in {zeros, ones}})
    handles = []
    for d in sorted(unique_ds, reverse=True):
        handles.append(Patch(facecolor=d_to_color[d], edgecolor="none", alpha=0.70, label=fr"$d_H = {d}$"))

    handles.append(Patch(facecolor=as_rgba_list(colors[ones])[0], edgecolor="none", alpha=0.70, label=r"$|1\ldots1\rangle$"))
    handles.append(Patch(facecolor=as_rgba_list(colors[zeros])[0], edgecolor="none", alpha=0.70, label=r"$|0\ldots0\rangle$"))

    ax.legend(handles=handles,
              loc="center left",
              bbox_to_anchor=(x_br + 0.05, 0.50),
              borderaxespad=0.0,
              fontsize=TICK_PT,
              handlelength=1.0, handleheight=1.0,
              handletextpad=0.6, labelspacing=0.6)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.spines["left"].set_bounds(0.0, 1.0)
    ax.spines["bottom"].set_bounds(0.0, xR1 + 0.01)

    ax.spines["left"].set_position(("axes", 0.005))
    ax.tick_params(axis="y", which="major", pad=0)

    fig.subplots_adjust(**MARGINS)

    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"[save] {out_png}")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyze",
        required=True,
        choices=tuple(TEST_CONFIG.keys()),
        help="Select dataset to analyze."
    )
    args = parser.parse_args()

    testname = args.analyze
    cfg = TEST_CONFIG[testname]
    n_qubits = int(cfg["n_qubits"])
    shots_expected = int(cfg["shots_expected"])

    here = Path(__file__).resolve().parent
    print(f"[cwd]  {Path.cwd()}")
    print(f"[file] {here}")

    test_dir = here / testname
    baseline_csv = test_dir / str(cfg["baseline_file"])
    conditioned_csv = test_dir / str(cfg["conditioned_file"])

    b, total_b = load_counts(baseline_csv, n_qubits)
    c, total_c = load_counts(conditioned_csv, n_qubits)

    if total_b != shots_expected:
        print(f"[warn] baseline total_shots={total_b} != expected {shots_expected} for {testname}")
    if total_c != shots_expected:
        print(f"[warn] conditioned total_shots={total_c} != expected {shots_expected} for {testname}")

    order, pb, pc, zeros, ones = compute_order_and_probs(b, c, n_qubits)
    int_b = stack_from_bottom(pb)
    int_c = stack_from_bottom(pc)

    colors, d_to_color = assign_colors(order, pb, zeros=zeros, ones=ones)

    title = f"[{testname}] - Distance-Binned Probability Mass Composition"
    out_png = here / f"{title}.png"

    plot_mass_flow(
        order=order,
        pb=pb,
        pc=pc,
        int_b=int_b,
        int_c=int_c,
        colors=colors,
        d_to_color=d_to_color,
        zeros=zeros,
        ones=ones,
        out_png=out_png,
        title=title,
    )

if __name__ == "__main__":
    main()
