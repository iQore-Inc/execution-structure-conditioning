# --------------------------------------------------------------
# Reproducibility Integrity Metadata
# Digest      : Generated at release (see repository checksums)
# Timestamp   : 2025–2026
# Purpose     : Deterministic, tamper-evident reproduction of
#               published execution-geometry analysis
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
# Title   : GHZ Manifold Distance Geometry — Reproducibility Script
# File    : ghz_geometry.py
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
#   python ghz_geometry.py --analyze 10Q-CBP
#   python ghz_geometry.py --analyze 15Q-MAIN
#   python ghz_geometry.py --analyze 15Q-MCE
#   python ghz_geometry.py --analyze 20Q-SBP
#
# Help:
#   python ghz_geometry.py --help
#
# Expected output (written next to this script):
#   "[<TESTNAME>] - Execution-Dependent Geometry of Near-Manifold Outcome Resolution.png"
#
# Input data layout (relative to this script):
#   ghz_geometry.py
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
# This script reproduces the execution-dependent geometry analysis
# presented in the paper, focusing on the distribution of outcomes
# near the GHZ manifold.
#
# For an n-qubit measurement outcome x, the distance from the GHZ
# manifold is defined as:
#
#     d(x) = min(wt(x), n - wt(x))
#
# where wt(x) is the Hamming weight of bitstring x.
#
# Using empirical shot-count data, the script:
#   1. Computes d(x) for all observed bitstrings
#   2. Expands counts into per-shot distance samples
#   3. Forms resampled shot batches via:
#        - partition (shuffle + block)
#        - bootstrap (sampling with replacement)
#   4. Estimates P(d) per batch
#   5. Renders paired 3D surfaces (Baseline vs Conditioned)
#
# The resulting figures correspond to the execution-geometry
# visualizations reported in the paper.
# --------------------------------------------------------------

# --------------------------------------------------------------
# SYSTEM OVERVIEW
# --------------------------------------------------------------
# Analysis Modules:
# ┌────────────────────────────────────────────────────────────┐
# │ 1. **Data Ingestion**                                      │
# │    - Loads empirical shot-count CSVs                       │
# │    - Validates bitstrings and qubit counts                 │
# │                                                            │
# │ 2. **Distance Mapping**                                    │
# │    - Computes GHZ-manifold distance d(x)                   │
# │    - Expands counts into per-shot samples                  │
# │                                                            │
# │ 3. **Resampling Engine**                                   │
# │    - Partition or bootstrap resampling                     │
# │    - Seeded RNG for deterministic reproduction             │
# │                                                            │
# │ 4. **Probability Estimation**                              │
# │    - Computes P(d) across resampled shot batches           │
# │                                                            │
# │ 5. **Visualization Layer**                                 │
# │    - Side-by-side 3D surface plots                         │
# │    - Fixed canvas, typography, and color normalization     │
# └────────────────────────────────────────────────────────────┘
# --------------------------------------------------------------

# --------------------------------------------------------------
# EXECUTION FLOW
# --------------------------------------------------------------
# 1. parse CLI args (--analyze)
# 2. load_counts() for baseline + conditioned CSVs
# 3. counts_to_shot_distances() expands counts into per-shot samples
# 4. build_Pd_blocks() resamples shots and estimates P(d) per block
# 5. plot_side_by_side() writes the final figure PNG
#
# Resampling modes (CONFIG: MODE):
# - Partition  : shuffle once, split into non-overlapping blocks
# - Bootstrap  : sample with replacement per block
# --------------------------------------------------------------

# --------------------------------------------------------------
# REPRODUCIBILITY CONTROLS
# --------------------------------------------------------------
# Determinism is enforced via:
# - Explicit RNG seed (CONFIG: SEED)
# - Fixed resampling strategy (CONFIG: MODE)
# - Fixed figure canvas size, DPI, and normalization rules
#
# All adjustable parameters are defined in the CONFIG section
# near the top of this file.
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

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# ============================
# CONFIG
# ============================
MODE = "partition"             # "partition" or "bootstrap"
BLOCK_SIZE = None              # None -> auto: total_shots // N_BLOCKS
SEED = 7
VMAX_FIXED = None            # None -> auto max across both surfaces, or set e.g. 0.8
SHOW = False                   # set True to display windows (optional)

# ============================
# Core geometry
# ============================
def clean_bitstring(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
    elif s.startswith("'"):
        s = s[1:]
    elif s.endswith("'"):
        s = s[:-1]
    return s.strip()


def ghz_distance(bitstring: str) -> int:
    wt = bitstring.count("1")
    n = len(bitstring)
    return min(wt, n - wt)


# ============================
# Load empirical counts
# ============================
def load_counts(csv_path: Path, n_qubits: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Bitstring", "Count"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path.name}: expected columns {sorted(required)}; got {df.columns.tolist()}")

    df["Bitstring"] = df["Bitstring"].map(clean_bitstring)
    df = df[df["Bitstring"].str.fullmatch(r"[01]+", na=False)].copy()
    if df.empty:
        raise ValueError(f"{csv_path.name}: no valid bitstrings after cleaning/filtering.")

    lengths = df["Bitstring"].str.len()
    if not (lengths == n_qubits).all():
        bad = df[lengths != n_qubits].iloc[0]["Bitstring"]
        raise ValueError(f"{csv_path.name}: bitstring length mismatch. Example: {bad} (len={len(bad)})")

    counts = df["Count"].astype(int).to_numpy()
    if np.any(counts < 0):
        raise ValueError(f"{csv_path.name}: negative counts found.")

    total = int(counts.sum())
    if total <= 0:
        raise ValueError(f"{csv_path.name}: total counts is {total}")

    distances = np.array([ghz_distance(b) for b in df["Bitstring"].to_numpy()], dtype=int)

    print(f"[load] {csv_path.name}: unique_states={len(df)}, total_shots={total}")
    return distances, counts, total


def counts_to_shot_distances(distances_per_row: np.ndarray, counts_per_row: np.ndarray) -> np.ndarray:
    return np.repeat(distances_per_row, counts_per_row)

# ============================
# Build replication axis
# ============================
def build_Pd_blocks(
    shot_distances: np.ndarray,
    n_qubits: int,
    n_blocks: int,
    block_size: int | None,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_d = n_qubits // 2
    Pd = np.zeros((max_d + 1, n_blocks), dtype=float)

    total_shots = int(shot_distances.size)
    if block_size is None:
        block_size = total_shots // n_blocks
        if block_size <= 0:
            raise ValueError("Computed block_size is 0. Reduce n_blocks or provide block_size explicitly.")

    print(f"[blocks] mode={mode}, n_blocks={n_blocks}, block_size={block_size}, total_shots={total_shots}")

    if mode not in {"partition", "bootstrap"}:
        raise ValueError("mode must be 'partition' or 'bootstrap'")

    if mode == "partition":
        usable = block_size * n_blocks
        if usable > total_shots:
            raise ValueError(
                f"Partition needs {usable} shots but only have {total_shots}. "
                f"Reduce block_size or n_blocks."
            )

        shuffled = shot_distances.copy()
        rng.shuffle(shuffled)
        shuffled = shuffled[:usable].reshape(n_blocks, block_size)

        for b in range(n_blocks):
            bc = np.bincount(shuffled[b], minlength=max_d + 1).astype(float)
            Pd[:, b] = bc / bc.sum()

    else:  # bootstrap
        for b in range(n_blocks):
            samp = rng.choice(shot_distances, size=block_size, replace=True)
            bc = np.bincount(samp, minlength=max_d + 1).astype(float)
            Pd[:, b] = bc / bc.sum()

    return Pd

# ============================
# Plotting (UPDATED to match Fig 1–2: canvas, margins, typography)
# ============================
def plot_side_by_side(P_base: np.ndarray, P_cond: np.ndarray, outpath: Path, title: str, vmax_fixed: float):
    print("[matplotlib]", matplotlib.__version__)

    # --- Enforce Fig 1–2 pixel canvas WITHOUT shrinking physical size ---
    TARGET_W_PX, TARGET_H_PX = 1948, 935
    DPI = 160
    FIGSIZE = (TARGET_W_PX / DPI, TARGET_H_PX / DPI)

    # SINGLE KNOB (everything except title)
    BASE_FS = 10
    TICK_FS = BASE_FS - 2  # keep ticks slightly smaller; adjust if you want equal

    # Serif only + font sizing (NO invalid rcParams)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": BASE_FS,          # general text default
        "axes.labelsize": BASE_FS,     # x/y/z label sizes
        "xtick.labelsize": TICK_FS,    # x tick labels
        "ytick.labelsize": TICK_FS,    # y tick labels
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    d_vals = np.arange(P_base.shape[0])
    blocks = np.arange(P_base.shape[1])

    D = np.repeat(d_vals[:, None], len(blocks), axis=1)
    B = np.repeat(blocks[None, :], len(d_vals), axis=0)

    norm = Normalize(vmin=0.0, vmax=float(vmax_fixed), clip=True)
    cmap = cm.inferno

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.suptitle(title, fontsize=12, y=0.965)  # title stays fixed

    gs = GridSpec(
        nrows=2, ncols=2,
        figure=fig,
        left=0.03, right=0.84,
        bottom=0.03, top=1.1,
        wspace=0.10,
        hspace=0.02,
        height_ratios=[0.94, 0.06]
    )

    def style_3d(ax):
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 1.0
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)

        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))

        # IMPORTANT: 3D z tick label size must be set per-axis (no ztick.labelsize rcParam)
        ax.zaxis.set_tick_params(labelsize=TICK_FS)

    def draw(ax, Z):
        facecolors = cmap(norm(Z))
        ax.plot_surface(
            D, B, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False
        )

        ax.set_xlabel("Hamming Distance from GHZ Manifold, d(x)", labelpad=12)
        ax.set_ylabel("Resampled Shot Batch Index", labelpad=14)
        ax.set_zlabel("P(d)", labelpad=10)

        ax.set_xlim(d_vals.min(), d_vals.max())
        ax.set_ylim(blocks.min(), blocks.max())
        ax.set_zlim(0.0, float(vmax_fixed))
        ax.set_yticks([0, 50, 100] if blocks.max() >= 100 else [blocks.min(), blocks.max()])
        ax.view_init(elev=25, azim=-65)

        style_3d(ax)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    draw(ax1, P_base)
    ax1.set_zlabel("")   # remove P(d) label from LEFT plot only

    label_fs = plt.rcParams["axes.labelsize"]
    ax1.text2D(
        0.87, 0.85, "Baseline",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=label_fs,
        fontweight="normal"
    )

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    draw(ax2, P_cond)
    ax2.text2D(
        0.82, 0.85, "Conditioned",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=label_fs,
        fontweight="normal"
    )

    # Bottom labels (kept so your layout safety math remains valid)
    ax1_label = fig.add_subplot(gs[1, 0])
    ax2_label = fig.add_subplot(gs[1, 1])
    for axl in (ax1_label, ax2_label):
        axl.set_axis_off()

    # Colorbar
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    m.set_clim(0.0, float(vmax_fixed))

    pos = ax2.get_position()
    CB_X = 0.905
    CB_WIDTH = 0.010
    CB_HEIGHT_SCALE = 0.60

    cb_height = pos.height * CB_HEIGHT_SCALE
    cb_y = pos.y0 + 0.5 * (pos.height - cb_height)

    cax = fig.add_axes([CB_X, cb_y, CB_WIDTH, cb_height])

    cb = fig.colorbar(m, cax=cax)
    cb.set_label("P(d)")  # will use axes.labelsize by default
    cb.ax.tick_params(labelsize=TICK_FS)

    # --- PIN (ax1, ax2, colorbar) UPWARD to remove whitespace under title ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    st = fig._suptitle
    st_bb = st.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    y_title_bottom = st_bb.y0

    p1 = ax1.get_position()
    p2 = ax2.get_position()
    pc = cax.get_position()

    y_plot_bottom = min(p1.y0, p2.y0, pc.y0)
    y_plot_top    = max(p1.y1, p2.y1, pc.y1)

    TITLE_GAP = 0.008
    dy = (y_title_bottom - TITLE_GAP) - y_plot_top

    lab1 = ax1_label.get_position()
    lab2 = ax2_label.get_position()
    y_labels_top = max(lab1.y1, lab2.y1)
    LABEL_GAP = 0.010

    dy_min = (y_labels_top + LABEL_GAP) - y_plot_bottom
    if dy < dy_min:
        dy = dy_min

    def shift_y(pos_, dy_):
        return [pos_.x0, pos_.y0 + dy_, pos_.width, pos_.height]

    ax1.set_position(shift_y(p1, dy))
    ax2.set_position(shift_y(p2, dy))
    cax.set_position(shift_y(pc, dy))

    fig.savefig(outpath, dpi=DPI)
    print(f"[save] {outpath}")

    if SHOW:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyze",
        required=True,
        choices=("15Q-MAIN", "15Q-MCE", "20Q-SBP", "10Q-CBP"),
        help="Select dataset to analyze."
    )
    args = parser.parse_args()

    TEST_CONFIG = {
        "15Q-MAIN": {
            "shots_expected": 10_000,
            "n_qubits": 15,
            "n_blocks": 100,
            "baseline_file": "15Q-MAIN - Shot Count - Baseline (d3kmirodd19c73966ud0).csv",
            "conditioned_file": "15Q-MAIN - Shot Count - Conditioned (d3kmis0dd19c73966udg).csv",
        },
        "15Q-MCE": {
            "shots_expected": 5_000,
            "n_qubits": 15,
            "n_blocks": 50,
            "baseline_file": "15Q-MCE - Shot Count - Baseline (d3kn6oj4kkus739bud1g).csv",
            "conditioned_file": "15Q-MCE - Shot Count - Conditioned (d3kn6oj4kkus739bud20).csv",
        },
        "20Q-SBP": {
            "shots_expected": 5_000,
            "n_qubits": 20,
            "n_blocks": 50,
            "baseline_file": "20Q-SBP - Shot Count - Baseline (d3knd903qtks738bjjdg).csv",
            "conditioned_file": "20Q-SBP - Shot Count - Conditioned (d3knd91fk6qs73e65s00).csv",
        },
        "10Q-CBP": {
            "shots_expected": 2_000,
            "n_qubits": 10,
            "n_blocks": 20,
            "baseline_file": "10Q-CBP - Shot Count - Baseline (d3nf1603qtks738eack0).csv",
            "conditioned_file": "10Q-CBP - Shot Count - Conditioned (d3nf1603qtks738eackg).csv",
        },
    }

    testname = args.analyze
    n_qubits = int(TEST_CONFIG[testname]["n_qubits"])
    n_blocks = int(TEST_CONFIG[testname]["n_blocks"])
    shots_expected = int(TEST_CONFIG[testname]["shots_expected"])

    here = Path(__file__).resolve().parent
    print(f"[cwd]  {Path.cwd()}")
    print(f"[file] {here}")

    # Data are stored in a per-test subdirectory:
    # analysis/python code/<TESTNAME>/
    test_dir = here / testname

    baseline_csv = test_dir / TEST_CONFIG[testname]["baseline_file"]
    conditioned_csv = test_dir / TEST_CONFIG[testname]["conditioned_file"]

    dist_b, counts_b, total_b = load_counts(baseline_csv, n_qubits)
    dist_c, counts_c, total_c = load_counts(conditioned_csv, n_qubits)

    if total_b != shots_expected:
        print(f"[warn] baseline total_shots={total_b} != expected {shots_expected} for {testname}")
    if total_c != shots_expected:
        print(f"[warn] conditioned total_shots={total_c} != expected {shots_expected} for {testname}")

    shots_b = counts_to_shot_distances(dist_b, counts_b)
    shots_c = counts_to_shot_distances(dist_c, counts_c)

    P_base = build_Pd_blocks(shots_b, n_qubits, n_blocks, BLOCK_SIZE, MODE, SEED)
    P_cond = build_Pd_blocks(shots_c, n_qubits, n_blocks, BLOCK_SIZE, MODE, SEED)

    print("[raw full-dist expected] baseline PGHZ=0.5964, conditioned PGHZ=0.7098 (from report)")
    print("[check] mean P(d=0) baseline   =", P_base[0, :].mean())
    print("[check] mean P(d=0) conditioned=", P_cond[0, :].mean())
    print("[check] max  P(d=0) baseline   =", P_base[0, :].max())
    print("[check] max  P(d=0) conditioned=", P_cond[0, :].max())

    vmax = VMAX_FIXED
    if vmax is None:
        vmax = float(max(P_base.mean(axis=1).max(), P_cond.mean(axis=1).max()))
        vmax *= 1.02

    print(f"[norm] vmin=0.0 vmax={vmax:.6f}")

    title = f"[{testname}] - Execution-Dependent Geometry of Near-Manifold Outcome Resolution"
    outfile_combined = f"{title}.png"

    plot_side_by_side(
        P_base,
        P_cond,
        here / outfile_combined,
        title,
        vmax_fixed=vmax
    )

if __name__ == "__main__":
    main()
