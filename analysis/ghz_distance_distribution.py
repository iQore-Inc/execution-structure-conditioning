#!/usr/bin/env python3
# --------------------------------------------------------------
# Reproducibility Integrity Metadata
# Digest      : Generated at release (see repository checksums)
# Timestamp   : 2026-02-09
# Purpose     : Deterministic, tamper-evident reproduction of
#               published GHZ-distance probability-mass analysis
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
# Title   : GHZ Distance Distribution — Reproducibility Script
# File    : ghz_distance_distribution.py
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
# This script is executed via one of:
#
#   --analyze {15Q-MAIN,15Q-MCE,20Q-SBP,10Q-CBP}
#   --compare {TEST_A TEST_B}
#
# Example commands (run from the directory containing this file):
#
#   python ghz_distance_distribution.py --analyze 10Q-CBP
#   python ghz_distance_distribution.py --analyze 15Q-MAIN
#   python ghz_distance_distribution.py --analyze 15Q-MCE
#   python ghz_distance_distribution.py --analyze 20Q-SBP
#
#   python ghz_distance_distribution.py --compare 15Q-MAIN 15Q-MCE
#
# Help:
#   python ghz_distance_distribution.py --help
#
# Expected output (written next to this script):
#   Single-test:
#     "[<TESTNAME>] - Geometric Concentration of Outcomes.pdf"
#     "[<TESTNAME>] - Geometric Concentration of Outcomes.png"
#   Compare:
#     "[COMPARE <TEST_A> vs <TEST_B>] - Geometric Concentration of Outcomes.pdf"
#     "[COMPARE <TEST_A> vs <TEST_B>] - Geometric Concentration of Outcomes.png"
#
# Input data layout (relative to this script):
#   ghz_distance_distribution.py
#   <TESTNAME>/
#     <baseline CSV>
#     <conditioned CSV>
#
# Each CSV must contain columns:
#   - Bitstring : 0/1 string of length n_qubits
#   - Count     : integer >= 0
# --------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

# Headless-safe backend for servers/CI. Must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


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

# ============================================================
# Compare plot styling rules (deterministic)
# ============================================================
# - Tests map to fixed colors (as requested)
# - Baseline is dashed; Conditioned is solid
TEST_COLOR: dict[str, str] = {
    "15Q-MAIN": "tab:blue",
    "15Q-MCE": "tab:orange",
}
BASELINE_LS = "--"
CONDITIONED_LS = "-"


# ============================================================
# Embedded QE figure standard (as provided)
# ============================================================
FigClass = Literal["single", "double"]
Mode = Literal["python", "latex"]


@dataclass(frozen=True)
class FigureSpec:
    width_in: float
    height_in: float
    dpi_png: int = 300

    title_pt: int = 12
    label_pt: int = 10
    tick_pt: int = 8
    legend_pt: int = 10
    panel_label_pt: int = 10

    spine_lw: float = 1.0
    line_lw: float = 1.5
    marker_size: float = 5.0

    grid: bool = True
    grid_axis: str = "y"
    grid_alpha: float = 0.30
    grid_lw: float = 0.8
    grid_color: str = "#B0B0B0"

    left: float = 0.11
    right: float = 0.98
    top: float = 0.88
    bottom: float = 0.16


SINGLE = FigureSpec(width_in=7.03, height_in=4.03)
DOUBLE = FigureSpec(width_in=14.06, height_in=6.00)


def _get_spec(fig_class: FigClass) -> FigureSpec:
    if fig_class == "single":
        return SINGLE
    if fig_class == "double":
        return DOUBLE
    raise ValueError(f"Unknown fig_class={fig_class!r} (expected 'single' or 'double').")


def apply_style(*, mode: Mode = "python", fig_class: FigClass = "single", use_constrained_layout: bool = False) -> FigureSpec:
    """Apply global rcParams for the QE figure standard."""
    spec = _get_spec(fig_class)

    if mode == "python":
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        })
    elif mode == "latex":
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })
    else:
        raise ValueError(f"Unknown mode={mode!r} (expected 'python' or 'latex').")

    mpl.rcParams.update({
        "figure.figsize": (spec.width_in, spec.height_in),
        "figure.dpi": 100,
        "savefig.dpi": spec.dpi_png,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        "axes.titlesize": spec.title_pt,
        "axes.labelsize": spec.label_pt,
        "xtick.labelsize": spec.tick_pt,
        "ytick.labelsize": spec.tick_pt,
        "legend.fontsize": spec.legend_pt,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": spec.spine_lw,

        "axes.grid": spec.grid,
        "axes.axisbelow": True,
        "grid.alpha": spec.grid_alpha,
        "grid.linewidth": spec.grid_lw,
        "grid.color": spec.grid_color,

        "lines.linewidth": spec.line_lw,
        "lines.markersize": spec.marker_size,

        "legend.frameon": False,
        "figure.constrained_layout.use": bool(use_constrained_layout),
    })

    return spec


def layout(fig: Optional[plt.Figure] = None, *, fig_class: FigClass = "single") -> None:
    """Apply standardized margins (use when NOT using constrained_layout)."""
    spec = _get_spec(fig_class)
    target = fig if fig is not None else plt.gcf()
    target.subplots_adjust(left=spec.left, right=spec.right, top=spec.top, bottom=spec.bottom)


def finalize_axes(
    ax: plt.Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid_axis: str = "y",
) -> None:
    """Apply per-axes best practices consistently."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis=grid_axis)
    ax.tick_params(direction="out")


def save(
    fig: plt.Figure,
    outpath_no_ext: str | Path,
    *,
    png: bool = True,
    pdf: bool = True,
    dpi_png: int = 300,
    transparent: bool = False,
) -> None:
    """Save PDF (vector-first) and PNG (300 dpi) deterministically."""
    outpath_no_ext = Path(outpath_no_ext)
    outpath_no_ext.parent.mkdir(parents=True, exist_ok=True)

    if pdf:
        fig.savefig(outpath_no_ext.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
    if png:
        fig.savefig(outpath_no_ext.with_suffix(".png"), dpi=dpi_png, bbox_inches="tight", transparent=transparent)


# ============================================================
# Analysis logic
# ============================================================
def _strip_quotes(s: str) -> str:
    s = str(s).strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def load_shotcount_csv(path: Path, *, n_qubits: int) -> Dict[str, int]:
    """Load {bitstring -> count} from CSV with headers Bitstring,Count (case-insensitive)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: missing header row.")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "bitstring" not in fields or "count" not in fields:
            raise ValueError(f"{path.name}: expected columns Bitstring,Count (found: {reader.fieldnames}).")

        bit_col = fields["bitstring"]
        cnt_col = fields["count"]

        data: Dict[str, int] = {}
        for row in reader:
            b = _strip_quotes(row.get(bit_col, ""))
            if not b:
                continue
            if any(ch not in "01" for ch in b):
                raise ValueError(f"{path.name}: non-binary bitstring encountered: {b!r}")
            if len(b) != n_qubits:
                raise ValueError(f"{path.name}: bitstring length mismatch: {b!r} (len={len(b)}), expected {n_qubits}")

            c_raw = row.get(cnt_col, "")
            try:
                c = int(float(str(c_raw).strip()))
            except Exception as e:
                raise ValueError(f"{path.name}: invalid Count value {c_raw!r}") from e
            if c < 0:
                raise ValueError(f"{path.name}: negative Count for bitstring {b!r}")
            data[b] = data.get(b, 0) + c

    if not data:
        raise ValueError(f"{path.name}: no data rows found.")
    return data


def pmass_by_ghz_distance(shotcounts: Dict[str, int], *, n_qubits: int) -> Tuple[list[float], int]:
    """Return (P(d) for d=0..floor(n/2), total_shots)."""
    d_max = n_qubits // 2
    counts_by_d = [0] * (d_max + 1)
    total = 0

    for b, c in shotcounts.items():
        ones = b.count("1")
        d = ones if ones <= (n_qubits - ones) else (n_qubits - ones)
        counts_by_d[d] += c
        total += c

    if total <= 0:
        raise ValueError("Total shot count is zero.")
    pmass = [cd / total for cd in counts_by_d]
    return pmass, total


def analyze_test(testname: str, *, here: Path) -> Tuple[int, int, list[float], list[float], int, int]:
    """
    Run the existing pipeline for a single test.

    Returns:
      (n_qubits, shots_expected, base_pm, cond_pm, base_total, cond_total)
    """
    cfg = TEST_CONFIG[testname]
    n_qubits = int(cfg["n_qubits"])
    shots_expected = int(cfg["shots_expected"])

    test_dir = here / testname
    baseline_path = test_dir / str(cfg["baseline_file"])
    conditioned_path = test_dir / str(cfg["conditioned_file"])

    base_counts = load_shotcount_csv(baseline_path, n_qubits=n_qubits)
    cond_counts = load_shotcount_csv(conditioned_path, n_qubits=n_qubits)

    base_pm, base_total = pmass_by_ghz_distance(base_counts, n_qubits=n_qubits)
    cond_pm, cond_total = pmass_by_ghz_distance(cond_counts, n_qubits=n_qubits)

    if base_total != shots_expected:
        print(f"[warn] baseline total_shots={base_total} != expected {shots_expected} for {testname}")
    if cond_total != shots_expected:
        print(f"[warn] conditioned total_shots={cond_total} != expected {shots_expected} for {testname}")

    return n_qubits, shots_expected, base_pm, cond_pm, base_total, cond_total


def _color_for_test(testname: str) -> Optional[str]:
    """
    Deterministic color mapping for compare plots.
    If a test isn't listed, return None to let Matplotlib pick a default color.
    """
    return TEST_COLOR.get(testname)


def _apply_clean_legend(ax: plt.Axes, *, tests: Tuple[str, ...], fig_class: FigClass) -> None:
    """
    Single, aligned legend block with two sections:
      Mode: Baseline (dashed), Conditioned (solid)
      Test: color swatches for each test
    """
    spec = _get_spec(fig_class)

    # "Invisible" handle so section headers align with a blank handle column.
    blank = Line2D([], [], linestyle="None", marker=None, linewidth=0, color="none")

    handles = [blank]
    labels = ["Mode"]

    legend_lw = spec.line_lw * 0.5

    # Mode rows
    handles += [
        Line2D([0], [0], color="black", linewidth=legend_lw, linestyle=BASELINE_LS),
        Line2D([0], [0], color="black", linewidth=legend_lw, linestyle=CONDITIONED_LS),
    ]
    labels += ["Baseline", "Conditioned"]



    # Test header
    handles += [blank]
    labels += ["Test"]

    # Test rows
    for t in tests:
        col = _color_for_test(t) or "tab:gray"
        handles.append(Patch(facecolor=col, edgecolor=col))
        labels.append(t)

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        borderaxespad=0.0,
        handlelength=1.0,   # good for dashed/solid samples
        handletextpad=0.8,
        labelspacing=0.4,
    )

    # Make "Mode" and "Test" look like section headers:
    # bold text, and no handle drawn (it's already blank).
    #for txt in leg.get_texts():
    #    if txt.get_text() in {"Mode", "Test"}:
    #       txt.set_weight("bold")

    # Ensure blank handles truly take no visual space
    handles_attr = getattr(leg, "legendHandles", None)
    if handles_attr is None:
        handles_attr = getattr(leg, "legend_handles", None)
    if handles_attr is None:
        handles_attr = []

    for h, lab in zip(handles_attr, labels):
        if lab in {"Mode", "Test", ""}:
            try:
                h.set_visible(False)
            except Exception:
                pass

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analyze",
        choices=tuple(TEST_CONFIG.keys()),
        help="Select dataset to analyze."
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("TEST_A", "TEST_B"),
        choices=tuple(TEST_CONFIG.keys()),
        help="Compare two datasets on one figure."
    )
    ap.add_argument("--mode", type=str, default="python", choices=["python", "latex"], help="Typography mode.")
    ap.add_argument("--fig-class", type=str, default="single", choices=["single", "double"], help="Figure class.")
    args = ap.parse_args()

    # Require exactly one of --analyze or --compare (preserves existing behavior while enabling compare)
    if (args.analyze is None) == (args.compare is None):
        ap.error("Specify exactly one of --analyze TEST or --compare TEST_A TEST_B.")

    here = Path(__file__).resolve().parent
    print(f"[cwd]  {Path.cwd()}")
    print(f"[file] {here}")

    figure_title = "Distance-Binned Mass Profile Comparison Under Distinct Compilation Configurations"

    # --------------------------------------------------------
    # Compare mode
    # --------------------------------------------------------
    if args.compare is not None:
        test_a, test_b = args.compare

        nA, shotsA, baseA, condA, base_totalA, cond_totalA = analyze_test(test_a, here=here)
        nB, shotsB, baseB, condB, base_totalB, cond_totalB = analyze_test(test_b, here=here)

        if nA != nB:
            raise ValueError(f"Cannot compare: n_qubits differ ({test_a}={nA}, {test_b}={nB}).")

        # Style + plot
        apply_style(mode=args.mode, fig_class=args.fig_class)
        fig, ax = plt.subplots()

        xs = list(range(len(baseA)))

        # Color coordinates by test; linestyle encodes baseline vs conditioned
        colA = _color_for_test(test_a)
        colB = _color_for_test(test_b)

        ax.plot(xs, baseA, label=f"{test_a} Baseline", linestyle=BASELINE_LS, color=colA)
        ax.plot(xs, condA, label=f"{test_a} Conditioned", linestyle=CONDITIONED_LS, color=colA)

        ax.plot(xs, baseB, label=f"{test_b} Baseline", linestyle=BASELINE_LS, color=colB)
        ax.plot(xs, condB, label=f"{test_b} Conditioned", linestyle=CONDITIONED_LS, color=colB)

        finalize_axes(
            ax,
            xlabel="Hamming distance from GHZ manifold $d$",
            ylabel="Probability mass $P(d)$",
            title=f"{figure_title}",
            grid_axis="y",
        )
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(bottom=0.0)

        # Single aligned legend block
        _apply_clean_legend(ax, tests=(test_a, test_b), fig_class=args.fig_class)

        layout(fig, fig_class=args.fig_class)

        outname = f"[COMPARE {test_a} vs {test_b}] - {figure_title}"
        outpath = here / outname
        save(fig, outpath, dpi_png=_get_spec(args.fig_class).dpi_png)
        plt.close(fig)

        print(f"[meta] compare_n_qubits={nA}")
        print(f"[meta] {test_a}_baseline_total_shots={base_totalA}")
        print(f"[meta] {test_a}_conditioned_total_shots={cond_totalA}")
        print(f"[meta] {test_b}_baseline_total_shots={base_totalB}")
        print(f"[meta] {test_b}_conditioned_total_shots={cond_totalB}")
        print(f"[save] {outpath.with_suffix('.pdf')}")
        print(f"[save] {outpath.with_suffix('.png')}")
        return

    # --------------------------------------------------------
    # Single-test mode (original behavior)
    # --------------------------------------------------------
    testname = args.analyze
    assert testname is not None  # for type-checkers

    n_qubits, shots_expected, base_pm, cond_pm, base_total, cond_total = analyze_test(testname, here=here)

    # Style + plot
    apply_style(mode=args.mode, fig_class=args.fig_class)
    fig, ax = plt.subplots()

    xs = list(range(len(base_pm)))

    # (test color if mapped; baseline dashed, conditioned solid)
    col = _color_for_test(testname)
    ax.plot(xs, base_pm, label="Baseline", linestyle=BASELINE_LS, color=col)
    ax.plot(xs, cond_pm, label="Conditioned", linestyle=CONDITIONED_LS, color=col)

    finalize_axes(
        ax,
        xlabel="Hamming distance from GHZ manifold $d$",
        ylabel="Probability mass $P(d)$",
        title=figure_title,
        grid_axis="y",
    )
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(bottom=0.0)

    # Single aligned legend block
    _apply_clean_legend(ax, tests=(testname,), fig_class=args.fig_class)

    layout(fig, fig_class=args.fig_class)

    outname = f"[{testname}] - {figure_title}"
    outpath = here / outname
    save(fig, outpath, dpi_png=_get_spec(args.fig_class).dpi_png)
    plt.close(fig)

    print(f"[meta] n_qubits={n_qubits}")
    print(f"[meta] baseline_total_shots={base_total}")
    print(f"[meta] conditioned_total_shots={cond_total}")
    print(f"[save] {outpath.with_suffix('.pdf')}")
    print(f"[save] {outpath.with_suffix('.png')}")


if __name__ == "__main__":
    main()
