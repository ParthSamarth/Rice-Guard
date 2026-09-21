"""
evaluation/plot_style.py

Shared matplotlib styling + a handful of chart helpers reused by every
script under evaluation/generate_*_plots.py, so the whole visualization
suite (YOLO, classifier, pipeline, final dashboard) reads as one consistent
set of figures rather than four differently-styled ones. Nothing in this
file computes or looks up a metric -- it only draws numbers it is given.

Class-to-color mapping is fixed and shared across every chart in the suite
(CLASS_COLORS below) specifically so "Blast" is always the same color
whether it appears in a YOLO chart, a classifier chart, or the pipeline --
color encodes class identity consistently, never chart-local ordering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated (scripts/validate_palette.js, light surface #fcfcfb) 6-slot
# categorical palette -- all pairs clear the CVD/contrast/chroma checks.
CLASS_COLORS = {
    "Healthy": "#2E8B57",
    "Blast": "#2A6FB0",
    "Brown Spot": "#A6631B",
    "Leaf Smut": "#7B5AA6",
    "Rice Tungro": "#C0472B",
    "Sheath Blight": "#0E9AA0",
}
YOLO_CLASSES = ["Blast", "Brown Spot", "Leaf Smut", "Rice Tungro", "Sheath Blight"]
CNN_CLASSES = ["Healthy", "Blast", "Brown Spot", "Leaf Smut", "Rice Tungro", "Sheath Blight"]

# Status colors -- reserved, never reused as a class color (dataviz good/warn/critical convention).
STATUS_COLORS = {
    "ok": "#2E8B57",
    "model_disagreement": "#B07A31",
    "low_confidence": "#8A6FB5",
    "error": "#B23A3A",
}

ACCENT_FINAL = "#1E7A52"     # converged / selected-as-final
ACCENT_PARTIAL = "#B07A31"   # partial / incomplete / not selected
ACCENT_COND = "#3B6E96"      # conditional metric (e.g. confident-only accuracy)

INK = "#1A2420"
MUTED = "#55625B"
FAINT = "#869086"
GRID = "#D9DED8"
SURFACE = "#FFFFFF"
BG = "#F4F6F3"

FONT_STACK = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def apply_style():
    plt.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 13.5,
        "axes.titlepad": 14,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
    })


apply_style()


def new_figure(figsize=(8, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(fig, out_stem: Path, formats=("png", "svg")) -> list:
    """Saves fig as PNG (crisp, PPT-ready) and SVG (publication quality),
    creating parent directories as needed. Returns the list of paths written."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_stem.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def annotate_source(ax, text: str):
    """Small provenance caption pinned to the bottom-right CORNER OF THE
    FIGURE (not the axes) -- every chart in this suite names the file it was
    computed from. Figure-fraction, not axes-fraction: a confusion matrix
    with rotated tick labels and an xlabel needs more room below the axes
    than a plain bar/line chart does, and an axes-relative offset tuned for
    the latter collides with the former's xlabel. Anchoring to the figure
    corner instead means this caption never competes with the axes' own
    ticks/labels for space, regardless of how tall those are. Call this
    LAST, after all other titles/labels are set, since tight_layout()
    generally does not know about text added straight to the figure."""
    fig = ax.figure
    fig.text(0.995, 0.008, text, ha="right", va="bottom", fontsize=7.5, color=FAINT, family=FONT_STACK)


def plot_confusion_matrix(matrix, row_labels, col_labels, out_stem: Path, title: str,
                           source: str, normalize: bool = False, row_axis_label="Predicted",
                           col_axis_label="True (ground truth)", cmap="Greens"):
    """One shared confusion-matrix renderer for YOLO / classifier / pipeline.
    `matrix[row_labels][col_labels]` -- rows are predicted classes, columns
    are true/ground-truth classes (matches Ultralytics' own ConfusionMatrix
    convention, verified against this project's saved per-class recall/
    precision numbers before this module was written).

    normalize=True divides each COLUMN by its sum (i.e. "of all true
    instances of class j, what fraction landed in each predicted row") --
    the same convention Ultralytics' own confusion_matrix_normalized.png
    uses, and the only orientation that gives every column a common 0-100%
    scale when column totals differ wildly (as they do here)."""
    m = np.asarray(matrix, dtype=float)
    nrows, ncols = m.shape
    if normalize:
        col_sums = m.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        disp = m / col_sums * 100
        fmt = lambda v: f"{v:.1f}%"
        vmax = 100
    else:
        disp = m
        fmt = lambda v: f"{int(v):,}"
        vmax = m.max() if m.max() > 0 else 1

    fig_w = max(6.5, 1.05 * ncols + 2.3)
    fig_h = max(5.5, 0.95 * nrows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_xlabel(col_axis_label, fontsize=10.5, labelpad=10)
    ax.set_ylabel(row_axis_label, fontsize=10.5, labelpad=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    thresh = vmax * 0.6
    for i in range(nrows):
        for j in range(ncols):
            val = disp[i, j]
            color = "white" if val > thresh else INK
            ax.text(j, i, fmt(val), ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    if normalize:
        cbar.set_label("% of column total", fontsize=9, color=MUTED)
    else:
        cbar.set_label("count", fontsize=9, color=MUTED)

    ax.set_title(title, loc="left")
    annotate_source(ax, source)
    fig.tight_layout()
    return save_fig(fig, out_stem)


def bar_chart(categories, values, out_stem: Path, title: str, ylabel: str, source: str,
              colors=None, ylim=None, value_fmt="{:.3f}", horizontal=False,
              figsize=None, rotate_xticks=0):
    figsize = figsize or (max(6.5, 0.9 * len(categories) + 2.5), 5.2)
    fig, ax = plt.subplots(figsize=figsize)
    colors = colors or [ACCENT_FINAL] * len(categories)
    if horizontal:
        y = np.arange(len(categories))
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        if ylim:
            ax.set_xlim(ylim)
        for yi, v in zip(y, values):
            ax.text(v, yi, f"  {value_fmt.format(v)}", va="center", ha="left", fontsize=9.5, color=INK)
    else:
        x = np.arange(len(categories))
        ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=rotate_xticks, ha="right" if rotate_xticks else "center")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        for xi, v in zip(x, values):
            ax.text(xi, v, f"{value_fmt.format(v)}", va="bottom", ha="center", fontsize=9.5, color=INK)
    ax.set_title(title, loc="left")
    annotate_source(ax, source)
    fig.tight_layout()
    return save_fig(fig, out_stem)


def line_chart(x, series: dict, out_stem: Path, title: str, xlabel: str, ylabel: str, source: str,
                colors=None, figsize=(8, 5.2), markers=False, subtitle: str | None = None):
    """series: {label: y-values}. colors: {label: hex} or None (uses ACCENT_FINAL/ACCENT_PARTIAL cycle).
    `subtitle`, if given, is a short caveat/context line rendered in small wrapped text below the
    title -- never concatenate long text into `title` itself, which doesn't wrap and will overflow
    the figure bounds under bbox_inches="tight"."""
    fig, ax = plt.subplots(figsize=figsize)
    default_cycle = [ACCENT_FINAL, ACCENT_PARTIAL, ACCENT_COND, "#7B5AA6", "#0E9AA0", "#C0472B"]
    for i, (label, y) in enumerate(series.items()):
        c = (colors or {}).get(label, default_cycle[i % len(default_cycle)])
        ax.plot(x, y, label=label, color=c, linewidth=2, marker="o" if markers else None, markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if subtitle:
        # Two independent, matplotlib-managed title slots (figure suptitle +
        # axes title) rather than hand-placed text -- letting each own its
        # normal vertical spacing is what actually prevents the two from
        # overlapping (a fixed axes-fraction offset for a second text block
        # does not reserve room and WILL collide with the title above it).
        fig.suptitle(title, x=0.01, ha="left", y=1.06, fontsize=13.5, fontweight="bold", color=INK)
        ax.set_title(subtitle, loc="left", fontsize=8.5, color=MUTED, fontweight="normal", pad=10)
    else:
        ax.set_title(title, loc="left")
    if len(series) > 1:
        ax.legend(loc="best")
    annotate_source(ax, source)
    fig.tight_layout()
    return save_fig(fig, out_stem)
