"""
figures/figstyle.py

The only styling module. Both plot scripts import apply_style() and
plot_band() from here so every figure in the dissertation shares one
visual language.

Design rules encoded below (deliberately minimal):
  - one line per algorithm, 1.0 pt, distinguished by colour, with the
    colour fixed per algorithm so the mapping carries between figures
  - one shaded band per algorithm, no edge, alpha 0.15
  - linear y throughout; the band's lower edge is truncated at zero
    because IGD is a distance and cannot be negative
  - no gridlines, no title (the caption carries the description),
    no annotation of derived quantities such as AUC
  - vector output (PDF) at single-column width
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Single-column width for a typical LaTeX/Word thesis body. Use
# WIDTH_FULL for the multi-panel grid.
WIDTH_1COL = 3.4
WIDTH_FULL = 7.0
HEIGHT_1COL = 2.4

# Fallback cycle for any label without an entry in ALGO_COLOURS.
# Colour-blind-safe (Okabe-Ito).
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# One colour per algorithm, fixed across every figure, so a reader who
# learns the mapping in the RQ1 figure keeps it for RQ2 and RQ3. Keys
# must match the labels used in each research question's series dict.
ALGO_COLOURS = {
    "adaptive_open=False": "#CC79A7",   # pink
    "adaptive_open=True":  "#1F3B73",   # navy
    "DQN":                 "#1F3B73",
    "PPO":                 "#D55E00",   # red
    "MA-DAC":              "#1B7F9E",   # blue-teal
    "SPACE (improvement)": "#4DAF4A",   # green
    "SPACE (absolute)":    "#E69F00",   # orange
}


def palette_for(series):
    """Colours in series order. Falls back to the COLOURS cycle for any
    label not in ALGO_COLOURS, so an ad-hoc comparison still plots."""
    return [ALGO_COLOURS.get(label, COLOURS[i % len(COLOURS)])
            for i, label in enumerate(series)]


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.color": "#cccccc",
        "grid.alpha": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def mean_and_band(Y, band="sd"):
    """
    Y: (n_seeds, n_points), one row per independently trained policy,
    each row already reduced over its evaluation repeats.

    Returns (centre, lower, upper).
      band="sd"    -> mean +/- 1 SD across seeds (ddof=1)
      band="range" -> mean, with min/max across seeds as the envelope.
                      Honest choice at n=3, where an SD is barely an
                      estimate; say which one you used in the caption.
    """
    Y = np.asarray(Y, dtype=float)
    centre = np.nanmean(Y, axis=0)
    if band == "range":
        return centre, np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(centre)
    return centre, centre - sd, centre + sd


def plot_band(ax, x, Y, label, index=0, band="sd", colours=None):
    """Draw one algorithm: mean line, plus an uncertainty band if band
    is not None. band=None draws the mean only -- use this wherever the
    figure doesn't have a variance source worth shading (e.g. per-seed
    variance was already discarded upstream, or a nearby panel in the
    same section shades a different quantity and a second shaded region
    here would silently mean something else)."""
    centre = np.nanmean(np.asarray(Y, dtype=float), axis=0)

    palette = colours or COLOURS
    colour = palette[index % len(palette)]
    (line,) = ax.plot(x, centre, color=colour, label=label)
    if band is not None:
        _, lo, hi = mean_and_band(Y, band=band)
        # IGD is a mean distance to the reference front, bounded below by zero.
        lo = np.clip(lo, 0.0, None)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
    return line


def finish(ax, xlabel, ylabel, legend=True, legend_loc="upper right"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    if legend:
        ax.legend(loc=legend_loc)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def panel_figure(panels, draw_fn, xlabel, ylabel, out_path,
                 ncols=4, compact_x=False):
    """
    Lays out one axes per entry in `panels` and calls
    draw_fn(ax, panel) to populate each. A single panel gives a
    single-column figure with no title; several give a grid titled by
    panel name, with the legend inside the first panel.

    This is the only layout code in the project - both plot scripts
    differ solely in draw_fn, i.e. in which files they read.
    """
    apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)

    if n == 1:
        fig, ax = plt.subplots(figsize=(WIDTH_1COL, HEIGHT_1COL))
        draw_fn(ax, panels[0])
        if compact_x:
            _compact_x(ax)
        finish(ax, xlabel, ylabel, legend=True)
        save(fig, out_path)
        return

    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(WIDTH_FULL, 1.9 * nrows), squeeze=False)
    flat = axes.ravel()
    for i, (ax, panel) in enumerate(zip(flat, panels)):
        draw_fn(ax, panel)
        ax.set_title(panel)
        if compact_x:
            _compact_x(ax)
        # Legend lives in the first panel, top right - conventional, and
        # it costs one panel's whitespace rather than a whole figure row.
        finish(ax, "", "", legend=(i == 0))
    for ax in flat[n:]:
        ax.set_visible(False)

    for ax in flat[max(0, n - ncols):n]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)

    fig.tight_layout()
    save(fig, out_path)


def _compact_x(ax):
    """1e5 prints as 100k, 1e6 as 1M, so the axis label stays a plain
    noun and the reader does no arithmetic to read a tick."""
    def fmt(v, _):
        if abs(v) >= 1e6:
            return f"{v / 1e6:g}M"
        if abs(v) >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_locator(
        mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))