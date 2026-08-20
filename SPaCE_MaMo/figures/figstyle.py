"""
Styling for plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


WIDTH_1COL = 3.4
WIDTH_FULL = 7.0
HEIGHT_1COL = 2.4


# One colour per algorithm
ALGO_COLOURS = {
    "adaptive_open=False": "#CC79A7",   # pink
    "adaptive_open=True": "#1F3B73",   # navy
    "DQN": "#1F3B73",
    "PPO": "#D55E00",   # red
    "MA-DAC": "#1B7F9E",   # blue
    "SPACE": "#4DAF4A",   # green
    "Just Sizes": "#E69F00",   # orange
}


def palette_for(series):
    return [ALGO_COLOURS[label] for label in series]


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
    # one row per seed, already reduced over repeats
    Y = np.asarray(Y, dtype=float)
    centre = np.nanmean(Y, axis=0)
    if band == "range":
        return centre, np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(centre)
    return centre, centre - sd, centre + sd


def plot_band(ax, x, Y, label, index=0, band="sd", colours=None):
    # mean line + sd band if not None
    centre = np.nanmean(np.asarray(Y, dtype=float), axis=0)

    palette = colours or COLOURS
    colour = palette[index % len(palette)]
    (line,) = ax.plot(x, centre, color=colour, label=label)
    if band is not None:
        _, lo, hi = mean_and_band(Y, band=band)
        lo = np.clip(lo, 0.0, None) #IGD can't be negative
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


def panel_figure(panels, draw_fn, xlabel, ylabel, out_path, ncols=4, compact_x=False):
    """
    Config for panel figures
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
        finish(ax, "", "", legend=(i == 0)) #legend in first panel only
    for ax in flat[n:]:
        ax.set_visible(False)

    for ax in flat[max(0, n - ncols):n]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)

    fig.tight_layout()
    save(fig, out_path)


def _compact_x(ax):
    # Format large axis vaues as e.g. 100k
    def fmt(v, _):
        if abs(v) >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_locator(
        mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))