"""
figures/report.py

The driver. Every research-question file describes what it is comparing
and calls run(); this module decides what that means in terms of files
on disk.

Nothing here is specific to any one comparison. A "series" is just a
dict of label -> (curve template, eval template, seeds), so the same
code produces the algorithm comparison, the adaptive_open comparison and
the SPACE ordering comparison without modification.

Outputs are prefixed by the research-question name, so running several
never overwrites another's results.
"""

from pathlib import Path

from figures import data, figstyle, results_tables
from functools import partial

OUT = Path("figures/output")

BAND = "sd"           # "sd" (mean +/- 1 SD across seeds) or "range"


def _panel(series, problems, loader, out_path, xlabel, compact_x=False, colours=None, band=BAND):
    colours = colours or figstyle.palette_for(series)

    def draw(ax, problem):
        for i, label in enumerate(series):
            x, Y = loader(series, label, problem)
            if x is None:
                continue
            figstyle.plot_band(ax, x, Y, label, index=i, band=band,
                               colours=colours)

    figstyle.panel_figure(problems, draw, xlabel=xlabel, ylabel="IGD",
                          out_path=out_path, compact_x=compact_x)


def _avg_panel(series, problems, out_path, xlabel, compact_x=False, colours=None, metric=None):
    # One line per algorithm, IGD averaged across `problems` instead of
    # split into one panel per problem.
    colours = colours or figstyle.palette_for(series)

    def draw(ax, _):
        for i, label in enumerate(series):
            x, Y = data.training_curve_matrix_avg(series, label, problems,
                                                   metric=metric or data.METRIC)
            if x is None:
                continue
            figstyle.plot_band(ax, x, Y, label, index=i, band=BAND,
                               colours=colours)

    figstyle.panel_figure(["avg"], draw, xlabel=xlabel, ylabel="IGD",
                          out_path=out_path, compact_x=compact_x)


def run(name, series, eval_problems, train_problems=None,
        within_episode=True, training_curve=False, training_curve_avg=False,
        tables=True, auc=False, metric=None,pairwise_stats=False, 
        pool_repeats=False, colours=None, training_curve_band=BAND):
  
    OUT.mkdir(parents=True, exist_ok=True)

    if within_episode:
        loader = partial(data.within_episode_matrix, pool_repeats=pool_repeats)
        _panel(series, eval_problems, loader,
               OUT / f"{name}_within_episode.pdf",
               "Generation", colours=colours)

    if training_curve:
        loader = partial(data.training_curve_matrix, metric=metric or data.METRIC)
        _panel(series, train_problems, loader,
               OUT / f"{name}_training_curve.pdf",
               "Timesteps", compact_x=True, colours=colours, band=training_curve_band)

    if training_curve_avg:
        _avg_panel(series, train_problems,
                   OUT / f"{name}_training_curve_avg.pdf",
                   "Timesteps", compact_x=True, colours=colours, metric=metric)

    if tables:
        results_tables.final_igd_tables(series, eval_problems, prefix=f"{name}_", metric=metric)

    if pairwise_stats:
        results_tables.rq2_statistics(series, eval_problems, prefix=f"{name}_")

    if auc:
        results_tables.auc_table(series, train_problems, prefix=f"{name}_")