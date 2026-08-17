"""
figures/rq1_plateau.py

Secondary RQ1 analysis: does adaptive_open=False plateau earlier than
adaptive_open=True? Operates on the same per-seed within-episode curves
data.within_episode_matrix() already produces (one mean IGD trajectory
per training seed, averaged over the 30 matched evaluation repeats) --
no new data loading, just a plateau definition applied to those curves.

Plateau generation: the earliest generation t after which the curve
never improves by more than eps again, i.e. y[t] minus the best value
still to come is less than eps (compared against the running suffix
minimum, not a fixed window -- see plateau_generation()'s docstring for
why a fixed window is the wrong tool on these specific curves).

eps is defined relative to each curve's own total drop (a fraction of
y[0] - y[-1]), not a fixed absolute IGD value -- problems sit at very
different IGD scales, so a fixed absolute eps under-detects genuine
late-stage improvement on large-range problems (see relative_eps()).

Two thresholds are run by default (eps=0.001 and eps=0.0005) so you can
check the qualitative conclusion ("False plateaus earlier than True")
isn't an artifact of one specific choice.

Run from rerun/:  python -m figures.rq1_plateau
"""

from pathlib import Path

import numpy as np

from figures import data, results_tables
from figures.rq1_adaptive_open import SERIES

OUT = Path("figures/output")


def plateau_generation(y, eps):
    """
    Earliest generation t such that the curve never again improves by
    more than eps after t -- i.e. y[t] minus the *best value the curve
    ever reaches from t onward* is less than eps. Equivalent to
    comparing y[t] against the running suffix minimum.

    A fixed-width window (y[t] - y[t+window] < eps) is NOT used here:
    these curves are stepped rather than smooth (periodic eval-hook
    firing produces flat treads between drops), so a short window finds
    the first flat tread and misreports it as convergence even when a
    later step-drop follows. Comparing against the suffix minimum
    instead requires the flatness to hold for the rest of the curve, so
    an early flat tread followed by a later drop is correctly not
    called a plateau.
    """
    y = np.asarray(y, dtype=float)
    suffix_min = np.minimum.accumulate(y[::-1])[::-1]
    for t in range(len(y)):
        if y[t] - suffix_min[t] < eps:
            return t
    return None


def relative_eps(y, frac):
    """
    frac of this curve's own total drop (y[0] - y[-1]), rather than a
    fixed absolute IGD value. Problems sit at very different IGD scales
    (WFG4 spans ~0.16, WFG5 spans ~0.06), so a fixed absolute eps is too
    coarse for large-range problems and too loose for small-range ones
    -- a genuinely still-declining tail can sit under a fixed eps just
    because the problem's total range is large, which is what happened
    with WFG4's near-tie despite a visibly later-plateauing True curve.
    """
    y = np.asarray(y, dtype=float)
    return frac * (y[0] - y[-1])


def plateau_table(series, problems, eps_fracs=(0.005, 0.0025), prefix="rq1_"):
    """
    One row per (problem, condition): the plateau generation of the
    MEAN curve across the 3 training seeds -- the same curve the
    within-episode plot draws (data.within_episode_matrix() averaged
    over seeds), not three separate per-seed plateau generations. This
    keeps the plateau number consistent with what the plot shows.

    eps is relative (a fraction of this curve's own total drop), not a
    fixed absolute IGD value -- see relative_eps()'s docstring.
    """
    rows = []
    for problem in problems:
        for label in series:
            _, Y = data.within_episode_matrix(series, label, problem)
            if Y is None:
                continue
            mean_curve = Y.mean(axis=0)
            row = [problem, label, Y.shape[0], len(mean_curve)]
            for frac in eps_fracs:
                eps = relative_eps(mean_curve, frac)
                gen = plateau_generation(mean_curve, eps)
                row.append(gen if gen is not None else "none")
            rows.append(row)

    header = ["problem", "condition", "n_seeds", "n_generations"]
    header += [f"plateau_gen_frac{frac}" for frac in eps_fracs]

    OUT.mkdir(parents=True, exist_ok=True)
    results_tables.write_csv(OUT / f"{prefix}plateau.csv", header, rows)


if __name__ == "__main__":
    plateau_table(SERIES, data.EVAL_PROBLEMS)