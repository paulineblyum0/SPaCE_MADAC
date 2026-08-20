"""
Secondary RQ1 analysis: does adaptive_open=False plateau earlier than adaptive_open=True?
"""

from pathlib import Path

import numpy as np

from figures import data, results_tables
from figures.rq1_adaptive_open import SERIES

OUT = Path("figures/output")


def plateau_generation(y, eps):
    """
    Earliest timestep where further improvements don't produce a value beyond the range of eps. 
    """
    y = np.asarray(y, dtype=float)
    suffix_min = np.minimum.accumulate(y[::-1])[::-1]
    for t in range(len(y)):
        if y[t] - suffix_min[t] < eps:
            return t
    return None


def relative_eps(y, frac):
    """
    Eps as a fraction of the curves total drop because different problems can have different scales which might make 
    eps too coarse.
    """
    y = np.asarray(y, dtype=float)
    return frac * (y[0] - y[-1])


def plateau_table(series, problems, eps_fracs=(0.005, 0.0025), prefix="rq1_"):
    """
    Plateau generation of the seed-mean curve
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