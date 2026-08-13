"""
plotting/training_curve_plot.py

Plots IGD vs env_step from igd_curve.npz, overlaying AO conditions with
a std band across seeds. AUC values (from analysis/training_curve_auc.py)
are folded into the title so the plot and the number it's summarizing
travel together.
"""

import matplotlib.pyplot as plt
import numpy as np

from analysis.training_curve_auc import (
    aggregate_auc_across_seeds,
    per_seed_curves,
    seed_curve_averaged_across_problems,
    training_curve_auc_across_problems,
)
from common.io_utils import all_problems_for_m, ensure_output_dir, find_training_curves

OUTPUT_DIR = ensure_output_dir(__file__)  # plotting/output/


def plot_training_curves(results_root, key, seeds, problem, metric="last_mean",
                          ao_conditions=(True, False), ax=None):
    if ax is None:
        _, ax = plt.subplots()

    for ao in ao_conditions:
        curves = per_seed_curves(results_root, key, seeds, ao, problem, metric)
        if not curves:
            continue
        # align on the union of step values via interpolation, since
        # early/late cadence can differ slightly if switch_step timing
        # lands on different env_steps across seeds
        all_steps = sorted(set(s for steps, _, _ in curves.values() for s in steps))
        interp_means = np.array([
            np.interp(all_steps, steps, means)
            for steps, means, _ in curves.values()
        ])
        mean_across_seeds = interp_means.mean(axis=0)
        std_across_seeds = interp_means.std(axis=0)

        _, auc_mean, auc_std = aggregate_auc_across_seeds(results_root, key, seeds, ao, problem, metric)

        label = f"adaptive_open={ao} (AUC={auc_mean:.1f}±{auc_std:.1f})"
        ax.plot(all_steps, mean_across_seeds, label=label)
        ax.fill_between(all_steps,
                         mean_across_seeds - std_across_seeds,
                         mean_across_seeds + std_across_seeds,
                         alpha=0.2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"IGD")
    ax.set_title("Training-progress curve")
    ax.legend()
    return ax


def plot_training_curves_across_problems(results_root, key, seeds, problems=None,
                                          metric="last_mean", ao_conditions=(True, False), ax=None):
    """
    Same as plot_training_curves, but averaged (raw mean) across all 8
    problems in the M-suite. Raw averaging is appropriate here because
    IGD across these problems, at a fixed M, stays within one order of
    magnitude (unlike BBOB's cross-function spread) -- see
    seed_curve_averaged_across_problems' docstring. AUC in the legend
    still applies the log transform once, as part of the AUC definition
    itself, not as a cross-problem correction.
    """
    problems = problems or all_problems_for_m(3)
    if ax is None:
        _, ax = plt.subplots()

    for ao in ao_conditions:
        paths = find_training_curves(results_root, key, seeds, ao)
        if not paths:
            continue
        seed_curves = {
            seed: seed_curve_averaged_across_problems(path, problems, metric)
            for seed, path in paths.items()
        }
        all_steps = sorted(set(s for steps, _ in seed_curves.values() for s in steps))
        interp = np.array([
            np.interp(all_steps, steps, curve)
            for steps, curve in seed_curves.values()
        ])
        mean_across_seeds = interp.mean(axis=0)
        std_across_seeds = interp.std(axis=0)

        _, auc_mean, auc_std = training_curve_auc_across_problems(
            results_root, key, seeds, ao, problems, metric)

        label = f"adaptive_open={ao} (AUC={auc_mean:.1f}±{auc_std:.1f})"
        ax.plot(all_steps, mean_across_seeds, label=label)
        ax.fill_between(all_steps,
                         mean_across_seeds - std_across_seeds,
                         mean_across_seeds + std_across_seeds,
                         alpha=0.2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Mean IGD across {len(problems)} problems")
    ax.set_title(f"{key}: training-progress curve, averaged across problems")
    ax.legend()
    return ax


if __name__ == "__main__":
    results_root = "results"
    key = "M_2_46_3"
    seeds = [42, 123, 2022]
    problem = "WFG6_3"

    plot_training_curves(results_root, key, seeds, problem)
    out_path = OUTPUT_DIR / f"training_curve_{key}_{problem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.figure()
    plot_training_curves_across_problems(results_root, key, seeds)
    out_path_all = OUTPUT_DIR / f"training_curve_{key}_all_problems.png"
    plt.savefig(out_path_all, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path_all}")