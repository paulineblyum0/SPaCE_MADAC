"""
plotting/within_episode_plot.py

Plots mean +/- std IGD-vs-generation across the evaluation repeats of
one trained policy, reading igd_his from rp30.npz files. AUC (from
analysis/within_episode_auc.py) is folded into the title.
"""

import matplotlib.pyplot as plt
import numpy as np

from analysis.within_episode_auc import aggregate_auc_across_seeds, per_seed_traces
from common.io_utils import ensure_output_dir

OUTPUT_DIR = ensure_output_dir(__file__)  # plotting/output/


def plot_within_episode(results_root, key, seeds, adaptive_open, n_repeats=30,
                         seed_to_plot=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    traces = per_seed_traces(results_root, key, seeds, adaptive_open, n_repeats)
    if not traces:
        raise FileNotFoundError("No eval files found for the given seeds/condition.")

    seed = seed_to_plot if seed_to_plot is not None else next(iter(traces))
    episode_traces = traces[seed]

    # pad defensively in case any repeat terminated early (early_stop)
    max_len = max(len(t) for t in episode_traces)
    padded = np.full((len(episode_traces), max_len), np.nan)
    for i, t in enumerate(episode_traces):
        padded[i, :len(t)] = t

    mean_curve = np.nanmean(padded, axis=0)
    std_curve = np.nanstd(padded, axis=0)
    gens = np.arange(max_len)

    per_seed_auc, _, _ = aggregate_auc_across_seeds(results_root, key, seeds, adaptive_open, n_repeats)
    auc_this_seed = per_seed_auc.get(seed)

    label = f"seed {seed}, adaptive_open={adaptive_open} (AUC={auc_this_seed:.1f})"
    ax.plot(gens, mean_curve, label=label)
    ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    ax.set_xlabel("MOEA/D generation")
    ax.set_ylabel("IGD")
    ax.set_title(f"{key}: within-episode convergence ({len(episode_traces)} repeats)")
    ax.legend()
    return ax


if __name__ == "__main__":
    results_root = "results"
    key = "WFG6_3"
    seeds = [42, 123, 2022]

    plot_within_episode(results_root, key, seeds, adaptive_open=True)
    out_path = OUTPUT_DIR / f"within_episode_{key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")