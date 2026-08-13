"""
analysis/within_episode_auc.py

Within-episode AUC: IGD integrated over MOEA/D generation, from the
final 30-repeat evaluation of a trained (frozen) policy. Answers "given
the trained policy, how fast does a single MOEA/D rollout converge" --
the axis that matters for RQ2 (PPO vs DQN-MA-DAC), distinguishing two
policies that reach similar final IGD but at different speeds.

Uses igd_his, which only exists in test_dqn.py's rp30.npz output -- see
common/io_utils.py's docstring for why igd_curve.npz can't be used here.
"""

import json

import numpy as np

from common.io_utils import ensure_output_dir, find_eval_files, load_within_episode_traces, trapz

OUTPUT_DIR = ensure_output_dir(__file__)  # analysis/output/


def episode_auc(igd_his):
    """AUC = integral of log(IGD_t + 1) over generation index (dt=1)."""
    return trapz(np.log(np.asarray(igd_his) + 1.0))


def per_seed_traces(results_root, key, seeds, adaptive_open, n_repeats=30):
    """Returns {seed: list_of_igd_his_arrays} for however many seeds exist."""
    paths = find_eval_files(results_root, key, seeds, adaptive_open, n_repeats)
    return {seed: load_within_episode_traces(path) for seed, path in paths.items()}


def aucs_per_repeat(results_root, key, seeds, adaptive_open, n_repeats=30):
    """
    Returns {seed: np.array of per-repeat AUCs}. Aggregate FROM this
    level -- mean-of-AUCs, not AUC-of-mean-curve (averaging traces first
    smooths out episode-to-episode variance before the log is applied,
    biasing the AUC down and losing the std you need for reporting).
    """
    traces = per_seed_traces(results_root, key, seeds, adaptive_open, n_repeats)
    return {seed: np.array([episode_auc(t) for t in seed_traces])
            for seed, seed_traces in traces.items()}


def aggregate_auc_across_seeds(results_root, key, seeds, adaptive_open, n_repeats=30):
    """
    Two-level aggregation: mean over repeats within a seed, then
    mean/std over seeds. Returns (per_seed_means: dict, grand_mean, grand_std).
    """
    per_repeat = aucs_per_repeat(results_root, key, seeds, adaptive_open, n_repeats)
    per_seed_means = {seed: aucs.mean() for seed, aucs in per_repeat.items()}
    values = np.array(list(per_seed_means.values()))
    return per_seed_means, values.mean(), values.std()


if __name__ == "__main__":
    results_root = "results"
    key = "WFG6_3"
    seeds = [42, 123, 2022]

    results = {}
    for ao in (True, False):
        per_seed, mean_auc, std_auc = aggregate_auc_across_seeds(results_root, key, seeds, ao)
        print(f"adaptive_open={ao}: per-seed mean AUC={per_seed}, "
              f"overall mean={mean_auc:.2f}, std={std_auc:.2f}")
        results[str(ao)] = {
            "per_seed_mean_auc": {str(s): float(v) for s, v in per_seed.items()},
            "mean": float(mean_auc),
            "std": float(std_auc),
        }

    out_path = OUTPUT_DIR / f"within_episode_auc_{key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")