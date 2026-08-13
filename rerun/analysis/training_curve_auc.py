"""
analysis/training_curve_auc.py

Training-progress AUC: IGD (best or last) integrated over env_step, from
igd_curve.npz. Answers "how fast does the RL training process improve" --
the axis that matters for RQ3 (SPACE curriculum vs round-robin), since
curriculum choice acts on training dynamics, not on a single frozen
policy's rollout.

mu_opt = 0 for IGD, so regret_t = IGD_t exactly, and there's no
infinite-runtime case to handle since DQN training runs a fixed,
predetermined step budget -- see plotting/training_curve_plot.py for the
visual counterpart of this module.
"""

import json

import numpy as np

from common.io_utils import (
    all_problems_for_m,
    ensure_output_dir,
    find_training_curves,
    load_training_curve,
    load_training_curve_all_problems,
    trapz,
)

OUTPUT_DIR = ensure_output_dir(__file__)  # analysis/output/


def seed_curve_averaged_across_problems(path, problems, metric="last_mean"):
    """
    For one seed's igd_curve.npz: average raw IGD across `problems`
    point-wise. Returns (steps, avg_igd_curve) in real IGD units.

    Raw averaging (no per-problem log/normalization) is fine here
    because, unlike BBOB's function-value scale (which spans orders of
    magnitude across functions), IGD across the 8 MaMo problems at a
    fixed M sits in the same order of magnitude -- e.g. M=3 problems
    all fall roughly in the 0.04-0.09 range per the MA-DAC paper's own
    tables. No single problem dominates a raw mean here.
    """
    steps, means_by_problem = load_training_curve_all_problems(path, problems, metric)
    raw_vals = np.array([means_by_problem[p] for p in problems])  # (n_problems, T)
    return steps, raw_vals.mean(axis=0)


def training_curve_auc_across_problems(results_root, key, seeds, adaptive_open,
                                        problems=None, metric="last_mean"):
    """
    One AUC per seed -- on the problem-averaged raw IGD curve, log-
    transformed once via training_curve_auc (same treatment as the
    single-problem case; the log here is the AUC definition itself, not
    a cross-problem scale correction) -- then mean/std across seeds.
    """
    problems = problems or all_problems_for_m(3)
    paths = find_training_curves(results_root, key, seeds, adaptive_open)
    per_seed_auc = {}
    for seed, path in paths.items():
        steps, avg_curve = seed_curve_averaged_across_problems(path, problems, metric)
        per_seed_auc[seed] = training_curve_auc(steps, avg_curve)
    values = np.array(list(per_seed_auc.values()))
    return per_seed_auc, values.mean(), values.std()


def training_curve_auc(steps, means):
    """AUC = integral of log(IGD_t + 1) over env_step (trapezoidal rule)."""
    return trapz(np.log(np.asarray(means) + 1.0), x=np.asarray(steps))


def per_seed_curves(results_root, key, seeds, adaptive_open, problem, metric="last_mean"):
    """Returns {seed: (steps, means, stds)} for one problem, one AO condition."""
    paths = find_training_curves(results_root, key, seeds, adaptive_open)
    return {
        seed: load_training_curve(path, problem, metric)
        for seed, path in paths.items()
    }


def aggregate_auc_across_seeds(results_root, key, seeds, adaptive_open, problem, metric="last_mean"):
    """
    One AUC per seed, then mean/std across seeds -- matches your seed
    protocol (report mean +/- std over the 3 training seeds), not a
    single AUC of an averaged curve.
    """
    curves = per_seed_curves(results_root, key, seeds, adaptive_open, problem, metric)
    aucs = {seed: training_curve_auc(steps, means) for seed, (steps, means, _) in curves.items()}
    values = np.array(list(aucs.values()))
    return aucs, values.mean(), values.std()


if __name__ == "__main__":
    results_root = "results"
    key = "M_2_46_3"
    seeds = [42, 123, 2022]
    problem = "WFG6_3"

    results = {}
    for ao in (True, False):
        aucs, mean_auc, std_auc = aggregate_auc_across_seeds(results_root, key, seeds, ao, problem)
        print(f"adaptive_open={ao}: per-seed AUC={aucs}, mean={mean_auc:.2f}, std={std_auc:.2f}")
        results[str(ao)] = {
            "per_seed_auc": {str(s): float(v) for s, v in aucs.items()},
            "mean": float(mean_auc),
            "std": float(std_auc),
        }

    out_path = OUTPUT_DIR / f"training_curve_auc_{key}_{problem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # averaged across all 8 problems in the M-suite, log-scale (see
    # seed_curve_averaged_across_problems docstring for why)
    results_all = {}
    for ao in (True, False):
        per_seed, mean_auc, std_auc = training_curve_auc_across_problems(results_root, key, seeds, ao)
        print(f"[all problems] adaptive_open={ao}: per-seed AUC={per_seed}, "
              f"mean={mean_auc:.2f}, std={std_auc:.2f}")
        results_all[str(ao)] = {
            "per_seed_auc": {str(s): float(v) for s, v in per_seed.items()},
            "mean": float(mean_auc),
            "std": float(std_auc),
        }
    out_path_all = OUTPUT_DIR / f"training_curve_auc_{key}_all_problems.json"
    with open(out_path_all, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"Saved: {out_path_all}")