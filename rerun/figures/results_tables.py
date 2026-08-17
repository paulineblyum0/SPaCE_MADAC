"""
figures/results_tables.py

Every number that goes in a Chapter 4 table.

  final_igd_tables()      RQ2/RQ3: pooled + per-seed descriptives, no test
  rq2_statistics()        RQ2/RQ3: pooled rank-sum per pair, secondary check
  rq1_final_igd_tables()  RQ1: paired per-seed signed-rank test
  rq3_statistics()        RQ3: paired per-seed signed-rank, specific pairs
  auc_table()              training-curve AUC per seed, no test

Run from rerun/:  python -m figures.results_tables
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import ranksums, wilcoxon

from figures import data

OUT = Path("figures/output")


def holm(pvals):
    # Holm-Bonferroni step-down, returns adjusted p-values in input order
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
def final_igd_tables(series, problems, prefix="", metric=None):
    # RQ2/RQ3: pooled + per-seed descriptives. No test here -- seeds
    # aren't comparable across algorithms, so no valid pairing.
    metric = metric or data.METRIC
    pooled_rows, per_seed_rows = [], []

    for problem in problems:
        loaded = {}
        for label in series:
            X, seeds = data.final_igd_matrix(series, label, problem, metric=metric)
            if X is not None:
                loaded[label] = (X, seeds)

        for label, (X, seeds) in loaded.items():
            flat = X.ravel()
            pooled_rows.append([problem, label, X.shape[0], X.shape[1],
                                f"{flat.mean():.4f}", f"{flat.std(ddof=1):.4f}"])
            for seed, row in zip(seeds, X):
                per_seed_rows.append([problem, label, seed,
                                      f"{row.mean():.4f}", f"{row.std(ddof=1):.4f}"])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"{prefix}final_igd.csv",
              ["problem", "algorithm", "n_seeds", "n_repeats", "mean", "sd"],
              pooled_rows)
    write_csv(OUT / f"{prefix}per_seed_igd.csv",
              ["problem", "algorithm", "train_seed", "mean", "sd"],
              per_seed_rows)


# ----------------------------------------------------------------------
def rq2_statistics(series, problems, prefix="rq2_"):
    # RQ2/RQ3 secondary check: pooled rank-sum per pair per problem,
    # Holm-corrected within each problem. Pseudo-replicates within a
    # seed (repeats share one trained policy) -- secondary, not primary.
    # seed_consistency = proxy for per-seed agreement since there's no
    # true pairing across algorithms.
    rows = []

    for problem in problems:
        loaded = {}
        for label in series:
            X, seeds = data.final_igd_matrix(series, label, problem)
            if X is not None:
                loaded[label] = (X, seeds)

        pairs = list(combinations(loaded, 2))
        pvals = []
        for a, b in pairs:
            Xa, _ = loaded[a]
            Xb, _ = loaded[b]
            _, p = ranksums(Xa.ravel(), Xb.ravel())
            pvals.append(p)
        if not pvals:
            continue
        p_adj = holm(pvals)

        for (a, b), p in zip(pairs, p_adj):
            Xa, seeds_a = loaded[a]
            Xb, seeds_b = loaded[b]
            mean_a, mean_b = Xa.ravel().mean(), Xb.ravel().mean()
            a_lower = mean_a < mean_b

            seed_means_a = Xa.mean(axis=1)
            seed_means_b = Xb.mean(axis=1)
            comparisons = [(sa < sb) == a_lower
                           for sa in seed_means_a for sb in seed_means_b]

            rows.append([
                problem, a, b,
                f"{mean_a:.4f}", f"{mean_b:.4f}",
                a if a_lower else b,
                f"{p:.4f}", "yes" if p < 0.05 else "no",
                f"{sum(comparisons)}/{len(comparisons)}",
                f"{len(seeds_a)}x{len(seeds_b)}",
            ])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"{prefix}statistics.csv",
              ["problem", "algorithm_a", "algorithm_b",
               "mean_a", "mean_b", "lower_mean",
               "p_adj", "sig", "seed_consistency", "n_seeds_a_x_b"],
              rows)


# ----------------------------------------------------------------------
def rq1_final_igd_tables(series, problems, prefix="rq1_"):
    # RQ1 only: adaptive_open True vs False, both DQN, same seeds, same
    # eval base -> columns are the same env draw, so this IS pairable
    # (unlike RQ2/RQ3). Paired per-seed signed-rank test, Holm-corrected
    # across the 3 seeds. Kept separate from the pooled test above since
    # that's the one MA-DAC's paper reports and stays on record too.
    labels = list(series)
    if len(labels) != 2:
        raise ValueError(f"rq1_final_igd_tables expects exactly 2 conditions, got {labels}")
    a, b = labels  # a = "adaptive_open=False", b = "adaptive_open=True" by convention

    igd_rows, stat_rows = [], []

    for problem in problems:
        Xa, seeds_a = data.final_igd_matrix(series, a, problem)
        Xb, seeds_b = data.final_igd_matrix(series, b, problem)
        if Xa is None or Xb is None:
            continue
        if seeds_a != seeds_b:
            raise ValueError(f"{problem}: seed sets differ between conditions "
                              f"({seeds_a} vs {seeds_b}) -- pairing would be invalid")

        # descriptive: pooled (90) + per-seed (30), both conditions
        for label, X in ((a, Xa), (b, Xb)):
            flat = X.ravel()
            row = [problem, label, f"{flat.mean():.4f}", f"{flat.std(ddof=1):.4f}"]
            for seed, srow in zip(seeds_a, X):
                row += [f"{srow.mean():.4f}", f"{srow.std(ddof=1):.4f}"]
            igd_rows.append(row)

        # inferential: paired per seed, one-sided (True < False), Holm across seeds
        pvals, directions = [], []
        for row_a, row_b in zip(Xa, Xb):
            _, p = wilcoxon(row_b, row_a, alternative="less")
            pvals.append(p)
            directions.append(row_b.mean() < row_a.mean())

        p_adj = holm(pvals)
        sig = [d and p < 0.05 for d, p in zip(directions, p_adj)]
        row = [problem]
        for p, s in zip(p_adj, sig):
            row += [f"{p:.4f}", "yes" if s else "no"]
        row.append(f"{sum(sig)}/{len(sig)}")
        stat_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)

    # column order follows seeds_a's order == data.SEEDS (guarded above)
    seed_igd_cols = [c for seed in data.SEEDS for c in (f"seed{seed}_mean", f"seed{seed}_sd")]
    write_csv(OUT / f"{prefix}final_igd.csv",
              ["problem", "condition", "pooled_mean", "pooled_sd"] + seed_igd_cols,
              igd_rows)

    seed_stat_cols = [c for seed in data.SEEDS for c in (f"seed{seed}_p_adj", f"seed{seed}_sig")]
    write_csv(OUT / f"{prefix}statistics.csv",
              ["problem"] + seed_stat_cols + ["significant_seeds"],
              stat_rows)


# ----------------------------------------------------------------------
def rq3_statistics(series, problems, pairs, prefix="rq3_"):
    # RQ3 only, specific pairs (DQN vs PPO, PPO vs MA-DAC): PAIRED
    # per-seed Wilcoxon signed-rank per seed per problem (not pooled
    # across seeds), Holm within each (pair, problem).
    #
    # Paired, not unpaired rank-sum: all three eval scripts seed each
    # repeat identically (np.random.seed(seed+run_idx) then
    # random.seed(seed+run_idx) immediately before constructing the
    # env, before .reset()), and DQN/PPO/MA-DAC's env wrappers all
    # construct the same underlying mamo.moead_env.MamoBase with no
    # extra RNG draws in between -- confirmed by reading saenv.py,
    # ppo_env.py, and madac's MOEAEnv wrapper. So repeat index j is the
    # same environment draw across all three algorithms, and column j
    # of algorithm a and column j of algorithm b are a true pair.
    #
    # Seeds missing on either side for a pair/problem are skipped.
    rows = []

    for problem in problems:
        loaded = {}
        for label in series:
            X, seeds = data.final_igd_matrix(series, label, problem)
            if X is not None:
                loaded[label] = dict(zip(seeds, X))  # seed -> 30 repeats, index-aligned

        for a, b in pairs:
            if a not in loaded or b not in loaded:
                continue
            common_seeds = [s for s in loaded[a] if s in loaded[b]]
            if not common_seeds:
                continue

            pvals, means = [], []
            for seed in common_seeds:
                ra, rb = loaded[a][seed], loaded[b][seed]
                _, p = wilcoxon(ra, rb)  # two-sided, paired by repeat index
                pvals.append(p)
                means.append((ra.mean(), rb.mean()))
            p_adj = holm(pvals)
            sig_flags = [p < 0.05 for p in p_adj]
            sig_count = sum(sig_flags)

            for seed, (mean_a, mean_b), p, sig in zip(
                    common_seeds, means, p_adj, sig_flags):
                rows.append([
                    problem, a, b, seed,
                    f"{mean_a:.4f}", f"{mean_b:.4f}",
                    a if mean_a < mean_b else b,
                    f"{p:.4f}", "yes" if sig else "no",
                    f"{sig_count}/{len(common_seeds)}",
                ])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"{prefix}statistics.csv",
              ["problem", "algorithm_a", "algorithm_b", "seed",
               "mean_a", "mean_b", "lower_mean",
               "p_adj", "sig", "significant_seeds"],
              rows)


# ----------------------------------------------------------------------
def auc_table(series, problems, prefix=""):
    # Time-averaged IGD over training (trapezoid / step span), one
    # value per seed. No test: at 3 seeds the min attainable two-sided
    # signed-rank p is 0.25, so significance is unreachable regardless
    # of effect size -- report direction/consistency instead.
    # All algorithms integrated over the same (shared) step interval.
    rows = []
    for problem in problems:
        curves = {}
        for label in series:
            steps, Y = data.training_curve_matrix(series, label, problem)
            if steps is not None:
                curves[label] = (steps, Y)
        if not curves:
            continue

        lo = max(s[0] for s, _ in curves.values())
        hi = min(s[-1] for s, _ in curves.values())
        grid = np.linspace(lo, hi, 512)
        span = hi - lo

        for label, (steps, Y) in curves.items():
            aucs = np.array([np.trapezoid(np.interp(grid, steps, y), grid) / span
                             for y in Y])
            rows.append([problem, label,
                         " ".join(f"{a:.4f}" for a in aucs),
                         f"{aucs.mean():.4f}",
                         f"{aucs.std(ddof=1):.4f}" if len(aucs) > 1 else "",
                         f"{lo:.0f}-{hi:.0f}"])

    write_csv(OUT / f"{prefix}auc.csv",
              ["problem", "algorithm", "per_seed", "mean", "sd", "step_range"],
              rows)