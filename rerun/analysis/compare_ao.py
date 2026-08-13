"""
analysis/compare_ao.py

Statistical comparison: adaptive_open=True vs adaptive_open=False, at
three different levels of rigor. Read the level descriptions before
picking which one goes in the dissertation as the primary claim.

1. seed_level_comparison  -- paired Wilcoxon signed-rank across your 3
   training seeds. This is the only one of the three whose unit of
   replication actually matches a claim about "the training procedure."
   Structurally capped at p=0.25 (two-sided) with n=3 -- cannot show
   significance at this sample size regardless of effect size. Report
   descriptively (direction + consistency across seeds), not as a
   significance result.

2. repeat_level_comparison -- Mann-Whitney U, pooling all 30 (or your
   n_repeats) evaluation repeats per seed across all seeds. Has real
   power, but repeats within one seed share a single trained policy --
   they are not independent draws of "the training procedure," only of
   evaluation-episode noise around one fixed policy. Report as
   supporting evidence with the pseudo-replication caveat stated
   explicitly, never as the headline number.

3. paper_style_comparison / paper_style_table -- reproduces MA-DAC's own
   methodology exactly: ONE trained policy per condition (their design
   only ever had one seed), 30 raw IGD values (not AUC -- IGD itself,
   matching what their tables report) from that one policy's evaluation
   repeats, tested with scipy.stats.ranksums (the literal Wilcoxon
   rank-sum test they cite, not mannwhitneyu). Use this for a table
   shaped like their Table 3/9, for direct comparability.

   Worth stating explicitly wherever this table appears: this is the
   SAME kind of comparison as (2) above, at the SAME level of rigor --
   n=1 training seed per condition, significance claimed only about
   that one trained policy's evaluation-outcome distribution, not about
   the algorithm robust to training-seed variance. The paper's own
   testing methodology has this limitation; reproducing it exactly
   (rather than silently using your stronger 3-seed test in a
   paper-style table) keeps the comparison honest about what MA-DAC's
   own numbers could and couldn't support.
"""

import numpy as np
from scipy import stats

from common.io_utils import eval_path, find_eval_files, load_final_igd, load_within_episode_traces
from analysis.within_episode_auc import aggregate_auc_across_seeds, episode_auc


def seed_level_comparison(auc_true: dict, auc_false: dict):
    """
    auc_true / auc_false: {seed: scalar AUC}, same seed set on both sides.
    Paired Wilcoxon signed-rank test.
    """
    seeds = sorted(set(auc_true) & set(auc_false))
    if len(seeds) < len(auc_true) or len(seeds) < len(auc_false):
        missing = set(auc_true) ^ set(auc_false)
        print(f"[warn] dropping unmatched seeds from paired test: {missing}")

    x = np.array([auc_true[s] for s in seeds])
    y = np.array([auc_false[s] for s in seeds])
    diff = x - y

    result = stats.wilcoxon(x, y) if len(seeds) >= 1 else None

    print(f"Seeds compared: {seeds}")
    print(f"AO=True:  {dict(zip(seeds, x))}")
    print(f"AO=False: {dict(zip(seeds, y))}")
    print(f"AO=True lower in {np.sum(diff < 0)}/{len(seeds)} seeds")
    print(f"Mean difference (True - False): {diff.mean():.3f}")
    if result is not None:
        print(f"Wilcoxon signed-rank: statistic={result.statistic:.3f}, p={result.pvalue:.3f}")
        if len(seeds) < 6:
            print(f"  NOTE: n={len(seeds)} < 6 -- p<0.05 is not reachable at this sample "
                  f"size regardless of effect size. Report descriptively.")
    return diff


def repeat_level_comparison(results_root, key, seeds, n_repeats=30):
    """
    Pools per-repeat within-episode AUCs across all seeds for each AO
    condition, Mann-Whitney U. See module docstring for the
    pseudo-replication caveat.
    """
    pooled = {True: [], False: []}
    for ao in (True, False):
        paths = find_eval_files(results_root, key, seeds, ao, n_repeats)
        for seed, path in paths.items():
            traces = load_within_episode_traces(path)
            pooled[ao].extend(episode_auc(t) for t in traces)

    x, y = np.array(pooled[True]), np.array(pooled[False])
    result = stats.mannwhitneyu(x, y, alternative="two-sided")

    print(f"\nPooled repeat-level comparison (n_True={len(x)}, n_False={len(y)}):")
    print(f"Mean AUC -- True: {x.mean():.3f} (+/-{x.std():.3f}), "
          f"False: {y.mean():.3f} (+/-{y.std():.3f})")
    print(f"Mann-Whitney U: statistic={result.statistic:.1f}, p={result.pvalue:.4f}")
    print("  Caveat: 30 repeats per seed share one trained policy -- treat this "
          "as supporting evidence, not an independent-samples guarantee at n=90.")
    return x, y, result


def paper_style_comparison(path_a, path_b, metric="last_igd"):
    """
    Reproduces MA-DAC's Table 3/9 comparison for a single problem: raw
    IGD (not AUC) from n_repeats evaluation episodes of ONE trained
    policy per condition, scipy.stats.ranksums (their stated test).

    Marker convention matches the paper: '+' if A significantly better
    (lower mean IGD, minimization) than B at p<0.05, '-' if
    significantly worse, '~' (paper uses '≈') if not significant.

    metric='last_igd' is used as the analog of "IGD after the full
    budget," which is what their tables appear to report -- swap to
    'best_igd' if you decide that's the better match for your own
    convention; the paper doesn't specify which they used explicitly,
    so this is an assumption worth stating in your methodology.
    """
    igd_a = load_final_igd(path_a, metric)
    igd_b = load_final_igd(path_b, metric)
    statistic, p = stats.ranksums(igd_a, igd_b)

    if p < 0.05:
        marker = "+" if igd_a.mean() < igd_b.mean() else "-"
    else:
        marker = "~"

    return {
        "mean_a": igd_a.mean(), "std_a": igd_a.std(),
        "mean_b": igd_b.mean(), "std_b": igd_b.std(),
        "statistic": statistic, "p": p, "marker": marker,
    }


def paper_style_table(results_root, keys, seed, adaptive_open_a, adaptive_open_b,
                       metric="last_igd", n_repeats=30):
    """
    Runs paper_style_comparison across multiple problems (`keys`) for a
    single seed pair, printed in a layout matching MA-DAC's tables.
    Condition A is the reference column (their convention marks A
    relative to A, i.e. read markers as "A vs B").
    """
    rows = []
    for key in keys:
        path_a = eval_path(results_root, key, seed, seed, adaptive_open_a, n_repeats)
        path_b = eval_path(results_root, key, seed, seed, adaptive_open_b, n_repeats)
        if not path_a.exists() or not path_b.exists():
            print(f"[warn] skipping {key}, missing eval file(s)")
            continue
        r = paper_style_comparison(path_a, path_b, metric)
        rows.append((key, r))

    print(f"{'Problem':<12} {'A mean(std)':<20} {'B mean(std)':<20} {'p':<8} marker")
    for key, r in rows:
        a_str = f"{r['mean_a']:.3e}({r['std_a']:.2e})"
        b_str = f"{r['mean_b']:.3e}({r['std_b']:.2e})"
        print(f"{key:<12} {a_str:<20} {b_str:<20} {r['p']:<8.3f} {r['marker']}")
    return rows


if __name__ == "__main__":
    results_root = "results"
    key = "WFG6_3"
    seeds = [42, 123, 2022]

    print("=== Within-episode AUC: seed-level paired comparison ===")
    auc_true_per_seed, _, _ = aggregate_auc_across_seeds(results_root, key, seeds, adaptive_open=True)
    auc_false_per_seed, _, _ = aggregate_auc_across_seeds(results_root, key, seeds, adaptive_open=False)
    seed_level_comparison(auc_true_per_seed, auc_false_per_seed)

    print("\n=== Within-episode AUC: pooled repeat-level comparison ===")
    repeat_level_comparison(results_root, key, seeds)

    print("\n=== Paper-style comparison (single seed, raw IGD, ranksums) ===")
    all_problems = ["DTLZ2_3", "DTLZ4_3", "WFG4_3", "WFG5_3", "WFG6_3",
                     "WFG7_3", "WFG8_3", "WFG9_3"]
    paper_style_table(results_root, all_problems, seed=2022,
                       adaptive_open_a=True, adaptive_open_b=False)