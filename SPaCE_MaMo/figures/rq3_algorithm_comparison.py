"""
Plots and figures for RQ3: how does PPO compare with MADAC and DQN? Across all seeds.
"""

from figures import data, report, results_tables

SERIES = {label: data.ALL_ALGORITHMS[label] for label in ("DQN", "PPO", "MA-DAC")}
PAIRS = [("DQN", "PPO"), ("PPO", "MA-DAC")]

if __name__ == "__main__":
    # Plots + averages pooled across all 90 values
    report.run("rq3", SERIES, eval_problems=data.EVAL_PROBLEMS,
                pairwise_stats=False, pool_repeats=True)

    # Paired per-seed Wilcoxon (repeat j = same env draw across algorithms)
    results_tables.rq3_statistics(SERIES, data.EVAL_PROBLEMS, PAIRS, prefix="rq3_")