"""
RQ5: SPACE curriculum learning applied to PPO. Compare DQN, PPO, MA-DAC, Just
Sizes, SPACE.
Produces IGD training curves + AUC. Final policy significance tested only against PPO.
"""

from figures import data, report, results_tables

SERIES = {label: data.ALL_ALGORITHMS[label]
          for label in ("DQN", "PPO", "MA-DAC", "Just Sizes", "SPACE")}
PAIRS = [("Just Sizes", "PPO"), ("SPACE", "PPO")]

if __name__ == "__main__":
    # Training curves + AUC
    report.run("rq5", SERIES,
                eval_problems=data.EVAL_PROBLEMS,
                train_problems=data.TRAIN_PROBLEMS,
                within_episode=False,
                training_curve=True,
                training_curve_avg=True,
                tables=False,
                auc=True,
                metric="last",
                pairwise_stats=False,
                pool_repeats=False,
                training_curve_band=None)

    # Final-policy descriptives
    results_tables.final_igd_tables(SERIES, data.EVAL_PROBLEMS,
                                    prefix="rq5_", metric="last")

    # Paired per-seed Wilcoxon, Just Sizes/SPACE vs PPO only
    results_tables.rq3_statistics(SERIES, data.EVAL_PROBLEMS, PAIRS, prefix="rq5_")