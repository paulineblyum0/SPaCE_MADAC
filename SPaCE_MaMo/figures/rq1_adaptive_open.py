"""
Tables and plots for RQ1: adaptive weights enabled vs disabled.
"""

from figures import data, report, results_tables

EVAL = "results/dqn/eval/{key}/seed_{seed}{ao}/{problem}{ao}_sd{eval_seed}_rp30.npz"


SERIES = {
    "adaptive_open=False": (None, EVAL.replace("{ao}", "_ao_false"), data.SEEDS),
    "adaptive_open=True":  (None, EVAL.replace("{ao}", ""), data.SEEDS),
}

if __name__ == "__main__":
    report.run("rq1", SERIES, eval_problems=data.EVAL_PROBLEMS,
                pairwise_stats=False, pool_repeats=True)
    results_tables.rq1_final_igd_tables(SERIES, data.EVAL_PROBLEMS, prefix="rq1_")
    