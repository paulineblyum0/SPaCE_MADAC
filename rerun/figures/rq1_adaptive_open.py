"""
RQ1: did the MA-DAC paper's DQN baseline have the weight-adaptation
action dimension structurally disabled, and what happens when it is
enabled?

Two conditions, identical in every other respect, differing only by the
'_ao_false' suffix test_dqn.py appends when adaptive_open is False. Both
are DQN; the labels name only the differing setting, so the legend fits
inside a panel. Say "both conditions are DQN" in the caption.

No training curve: the claim is about what the corrected policy does,
so the evidence is the final policy's within-episode convergence and
its final IGD, not how either run got there.

Colours come from figstyle.ALGO_COLOURS, keyed on the labels below.

Statistics: unlike RQ2/RQ3, the two conditions here share the same
training seeds and the same evaluation base seed, so column j of one
condition and column j of the other are the same environment draw
(see test_dqn.py's step_in_env) -- a true pairing exists. So this
script calls results_tables.rq1_final_igd_tables directly (paired,
per-seed Wilcoxon signed-rank, Holm-corrected across the 3 seeds)
instead of going through report.run's pairwise_stats=True, which
would only ever call the pooled, unpaired rq2_statistics -- correct
for RQ2/RQ3's cross-algorithm comparisons, not for this one.

Plots and descriptive tables still pool across all 90 (seed x repeat)
values per condition, per supervisor sign-off -- only the significance
test is kept per-seed.

Run from rerun/:  python -m figures.rq1_adaptive_open
"""

from figures import data, report, results_tables

EVAL = "results/dqn/eval/{key}/seed_{seed}{ao}/{problem}{ao}_sd{eval_seed}_rp30.npz"

# Order matters twice over: it fixes the legend order and, via
# figstyle.palette_for, the colour assignment.
SERIES = {
    "adaptive_open=False": (None, EVAL.replace("{ao}", "_ao_false"), data.SEEDS),
    "adaptive_open=True":  (None, EVAL.replace("{ao}", ""), data.SEEDS),
}

if __name__ == "__main__":
    report.run("rq1", SERIES, eval_problems=data.EVAL_PROBLEMS,
                pairwise_stats=False, pool_repeats=True)
    results_tables.rq1_final_igd_tables(SERIES, data.EVAL_PROBLEMS, prefix="rq1_")