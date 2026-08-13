"""
common/io_utils.py

Shared loading utilities for the plotting/ and analysis/ packages.

Directory conventions (from dqn.py / test_dqn.py):

  Training curves (igd_curve.npz, no igd_his -- dqn.py's save_history
  defaults False and DQNIGDEvalHook never overrides it):
      results/dqn/<key>/seed_<seed>{ao_tag}/igd_curve.npz

  Final 30-repeat evaluation (rp30.npz, HAS igd_his -- test_dqn.py's own
  get_args() defaults save_history=True):
      results/dqn/eval_trainsd<train_seed>{ao_tag}/
          <key>_trainsd<train_seed>_evalsd<eval_seed>_rp<repeat>.npz

  ao_tag is '' when adaptive_open=True, '_ao_false' when adaptive_open=False.
  This is the only thing distinguishing the two conditions on disk --
  ao_true and ao_false runs for the same seed live in sibling
  directories, never the same one.
"""

from pathlib import Path

import numpy as np

# NumPy >=2.0 renamed trapz -> trapezoid and removed the old name;
# NumPy <2.0 only has trapz. This picks whichever exists so the AUC
# functions in analysis/ don't care which NumPy version is installed.
trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def ensure_output_dir(script_path) -> Path:
    """
    Returns <the calling script's own directory>/output, creating it if
    needed. Call as ensure_output_dir(__file__) from each entry-point
    script so plotting/ writes to plotting/output/ and analysis/ writes
    to analysis/output/ regardless of the directory you ran python from.
    """
    d = Path(script_path).resolve().parent / "output"
    d.mkdir(exist_ok=True)
    return d


def load_training_curve_all_problems(path, problems, metric="last_mean"):
    """
    Returns (steps, {problem: means_array}) for a set of problems from
    one igd_curve.npz. All 8 problems in the M-suite share the same
    `steps` array within one file (DQNIGDEvalHook evaluates all of them
    at every trigger), so no interpolation is needed across problems --
    only later, across seeds.
    """
    data = np.load(path)
    steps = data["steps"]
    means = {p: data[f"{p}_{metric}"] for p in problems}
    return steps, means


def all_problems_for_m(m):
    """
    Reproduces mamo.mamo_register.Task.get_task's "all{m}" construction
    without importing mamo, so common/io_utils.py stays dependency-free.
    Keep in sync if Task.get_task's DTLZ/WFG lists ever change.
    """
    dtlz = [f"DTLZ{i}_{m}" for i in (2, 4)]
    wfg = [f"WFG{i}_{m}" for i in range(4, 10)]
    return dtlz + wfg


def ao_tag(adaptive_open: bool) -> str:
    return "" if adaptive_open else "_ao_false"


def training_curve_path(results_root, key, seed, adaptive_open):
    return Path(results_root) / "dqn" / key / f"seed_{seed}{ao_tag(adaptive_open)}" / "igd_curve.npz"


def find_training_curves(results_root, key, seeds, adaptive_open):
    """Returns {seed: path} for whichever of `seeds` actually exist on disk."""
    paths = {}
    for seed in seeds:
        p = training_curve_path(results_root, key, seed, adaptive_open)
        if p.exists():
            paths[seed] = p
        else:
            print(f"[warn] missing training curve: {p}")
    return paths


def eval_path(results_root, key, train_seed, eval_seed, adaptive_open, n_repeats=30):
    tag = ao_tag(adaptive_open)
    return (Path(results_root) / "dqn" / f"eval_trainsd{train_seed}{tag}"
            / f"{key}_trainsd{train_seed}_evalsd{eval_seed}_rp{n_repeats}.npz")


def find_eval_files(results_root, key, seeds, adaptive_open, n_repeats=30):
    """
    Assumes train_seed == eval_seed for each run, matching your seed
    protocol. Returns {seed: path}.
    """
    paths = {}
    for seed in seeds:
        p = eval_path(results_root, key, seed, seed, adaptive_open, n_repeats)
        if p.exists():
            paths[seed] = p
        else:
            print(f"[warn] missing eval file: {p}")
    return paths


def load_training_curve(path, problem, metric="last_mean"):
    """
    Returns (steps, values, std) for one problem from an igd_curve.npz.
    metric is one of best_mean / last_mean -- std is fetched
    automatically for the matching *_std key.
    """
    data = np.load(path)
    steps = data["steps"]
    mean_key = f"{problem}_{metric}"
    std_key = f"{problem}_{metric.replace('mean', 'std')}"
    return steps, data[mean_key], data[std_key]


def load_within_episode_traces(path):
    """
    Returns a list of 1D igd_his arrays, one per repeat, from a
    test_dqn.py rp30.npz file. Raises KeyError with a clear message if
    save_history was off when this file was generated (no igd_his
    present) -- that would mean it's not the file you want.
    """
    data = np.load(path, allow_pickle=True)
    info_stack = data["info_stack"]
    try:
        return [np.asarray(info["igd_his"]) for info in info_stack]
    except KeyError as e:
        raise KeyError(
            f"{path} has no 'igd_his' in its info dicts -- it was likely "
            f"generated with save_history=False. Within-episode AUC needs "
            f"test_dqn.py's default (save_history=True) evaluation output."
        ) from e


def load_final_igd(path, metric="last_igd"):
    """
    Returns a 1D array of the raw scalar IGD (not igd_his) from each of
    the n_repeats evaluation episodes in an rp30.npz -- metric is
    'best_igd' or 'last_igd'. This is the quantity the MA-DAC paper's
    own Wilcoxon rank-sum tables report (mean/std of 30 runs), as
    opposed to the AUC construct, which is our own addition.
    """
    data = np.load(path, allow_pickle=True)
    info_stack = data["info_stack"]
    return np.array([info[metric] for info in info_stack])