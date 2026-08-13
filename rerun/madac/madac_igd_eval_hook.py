"""
Periodic IGD evaluation for MA-DAC (pymarl-style) training.

Mirrors dqn/igd_eval_hook.py and ppo/ppo_igd_eval_hook.py: same early/late
cadence split, same n_repeats Ray-parallel repeats per problem evaluated
consecutively, same seed + run_idx per-repeat seeding, same running-history
npz saved on every firing. Reads best_igd/last_igd from moead_env.py's
moead_step info dict, same keys the other two hooks use -- MA-DAC's
multi-agent env wrapper (madac/envs/maenv.py) forwards that same dict
unchanged.

Scope is the THREE training problems for the key (get_maenv -> e.g.
M_2_46_3 -> DTLZ2_3, WFG4_3, WFG6_3), matching ppo_igd_eval_hook.py's
scope, not DQN's full 8-problem suite -- this curve is a training
diagnostic, held-out problems are covered by the final evaluation.

Hook mechanism differs from both other versions because MA-DAC has neither
Tianshou's train_fn nor SB3's BaseCallback -- run_sequential's training
loop in run.py is a plain while-loop. So this is not a callback object;
it's called directly inside that loop, same place the existing
test_interval block already lives (see run.py's `if (runner.t_env -
last_test_T) / args.test_interval >= 1.0:` block) -- call
`igd_hook.maybe_eval(runner.t_env, mac)` there.

Serialization follows PPO's CURRENT approach (ray.put + load_state_dict,
no disk touched), not the disk-checkpoint approach either hook used
before. MAC classes (BasicMAC/NonSharedMAC) mutate self.hidden_states in
place on every forward() call, so passing the live mac object into
several Ray remote calls directly is out for the same reason DQN's
pass-the-object pattern isn't used here -- but this is not a memory-bloat
concern the way SB3's model.save() was: mac.save_models() only ever wrote
self.agent.state_dict() (a single small nn.Module -- no optimizer state,
no mixer, no target network; those live in learner.save_models(), which
this hook never touches). The reason for dropping the checkpoint file
anyway is the *disk* round-trip itself: n_repeats workers x n_problems
reading from the same NFS-shared path on every firing is the same
mechanism already implicated in the gRPC keepalive instability noted
elsewhere for this cluster, independent of payload size.

Run from rerun/ -- imported by madac/run.py, not run standalone.
"""
import os

import numpy as np
import ray

from mamo.mamo_register import get_maenv


@ray.remote
def _run_episode(mac_type, agent_state_dict, scheme, groups, preprocess,
                  args_dict, key, seed, run_idx):
    """
    Runs one evaluation episode of a freshly-reconstructed MAC (weights
    loaded directly from agent_state_dict, no disk touched) on the
    problem `key`, replicating episode_runner.py's run(test_mode=True)
    stepping logic standalone (no logger/buffer/sacred ties -- this is a
    Ray worker, not the training runner). Seeded with seed + run_idx,
    matching test_dqn.py/test_ppo.py's per-repeat convention. Returns
    {"best_igd": ..., "last_igd": ...}.
    """
    import random
    from types import SimpleNamespace as SN

    import numpy as np
    from madac.controllers import REGISTRY as mac_REGISTRY
    from madac.envs import REGISTRY as env_REGISTRY
    from madac.components.episode_buffer import EpisodeBatch

    args = SN(**args_dict)
    np.random.seed(seed + run_idx)
    random.seed(seed + run_idx)

    # replay=False regardless of training config -- firing every few
    # thousand steps with replay writes on would flood disk for a curve
    # that's discarded after the training run anyway.
    env_args = {**args.env_args, "key": key, "seed": seed + run_idx, "replay": False}
    env = env_REGISTRY[args.env](**env_args)
    episode_limit = env.episode_limit

    mac = mac_REGISTRY[mac_type](scheme, groups, args)
    mac.agent.load_state_dict(agent_state_dict)

    batch = EpisodeBatch(scheme, groups, 1, episode_limit + 1,
                          preprocess=preprocess, device="cpu")
    env.reset()
    mac.init_hidden(batch_size=1)

    terminated = False
    t = 0
    env_info = {}
    while not terminated:
        batch.update({
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()],
        }, ts=t)

        actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
        reward, terminated, env_info = env.step(actions[0])

        batch.update({
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }, ts=t)
        t += 1

    return {"best_igd": env_info.get("best_igd"), "last_igd": env_info.get("last_igd")}


class MADACIGDEvalHook:
    def __init__(self, args, scheme, groups, preprocess,
                 eval_freq_early=5000, eval_freq_late=10000, switch_step=100000,
                 n_repeats=10, curve_path=None, verbose=True):
        self.args = args
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.eval_freq_early = eval_freq_early
        self.eval_freq_late = eval_freq_late
        self.switch_step = switch_step
        self.n_repeats = n_repeats
        self.verbose = verbose

        results_dir = os.path.join(args.local_results_path, "madac")
        self.curve_path = curve_path or os.path.join(
            results_dir, "igd_curves", f"{args.unique_token}.npz")
        os.makedirs(os.path.dirname(self.curve_path), exist_ok=True)

        self._last_eval_step = 0
        self._steps = []
        # same 3-problem scope as ppo_igd_eval_hook.py: the training key's
        # own constituent problems, not the full 8-problem M-suite
        func_list, nobjs_list = get_maenv(args.env_args["key"])
        self._task = [f"{f}_{n}" for f in func_list for n in nobjs_list]
        self._history = {
            t: {"best_mean": [], "best_std": [], "last_mean": [], "last_std": []}
            for t in self._task
        }

        self._ray_owns_init = False
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_repeats)
            self._ray_owns_init = True

    def _current_eval_freq(self, t_env):
        return self.eval_freq_early if t_env < self.switch_step else self.eval_freq_late

    def maybe_eval(self, t_env, mac):
        """Call once per training-loop iteration with runner.t_env and the
        live mac. No-ops unless the cadence threshold has been crossed."""
        if t_env - self._last_eval_step < self._current_eval_freq(t_env):
            return
        self._last_eval_step = t_env
        self._steps.append(t_env)

        # weights only, in Ray's object store -- no disk touched, matching
        # ppo_igd_eval_hook.py's ray.put(model.policy.state_dict())
        agent_state_ref = ray.put(mac.agent.state_dict())
        args_dict = vars(self.args)

        for t in self._task:
            futures = [
                _run_episode.remote(
                    self.args.mac, agent_state_ref, self.scheme, self.groups,
                    self.preprocess, args_dict, t, self.args.seed, i)
                for i in range(self.n_repeats)
            ]
            results = ray.get(futures)
            best = np.array([r["best_igd"] for r in results])
            last = np.array([r["last_igd"] for r in results])
            self._history[t]["best_mean"].append(best.mean())
            self._history[t]["best_std"].append(best.std())
            self._history[t]["last_mean"].append(last.mean())
            self._history[t]["last_std"].append(last.std())
            if self.verbose:
                print(f"[MA-DAC IGD eval @ {t_env}] {t} done")

        del agent_state_ref

        # overwritten in full on every firing, same convention as the
        # DQN/PPO hooks' igd_curve.npz
        flat = {
            f"{t}_{metric}": np.array(values)
            for t, hist in self._history.items()
            for metric, values in hist.items()
        }
        np.savez(self.curve_path, steps=np.array(self._steps), **flat)

    def close(self):
        if self._ray_owns_init:
            ray.shutdown()