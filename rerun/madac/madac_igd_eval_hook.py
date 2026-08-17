"""
Periodic IGD evaluation for MA-DAC.
"""
import os

import numpy as np
import ray

from mamo.mamo_register import get_maenv


@ray.remote
def _run_episode(mac_type, agent_state_dict, scheme, groups, preprocess,
                  args_dict, key, seed, run_idx):
    """
    Runs one eval episode of a freshly-reconstructed MAC on `key`, seeded with seed + run_idx
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

    # replay=False regardless of training config to not flood disk 
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
        run_name = f"{args.name}_{args.env_args['key']}_seed_{args.seed}"
        self.curve_path = curve_path or os.path.join(
            results_dir, "igd_curves", f"{run_name}.npz")
        os.makedirs(os.path.dirname(self.curve_path), exist_ok=True)

        self._last_eval_step = 0
        self._steps = []
        # 3 training problems only
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
        # fires _do_eval once the eval-frequency threshold is crossed since last firing
        if t_env - self._last_eval_step < self._current_eval_freq(t_env):
            return
        self._do_eval(t_env, mac)

    def eval_now(self, t_env, mac):
        # used only by eval_t0 in run.py to evaluate starting policy
        self._do_eval(t_env, mac)

    def _do_eval(self, t_env, mac):
        self._last_eval_step = t_env
        self._steps.append(t_env)

    
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

        # overwritten in full on every firing
        flat = {
            f"{t}_{metric}": np.array(values)
            for t, hist in self._history.items()
            for metric, values in hist.items()
        }
        np.savez(self.curve_path, steps=np.array(self._steps), **flat)

    def close(self):
        if self._ray_owns_init:
            ray.shutdown()