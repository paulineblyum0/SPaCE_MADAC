from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

from point_mass_wrapper import CPMWrapper
import csv
import json
import numpy as np
import logging
import os
import argparse
from functools import partial


def env_creator(env_config):
    path = env_config["path"]
    wrapper_func = env_config["wrapper_func"]
    feats = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            f = []
            for i in range(len(row)):
                if i != 0:
                    try:
                        f.append(float(row[i]))
                    except:
                        f.append(row[i])
            feats.append(f)
    return wrapper_func(
        instance_feats=feats,
        test=env_config["test"],
        external_eval=env_config["ev"],
    )


def get_instance_evals(learner, env, num_instances):
    evals = []
    cur_set, _ = env.get_instance_set()
    for i in range(num_instances):
        env.set_instance_set([i])
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        val = learner.policy.predict_values(obs_tensor).item()
        evals.append(val)
    env.set_instance_set(cur_set)
    return np.array(evals)


def order_instances_qvals(learner, env, num_instances):
    evals = get_instance_evals(learner, env, num_instances)
    return np.argsort(evals)


def order_instances_improvement(learner, env, num_instances, last_evals):
    evals = get_instance_evals(learner, env, num_instances)
    improvement = evals - last_evals
    return np.argsort(improvement)[::-1], evals


def order_instances_relative_improvement(learner, env, num_instances, last_evals):
    evals = get_instance_evals(learner, env, num_instances)
    absolute_improvement = evals - last_evals
    relative_improvement = absolute_improvement / last_evals
    return np.argsort(relative_improvement)[::-1], evals


def order_instances_distance(env, num_insts):
    instances = env.env_method("get_instances")[0]
    indices, training_set = env.env_method("get_instance_set")[0]
    mean_train_instance = np.mean(training_set, axis=0)
    distances = []
    for i in range(num_insts):
        distances.append(
            np.mean(
                np.abs(
                    np.array(instances[i])
                    - np.array(mean_train_instance)
                )
            )
        )
    return np.argsort(distances)


def get_mean_q(model):
    qs = []
    n_insts = model.env.env_method("get_instance_size")[0]
    env = model.env
    for i in range(n_insts):
        obs = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        val = model.policy.predict_values(obs_tensor).item()
        qs.append(val)
    return np.mean(qs)


def eval_hook(model, eval_env, s, outdir, name):
    step = 1
    to_eval = eval_env.get_feats()
    train_reward = 0
    policies = []
    for _ in range(to_eval):
        policies.append([])
    rewards = np.zeros(to_eval)
    for n in range(to_eval):
        logging.info(f"Evaluating instance {n}")
        obs, _ = eval_env.reset()
        done = False
        rews = 0
        pol = [eval_env.inst_id]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            pol.append(action)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            rews = rews * 0.95 + r
        rewards[eval_env.inst_id] = rews
        train_reward += rews
        policies[eval_env.inst_id] = pol
    train_reward = train_reward / to_eval
    with open(os.path.join(outdir, f"{name}_reward.txt"), "a") as fh:
        fh.writelines(
            str(train_reward)
            + "\t"
            + str(step)
            + "\t"
            + str(s)
            + "\t"
            + str(to_eval)
            + "\n"
        )
 
    with open(os.path.join(outdir, f"{name}_per_instance.txt"), "a") as fh:
        r_string = ""
        for r in rewards:
            r_string += str(r) + "\t"
        r_string += str(step) + "\t" + str(s) + "\n"
        fh.writelines(r_string)
 
    with open(os.path.join(outdir, f"{name}_policies.txt"), "a") as fh:
        for p in policies:
            p_string = str(p[0]) + " "
            for i in range(1, len(p)):
                p_string += str(p[i])
            fh.writelines(p_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPaCE CPM")
    parser.add_argument(
        "--spl_mode",
        choices=["absolute", "improvement", "rel-improvement"],
        default="improvement",
        help="Q-value comparison method",
    )
    parser.add_argument(
        "--steps", choices=["fixed", "adaptive"], default="adaptive", help=" "
    )
    parser.add_argument(
        "--mode",
        choices=["rr", "cl", "spl"],
        default="spl",
        help=" ",
    )
    parser.add_argument(
        "--outdir",
        default="./results",
        help="Directory in which to save test and eval",
    )
    parser.add_argument(
        "--setfactor",
        default=1,
        type=int,
        help="Number of runs through training set before reevaluating",
    )
    parser.add_argument(
        "--kappa",
        default=1,
        type=int,
        help="Number of instances to add at curriculum update",
    )
    parser.add_argument("--multi", action="store_true")
    parser.add_argument(
        "--eta",
        default=0.1,
        type=float,
        help="Performance threshold for curriculum update",
    )
    parser.add_argument(
        "--eval", default=1, type=int, help="Evaluation and Test interval"
    )
    parser.add_argument("--seed", default=0, type=int, help="Environment seed")
    parser.add_argument(
        "--warmup", default=0, type=int, help="Number of RR warmup episodes"
    )
    parser.add_argument(
        "--test",
        default="cpm_test_100.csv",
        help="Test instance file",
    )
    parser.add_argument(
        "--instances",
        default="cpm_train_100.csv",
        help="Instance file",
    )
    args = parser.parse_args()
 
    env_func = partial(CPMWrapper)
    env_init = partial(env_creator)
 
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
 
    if args.mode == "rr":
        test = True
    else:
        test = False
 
    env = env_init(
        {
            "path": args.test,
            "wrapper_func": env_func,
            "test": test,
            "ev": False,
        },
    )
    eval_env = env_init(
        {
            "path": args.instances,
            "wrapper_func": env_func,
            "test": True,
            "ev": True,
        }
    )
    test_env = env_init(
        {
            "path": args.test,
            "wrapper_func": env_func,
            "test": True,
            "ev": True,
        },
    )

       # Parameters from Klink et.al.
    policy_kwargs = dict(net_arch=[21], activation_fn=torch.nn.Tanh)
 
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
 
    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.95,
        gae_lambda=0.99,
        n_epochs=4,
        vf_coef=1.0,
        ent_coef=0.0,
        seed=args.seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
 
    total_steps = 0
    eval_hook(model, eval_env, total_steps, outdir, "eval")
    eval_hook(model, test_env, total_steps, outdir, "test")
    if args.warmup:
        cur_set, _ = model.env.env_method("get_instance_set")[0]
        model.env.env_method(
            "set_instance_set", np.arange(eval_env.get_instance_size())
        )
        model.learn(total_timesteps=args.warmup, reset_num_timesteps=False)
        total_steps += args.warmup
        model.env.env_method("set_instance_set", cur_set)
        eval_hook(model, eval_env, total_steps, outdir, "eval")
        eval_hook(model, test_env, total_steps, outdir, "test")
 
    set_factor = args.setfactor
    delta_q = -np.inf
    last_q = 0
    n_instances = model.env.env_method("get_instance_size")[0]
    total_instances = eval_env.get_feats()
    last_evals = np.zeros(total_instances)
    for i in range(2000):
        print(f"This is iteration {i}")
        if args.steps == "fixed" or args.mode == "rr":
            print("Fixed timesteps")
            timesteps = 2048
        else:
            timesteps = n_instances * 100
            print(f"timesteps: {timesteps}")
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        total_steps += timesteps
        print("Train done")
        if not args.mode == "rr":
            mean_q = get_mean_q(model)
            delta_q = np.abs(np.abs(mean_q) - np.abs(last_q))
            last_q = mean_q
            print("Mean computed")
            if (
                delta_q <= args.eta * np.abs(last_q)
                and n_instances < total_instances
            ):
                print("Increasing instance set size")
                model.env.env_method(
                    "increase_set_size", args.kappa, args.multi
                )
                n_instances = model.env.env_method("get_instance_size")[0]
            if args.mode == "cl":
                indices = order_instances_distance(
                    env, eval_env.get_instance_size()
                )
                print(indices)
            elif args.spl_mode == "absolute":
                indices = order_instances_qvals(
                    model, eval_env, eval_env.get_instance_size()
                )
            elif args.spl_mode == "improvement":
                indices, last_evals = order_instances_improvement(
                    model,
                    eval_env,
                    eval_env.get_instance_size(),
                    last_evals,
                )
            else:
                indices, last_evals = order_instances_relative_improvement(
                    model,
                    eval_env,
                    eval_env.get_instance_size(),
                    last_evals,
                )
 
            model.env.env_method("set_instance_set", indices)
            with open(
                os.path.join(outdir, f"instance_curriculum.txt"), "a"
            ) as fh:
                fh.writelines(f"{indices[:n_instances]}\n")
            with open(os.path.join(outdir, f"curriculum_size.txt"), "a") as fh:
                fh.writelines(f"{n_instances}\n")
 
        print("Evaluating")
        eval_hook(model, eval_env, total_steps, outdir, "eval")
        eval_hook(model, test_env, total_steps, outdir, "test")