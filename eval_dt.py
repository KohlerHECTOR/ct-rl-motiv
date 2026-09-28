"""Evaluate, at every frametime in dt_utils.MULTS:
  - transfer: the policies trained at the default frametime (models/ppo_mult1_seed*.zip)
  - trained:  the policies trained at that frametime (models/ppo_mult<m>_seed*.zip)
(all trained by train_dt.py; at the default frametime transfer and trained are the same model)
  - random:   uniformly random actions
and save one row per episode to eval_dt_episodes.csv.

Usage: python eval_dt.py [--episodes 50] [--jobs N] [--seeds 0 1 2]
"""
import argparse
import os
from fractions import Fraction
from multiprocessing import Pool

import pandas as pd
import torch
from stable_baselines3 import PPO

from dt_utils import ROOT, MULTS, make_env, model_path, frametime, max_steps

TRANSFER_MULT = Fraction(1)  # the transferred policy is the one trained at the default frametime
OUT_CSV = os.path.join(ROOT, 'eval_dt_episodes.csv')


def rollout(env, act, n_episodes, seed):
    """Run n_episodes, return a list of (return, length, unstable) per episode."""
    episodes = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_episodes):
        ret, length, done, info = 0.0, 0, False, {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs))
            ret += reward
            length += 1
            done = terminated or truncated
        episodes.append((ret, length, info['unstable']))
        obs, _ = env.reset()
    return episodes


def evaluate_mult(args):
    mult, n_episodes, seeds = args
    torch.set_num_threads(1)
    env = make_env(mult)

    policies = {}
    for seed in seeds:
        for policy, train_mult in [('transfer', TRANSFER_MULT), ('trained', mult)]:
            if os.path.exists(model_path(train_mult, seed)):
                policies[(policy, seed)] = PPO.load(model_path(train_mult, seed), device='cpu')
            else:
                print(f'WARNING: missing {model_path(train_mult, seed)}, run train_dt.py first', flush=True)

    rows = []
    for (policy, seed), model in list(policies.items()) + [(('random', 0), None)]:
        if model is None:
            env.action_space.seed(seed)
            act = lambda obs: env.action_space.sample()
        else:
            act = lambda obs, model=model: model.predict(obs, deterministic=True)[0]
        for ep, (ret, length, unstable) in enumerate(rollout(env, act, n_episodes, seed=1000 + seed)):
            rows.append({'mult': float(mult), 'mult_str': str(mult), 'frametime': frametime(mult),
                         'dt': env.unwrapped.dt, 'policy': policy, 'seed': seed, 'episode': ep,
                         'return': ret, 'length': length, 'return_per_step': ret / length,
                         'unstable': unstable})
    env.close()
    print(f'done mult={mult} (frametime={frametime(mult):g}, {max_steps(mult)} steps/episode)', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    with Pool(args.jobs) as pool:
        results = pool.map(evaluate_mult, [(mult, args.episodes, args.seeds) for mult in MULTS])
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(OUT_CSV, index=False)
    print(f'saved {OUT_CSV}')
    print(df.groupby(['mult_str', 'policy'], sort=False)[['return_per_step', 'unstable']].mean().unstack())
