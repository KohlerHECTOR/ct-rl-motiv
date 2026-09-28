"""Train one default PPO agent per frametime in dt_utils.MULTS (skips models that already exist).

Usage: python train_dt.py [--jobs N] [--seeds 0 1 2] [--timesteps 1000000]
"""
import argparse
import os
from multiprocessing import Pool

import torch
from stable_baselines3 import PPO

from dt_utils import MULTS, MODEL_DIR, make_env, model_path, frametime, max_steps


def train(args):
    mult, seed, timesteps = args
    path = model_path(mult, seed)
    if os.path.exists(path):
        return f'skip {path} (exists)'
    torch.set_num_threads(1)  # one core per worker, parallelism comes from the pool
    env = make_env(mult)
    model = PPO('MlpPolicy', env=env, seed=seed, device='cpu', verbose=0)  # same default PPO as train_halfcheetah.py
    model.learn(timesteps)
    model.save(path)
    env.close()
    return f'done {path} (frametime={frametime(mult):g}, {max_steps(mult)} steps/episode)'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--timesteps', type=int, default=int(1e6))
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    jobs = [(mult, seed, args.timesteps) for mult in MULTS for seed in args.seeds]
    with Pool(args.jobs) as pool:
        for msg in pool.imap_unordered(train, jobs):
            print(msg, flush=True)
