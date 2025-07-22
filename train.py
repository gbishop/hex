from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from gymnasium import Env
from hexenv import HexSelfPlayEnv
import os
from typing import Callable, cast
import numpy as np
import argparse
import re
from glob import glob

parser = argparse.ArgumentParser(
    description="Train PPO on Hex",
)
parser.add_argument("--size", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_games", type=int, default=600)
parser.add_argument("--evaluate_games", type=int, default=100)
parser.add_argument("--environments", type=int, default=6)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--dir", default="models")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

print(args)

set_random_seed(args.seed)


class Files:
    def __init__(self, dir, base="model"):
        self.dir = dir
        self.base = base
        self.generation = 0
        os.makedirs(self.dir, exist_ok=True)

    def latest(self):
        files = sorted(glob(f"{self.dir}/*"))
        self.generation = 0
        last = ""
        if files:
            last = files[-1]
            nums = re.findall(r"\d\d\d", last)
            print(f"{nums=}")
            if nums:
                self.generation = int(nums[0])
                print(f"{self.generation=}")

        return last, self.generation

    def save(self, model):
        self.generation += 1
        name = f"{self.dir}/{self.base}{self.generation:03d}"
        model.save(name)
        return name


def make_hex_env(seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(**kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")
    fm = Files(f"{args.dir}-{args.size}")

    latest, generation = fm.latest()
    print(f"{latest=} {generation=}")

    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            size=args.size,
            render_mode="human",
            random_moves=i,
            seed=args.seed + i,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    if latest:
        model = MaskablePPO.load(latest, env, verbose=args.verbose)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=args.verbose)

    for round in range(args.rounds):
        if latest:
            opponent = MaskablePPO.load(latest, verbose=args.verbose)
            for i in range(len(env.envs)):
                cast(HexSelfPlayEnv, env.envs[i]).opponent = opponent

        print(f"learning {round=}")

        max_episodes = args.train_games // args.environments
        callback_max_episodes = StopTrainingOnMaxEpisodes(
            max_episodes=max_episodes, verbose=args.verbose
        )

        ts0 = model.num_timesteps
        model.learn(
            1_000_000_000, callback=callback_max_episodes, reset_num_timesteps=False
        )
        ts1 = model.num_timesteps
        print(f"timesteps/game = {(ts1 - ts0) / args.train_games}")
        latest = fm.save(model)

        print("evaluating")
        env.seed(args.seed)
        obs = env.reset()
        total_wins = np.zeros(args.environments)
        total_games = np.zeros(args.environments)
        for g in range(args.evaluate_games * args.environments * 10):
            assert isinstance(obs, np.ndarray)
            action, _ = model.predict(obs, action_masks=obs == 0)
            obs, rewards, dones, info = env.step(action)
            total_wins += rewards == 1
            total_games += dones
            if np.sum(total_games) >= args.evaluate_games:
                break

        print(f"games = {total_games}")
        rate = 100 * total_wins / total_games
        mean_rate = f"{100 * np.sum(total_wins) / np.sum(total_games):.1f}"
        print(
            mean_rate,
            np.array2string(rate, precision=2),
        )
