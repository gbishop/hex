from stable_baselines3.common.utils import set_random_seed
from PPO import HexSelfPlayEnv
import os
from typing import Callable
from gymnasium import Env
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
import argparse

parser = argparse.ArgumentParser(
    description="Train PPO on Hex",
)
parser.add_argument("--size", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_games", type=int, default=100)
parser.add_argument("--evaluate_games", type=int, default=100)
parser.add_argument("--environments", type=int, default=6)
parser.add_argument("--opponents", type=int, default=7)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--dir", default="MaskablePPO")
parser.add_argument("-canon_identity", action="store_true")
parser.add_argument("-canon_byvalue", action="store_false")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

print(args)

set_random_seed(args.seed)

TIMESTEPS = 10 * args.environments * args.train_games


def fname(i: int):
    return f"{args.dir}/model{i:03d}"


def make_hex_env(seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(**kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")
    os.makedirs(args.dir, exist_ok=True)

    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            size=args.size,
            render_mode="human",
            random_moves=i,
            seed=args.seed + i,
            canon_identity=args.canon_identity,
            canon_byvalue=args.canon_byvalue,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    model = MaskablePPO("MlpPolicy", env, verbose=args.verbose)
    print("learning")
    model.learn(TIMESTEPS)

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
        if np.all(total_games >= args.evaluate_games):
            break

    print(f"games = {total_games}")
    rate = 100 * total_wins / total_games
    print(np.array2string(rate, precision=2))
