"Quick hack to look at the win stats vs number of random moves"

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from gymnasium import Env
from hexenv import HexSelfPlayEnv, Opponent
from typing import Callable
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(
    description="Evaluate model win stats",
)
parser.add_argument("file")
parser.add_argument("size", type=int)
parser.add_argument("evaluate_games", type=int)
parser.add_argument("--opponent")
parser.add_argument("--seed", type=int)
parser.add_argument("--environments", type=int, default=6)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

print(args)

if args.seed is not None:
    set_random_seed(args.seed)


def make_hex_env(size: int, opponent: Opponent, seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(size, opponent, **kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")

    model_name = args.file
    size = args.size

    opponent = Opponent(size)
    if args.opponent:
        opponent_name = args.opponent
    else:
        opponent_name = model_name

    opponent.update_model(MaskablePPO.load(opponent_name))

    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            size,
            opponent,
            render_mode="human",
            random_moves=i,
            seed=args.seed + i if args.seed is not None else None,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    if model_name:
        model = MaskablePPO.load(model_name, env, verbose=args.verbose)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=args.verbose)

    env.seed(args.seed)
    obs = env.reset()
    total_wins = np.zeros(args.environments)
    total_games = np.zeros(args.environments)
    episode_moves: list[list[int]] = [[] for _ in range(args.environments)]
    games: list[dict[tuple, int]] = [{} for _ in range(args.environments)]
    tstart = time.time()
    while True:
        assert isinstance(obs, np.ndarray)
        actions, _ = model.predict(obs, action_masks=obs == 0)
        for i, action in enumerate(actions):
            episode_moves[i].append(int(action))
        obs, rewards, dones, info = env.step(actions)
        for i, done in enumerate(dones):
            if done:
                t = tuple(episode_moves[i])
                episode_moves[i] = []
                games[i][t] = 1 + games[i].get(t, 0)
        total_wins += rewards == 1
        total_games += dones
        if np.all(total_games >= args.evaluate_games):
            break
    tend = time.time()

    print(f"games = {np.sum(total_games)} {total_games}")
    rates = np.round(100 * total_wins / total_games, 1)
    rate = np.round(100 * np.sum(total_wins) / np.sum(total_games), 1)
    print(f"rate = {rate} {rates}")
    gps = np.sum(total_games) / (tend - tstart)
    print(f"{gps:.2f} games/second")

    for i in range(args.environments):
        game = games[i]
        dups = 0
        maxdup = 1
        sumdup = 0
        for t in game:
            if game[t] > 1:
                dups += 1
                sumdup += game[t]
                maxdup = max(maxdup, game[t])
        print(f"env {i} {dups=} {sumdup=} {maxdup=}")
