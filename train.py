from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from gymnasium import Env
from hexenv import HexSelfPlayEnv, Opponent
from typing import Callable
import numpy as np
import argparse
from modelmanager import ModelManager
from table import Table

parser = argparse.ArgumentParser(
    description="Train PPO on Hex",
)
parser.add_argument("base", type=str)
parser.add_argument("size", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--train", type=int, default=1200)
parser.add_argument("--evaluate", type=int, default=100)
parser.add_argument("--threshold", type=float, default=0.75)
parser.add_argument("--environments", type=int, default=6)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--learning_rate", type=float, default=0.0003)

args = parser.parse_args()

if args.verbose:
    print(args)

if args.seed is not None:
    set_random_seed(args.seed)


class EpisodeEvalCallback(BaseCallback):
    """
    Call a function when the moving average of rewards exceeds a threshold

    :param update: The function to call.
    :param period: The window for the moving average.
    :param threshold: The moving average threshold.
    :param max_episodes: Quit after this many.
    :param verbose: Verbosity level.
    """

    def __init__(
        self,
        update: Callable,
        period: int,
        threshold: float,
        max_episodes: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.update = update
        self.period = period
        self.threshold = threshold
        self.max_episodes = max_episodes
        self.n_episodes = 0
        self.total_episodes = 0
        self.recent_rewards = np.zeros((period,))

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        self.total_episodes += np.sum(dones)

        for reward in rewards[dones == 1]:
            self.recent_rewards[self.n_episodes % self.period] = reward
            self.n_episodes += 1

        self.recent_rewards = self.recent_rewards[-self.period :]

        if self.n_episodes > self.period:
            self.mean = (np.mean(self.recent_rewards) + 1) * 0.5
            if self.mean > self.threshold:
                self.update(
                    self.model,
                    self.n_episodes,
                    self.mean,
                    100 * self.total_episodes / self.max_episodes,
                    self.verbose,
                )
                self.n_episodes = 0

        if self.total_episodes > self.max_episodes:
            if self.verbose:
                print(f"Quitting after {self.total_episodes}.")
            return False

        return True


def make_hex_env(size: int, opponent: Opponent, seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(size, opponent, **kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")
    manager = ModelManager(args.base, args.size)

    latest, latest_path = manager.latest()

    size = args.size
    opponent = Opponent(size)
    if latest:
        opponent.update_model(MaskablePPO.load(latest_path, verbose=args.verbose))

    random_moves = [0, 0, 0, 0, 2, 4]
    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            size,
            opponent,
            render_mode="human",
            random_moves=random_moves[i],
            seed=args.seed + i if args.seed is not None else None,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    if latest:
        model = MaskablePPO.load(
            latest_path, env, learning_rate=args.learning_rate, verbose=args.verbose
        )
    else:
        model = MaskablePPO(
            "MlpPolicy", env, learning_rate=args.learning_rate, verbose=args.verbose
        )

    saveTable = Table(
        ["Name", "Episodes", "Mean", "Progress"], ["", "5d", "4.2f", "5.1f"]
    )

    def saveAndUpdateOpponent(model, episodes, mean, progress, verbose=1):
        latest, latest_path = manager.save(model)
        if verbose:
            saveTable.print(latest, episodes, mean, progress)
        opponent.update_model(MaskablePPO.load(latest_path))

    callback_win_threshold = EpisodeEvalCallback(
        update=saveAndUpdateOpponent,
        period=args.evaluate,
        threshold=args.threshold,
        max_episodes=args.train,
        verbose=1,
    )

    model.learn(
        1_000_000_000,
        callback=callback_win_threshold,
        reset_num_timesteps=False,
    )
    manager.save(model)
