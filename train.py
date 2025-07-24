from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from gymnasium import Env
from hexenv import HexSelfPlayEnv, Opponent
from typing import Callable
import numpy as np
import argparse
from filemanager import FileManager

parser = argparse.ArgumentParser(
    description="Train PPO on Hex",
)
parser.add_argument("--size", type=int, default=5)
parser.add_argument("--seed", type=int)
parser.add_argument("--train_games", type=int, default=1200)
parser.add_argument("--evaluate_games", type=int, default=400)
parser.add_argument("--threshold", type=float, default=0.55)
parser.add_argument("--environments", type=int, default=6)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--dir", default="models")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

print(args)

if args.seed is not None:
    set_random_seed(args.seed)


class EpisodeEvalCallback(BaseCallback):
    """
    Call a function when the moving average of rewards exceeds a threshold

    :param update: The function to call.
    :param period: The window for the moving average.
    :param threshold: The EMA threshold.
    :param verbose: Verbosity level.
    """

    def __init__(
        self, update: Callable, period: int, threshold: float, verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.update = update
        self.period = period
        self.threshold = threshold
        self.n_episodes = 0
        self.recent_rewards = np.zeros((period,))

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        for reward in rewards[dones == 1]:
            self.recent_rewards[self.n_episodes % self.period] = reward
            self.n_episodes += 1

        self.recent_rewards = self.recent_rewards[-self.period :]

        if self.n_episodes > self.period:
            self.mean = (np.mean(self.recent_rewards) + 1) * 0.5
            if self.mean > self.threshold:
                self.update(self.model, self.n_episodes, self.mean, self.verbose)
                self.n_episodes = 0
                self.ema = 0

        return True


def make_hex_env(size: int, opponent: Opponent, seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(size, opponent, **kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")
    fm = FileManager(f"{args.dir}-{args.size}")

    latest, generation = fm.latest()
    print(f"{latest=} {generation=}")

    size = args.size
    opponent = Opponent(size)

    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            size,
            opponent,
            render_mode="human",
            random_moves=i // 2,
            seed=args.seed + i if args.seed is not None else None,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    if latest:
        model = MaskablePPO.load(latest, env, verbose=args.verbose)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=args.verbose)

    def saveAndUpdateOpponent(model, episodes, mean, verbose=1):
        latest = fm.save(model)
        if verbose:
            print(f"save {latest=} {episodes=} {mean=}")
        opponent.update_model(MaskablePPO.load(latest))

    callback_ema_rewards = EpisodeEvalCallback(
        update=saveAndUpdateOpponent,
        period=args.evaluate_games,
        threshold=args.threshold,
        verbose=1,
    )

    max_episodes = args.train_games // args.environments
    callback_max_episodes = StopTrainingOnMaxEpisodes(
        max_episodes=max_episodes,
        verbose=1,
    )

    model.learn(
        1_000_000_000,
        callback=[callback_ema_rewards, callback_max_episodes],
        reset_num_timesteps=False,
    )
    fm.save(model)
