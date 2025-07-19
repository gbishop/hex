from gymnasium import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from hexenv import HexEnv
from hexgame import player1, player2
from sb3_contrib import MaskablePPO
import numpy as np
import numpy.typing as npt
import os
from typing import Callable, Any

SEED = 42
set_random_seed(SEED)
SIZE = 5

DIR = "MaskablePPO"

TRAIN_GAMES = 100
EVALUATE_GAMES = 100
ENVIRONMENTS = 6
TIMESTEPS = 10 * ENVIRONMENTS * TRAIN_GAMES
OPPONENTS = 7
ROUNDS = 10


class HexSelfPlayEnv(HexEnv):
    def __init__(
        self,
        size: int = 0,
        opponent: MaskablePPO | None = None,
        random_moves=0,
        **kwargs,
    ):
        super().__init__(size, **kwargs)
        self.first = True
        self.opponent = opponent
        self.random_moves = random_moves

    def opponent_move(self, obs: npt.NDArray[np.float32], info: dict[str, Any]):
        if self.opponent:
            action, _ = self.opponent.predict(
                obs, action_masks=obs == 0, deterministic=False
            )
            action = self.info["inverse"][action]
        else:
            action = self.np_random.choice(self.game.legal_moves())

        win = self.game.move(action, player2)
        obs, info = self.get_obs_info()
        return win, obs, info

    def reset(self, *, seed: int | None = None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)

        for _ in range(self.random_moves):
            move = self.np_random.choice(self.game.legal_moves())
            if self.first:
                self.game.move(move, player1)
            else:
                self.game.move(move, player2)
            self.first = not self.first
            obs, info = self.get_obs_info()

        if not self.first:
            # let the opponent go first
            _, obs, info = self.opponent_move(obs, info)

        self.first = not self.first

        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated:
            win, obs, info = self.opponent_move(obs, info)
            if win:
                terminated = True
                reward = -1

        return obs, reward, terminated, truncated, info


def fname(i: int):
    return f"{DIR}/model{i:03d}"


def make_hex_env(seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(**kwargs)
        env.reset(seed=seed)
        return env

    return thunk


if __name__ == "__main__":
    print("start")
    os.makedirs(DIR, exist_ok=True)

    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(size=SIZE, render_mode="human", random_moves=i, seed=SEED + i)
        for i in range(ENVIRONMENTS)
    ]

    env = DummyVecEnv(env_fns)

    model = MaskablePPO("MlpPolicy", env, verbose=1)
    print("learning")
    model.learn(TIMESTEPS)

    print("evaluating")
    env.seed(SEED)
    obs = env.reset()
    total_wins = np.zeros(ENVIRONMENTS)
    total_games = np.zeros(ENVIRONMENTS)
    for g in range(EVALUATE_GAMES * ENVIRONMENTS * 10):
        assert isinstance(obs, np.ndarray)
        action, _ = model.predict(obs, action_masks=obs == 0)
        obs, rewards, dones, info = env.step(action)
        total_wins += rewards == 1
        total_games += dones
        if np.all(total_games >= TRAIN_GAMES):
            break

    print(f"games = {total_games}")
    rate = 100 * total_wins / total_games
    print(np.array2string(rate, precision=2))
