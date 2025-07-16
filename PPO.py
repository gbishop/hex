from hexenv import HexEnv
from hexgame import HexGame, player2
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EventCallback
import numpy as np
from typing import cast
import os
import os.path as osp
import random
from glob import glob

SEED = 42
SIZE = 5

DIR = "MaskablePPO"

NUM_TIMESTEPS = int(1e9)
WIN_RATE_WINDOW = 400
WIN_RATE_THRESHOLD = 0.65
NUM_OPPONENTS = 7
ROUNDS = 10


class Opponent:
    """Always player2"""

    def __init__(self, model: MaskablePPO | None = None):
        self.model = model

    def move(self, game: HexGame) -> bool:
        """Make the opponents move"""
        if self.model:
            # transpose the board
            obs = -game.board.reshape((game.size, game.size)).transpose().flatten()
            action, _ = self.model.predict(
                obs, action_masks=obs == 0, deterministic=False
            )
            action = int(action)
            # transpose the action back
            r, c = game.rc(action)
            action = game.index(c, r)
        else:
            action = random.choice(game.legal_moves())

        win = game.move(action, player2)
        return win

    def set_model(self, model: MaskablePPO | None):
        self.model = model


class HexSelfPlayEnv(HexEnv):
    def __init__(self, size: int, opponent: Opponent, *args, **kwargs):
        super().__init__(size, *args, **kwargs)
        self.wins = 0
        self.games = 0
        self.first = True
        self.opponent = opponent

    def reset(self, *, seed: int | None = None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)

        if not self.first:
            # let the opponent go first
            self.opponent.move(self.game)
            obs = self._get_obs()
            info = self._get_info()

        self.first = not self.first

        return obs, info

    def step(self, action):
        obs, reward, self.terminated, truncated, info = super().step(action)

        if not self.terminated:
            owin = self.opponent.move(self.game)
            if owin:
                self.terminated = True
                reward = -1

        if self.terminated:
            self.games += 1
            if reward == 1:
                self.wins += 1

        return obs, reward, self.terminated, truncated, info

    def set_opponent(self, opponent: Opponent):
        self.opponent = opponent


class HexEvalCallback(EventCallback):
    def __init__(self, env: HexSelfPlayEnv, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.env = env

    def _on_step(self) -> bool:
        env = self.env
        if env.terminated and env.games >= WIN_RATE_WINDOW:
            win_rate = env.wins / env.games
            if win_rate > WIN_RATE_THRESHOLD:
                if self.verbose:
                    print(
                        f"Quit after {env.games} and {self.num_timesteps} steps, rate={win_rate}"
                    )
                return False
        return True

    def _on_training_start(self) -> None:
        self.env.games = 0
        self.env.wins = 0
        return super()._on_training_start()


def fname(i):
    return f"{DIR}/model{i:03d}"


if __name__ == "__main__":
    os.makedirs(DIR, exist_ok=True)

    envs = [
        HexSelfPlayEnv(SIZE, Opponent(), render_mode="human")
        for _ in range(NUM_OPPONENTS)
    ]

    # create initial opponents
    print("initialize opponents")
    opponents = []
    for i in range(NUM_OPPONENTS):
        name = fname(i)
        if osp.exists(name + ".zip"):
            print("loading", name)
            model = MaskablePPO.load(name)
        else:
            print("training", name)
            model = MaskablePPO("MlpPolicy", envs[i], verbose=0)
            callback = HexEvalCallback(envs[i])
            model.learn(total_timesteps=NUM_TIMESTEPS, callback=callback)
            model.save(name)
            model = MaskablePPO.load(name)
        opponents.append(Opponent(model))

    print("training")
    for round in range(ROUNDS):
        # train a new model until it bests all the opponents
        env = envs[round % NUM_OPPONENTS]
        model = MaskablePPO("MlpPolicy", env, verbose=0)
        for i in range(NUM_OPPONENTS):
            env.set_opponent(opponents[i])
            callback = HexEvalCallback(env, verbose=1)
            env.reset()
            model.learn(total_timesteps=NUM_TIMESTEPS, callback=callback)
            print(
                round,
                i,
                callback.num_timesteps / env.games,
                env.games,
                callback.num_timesteps,
            )

        # replace a random opponent
        replace = round % NUM_OPPONENTS
        print("replacing", replace)
        name = fname(replace)
        model.save(name)
        opponents[replace] = Opponent(MaskablePPO.load(name))
