from hexenv import HexEnv
from hexgame import HexGame, player2
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.utils import set_random_seed
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
WIN_RATE_WINDOW = 1000
WIN_RATE_THRESHOLD = 0.55


class Opponent:
    """Always player2"""

    def __init__(self):
        self.model: MaskablePPO | None = None
        self.model_generation = 0
        self.model_weights = {0: 0.5}

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

    def update(self, lose_rate: float):
        self.model_weights[self.model_generation] = lose_rate
        model_file_list = sorted(os.listdir(DIR))
        weights = [self.model_weights.get(i, 0.5) for i in range(len(model_file_list))]
        choice = random.choices(range(len(model_file_list)), weights=weights, k=1)[0]
        print(f"{choice=} {weights=}")
        self.model_generation = choice
        filename = osp.join(DIR, model_file_list[choice])
        self.model = MaskablePPO.load(filename)


class HexSelfPlayEnv(HexEnv):
    def __init__(self, size: int, opponent: Opponent, *args, **kwargs):
        super().__init__(size, *args, **kwargs)
        self.wins = 0
        self.games = 0
        self.generation = 0
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
        obs, reward, terminated, truncated, info = super().step(action)

        if reward != 1:
            owin = self.opponent.move(self.game)
            if owin:
                terminated = True
                reward = -1

        if terminated:
            self.games += 1
            if reward == 1:
                self.wins += 1
            if self.games >= WIN_RATE_WINDOW:
                win_rate = self.wins / self.games
                if win_rate > WIN_RATE_THRESHOLD:
                    self.generation += 1
                    print(
                        f"win rate = {win_rate*100:.1f}% {self.generation} vs {self.opponent.model_generation}"
                    )
                    save_file = os.path.join(DIR, f"history{self.generation:03d}")
                    model.save(save_file)
                    self.opponent.update(1 - win_rate)

                    self.wins = 0
                    self.games = 0

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    set_random_seed(SEED)
    os.makedirs(DIR, exist_ok=True)

    env = HexSelfPlayEnv(SIZE, Opponent(), render_mode="human")

    obs, info = env.reset()

    savedModels = sorted(glob(f"{DIR}/*"))
    env.generation = len(savedModels)
    if env.generation:
        latest = savedModels[-1]
        model = MaskablePPO.load(latest, env=env)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=0)

    model.learn(total_timesteps=NUM_TIMESTEPS)

    model.save(osp.join(DIR, "final_model"))

    rewards = []
    reward = 0
    for i in range(10):
        done = False
        env.verbose = True
        obs, info = env.reset()
        while not done:
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
            )
            obs, reward, done, _, info = env.step(action)
        rewards.append(int(reward))
    print(rewards)
