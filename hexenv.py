from hexgame import HexGame, player1, player2
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
import numpy as np
import numpy.typing as npt
from typing import Any


class HexEnv(gym.Env):
    """Always player1"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(size * size,))
        self.size = size
        self.game = HexGame(size)
        self.render_mode = render_mode
        self.verbose = False
        self.info = {}

    def get_obs_info(self):
        """Convert to observation format"""
        return self.game.board, self.info

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed, *kwargs)
        self.game = HexGame(self.size)

        observation, info = self.get_obs_info()
        return observation, info

    def step(self, action):
        """Execute one timestep"""

        action = int(action)
        win = self.game.move(action, player1)

        reward = 1.0 if win else 0.0
        obs, info = self.get_obs_info()
        truncated = False
        terminated = win

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(self.game)

    def action_masks(self):
        """Mask off illegal moves"""
        return self.game.board == 0

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class HexSelfPlayEnv(HexEnv):
    """Add an opponent to the HexEnv"""

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
        self.transpose = (
            np.arange(size * size).reshape((size, size)).transpose().flatten().tolist()
        )

    def opponent_move(self, obs: npt.NDArray[np.float32], info: dict[str, Any]):
        if self.opponent:
            obs = -obs[self.transpose]
            action, _ = self.opponent.predict(
                obs, action_masks=obs == 0, deterministic=False
            )
            action = self.transpose[action]
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
