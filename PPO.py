from hexenv import HexEnv
from hexgame import player1, player2
from sb3_contrib import MaskablePPO
import numpy as np
import numpy.typing as npt
from typing import Any

transpose = [
    0,
    5,
    10,
    15,
    20,
    1,
    6,
    11,
    16,
    21,
    2,
    7,
    12,
    17,
    22,
    3,
    8,
    13,
    18,
    23,
    4,
    9,
    14,
    19,
    24,
]


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
            obs = -obs[transpose]
            action, _ = self.opponent.predict(
                obs, action_masks=obs == 0, deterministic=False
            )
            action = transpose[action]
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
