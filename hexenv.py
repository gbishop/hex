from hexgame import HexGame, player1
import gymnasium as gym
from gymnasium import spaces
from canonicalize import Canonicalizer


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
        self.canon = Canonicalizer(size)

    def get_obs_info(self, player=player1):
        """Convert to observation format"""
        obs, inverse, sign = self.canon.canocalize(self.game.board, player)
        self.info.update(
            dict(inverse=inverse, sign=sign, action_masks=obs == 0, obs=obs)
        )
        return obs, self.info

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed, *kwargs)
        self.game = HexGame(self.size)

        observation, info = self.get_obs_info()
        return observation, info

    def step(self, action):
        """Execute one timestep"""

        naction = self.info["inverse"][int(action)]
        try:
            win = self.game.move(naction, player1)
        except AssertionError:
            print(f"{action=} {naction=}", self.info)
            raise

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
        return self.info["action_masks"]

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
