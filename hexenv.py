from hexgame import HexGame, player1
import gymnasium as gym
from gymnasium import spaces


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
