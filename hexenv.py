from hexgame import HexGame
import gymnasium as gym
from gymnasium import spaces
import random


class Opponent:
    def __init__(self, model):
        self.model = model

    def predict(self, game, player, **options):
        if self.model:
            obs = player * game.board
            action, _ = self.model.predict(obs, **options)
            action = int(action)
        else:
            action = random.choice(game.legal_moves())

        return action


class HexEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(size * size,))
        self.size = size
        self.opponent = Opponent(None)
        self.game = HexGame(size)
        self.player = 1
        self.go_first = False
        self.render_mode = render_mode
        self.info = {
            "wins": {-1: 0, 1: 0},
            "winner": 0,
            "board": "",
        }

    def _get_obs(self):
        """Convert to observation format"""
        return self.player * self.game.board

    def _get_info(self):
        """Get auxilary info for debugging"""
        return self.info

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.game = HexGame(self.size)
        self.go_first = not self.go_first
        if not self.go_first:
            action = self.opponent.predict(
                self.game,
                -self.player,
                deterministic=True,
                action_masks=self.game.board == 0,
            )
            self.game.move(action, -self.player)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep"""

        win = self.game.move(action, self.player)
        owin = False

        reward = 0
        winner = 0
        if not win:
            oaction = self.opponent.predict(
                self.game,
                -self.player,
                deterministic=True,
                action_masks=self.game.board == 0,
            )
            owin = self.game.move(oaction, -self.player)
            if owin:
                reward = -1
                winner = -1
        else:
            reward = 1
            winner = 1

        obs = self._get_obs()
        self.info["board"] = str(self.game)
        info = self._get_info()
        truncated = False
        terminated = winner != 0

        if winner:
            wins = self.info["wins"]
            wins[winner] += 1
            N = sum(wins.values())
            if N % 100 == 0:
                print(f"Win rate = {100 * wins[self.player] / N:.1f}%")

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(self.game)

    def action_masks(self):
        """Mask off illegal moves"""
        return self.game.board == 0

    def set_opponent(self, opponent):
        """Install a new opponent"""
        self.opponent = opponent
