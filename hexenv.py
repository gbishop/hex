from hexgame import HexGame, player1, player2
import gymnasium as gym
from gymnasium import spaces
import random


class Opponent:
    """Always player2"""

    def __init__(self, model):
        self.model = model

    def move(self, game: HexGame) -> bool:
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


class HexEnv(gym.Env):
    """Always player1"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(size * size,))
        self.size = size
        self.opponent = None
        self.game = HexGame(size)
        self.render_mode = render_mode
        self.info = {
            "wins": {-1: 0, 1: 0, "first": 0, "second": 0},
            "winner": 0,
            "board": "",
            "order": "first",
        }
        self.verbose = False

    def _get_obs(self):
        """Convert to observation format"""
        return self.game.board

    def _get_info(self):
        """Get auxilary info for debugging"""
        return self.info

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.game = HexGame(self.size)

        # let player2 go first half the time
        if self.opponent and random.random() < 0.5:
            # it's move will be random or deterministic
            self.info["order"] = "second"
            if random.random() < 0.5:
                self.log("\nO random")
                self.game.move(random.choice(self.game.legal_moves()), player2)
            else:
                self.log("\nO predicted")
                self.opponent.move(self.game)
            self.log(self.game)
        else:
            self.info["order"] = "first"
            self.log("\nfirst")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep"""

        action = int(action)
        win = self.game.move(action, player1)

        self.log("X\n", self.game)

        reward = -0.01
        winner = 0
        if not win:
            if self.opponent:
                owin = self.opponent.move(self.game)
                self.log("O\n", self.game)
                if owin:
                    reward = -1
                    winner = player2
        else:
            reward = 1
            winner = player1

        if winner:
            self.log(f"{winner=}")

        obs = self._get_obs()
        self.info["board"] = str(self.game)
        info = self._get_info()
        truncated = False
        terminated = winner != 0

        if winner:
            wins = info["wins"]
            wins[winner] += 1
            wins[info["order"]] += 1
            N = wins[player1] + wins[player2]
            first = wins["first"]
            if N % 100 == 0:
                print(f"player 1 win = {100 * wins[player1] / N:.1f}%")
                print(f"first win = {100 * first / N:.1f}%")
                for key in wins:
                    wins[key] = 0

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

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
