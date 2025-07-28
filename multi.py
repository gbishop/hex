"""Try training against multiple opponents"""

from myArgs import Parse
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from hexenv import HexSelfPlayEnv, Opponent
from typing import Callable
from gymnasium import Env
import os


def main():
    args = Parse(
        dir=str,
        size=int,
        seed=None,
        train=12000,
        evaluate=100,
        threshold=0.65,
        environments=6,
        learning_rate=0.0003,
        rounds=10,
        verbose=False,
        progress=True,
        _config="{dir}/config.json",
    )

    os.makedirs(args.dir, exist_ok=True)
    os.chdir(args.dir)

    bootstrap(args)

    if args.verbose:
        print(args)

    if args.seed is not None:
        set_random_seed(args.seed)

    ordered_models = OrderModels(args)

    opponents = []
    for name in ordered_models.names:
        model = MaskablePPO.load(name, verbose=args.verbose)
        opponents.append(Opponent(args.size, model))

    for round in range(args.rounds):
        least = ordered_models.names[0]
        model = train(least, opponents, args)
        model.save(least)
        opponent = Opponent(
            args.size, MaskablePPO.load(least, device="cpu", verbose=args.verbose)
        )
        opponents[0] = opponent
        oscore = ordered_models.scores[least]
        ordered_models.update_scores(least)
        nscore = ordered_models.scores[least]
        if args.progress:
            print(f"{least} {oscore:.2f}->{nscore:.2f}")


class OrderModels:
    def __init__(self, args):
        # collect the models
        self.args = args
        self.names = [f"model{i:02d}" for i in range(args.environments)]
        self.pair_scores: dict[tuple[str, str], float] = {}

        self.scores = {name: 0.0 for name in self.names}
        if args.progress:
            print("comparing opponents")
        for i in range(args.environments - 1):
            m1 = self.names[i]
            for j in range(i + 1, args.environments):
                m2 = self.names[j]
                rate = self.compare(m1, m2)
                self.pair_scores[(m1, m2)] = rate
                self.pair_scores[(m2, m1)] = 1 - rate
                self.scores[m1] += rate
                self.scores[m2] += 1 - rate

        self.names.sort(key=lambda name: self.scores[name])
        # for name in self.names:
        #     print(name, self.scores[name])

    def update_scores(self, m1: str):
        for m2 in self.names:
            if m1 == m2:
                continue
            nrate = self.compare(m1, m2)
            orate = self.pair_scores[(m1, m2)]
            self.pair_scores[(m1, m2)] = nrate
            self.pair_scores[(m2, m1)] = 1 - nrate
            self.scores[m1] += nrate - orate
            self.scores[m2] += orate - nrate
        self.names.sort(key=lambda name: self.scores[name])
        # for name in self.names:
        #     print(name, self.scores[name])

    def compare(self, m1: str, m2: str):
        args = self.args
        opponent = Opponent(
            args.size, MaskablePPO.load(m2, device="cpu", verbose=args.verbose)
        )

        def make_hex_env(
            size: int, opponent: Opponent, seed: int | None = None, **kwargs
        ):
            def thunk():
                env = HexSelfPlayEnv(size, opponent, **kwargs)
                env.reset(seed=seed)
                return env

            return thunk

        env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
            make_hex_env(
                args.size,
                opponent,
                render_mode=None,
                random_moves=0,
                seed=args.seed + i if args.seed is not None else None,
            )
            for i in range(args.environments)
        ]

        env = DummyVecEnv(env_fns)

        env.seed(args.seed)

        model = MaskablePPO.load(m1, env, device="cpu", verbose=args.verbose)

        obs = env.reset()
        wins = 0
        games = 0
        while True:
            assert isinstance(obs, np.ndarray)
            actions, _ = model.predict(obs, action_masks=obs == 0)
            obs, rewards, dones, _ = env.step(actions)
            for i, done in enumerate(dones):
                if done:
                    games += 1
                    if rewards[i] == 1:
                        wins += 1

            if games >= args.evaluate:
                break
        win_rate = wins / games
        return win_rate

    def make_hex_env(
        self, size: int, opponent: Opponent, seed: int | None = None, **kwargs
    ):
        def thunk():
            env = HexSelfPlayEnv(size, opponent, **kwargs)
            env.reset(seed=seed)
            return env

        return thunk


class EpisodeEvalCallback(BaseCallback):
    """
    Call a function when the moving average of rewards exceeds a threshold

    :param period: The window for the moving average.
    :param threshold: The moving average threshold.
    :param max_episodes: Quit after this many.
    :param verbose: Verbosity level.
    """

    def __init__(
        self,
        period: int,
        threshold: float,
        max_episodes: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.period = period
        self.threshold = threshold
        self.max_episodes = max_episodes
        self.n_episodes = 0
        self.total_episodes = 0
        self.recent_rewards = np.zeros((period,))

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        self.total_episodes += np.sum(dones)

        for reward in rewards[dones == 1]:
            self.recent_rewards[self.n_episodes % self.period] = reward
            self.n_episodes += 1

        self.recent_rewards = self.recent_rewards[-self.period :]

        if self.n_episodes > self.period:
            self.mean = (np.mean(self.recent_rewards) + 1) * 0.5
            if self.mean > self.threshold:
                if self.verbose:
                    print(f"Quitting on threshold exceeded")
                return False

        if self.total_episodes > self.max_episodes:
            if self.verbose:
                print(f"Quitting after {self.total_episodes}.")
            return False

        return True


def train(name: str | None, opponents: list[Opponent], args, **kwargs):
    def make_hex_env(size: int, opponent: Opponent, seed: int | None = None, **kwargs):
        def thunk():
            env = HexSelfPlayEnv(size, opponent, **kwargs)
            env.reset(seed=seed)
            return env

        return thunk

    random_moves = [0, 0, 0, 0, 0, 0]
    env_fns: list[Callable[[], Env[np.ndarray, int]]] = [
        make_hex_env(
            args.size,
            opponents[i % len(opponents)],
            render_mode="human",
            random_moves=random_moves[i],
            seed=args.seed + i if args.seed is not None else None,
        )
        for i in range(args.environments)
    ]

    env = DummyVecEnv(env_fns)

    if name:
        model = MaskablePPO.load(
            name, env, learning_rate=args.learning_rate, verbose=args.verbose
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            verbose=args.verbose,
            **kwargs,
        )

    callback_win_threshold = EpisodeEvalCallback(
        period=args.evaluate,
        threshold=args.threshold,
        max_episodes=args.train,
        verbose=0,
    )

    model.learn(
        1_000_000_000,
        callback=callback_win_threshold,
        reset_num_timesteps=False,
    )
    return model


def bootstrap(args):
    """Initialize the opponents"""
    if (
        len([name for name in os.listdir(".") if name.endswith(".zip")])
        < args.environments
    ):
        for i in range(args.environments):
            name = f"model{i:02d}"
            if args.progress:
                print("bootstrap", name)
            model = train(None, [Opponent(args.size)], args)
            model.save(name)


if __name__ == "__main__":
    main()
