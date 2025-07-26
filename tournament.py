"""Play models against one another"""

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env
from hexenv import HexSelfPlayEnv, Opponent
from typing import Callable
import numpy as np
import argparse
import sqlite3
import os.path as osp
from table import Table
from modelmanager import ModelManager

parser = argparse.ArgumentParser(
    description="Evaluate model win stats",
)
parser.add_argument("base")
parser.add_argument("size", type=int)
parser.add_argument("evaluate_games", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--environments", type=int, default=5)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

if args.verbose:
    print(args)

manager = ModelManager(args.base, args.size)

connection = sqlite3.connect(osp.join(manager.dir, "tournament.db"))
cursor = connection.cursor()

# create tables
cursor.execute(
    """
create table if not exists model(
    id integer primary key,
    name text unique
)
"""
)

cursor.execute(
    """
create table if not exists pairings(
    player1 integer,
    player2 integer,
    score real,
    foreign key(player1) references model(id),
    foreign key(player2) references model(id)
)
"""
)

# add models to be compared
fnames = manager.list()[:: args.step]
cursor.executemany(
    """
insert or ignore into model (name) values (?)
""",
    [(fname,) for fname in fnames],
)

# list the pairings that have not been evaluated
cursor.execute(
    """
select m1.id, m1.name, m2.id, m2.name
from
    model as m1
cross join
    model as m2
left join
    pairings as p
on
    m1.id = p.player1 and m2.id = player2
where 
    p.player1 is null and m1.id > m2.id;
"""
)


if args.seed is not None:
    set_random_seed(args.seed)


def make_hex_env(size: int, opponent: Opponent, seed: int | None = None, **kwargs):
    def thunk():
        env = HexSelfPlayEnv(size, opponent, **kwargs)
        env.reset(seed=seed)
        return env

    return thunk


def compare(m1: str, m2: str, verbose: int = 0):

    opponent = Opponent(args.size)
    opponent.update_model(manager.load(m2, device="cpu"))

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

    model = manager.load(m1, env=env, verbose=verbose, device="cpu")

    obs = env.reset()
    wins = 0
    games = 0
    while True:
        assert isinstance(obs, np.ndarray)
        actions, _ = model.predict(obs, action_masks=obs == 0)
        obs, rewards, dones, _ = env.step(actions)
        # games += np.sum(dones)
        # reward_sum += np.sum(rewards[dones == 1])
        for i, done in enumerate(dones):
            if done:
                games += 1
                if rewards[i] == 1:
                    wins += 1

        if games >= args.evaluate_games:
            break
    win_rate = wins / games
    return win_rate


T = Table(["Model 1", "Model 2", "Score"], ["s", "s", "4.2f"])
for id1, name1, id2, name2 in cursor.fetchall():
    score = compare(name1, name2)
    T.print(name1, name2, score)
    cursor.execute(
        """
        insert into pairings (player1, player2, score)
        values (?, ?, ?)
    """,
        (id1, id2, score),
    )

    connection.commit()
