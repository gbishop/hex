"""Display model rank"""

import sqlite3
import argparse
import os.path as osp
from table import Table
from modelmanager import ModelManager

parser = argparse.ArgumentParser(
    description="Evaluate model win stats",
)
parser.add_argument("base")
parser.add_argument("size")
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

if args.verbose:
    print(args)

manager = ModelManager(args.base, args.size)

connection = sqlite3.connect(osp.join(manager.dir, "tournament.db"))
cursor = connection.cursor()

cursor.execute(
    """
    select id, name from model
    order by name
    """
)
ids = {}
names = {}
for id, name in cursor.fetchall():
    ids[name] = id
    names[id] = name

scores = {}
cursor.execute(
    """
    select player1, player2, score from pairings
    """
)
for p1, p2, score in cursor.fetchall():
    scores[p1] = scores.get(p1, 0) + score
    scores[p2] = scores.get(p2, 0) + (1 - score)

ranking = sorted(scores.items(), key=lambda t: t[1], reverse=True)
T = Table(["Name", "Score"], ["", "5.2f"])
for id, score in ranking:
    T.print(names[id], score)
