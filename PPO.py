from hexenv import HexEnv, Opponent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
from typing import cast
import os
import os.path as osp

SIZE = 5
GENERATIONS = 5
STEPS = 1000
DIR = "MaskablePPO"
os.makedirs(DIR, exist_ok=True)

env = HexEnv(5, render_mode="human")

obs, info = env.reset()

savedModels = sorted(os.listdir(DIR))
print(savedModels)
firstGen = 0
if savedModels:
    latest = savedModels[-1]
    model = MaskablePPO.load(osp.join(DIR, latest), env=env)
    firstGen = len(savedModels)
else:
    model = MaskablePPO("MlpPolicy", env, verbose=1)

for gen in range(firstGen, firstGen + GENERATIONS):
    print(f"{gen=}")
    model.learn(total_timesteps=STEPS)
    name = osp.join(DIR, f"model_{gen:03d}")
    model.save(name)

    opponent = Opponent(MaskablePPO.load(name))
    env.set_opponent(opponent)

vec_env = model.get_env()
assert vec_env
obs = vec_env.reset()

done = False
reward = 0
vec_env.render("human")
for i in range(100):
    action_masks = get_action_masks(env)
    action, _ = model.predict(
        cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
    )
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

    if reward == 1:
        print("win")
        break
