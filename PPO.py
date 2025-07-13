from hexenv import HexEnv, Opponent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
from typing import cast
import os
import os.path as osp
from glob import glob
import random

SIZE = 5
GENERATIONS = 50
STEPS = 5000
DIR = "MaskablePPO"

if __name__ == "__main__":
    os.makedirs(DIR, exist_ok=True)

    env = HexEnv(SIZE, render_mode="human")

    obs, info = env.reset()

    savedModels = sorted(glob(f"{DIR}/*"))
    print(savedModels)
    firstGen = 0
    if len(savedModels) > 1:
        latest = savedModels[-1]
        model = MaskablePPO.load(latest, env=env)
        env.set_opponent(Opponent(None))
        firstGen = len(savedModels)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=0)
        env.set_opponent(Opponent(None))

    ogen = -1
    for gen in range(firstGen, firstGen + GENERATIONS):
        print(f"{gen=} opponent={ogen}")
        model.learn(total_timesteps=STEPS)
        name = osp.join(DIR, f"model_{gen:03d}")
        model.save(name)

        if len(savedModels) > 0:
            ogen = random.randint(0, len(savedModels) - 1)
            opponent = Opponent(MaskablePPO.load(savedModels[ogen]))
            env.set_opponent(opponent)
        savedModels.append(name)

    F = True

    if F:
        rewards = []
        reward = 0
        for i in range(10):
            done = False
            env.verbose = True
            obs, info = env.reset()
            while not done:
                action_masks = get_action_masks(env)
                action, _ = model.predict(
                    cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
                )
                obs, reward, done, _, info = env.step(action)
            rewards.append(int(reward))
        print(rewards)

    else:
        vec_env = model.get_env()
        assert vec_env
        obs = vec_env.reset()

        done = False
        rewards = []

        while 1 not in rewards or -1 not in rewards:
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
            )
            obs, reward, done, info = vec_env.step(action)
            # vec_env.render("human")
            # print(info[0]["board"])

            if done:
                rewards += [int(r) for r in reward]

        print(rewards)
