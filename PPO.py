from hexenv import HexEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
from typing import cast

env = HexEnv(5, render_mode="human")

obs, info = env.reset()

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
assert vec_env
obs = vec_env.reset()

done = False
reward = 0
for i in range(100):
    action_masks = get_action_masks(env)
    action, _ = model.predict(
        cast(np.ndarray, obs), action_masks=action_masks, deterministic=True
    )
    obs, reward, done, info = vec_env.step(action)

    if reward == 1:
        print("win")
        print(info[0]["board"])
        break
