import os 

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np

from mujoco_playground import registry

ENV_NAME = "Go1JoystickFlatTerrain"
DURATION_SECONDS = 5.0
RENDER_EVERY = 2

CAMERA = "track"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

OUTPUT_PATH = Path("outputs/go1_zero_action.mp4")

def main():
    env_config = registry.get_default_config(ENV_NAME)

    env = registry.load(ENV_NAME, config=env_config)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = reset_fn(rng)

    print("\nObservation")

    command = jp.array(
        [0.0, 0.0, 0.0],
        dtype=jp.float32,
    )

    if "command" in state.info:
        state.info["command"] = command

    action = jp.zeros(
        env.action_size,
        dtype=jp.float32
    )

    num_steps = int(DURATION_SECONDS/ env.dt)

    num_steps = min(num_steps, int(env_config.episode_length))

    rollout = [state]
    rewards = []

    print("\nRollout")
    print("number of steps:", num_steps)

    for step_index in range(num_steps):
        state = step_fn(state,action)

        # 防止环境内部重新采样 command。
        if "command" in state.info:
            state.info["command"] = command

        rollout.append(state)
        rewards.append(float(state.reward))

        if bool(state.done):
            print("terminated at step:", step_index + 1)
            break
    print("collected states:", len(rollout))
    print("mean reward:", float(np.mean(rewards)))
    print("final reward:", rewards[-1])
    print("final done:", bool(state.done))

    trajectory = rollout[::RENDER_EVERY]
    fps = 1.0 / env.dt / RENDER_EVERY

    print("\nRendering")
    print("number of frames:", len(trajectory))
    print("fps:", fps)

    frames = env.render(trajectory, camera=CAMERA, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    media.write_video(
        OUTPUT_PATH,
        frames,
        fps=fps,
    )

    print("\nVideo saved to:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()