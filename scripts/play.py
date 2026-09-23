import os 
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import torch

from rsl_rl.runners import OnPolicyRunner
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params

ENV_NAME = "Go1JoystickFlatTerrain"

DEVICE = "cuda:0"
DEVICE_RANK = 0
SEED = 1

NUM_ENVS = 1

CHECKPOINT_PATH = Path(
    "runs/"
    "Go1JoystickFlatTerrain-20260922-144956-rsl-baseline/"
    "model_4999.pt"
)

OUTPUT_PATH = Path(
    "outputs/go1_checkpoint_rollout.mp4"
)

DURATION_SECONDS = 5.0
RENDER_EVERY = 2
CAMERA = "track"

def main():

    # 加载环境
    env_config = registry.get_default_config(ENV_NAME)
    env_config.impl = "jax"

    env = registry.load(ENV_NAME, config=env_config)

    randomizer = registry.get_domain_randomizer(ENV_NAME)

    rsl_env = wrapper_torch.RSLRLBraxWrapper(
        env=env,
        num_actors=NUM_ENVS,
        seed=SEED,
        episode_length=env_config.episode_length,
        action_repeat=1,
        randomization_fn=randomizer,
        render_callback=None,
        device_rank=DEVICE_RANK
    )

    train_config = locomotion_params.rsl_rl_config(ENV_NAME)

    if isinstance(env.observation_size,dict):
        train_config.obs_groups = {
            "policy":["state"],
            "critic": ["privileged_state"],
        }
    else:
        train_config.obs_groups = {
            "policy": ["state"],
            "critic": ["state"],
        }


    train_config.seed = SEED
    train_config.resume = False

    runner = OnPolicyRunner(
        env=rsl_env,
        train_cfg=train_config.to_dict(),
        log_dir=None,
        device=DEVICE,
    )

    runner.load(str(CHECKPOINT_PATH))

    policy = runner.get_inference_policy(device=DEVICE)

    # policy.eval()

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(SEED)
    

    command = jp.array(
        [0.1, 0.0, 0.0],
        dtype=jp.float32,
    )
    state = reset_fn(rng)

    if "command" in state.info:
        state.info["command"] = command

    rollout = [state]

    num_steps = min(
        int(DURATION_SECONDS/env.dt),
        int(env_config.episode_length),
    )

    for step_index in range(num_steps):
        obs = state.obs["state"]
        obs_torch = wrapper_torch._jax_to_torch(obs)
        policy_obs = {
            "state":obs_torch
        }
        with torch.no_grad():
            action_torch = policy(policy_obs)
            action_torch = torch.clip(action_torch, -1.0,1.0)
        
        action_jax = wrapper_torch._torch_to_jax(action_torch.flatten())
        state = step_fn(state, action_jax)
        rollout.append(state)

        if bool(state.done):
            print(
                "Episode terminated at step:",
                step_index + 1,
            )
            break
    print("Rollout states:", len(rollout))
    print("Final reward:", float(state.reward))
    print("Done:", bool(state.done))

    # 渲染
    trajectory = rollout[::RENDER_EVERY]
    fps = 1.0/env.dt/RENDER_EVERY
    frames = env.render(
        trajectory=trajectory,
        camera=CAMERA,
        width=640,
        height=480
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    media.write_video(OUTPUT_PATH, frames, fps=fps)

    print("Video saved to:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()