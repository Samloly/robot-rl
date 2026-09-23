import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
from datetime import datetime
from pathlib import Path

import jax
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as on_policy_runner_module
from algorithms.my_ppo import MyPPO
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params

# 实验配置
ENV_NAME = "Go1JoystickFlatTerrain"

SEED = 1
NUM_ENVS = 1024

DEVICE = "cuda:0"
DEVICE_RANK = 0

# 第一次运行自己的 train.py 时设为 10。
# 验证通过后改为 1000。
MAX_ITERATIONS = 5

# smoke test 时每 5 次迭代保存一次。
# 正式训练可以改为 50。
SAVE_INTERVAL = 1

USE_DOMAIN_RANDOMIZATION = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"

WARP_CACHE_DIR = PROJECT_ROOT / ".cache" / "warp"

def create_run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{ENV_NAME}"
        f"-{timestamp}"
        f"-rsl-baseline"
    )

    run_directory = RUNS_ROOT/run_name
    run_directory.mkdir(parents=True,exist_ok=False)

    return run_directory

def create_environment():
    env_config = registry.get_default_config(env_name=ENV_NAME)

    env_config.impl = "jax"

    env = registry.load(ENV_NAME, config=env_config)
    return env,env_config

def create_randomizer():
    if not USE_DOMAIN_RANDOMIZATION:
        return None

    return registry.get_domain_randomizer(ENV_NAME)

def create_vectorized_environment(env, env_config, randomizer):
    vectorized_env = wrapper_torch.RSLRLBraxWrapper(
        env=env,
        num_actors=NUM_ENVS,
        seed=SEED,
        episode_length=env_config.episode_length,
        action_repeat=1,
        randomization_fn=randomizer,
        render_callback=None,
        device_rank=DEVICE_RANK,
    )

    return vectorized_env

def create_training_config(env):
    train_config = locomotion_params.rsl_rl_config(ENV_NAME)

    observation_size =env.observation_size

    if isinstance(observation_size,dict):
        train_config.obs_groups = {
            "policy": ["state"],
            "critic": ["privileged_state"],
        }
    else:
        train_config.obs_groups = {
            "policy": ["state"],
            "critic": ["state"],
        }

    train_config.seed = SEED
    train_config.max_iterations = MAX_ITERATIONS
    train_config.save_interval = SAVE_INTERVAL
    train_config.experiment_name = ENV_NAME
    train_config.run_name = "rsl-baseline"

    train_config.algorithm.class_name = "MyPPO"

    train_config.resume = False
    train_config.load_run ="-1"
    train_config.checkpoint = -1
    return train_config

def save_configs(
    run_directory:Path,
    env_config,
    train_config,
)->None:
    env_config_path = run_directory/"env_config.json"
    train_config_path = run_directory/"train_config.json"

    with env_config_path.open("w", encoding="utf-8") as file:
        json.dump(env_config.to_dict(), file, indent=2)

    with train_config_path.open("w", encoding="utf-8") as file:
        json.dump(train_config.to_dict(), file, indent=2)

def main():
    RUNS_ROOT.mkdir(parents=True,exist_ok=True)
    WARP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(WARP_CACHE_DIR)

    run_directory = create_run_directory()
    print("\nExperiment")
    print("  environment:", ENV_NAME)
    print("  seed:", SEED)
    print("  num envs:", NUM_ENVS)
    print("  iterations:", MAX_ITERATIONS)
    print("  device:", DEVICE)
    print("  domain randomization:", USE_DOMAIN_RANDOMIZATION)
    print("  output:", run_directory)

    env, env_config = create_environment()
    print("\nEnvironment")
    print("  action size:", env.action_size)
    print("  observation size:", env.observation_size)
    print("  episode length:", env_config.episode_length)
    print("  control dt:", env.dt)

    randomizer = create_randomizer()

    vectorized_env = create_vectorized_environment(env=env, env_config=env_config, randomizer=randomizer)
    
    train_config = create_training_config(env)

    on_policy_runner_module.MyPPO = MyPPO
    
    train_config_dict = train_config.to_dict()

    save_configs(run_directory=run_directory, env_config=env_config,train_config=train_config)
    
    print("\nTraining configuration")
    print(train_config)

    runner = OnPolicyRunner(
        env= vectorized_env,
        train_cfg=train_config_dict,
        log_dir=str(run_directory),
        device=DEVICE,
    )

    runner.learn(
        num_learning_iterations=MAX_ITERATIONS,
        init_at_random_ep_len=False,
    )

    print("\nTraining finished")
    print("Results saved to:")
    print(run_directory)

if __name__ == "__main__":
    main()