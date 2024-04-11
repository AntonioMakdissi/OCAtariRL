import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FrameStack
from tqdm import tqdm
from stable_baselines3.common.env_util import make_atari_env

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

def make_env(game: str, save: str, rank: int, seed: int = 0, is_eval: bool = False):
    def _init() -> gym.Env:
        #env = gym.make(game)
        env = make_atari_env("ALE/Seaquest-v5", n_envs=1, seed=0)
        #env.num_envs=1 
        env = Monitor(env, save + "/seed{}{}".format(rank, is_eval))
        env = gym.wrappers.NormalizeObservation(env)
        env = FrameStack(env, 4)
        env.reset()
        return env

    return _init

# #############################
# env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4,
#                      seed=0, env_kwargs={"render_mode": "rgb_array"})
# env = VecFrameStack(env, n_stack=4)
# obs = env.reset()
# model = DQN("CnnPolicy", env, **hyperparams, verbose=1)
# model.learn(total_timesteps=10_000)
# eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4,
#                           seed=0, env_kwargs={"render_mode": "rgb_array"})
# eval_env = VecFrameStack(eval_env, n_stack=4)
#######################33

# def make_env(game: str,save: str, rank: int, seed: int = 0, is_eval: bool=False):
#     def _init() -> gym.Env:
#         # env = PositionHistoryEnv(game)
#         # env= make_atarienv(game) #for comparision
#         # env = OCAtari(game, mode="revised", hud=False, obs_mode=None, render_mode=None)
#         env = gym.make(game, render_mode=None)
#         env= VecFrameStack(env, n_stack=4)
#         env = Monitor(env, save + "/seed{}{}".format(rank, is_eval))
#         #env = gym.wrappers.NormalizeObservation(env) #add or remove normilzation
#         env.reset(seed=seed + rank)
#         return env

#     set_random_seed(seed)
#     return _init

if __name__ == "__main__":
    cores = 1
    exp_name = "ALE/gym"
    env_str = "ALE/Seaquest-v5"
    env = SubprocVecEnv([make_env(game=env_str,save=exp_name, rank=i, seed=42) for i in range(cores)])
    print("#################################################################")
    print(type(env.action_space))

    
    adam_step_size = 0.00025
    clipping_eps = 0.1
    model = PPO(
        "CnnPolicy",
        n_steps=128,
        learning_rate=linear_schedule(adam_step_size),
        n_epochs=3,
        batch_size=32*8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(clipping_eps),
        vf_coef=1,
        ent_coef=0.01,
        env=env,
        verbose=1)
    
    total_iterations = 1e7
    with tqdm(total=total_iterations, desc="Training") as pbar:
        def update_progress(local_, _globals_):
            pbar.update(1)  # Increment progress bar by 1 step
        model.learn(total_timesteps=total_iterations, callback=update_progress)
    model.save(exp_name + "/")

print("done")
