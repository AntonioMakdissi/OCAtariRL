import numpy as np
from ocatari.core import OCAtari
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

class PositionHistoryEnv(gym.Env):
    """OCAtari Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected. No hud information.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.

    """
    def __init__(self, env_name : str, render_mode : str=None) -> None:
        self.render_mode=render_mode
        self.ocatari_env = OCAtari(env_name, mode="revised", hud=False, obs_mode=None, render_mode=render_mode)
        self.reference_list = self._init_ref_vector()
        self.current_vector = np.zeros(4 * len(self.reference_list), dtype=np.float32) #here 4 or 6

    @property
    def observation_space(self):
        vl = len(self.reference_list) * 4 #here 4 or 6
        return gym.spaces.Box(low=-2**63, high=2**63 - 2, shape=(vl, ), dtype=np.float32)

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        _, reward, truncated, terminated, info = self.ocatari_env.step(*args, **kwargs)
        self._obj2vecOg()
        return self.current_vector, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        _, info = self.ocatari_env.reset(*args, **kwargs)
        self._obj2vecOg()
        return self.current_vector, info

    def render(self, *args, **kwargs):
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.ocatari_env.close(*args, **kwargs)

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

    def _obj2vecOg(self):#no speed
        temp_ref_list = self.reference_list.copy()
        for o in self.ocatari_env.objects: # populate out_vector with object instance
            idx = temp_ref_list.index(o.category) #at position of first category occurance
            start = idx * 4
            flat = [item for sublist in o.h_coords for item in sublist]
            self.current_vector[start:start + 4] = flat #write the slice
            temp_ref_list[idx] = "" #remove reference from reference list
        for i, d in enumerate(temp_ref_list):
            if d != "": #fill not populated category instances wiht 0.0's
                self.current_vector[i*4:i*4+4] = [-1.0, -1.0, -1.0, -1.0] #replace with 0.0

def make_env(game: str, save: str, rank: int, seed: int = 0, is_eval: bool = False):
    def _init() -> gym.Env:
        env = PositionHistoryEnv(game)
        env = Monitor(env, save + "/seed{}{}".format(rank, is_eval))
        env.reset(seed=seed + rank)
        return env

    return _init

def simulate(env, num_steps):
    total_reward = 0
    obs = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
    return total_reward

if __name__ == "__main__":
    game_names = ["Assault-v4", "Atlantis-v4", "Berzerk-v4", "Bowling-v4", "Boxing-v4", "Carnival-v4", "Centipede-v4", "Freeway-v4", "Breakout-v4", "Skiing-v4"]
    save_dir="/content/drive/MyDrive/Colab Notebooks/ALE/gym"
    for game_name in game_names:
        print(game_name)
        # env = SubprocVecEnv([make_env(game=game_name, save=game_name, rank=i, seed=42) for i in range(cores)])
        num_steps = 1000

        env = make_env(game=game_name, save=save_dir, rank=0, seed=42)()
        total_reward = simulate(env, num_steps)
        print("Total reward:", total_reward)
        env.close()
