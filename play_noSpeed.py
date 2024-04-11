import gymnasium as gym
import numpy as np
from ocatari.core import OCAtari
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
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


    
                

    # def _obj2vec(self):#with speed
    #     temp_ref_list = self.reference_list.copy()
    #     for o in self.ocatari_env.objects: # populate out_vector with object instance
    #         idx = temp_ref_list.index(o.category) # at position of first category occurrence
    #         start = idx * 6
    #         flat = [item for sublist in o.h_coords for item in sublist]
    #         # Calculate speed
    #         if len(o.h_coords) == 2: # Check if there are previous coordinates
    #             prev_x, prev_y = o.h_coords[1] # Previous coordinates
    #             curr_x, curr_y = o.h_coords[0] # Current coordinates
    #             speed_x = (curr_x - prev_x)
    #             speed_y = (curr_y - prev_y)
    #         else:
    #             speed_x = 0
    #             speed_y = 0
    #         # Add speed to the flat list
    #         flat.extend([speed_x, speed_y])
    #         self.current_vector[start:start + 6] = flat # write the slice
    #         temp_ref_list[idx] = "" # remove reference from reference list
    #     for i, d in enumerate(temp_ref_list):
    #         if d != "": # fill not populated category instances with 0.0's
    #             self.current_vector[i*6:i*6+6] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] # replace with -1
    #     #print(self.current_vector)

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

def make_env(game: str,save: str, rank: int, seed: int = 0, is_eval: bool=False):
    def _init() -> gym.Env:
        env = PositionHistoryEnv(game)
        #env= make_atarienv() #for comparision
        env = Monitor(env, save + "/seed{}{}".format(rank, is_eval))
        env = gym.wrappers.NormalizeObservation(env) #add or remove normilzation
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    cores = 1
    exp_name = "ALE/nomralize_TRUE_missingOb_NEG1_speeds_FALSE"
    env_str = "ALE/Seaquest-v5"
    env = SubprocVecEnv([make_env(game=env_str,save=exp_name, rank=i, seed=42) for i in range(cores)])
    

    # eval_callback = EvalCallback(
    #     eval_env,
    #     n_eval_episodes=4,
    #     best_model_save_path=env_str + "/",
    #     log_path=env_str + "/",
    #     eval_freq=max(100_000 // cores, 1),
    #     deterministic=True,
    #     render=False)
    
    adam_step_size = 0.00025
    clipping_eps = 0.1
    model = PPO(
        "MlpPolicy",
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
    
    # Create tqdm progress bar
    with tqdm(total=total_iterations, desc="Training") as pbar:
        # Define a function to update the tqdm progress bar
        def update_progress(local_, _globals_):
            pbar.update(1)  # Increment progress bar by 1 step
        
        # Learn with tqdm progress bar
        model.learn(total_timesteps=total_iterations, callback=update_progress)
    #model.learn(100, callback=eval_callback)#2e7
    model.save(exp_name + "/")


print("done")