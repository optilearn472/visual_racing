import gym
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import sys
import argparse
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.gym_monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os

from f110_gym.envs.base_classes import Integrator
import yaml
from argparse import Namespace

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 10 episodes
              mean_reward = np.mean(y[-44:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def main(model_name, load_model=None):
    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    map_path = './examples/rl_race/f1tenth_racetracks/ex1/ex'
    map_ext = '.png'
    wp_path = './examples/rl_race/f1tenth_racetracks/ex1/ex_centerline.npy'
    sdf_path = './examples/rl_race/f1tenth_racetracks/ex1/ex.sdf'

    freq = 30
    dt = 1.0 / (freq * 10.0)

    env = gym.make('f110_gym:f110-v0',
                    map=map_path,
                    map_ext=map_ext,
                    waypoint=wp_path,
                    sdf_path=sdf_path,
                    num_agents=1,
                    timestep=dt,
                    integrator=Integrator.RK4,
                    eval_flag=0,
                    depth_render=0,
                    max_time=30)
    env.add_render_callback(render_callback)
    

    filename = "./examples/rl_race/model/" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + '/'


    try:
        if load_model is not None:
            print("********************yes*************************")
            model = PPO.load(
                load_model,
                env,
            )
        else:
            model = PPO(
                MlpPolicy,
                env,
                verbose=2,
                n_steps=10240,
                batch_size=512,
                # gae_lambda=0.98,
                device="cuda",
                tensorboard_log=filename,
                policy_kwargs={
                    "net_arch": [256, 256],
                    "activation_fn": torch.nn.ELU,
                },
            )

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)
        eval_callback = EvalCallback(env,
                                    callback_on_new_best=callback_on_best,
                                    verbose=1,
                                    n_eval_episodes=50,
                                    best_model_save_path=filename,
                                    log_path=filename,
                                    eval_freq=10240,
                                    deterministic=True,
                                    render=False
                                    )

        model.learn(total_timesteps=20000000,
                    log_interval=1,
                    callback=eval_callback
                    )
        model.save(filename +'/'+ model_name)
    finally:
        env.close()


if __name__ == "__main__":
    main(model_name='ppo')