import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
import panda_gym
import time

model = PPO.load('ppo_pusher_logs/best_model.zip')

env_id = "Pusher-v5"  # Gymnasium Mujoco environment
env = gym.make(env_id, render_mode="rgb_array")

obs = env.reset()[0]
trigger = lambda t: t % 10 == 0
env = RecordVideo(env, video_folder='.', episode_trigger=trigger, disable_logger=True)
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, terminated, _ = env.step(action)
    if done or terminated:
        env.reset()[0]
env.close()
