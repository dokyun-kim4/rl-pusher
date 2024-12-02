"""
Functions for loading an expert policy and creating expert demonstration data for a Panda-Gym environment
"""
from tqdm import tqdm
from sb3_contrib import TQC
import gymnasium as gym
from minari import DataCollector
import time
import panda_gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def get_expert_demo(dataset_id: str, env_id: str, n_episodes:int=100, visualize:bool = True) -> None:
    """
    Generates expert demonstration data and saves it locally

    Args:
        dataset_id (str): Name of dataset; It has to follow the format `env_id/DATASET_NAME-v(VERSION). Ex: `PandaSlide/expert-v0`. 
                          Note: It will raise an exception if the file already exists. You can check the datasets with `$ minari list local`
        env_id (str): Name of environment. Ex: 'PandaSlide-v3'
        n_episodes (int): Number of episodes the expert will demonstrate
        visualize (bool): Visualizes the expert demonstration when True
    """

    render = 'human' if visualize else 'rgb_array'
    base_env = DataCollector(gym.make(env_id, render_mode=render))
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(f'logs/tqc/{env_id}_1/{env_id}/vecnormalize.pkl', env)
    env.norm_obs = True
    env.training = False
    env.norm_reward = False  # Disable reward normalization

    # Load expert agent
    expert = TQC.load(f"logs/tqc/{env_id}_1/{env_id}.zip", env)
    
    for _ in tqdm(range(n_episodes)):
        observation = env.reset()
        while True:
            action, _ = expert.predict(observation = observation) # type: ignore
            result = env.step(action)
            observation = result[0]
            terminated = result[2][0]
            truncated = result[3][0]['TimeLimit.truncated']
            if visualize:
                time.sleep(0.1)
            if terminated or truncated:
                break

    base_env.create_dataset(
        dataset_id = dataset_id,
        algorithm_name="ExpertPolicy",
        code_permalink="https://github.com/dokyun-kim4/rl-pusher",
        author="dokyun-kim4",
        author_email="dkim4@olin.edu",
        description="Expert policy",
        eval_env= env_id
    )

    env.close()

    print(f"----------Dataset successfully created at /home/dokyun/.minari/datasets/{dataset_id}----------")


if __name__ == "__main__":
    env_id = "PandaSlide-v3"
    dataset_id = "PandaSlide/test-v0"
    get_expert_demo(dataset_id, env_id, n_episodes=10, visualize=True)
