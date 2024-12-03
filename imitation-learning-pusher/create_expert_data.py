"""
Functions for loading an expert policy and creating expert demonstrations
"""
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub
import gymnasium as gym
from tqdm import tqdm
from minari import DataCollector

REPO_ID = "farama-minari/Pusher-v5-SAC-expert"
FILENAME = "pusher-v5-sac-expert.zip"
ENV_ID = "Pusher-v5"


def get_expert_demo(dataset_id: str, n_episodes:int=100, visualize:bool = True) -> None:
    """
    Generates expert demonstration data and saves it locally

    Args:
        dataset_id (str): Name of dataset; It has to follow the format `ENV_NAME/DATASET_NAME-v(VERSION). Ex: `pusher/expert-v0`. 
                          Note: It will raise an exception if the file already exists. You can check the datasets with `$ minari list local`
        n_episodes (int): Number of episodes the expert will demonstrate
        visualize (bool): Visualizes the expert demonstration when True
    """


    render_mode = 'human' if visualize else "rgb_array"
    env = DataCollector(gym.make(ENV_ID, render_mode = render_mode))
    checkpoint = load_from_hub(repo_id=REPO_ID, filename=FILENAME)
    expert = SAC.load(checkpoint)

    print("----------Getting expert demonstration----------")
    for i in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        while True:
            action, _ = expert.predict(obs)
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    # Change this if needed
    env.create_dataset(
        dataset_id = dataset_id,
        algorithm_name="ExpertPolicy",
        code_permalink="https://github.com/dokyun-kim4/rl-pusher",
        author="dokyun-kim4",
        author_email="dkim4@olin.edu",
        description="Pusher expert policy",
        eval_env=ENV_ID
    )
    env.close()

    print(f"----------Dataset successfully created at /home/dokyun/.minari/datasets/{dataset_id}----------")


if __name__ == "__main__":
    dataset_id = "pusher/expert-v2"
    get_expert_demo(dataset_id, n_episodes=5_000, visualize=False)
