import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm
import panda_gym
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bc_network import BCnetworkPanda
from loader import load_dataset

# TODO Modify to work with panda-gym dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bc(model_name: str, data_path: str, batch_size: int, num_epochs: int):

    model_path = f"models/{model_name}.pth"

    # Load in expert dataset
    dataloader, env = load_dataset(path = data_path, batch_size = batch_size)
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, spaces.Dict)
    assert isinstance(act_space, spaces.Box)
    input_dim, output_dim = np.prod(obs_space['achieved_goal'].shape) + np.prod(obs_space['desired_goal'].shape) + np.prod(obs_space['observation'].shape), np.prod(act_space.shape) #type: ignore


    # Initialize model, loss function, optimizer
    model = BCnetworkPanda(input_dim=input_dim, output_dim=output_dim).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    print(f"Training on {'GPU' if device.type == 'cuda' else 'CPU'}")

    for epoch in tqdm(range(num_epochs)):
        for batch in dataloader:
            cur_state = batch["observations"][:, :-1].to(device) # We dont need last observation, remove it
            agent_action = model(cur_state.to(torch.float32))
            expert_action = batch["actions"].to(device)

            batch_loss = loss_fn(agent_action, expert_action)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f"Current Epoch {epoch}/{num_epochs}; Loss: {batch_loss}")

    torch.save(model.state_dict(), model_path)
    return model_path



if __name__ == "__main__":

    # model_pth = train_bc(model_name="panda_slide_v2", data_path="PandaSlide/expert-v0", batch_size=50, num_epochs=100)
    model = BCnetworkPanda(24,3)
    model.load_state_dict(torch.load('./models/panda_slide_v2.pth', weights_only=True))
    model.eval()

    model.to(device)

    ENV_ID = 'PandaSlide-v3'
    n_episodes = 10
    base_env = (gym.make(ENV_ID, render_mode='human'))
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(f'logs/tqc/{ENV_ID}_1/{ENV_ID}/vecnormalize.pkl', env)
    env.norm_obs = True
    env.training = False
    env.norm_reward = False  # Disable reward normalization
    for i in tqdm(range(n_episodes)):
        obs_dict = env.reset()

        while True:
            action = env.action_space.sample()
            obs = [
                    [*ag, *dg, *obs] for ag, dg, obs in zip(obs_dict['achieved_goal'], obs_dict['desired_goal'], obs_dict['observation']) # type:ignore
                ]
            with torch.no_grad():
                action = model(torch.Tensor(obs).to(device))
                action = action.cpu().numpy()

            result = env.step(action)
            obs_dict = result[0]
            terminated = result[2][0]
            truncated = result[3][0]['TimeLimit.truncated']
            time.sleep(0.1)

            if terminated or truncated:
                break
    env.close()
