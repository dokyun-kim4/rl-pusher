import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm

from bc_network import BCnetwork
from loader import load_dataset

# TODO Modify to work with panda-gym dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bc(model_name: str, data_path: str, batch_size: int, num_epochs: int):

    model_path = f"models/{model_name}.pth"

    # Load in expert dataset
    dataloader, env = load_dataset(path = data_path, batch_size = batch_size)
    obs_space, act_space = env.observation_space, env.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(act_space, spaces.Box)
    input_dim, output_dim = np.prod(obs_space.shape), np.prod(act_space.shape)


    # Initialize model, loss function, optimizer
    model = BCnetwork(input_dim=input_dim, output_dim=output_dim).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    print(f"Training on {'GPU' if device == 'cuda' else 'CPU'}")

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

    model_pth = train_bc(model_name="test_bc", data_path="pusher/expert-v0", batch_size=100, num_epochs=10)
    model = BCnetwork(23,7)
    model.load_state_dict(torch.load(model_pth, weights_only=True))
    model.eval()

    n_episodes = 10
    env = gym.make('Pusher-v5', render_mode="human")
    for i in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        while True:
            action = env.action_space.sample()
            
            with torch.no_grad():
                action = model(torch.Tensor(obs).to(device))
                action = action.cpu().numpy()

            obs, rew, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                break
    env.close()
