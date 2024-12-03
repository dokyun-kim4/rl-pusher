import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from bc_network import BCnetwork
from loader import load_dataset

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

    print(f"Training on {'GPU' if device.type == 'cuda' else 'CPU'}")

    losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for batch in dataloader:
            cur_state = batch["observations"][:, :-1].to(device) # We dont need last observation, remove it
            agent_action = model(cur_state.to(torch.float32))
            expert_action = batch["actions"].to(device)

            batch_loss = loss_fn(agent_action, expert_action)
            running_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f"Current Epoch {epoch}/{num_epochs}; Loss: {batch_loss}")
        losses.append(running_loss / len(dataloader))

    torch.save(model.state_dict(), model_path)
    print(f"---------- Model saved at {model_path}----------")

    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.title(f"Training Loss over {num_epochs} epochs")
    plt.show()


def evaluate_model(model_pth: str, env_id: str, n_episodes: int):
    model = BCnetwork(23,7).to(device)
    model.load_state_dict(torch.load(f"models/{model_pth}.pth", weights_only=True))
    model.eval()

    env = gym.make('Pusher-v5', render_mode="human")
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                action = model(torch.Tensor(obs).to(device))
                action = action.cpu().numpy()

            obs, rew, terminated, truncated, info = env.step(action)
            time.sleep(0.025)
            if terminated or truncated:
                break
    env.close()


if __name__ == "__main__":

    n_epochs = 500
    model_pth = "mujoco-pusher-v1"

    # Uncomment to train a new model
    # losses = train_bc(model_name=model_pth, data_path="pusher/expert-v2", batch_size=20, num_epochs=n_epochs)
    
    n_eps = 10
    evaluate_model(model_pth, "Pusher-v5", n_eps)
