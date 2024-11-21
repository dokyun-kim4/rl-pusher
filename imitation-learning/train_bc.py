import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm


from bc_network import BCnetwork
from loader import load_dataset

# Load in expert dataset
dataloader, env = load_dataset("pusher/expert-v1", batch_size=100)
obs_space, act_space = env.observation_space, env.action_space
assert isinstance(obs_space, spaces.Box)
assert isinstance(act_space, spaces.Box)
input_dim, output_dim = np.prod(obs_space.shape), np.prod(act_space.shape)



NUM_EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer
model = BCnetwork(input_dim=input_dim, output_dim=output_dim).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

print(device)


for epoch in tqdm(range(NUM_EPOCHS)):
    for batch in dataloader:
        cur_state = batch["observations"][:, :-1].to(device) # We dont need last observation, remove it
        agent_action = model(cur_state.to(torch.float32))
        expert_action = batch["actions"].to(device)

        batch_loss = loss_fn(agent_action, expert_action)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    print(f"Current Epoch {epoch}/{NUM_EPOCHS}; Loss: {batch_loss}")


n_episodes = 10
env = gym.make('Pusher-v5', render_mode="human")
for i in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    while True:
        # action = env.action_space.sample()
        
        with torch.no_grad():
            action = model(torch.Tensor(obs).to(device))
            action = action.cpu().numpy()
            obs, rew, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break
env.close()
