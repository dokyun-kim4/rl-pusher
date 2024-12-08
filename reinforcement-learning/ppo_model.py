import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch.distributions.categorical import Categorical

class PPOMemory:
    """
        A class to handle memory for the PPO reinforcement learning algorithm.
    
    Attributes:
        batch_size (int): The size of each batch for training.
        states (List): List of stored states.
        probs (List): List of action probabilities from the policy.
        vals (List): List of value function estimates.
        actions (List): List of stored actions.
        rewards (List): List of rewards received for taking actions.
        dones (List): List of boolean flags indicating episode termination.
    """
    def __init__(self, batch_size: int):
        """
        Initialize the memory buffer and set batch size.

        Args:
            batch_size (int): The size of each training batch.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size: int = batch_size

    def generate_batches(self):
        """
        Generate randomized batches for training.

        Returns:
            Tuple containing:
                - states (np.ndarray): Array of stored states.
                - actions (np.ndarray): Array of stored actions.
                - probs (np.ndarray): Array of action probabilities.
                - vals (np.ndarray): Array of value function estimates.
                - rewards (np.ndarray): Array of rewards.
                - dones (np.ndarray): Array of done flags.
                - batches (List[np.ndarray]): List of indices for each batch.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        idx = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(idx) # Add randomness for Random SGD
        batches = [idx[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Store an experience in the memory buffer.

        Args:
            state (any): The observed state.
            action (any): The action taken.
            probs (any): The probability of the action from the policy.
            vals (any): The estimated value of the state.
            reward (float): The reward received.
            done (bool): Flag indicating episode termination.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
        Clear all stored memory from the buffer.
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    """
    Neural network representing the actor in the PPO algorithm.

    The actor generates a policy (probability distribution) over actions given an input state.

    Attributes:
        checkpoint_file (str): Path to the file where the model's weights are saved.
        action (nn.Sequential): Sequential layers defining the policy network.
        optimizer (torch.optim.Optimizer): Optimizer for training the actor network.
        device (torch.device): Device (CPU or GPU) where the network is hosted.
    """
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/ppo'):
        """
        Initialize the ActorNetwork.

        Args:
            n_actions (int): Number of possible actions.
            input_dims (tuple): Dimensions of the input state.
            alpha (float): Learning rate for the optimizer.
            fc1_dims (int, optional): Number of neurons in the first fully connected layer. Defaults to 256.
            fc2_dims (int, optional): Number of neurons in the second fully connected layer. Defaults to 256.
            chkpt_dir (str, optional): Directory to save model checkpoints. Defaults to 'checkpoints/ppo'.
        """
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims), # unpack for linear input dims.
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions), # Select one of the possible actions to take 
            nn.Softmax(dim=-1) 
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # GOAT optimizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        """
        Perform a forward pass through the network to generate an action distribution.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            Categorical: A categorical distribution over possible actions.
        """
        dist = self.actor(state)
        dist = Categorical(dist) # convert the state space into a categorical distribution

        return dist
    
    def save_checkpoint(self):
        """
        Save the model weights to a checkpoint file.
        """
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """
        Load the model weights from a checkpoint file.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    """
    Neural network representing the critic in the PPO algorithm.

    The critic estimates the value of a given state. Used to calculate
    advantages during training.

    Attributes:
        checkpoint_file (str): Path to the file where the model's weights are saved.
        critic (nn.Sequential): Sequential layers defining the value network.
        optimizer (torch.optim.Optimizer): Optimizer for training the critic network.
        device (torch.device): Device (CPU or GPU) where the network is hosted.
    """
    def __init__ (self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/ppo'):
        """
        Initialize the CriticNetwork.

        Args:
            input_dims (tuple): Dimensions of the input state.
            alpha (float): Learning rate for the optimizer.
            fc1_dims (int, optional): Number of neurons in the first fully connected layer. Defaults to 256.
            fc2_dims (int, optional): Number of neurons in the second fully connected layer. Defaults to 256.
            chkpt_dir (str, optional): Directory to save model checkpoints. Defaults to 'checkpoints/ppo'.
        """
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo') 
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1) # Estimated value of the input state.
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Perform a forward pass through the network to estimate the value of the input state.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            value (torch.Tensor): Estimated value of the input state.
        """
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        """
        Save the model weights to a checkpoint file.
        """
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """
        Load the model weights from a checkpoint file.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self):
        """
        Save checkpoints for the actor and critic model.
        """
        print('... saving models')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        """
        Load updates for the actor and critic model.
        """
        print("... loading model")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """
        Samples an action from the current action space distribution.

        Args:
            observation (any): An observation of the current state.
        
        Returns:
            action 
        """
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                advantage_t = 0
                for k in range(t, len(reward_arr)-1):
                    advantage_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = advantage_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                # pi(theta_old)
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                # pi(theta_new)
                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                # pi(theta_new)/pi(theta_old)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # A^t
                weighted_probs = advantage[batch] * prob_ratio
                # Clipping function from fig.7 of PPO paper.
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            self.memory.clear_memory()

