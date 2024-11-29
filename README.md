# RL Pusher  
By Dexter Friis-Hecht and Dokyun Kim  

## Project Description  
In this project, we aim to deepen our understanding of reinforcment learning by implementing Imitation Learning and Deep Reinforcement Learning in the [Minari Pusher environment](https://gymnasium.farama.org/environments/mujoco/pusher/). The pusher environment has a 7-DOF manipulator whose task is to push a cylindrical object into the goal (marked in red).  

**INSERT IMAGE OF PUSHER ENVIRONMENT**

## Methodology
In this section, we will cover basic concepts of reinforcement learning and explain the two methods we used. 

### Terminology
We will define some terminology that will be used in the following sections.  

**Agent**: The decision-maker that interacts with the environment (This is the arm in our case)

**Environment**: The system or world in which the agent operates. (This is the table)

**Action Space**: The set of all possible actions the agent can take in the environment.  

**Observation Space**: The set of all possible observations or states the agent can perceive.

**Policy**: The agent’s strategy for deciding which action to take given an observation.  

**Reward**: A value indicating the "quality" of an action. The agent adjusts its policy to maximize this.

### Imitation Learning (Behavior Cloning)
The idea behind imitation learning is quite straightforward. Given an expert demonstration of the task we want the model to perform, can we make the model learn the expert's policy? We implement this using Behavior Cloning, which has one of the simplest architectures.  

In behavior cloning, the agent is initialized with no information about the environment. Given an observation, it will take some action, which will most likely be wrong. However, since we have the expert demonstration, aka the "correct action" to take at a given observation, we can compare the expert's action with our agent's action and make our agent learn the expert policy.  

Pusher's action consists of 7 different torques applied at different joints, which can be represented as a $7 x 1$ vector. The observation consists of 23 values containing information about the arm, cylinder and goal, which can be represented as a $23 x 1$ vector. We will now define our network.

<div style="text-align: center;">
  <img src="img/bc_network.png" alt="Behavior Cloning Architecture" width="600">
</div>



### Deep Reinforcement Learning (DQN?)


## Lessons Learned

