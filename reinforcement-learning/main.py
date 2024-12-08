import os
import gymnasium as gym
import numpy as np
from ppo_model import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = 'LunarLander-v3'
    env = gym.make(env_name, continuous=False)
    N = 10
    batch_size = 16
    n_epochs = 8
    alpha = 0.003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, N=2048, alpha=alpha, policy_clip=0.3, n_epochs=n_epochs, input_dims=env.observation_space.shape)

    n_games = 10000

    if not os.path.exists('plots'):
        os.makedirs('plots')
    figure_file = f'plots/{env_name}.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    best_score = -float('inf')
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            done = done or truncated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

