import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


env_id = "Pusher-v5"
env = gym.make(env_id, render_mode="rgb_array")


log_dir = "./ppo_pusher_logs/"
model_save_path = log_dir + "ppo_pusher_model"
os.makedirs(log_dir, exist_ok=True)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
)

eval_env = gym.make(env_id)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,  # Evaluate every 10k steps
    deterministic=True,
    render=False,
)

# Train the PPO model
time_steps = 1_000_000
print(f"Training for {time_steps} time steps...")
#model.learn(total_timesteps=time_steps, callback=eval_callback)

# Save the trained model
#model.save(model_save_path)
print(f"Model saved at {model_save_path}")

model.load(model_save_path)
# Extract training loss from TensorBoard logs
def extract_rewards_from_logs(log_dir):
    event_file = None

    # Locate the event file
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                break

    if not event_file:
        print("No event file found in the log directory.")
        return None, None

    # Load the event data
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    
    if "train/mean_reward" not in ea.Tags().get("scalars", []):
        print("No 'train/reward' tag found in the event logs.")
        return None, None

    events = ea.Scalars("train/mean_reward")
    steps = [e.step for e in events]
    rewards = [e.value for e in events]

    return steps, rewards

# Generate the loss graph
steps, losses = extract_rewards_from_logs(log_dir)

if steps and losses:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Rewads")
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_dir, "reward_over_time.png"))
    plt.show()
else:
    print("Could not plot loss graph.")
