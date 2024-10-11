import numpy as np
import os  # Import the os module for file operations
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import simple_env  # Import the custom environment
from datetime import datetime
import torch as th  # Import PyTorch for setting device


# Define the filename for the final saved model
model_filename = "ppo_simple.zip"

# Initialize the environment
env = simple_env

# Create a PPO model for training the agent
# MlpPolicy means using a multi-layer perceptron policy (neural network)
model = PPO("MlpPolicy", env, verbose=1, device='cpu')


# Function to save a model checkpoint
def save_checkpoint(model, steps):
    """
    Save the model checkpoint at the given step number.
    """
    # Get the current time and date in the format HH:MM_DD_MM
    current_time = datetime.now().strftime("%H:%M_%d_%m")
    # Define the checkpoint filename with time, date, and steps in 'k' (thousands)
    checkpoint_filename = f"checkpoint_{current_time}_{steps // 1000}k.zip"
    # Save the model at the checkpoint
    model.save(checkpoint_filename)
    print(f"Checkpoint saved: {checkpoint_filename}")



# Train the agent and save checkpoints at specified timesteps
total_timesteps = 3000000  # Total timesteps for training
checkpoint_timesteps = [1000000, 2000000, 3000000]  # Define checkpoints
current_timesteps = 0  # Initialize timesteps tracker


# Training loop with checkpoint saving
while current_timesteps < total_timesteps:
    # Find the next checkpoint to train until
    next_checkpoint = min(checkpoint_timesteps)

    # Calculate the timesteps to train until the next checkpoint
    timesteps_to_train = next_checkpoint - current_timesteps
    model.learn(total_timesteps=timesteps_to_train)

    # Update the total timesteps trained so far
    current_timesteps += timesteps_to_train

    # Save the checkpoint at this stage
    save_checkpoint(model, current_timesteps)

    # Remove the checkpoint from the list after saving
    checkpoint_timesteps.remove(next_checkpoint)

# Save the final trained model
model.save(model_filename)
print(f"Final model saved as {model_filename}")

