import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import random
from environment import simple_env  # Import the environment you created

# Initialize the environment
env = simple_env  # Using the previously created environment

# Reset the environment to get the initial observation
obs = env.reset()
print("Initial Observation:", obs)

# Run the environment for 100 steps by taking random actions
for _ in range(100):
    # Sample a random action from the action space
    action = env.action_space.sample()

    # Apply the action and observe the new state, reward, and other info
    obs, reward, done, info = env.step(action)

    # Print out the action, observation, reward, termination status, and additional info
    print("Action:", action)
    print("Observation:", obs, "Reward:", reward, "Done:", done, "Info:", info)

    # If the environment reaches a terminal state, reset it and print the reset state
    if done:
        obs = env.reset()
        print("Environment reset")

# Close the environment after use
env.close()
