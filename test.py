# Import necessary libraries
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from environment import simple_env  # Import the custom environment
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')  # Set the backend for Matplotlib

# Initialize the environment
env = simple_env

# Check if the saved model file exists
model_file = Path("ppo_simple.zip")
if model_file.is_file():
    # Load the trained model if it exists
    model = PPO.load("ppo_simple", env=env)
else:
    # Raise an error if the model file doesn't exist
    raise FileNotFoundError(
        "The trained model 'ppo_simple.zip' was not found. Please ensure the model is available before running the code.")

# Test the trained agent
num_episodes = 1  # Number of episodes to test the trained agent

for episode in range(num_episodes):
    obs = env.reset()  # Reset the environment for a new episode
    done = False
    episode_reward = 0
    trajectory = []  # Store the drone's positions over time
    velocities = []  # Store the drone's velocities over time
    distances = []  # Store the drone's distance to the landing pad over time
    rewards = []  # Store rewards over time

    # Print the landing pad's position
    landing_pad_position = env.envs[0].landing_pad.position
    print(f"Landing Pad Position: {landing_pad_position}")

    while not done:
        # Track the position, velocity, distance, and rewards
        trajectory.append(env.envs[0].drone.position.tolist())
        velocities.append(np.linalg.norm(env.envs[0].drone.velocity))  # Track velocity magnitude
        distance = np.linalg.norm(env.envs[0].drone.position - env.envs[0].landing_pad.position)
        distances.append(distance)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]  # Aggregate the rewards
        rewards.append(reward[0])  # Store the reward for each step

        # Add final position after step
        #if done:
            #trajectory.append(env.envs[0].drone.position.tolist())  # Add the final position here
            #print(f"Final UAV Position (from environment): {env.envs[0].drone.position}")

    # Display final episode results
    print(f"Episode {episode + 1}: Final Reward = {episode_reward}, "
          f"Successful Landing = {info[0]['successful_landing']}, "
          f"Failed to Land = {info[0]['failed']}")

    # After the episode ends, print the final position of the UAV
    final_uav_position = trajectory[-1]  # The last entry in trajectory is the final position
    print(f"Final UAV Position: {final_uav_position}")

    # Convert trajectory to NumPy array for easier slicing
    trajectory = np.array(trajectory)

    # Create two separate figures
    fig_traj = plt.figure(figsize=(6, 6))  # Figure for 3D trajectory
    fig_data = plt.figure(figsize=(10, 10))  # Figure for velocity, distance, and reward

    # 3D Trajectory Plot
    ax3d = fig_traj.add_subplot(111, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.set_xlim(-5, 5)
    ax3d.set_ylim(-5, 5)
    ax3d.set_zlim(0, 5)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    # Scatter the landing pad as a red point
    ax3d.scatter(*landing_pad_position, color='red')

    # 2D Subplots for velocity, distance, and reward in the second figure
    ax_vel = fig_data.add_subplot(311)  # 3x1 grid, first plot
    ax_vel.set_title("Velocity over Time")
    ax_vel.set_xlabel("Timestep")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.grid(True)  # Add grid to the velocity plot

    ax_dist = fig_data.add_subplot(312)  # 3x1 grid, second plot
    ax_dist.set_title("Distance to Landing Pad over Time")
    ax_dist.set_xlabel("Timestep")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.grid(True)  # Add grid to the distance plot

    ax_reward = fig_data.add_subplot(313)  # 3x1 grid, third plot
    ax_reward.set_title("Reward over Time")
    ax_reward.set_xlabel("Timestep")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True)  # Add grid to the reward plot

    # Adjust the spacing between the subplots (increase hspace for more vertical space)
    plt.subplots_adjust(hspace=0.5)

    # Initialize empty lines for updating during the animation
    line3d, = ax3d.plot([], [], [], color='blue')  # Line in 3D trajectory
    line_vel, = ax_vel.plot([], [], color='green')  # Line for velocity
    line_dist, = ax_dist.plot([], [], color='orange')  # Line for distance
    line_reward, = ax_reward.plot([], [], color='purple')  # Line for reward


    # Function to update the 3D trajectory plot during the animation
    def update_traj(num):
        # Update 3D trajectory
        line3d.set_data(trajectory[:num, 0], trajectory[:num, 1])
        line3d.set_3d_properties(trajectory[:num, 2])
        return line3d,

    # Function to update the 2D plots (velocity, distance, reward) during the animation
    def update_data(num):
        # Update velocity plot
        line_vel.set_data(range(num), velocities[:num])
        ax_vel.set_xlim(0, len(velocities))
        ax_vel.set_ylim(0, max(velocities) * 1.1)

        # Update distance to landing pad plot
        line_dist.set_data(range(num), distances[:num])
        ax_dist.set_xlim(0, len(distances))
        ax_dist.set_ylim(0, max(distances) * 1.1)

        # Update reward plot
        line_reward.set_data(range(num), rewards[:num])
        ax_reward.set_xlim(0, len(rewards))
        ax_reward.set_ylim(min(rewards) * 1.1, max(rewards) * 1.1)

        return line_vel, line_dist, line_reward


    # Create the animation for the 3D trajectory
    ani_traj = FuncAnimation(fig_traj, update_traj, frames=len(trajectory), interval=50, blit=True, repeat=True)

    # Create the animation for the 2D data plots
    ani_data = FuncAnimation(fig_data, update_data, frames=len(trajectory), interval=50, blit=True, repeat=False)

    # Show the separate figures
    plt.show()
