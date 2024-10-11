import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import random

# Environment dimensions
world_width = 10
world_height = 10
world_depth = 5

# Drone properties
drone_radius = 0.1
MAX_VELOCITY = 0.5  # Maximum norm velocity for the drone
MAX_ACCELERATION = 0.5  # Maximum acceleration for the drone

# Timestep duration
TIMESTEP = 0.1  # Timestep in seconds

# Landing pad properties
landing_pad_radius = 0.2
# Create the landing pad positions
landing_pad_position = [0.0, 0.0, 0.1] # Static landing pad position

# Landing parameters
landing_distance = 0.1
landing_velocity = 0.1

class Drone:
    def __init__(self, position, mode):
        """
        Initialize the drone with a given position and mode (0 for takeoff, 1 for landing).
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.mode = mode  # Mode is either 0 (taking off) or 1 (landing)

    @staticmethod
    def random_mode():
        return random.randint(0, 1)  # Randomly returns 0 or 1

class LandingPad:
    def __init__(self, position):
        """
        Initialize the landing pad at a given position.
        """
        self.position = np.array(position)

# SingleAgentEnv class represents the custom Gym environment for drone landing/takeoff
class SingleAgentEnv(gym.Env):
    """
    Initialize the environment, including drone, landing pad, and action/observation spaces.
    """
    def __init__(self):
        super(SingleAgentEnv, self).__init__()
        # Define action space: Acceleration in 3D space
        self.action_space = spaces.Box(low=-MAX_ACCELERATION, high=MAX_ACCELERATION, shape=(3,), dtype=np.float32)

        # Define observation space: Drone's relative position to the landing pad, velocity, and mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Initialize a drone and landing pad
        self.drone = Drone(self.create_random_drone(), Drone.random_mode())
        self.landing_pad = LandingPad(self.create_random_target())

        # Initialize episode step counter
        self.current_step = 0
        self.max_episode_steps = 1000

    def create_random_drone(self):
        """
        Create a random starting position for the drone within the world bounds.
        """
        random_x = int(random.uniform(-world_width/2, world_width/2))
        random_y = int(random.uniform(-world_height/2, world_height/2))
        random_z = int(random.uniform(2, world_depth))

        return np.array([random_x, random_y, random_z], dtype=np.float64)

    def create_random_target(self):
        """
        Create a random position for the landing pad within the world bounds.
        """
        random_x = int(random.uniform(-(world_width-2)/2, (world_width-2)/2))
        random_y = int(random.uniform(-(world_height-2)/2, (world_height-2)/2))
        random_z = int(random.uniform(0, (world_depth-2)))

        return np.array([random_x, random_y, random_z], dtype=np.float64)

    def reset(self):
        """
        Reset the environment to a new random state at the beginning of each episode.
        """
        # Reset drone position, velocity, and mode
        self.drone = Drone(self.create_random_drone(), Drone.random_mode())
        self.landing_pad = LandingPad(self.create_random_target())
        self.drone.velocity = np.array([0, 0, 0])
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """
        Apply the action (acceleration) to the drone, update its state, and return the new state.
        """
        reward = 0
        initial_distance = np.linalg.norm(self.drone.position - self.landing_pad.position)
        acceleration_norm = np.linalg.norm(action)

        # Penalize if the action exceeds maximum acceleration
        if acceleration_norm > MAX_ACCELERATION:
            reward -= 10

        current_velocity = self.drone.velocity

        # Update drone velocity with the given acceleration
        new_velocity = self.drone.velocity + action * TIMESTEP
        new_velocity_norm = np.linalg.norm(new_velocity)

        # Penalize and clip the velocity if it exceeds the maximum allowed
        if new_velocity_norm > MAX_VELOCITY:
            reward -= 10
            self.drone.velocity = (new_velocity / new_velocity_norm) * MAX_VELOCITY
        else:
            self.drone.velocity = new_velocity

        # Update drone's position
        self.drone.position += (current_velocity + self.drone.velocity) * TIMESTEP / 2

        # Calculate the new distance to the landing pad
        current_distance = np.linalg.norm(self.drone.position - self.landing_pad.position)

        # Reward if the drone moves closer to the landing pad, penalize otherwise
        if initial_distance - current_distance > 0:
            reward += 1
        else:
            reward -= 1

        # Time step penalty
        reward -= 2
        self.current_step += 1

        # Check if the episode is over (terminated) and if the landing was successful
        terminated = False
        info = {}
        is_successful_landing = False

        # Successful criteria based on drone mode
        if self.drone.mode == 1:  # If mode is landing
            if current_distance <= landing_distance and np.linalg.norm(self.drone.velocity) <= landing_velocity and \
                    self.drone.position[2] >= self.landing_pad.position[2]:
                is_successful_landing = True
        else:
            if current_distance <= landing_distance and np.linalg.norm(self.drone.velocity) <= landing_velocity:
                is_successful_landing = True

        # Reward for successful landing
        if is_successful_landing:
            reward += 100
            terminated = True
            info['successful_landing'] = True
        else:
            info['successful_landing'] = False

        # Terminate if maximum episode steps are reached
        if self.current_step >= self.max_episode_steps:
            terminated = True
            info['failed'] = True
        else:
            info['failed'] = False

        return self._get_observation(), reward, terminated, info

    def _get_observation(self):
        """
        Get the current observation (relative position, velocity, mode) for the agent.
        """
        relative_position = self.drone.position - self.landing_pad.position
        drone_mode = np.array([self.drone.mode], dtype=np.float32)
        return np.concatenate((relative_position, self.drone.velocity, drone_mode))


# Create an instance of the custom environment
env = SingleAgentEnv()

# Wrap the environment with a VecEnv to handle the single agent
simple_env = DummyVecEnv([lambda: env])