# UAV Control using Reinforcement Learning (RL)

## Introduction
This is a small example project that demonstrates how to use reinforcement learning (RL) to control an unmanned aerial vehicle (UAV). The UAV is controlled by adjusting its acceleration to reach a target destination while minimizing time and maintaining safe motion parameters such as speed and acceleration.

## Environment
- **Agent**: Single UAV
- **State**: The UAV's position and velocity.
- **Observation**: The relative position of the UAV to its destination and the current mode of the UAV.
- **Action**: Acceleration applied to the UAV.

### Reward Function:
- `+100` for reaching the target.
- `-2` time penalty for each step taken.
- `+1` if the UAV moves closer to the target.
- `-1` if the UAV moves away from the target.
- `-10` if acceleration exceeds the maximum limit.
- `-10` if speed exceeds the maximum limit.

## Virtual Environment Setup
1. Install necessary packages:
   ```bash
   pip install stable_baselines3
   pip install gym==0.26.0
   pip install gymnasium==0.28.1
   pip install shimmy
   ```2.Important Notice:
You may need to reinstall `gymnasium==0.28.1` because the default version of gymnasium(v1.0.0) installed with stable_baselines3 does not work well with this environment.

## How to Use the Code
1.Training the Model
   ```bash
   python train.py
   ```
2.Testing and Visualization
   ```bash
   python test.py
   ```
3. You can find all the parameter settings in the `environment.py` file.

