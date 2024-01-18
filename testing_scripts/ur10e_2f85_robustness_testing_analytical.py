import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
import time
import math


# Load the UR10e model with the 2F85 gripper from the provided XML file
model = mujoco.MjModel.from_xml_path('gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml')
data = mujoco.MjData(model)

# Create a renderer object for rendering
renderer = MujocoRenderer(model, data)

# Simulation parameters
duration = 60  # Duration of the simulation in seconds
time_step = 0.01  # Simulation time step
n_steps = int(duration / time_step)

# Get the indices of the robot and gripper actuators
robot_actuator_indices = [i for i in range(model.nu - 2)]
gripper_actuator_indices = [i for i in range(model.nu - 2, model.nu)]

# Control loop
for step in range(n_steps):
    # Sinusoidal control input for each joint of the robot
    current_time = step * time_step
    robot_control_input = [math.sin(2 * math.pi * 0.1 * current_time + i * math.pi / 4) for i in range(model.nu - 2)]

    # Control for the 2F85 gripper: alternating between opening and closing
    gripper_phase = math.pi * (step // (n_steps // 2))  # Switch between 0 and π halfway through
    gripper_control_input = [math.sin(gripper_phase) for _ in range(2)]  # Assuming 2 DOF for the gripper

    # Apply control input to the robot and the gripper
    data.ctrl[robot_actuator_indices] = robot_control_input
    data.ctrl[gripper_actuator_indices] = gripper_control_input

    # Step the simulation
    mujoco.mj_step(model, data)

    # Render the scene using Gymnasium's renderer
    renderer.render(render_mode="human")

    # Optionally, add a delay for real-time visualization
    time.sleep(time_step)

# Save the simulation results to a CSV file

# Note: Ensure that the path to the XML file is correct and that your environment supports rendering.