import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
import time
import math
import csv
import os

# Load the UR10e model with the 2F85 gripper from the provided XML file
model = mujoco.MjModel.from_xml_path('gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml')
data = mujoco.MjData(model)

# Create a renderer object for rendering
renderer = MujocoRenderer(model, data)

# Simulation parameters
duration = 60  # Duration of the simulation in seconds
time_step = 0.01  # Simulation time step
n_steps = int(duration / time_step)


# Griper control parameters
max_control = 255
min_control = 0
frequency = 1  # Frequency of opening and closing in seconds

# Helper function to actuate the gripper
def actuate_gripper(control_input):
    data.ctrl[-1] = control_input
    mujoco.mj_step(model, data)


# Get the indices of the robot and gripper actuators
robot_actuator_indices = [i for i in range(model.nu - 1)]
gripper_actuator_indices = [i for i in range(model.nu - 1, model.nu)]


'''# Ensure the directory for the CSV file exists
os.makedirs(os.path.dirname('simulation_data.csv'), exist_ok=True)'''


# Open a CSV file for writing
with open('simulation_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['time_step', 'control_input_1', 'control_input_2', 'gripper_state_1', 'gripper_state_2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    start_time = time.time()
    # Control loop
    for step in range(n_steps):
        # Sinusoidal control input for each joint of the robot
        current_time = step * time_step
        robot_control_input = [math.sin(2 * math.pi * 0.1 * current_time + i * math.pi / 4) for i in range(model.nu - 1)]

        # Control for the 2F85 gripper: alternating between opening and closing
        #gripper_phase = np.random.uniform(0, 0.8)  # the phase takes a random number between 0 and 0.8
        gripper_control_input = (np.sin(2 * np.pi * (time.time() - start_time) / frequency) + 1) / 2 * (max_control - min_control) + min_control # Assuming 1 DOF for the gripper

        # Apply control input to the robot and the gripper
        data.ctrl[robot_actuator_indices] = robot_control_input
        data.ctrl[gripper_actuator_indices] = gripper_control_input

        # Print the control inputs, the gripper's state, and the gripper phase
        print(f"Control inputs: {gripper_control_input}")
        print(f"Gripper state: {data.qpos[gripper_actuator_indices]}")
        #print(f"Gripper phase: {gripper_phase}")

        actuate_gripper(gripper_control_input)

        # Write the control inputs, the gripper's state, and the gripper phase to the CSV file
        writer.writerow({'time_step': current_time, 'control_input_1': gripper_control_input, 'gripper_state_1': data.qpos[gripper_actuator_indices],})

        # Step the simulation
        mujoco.mj_step(model, data)

        # Render the scene using Gymnasium's renderer
        renderer.render(render_mode="human")

        # Optionally, add a delay for real-time visualization
        time.sleep(time_step)