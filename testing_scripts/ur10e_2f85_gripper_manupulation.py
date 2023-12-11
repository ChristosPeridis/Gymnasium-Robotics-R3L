import mujoco
import numpy as np
import time
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Path to the XML file for the gripper
xml_path = 'gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml'

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create a renderer object for rendering
renderer = MujocoRenderer(model, data)

# Simulation time step
time_step = model.opt.timestep

# Control parameters
max_control = 255  # Maximum control input
min_control = 0    # Minimum control input
duration = 60      # Duration of the script in seconds

# Function to smoothly interpolate between min and max control values
def smooth_control(elapsed_time, duration):
    # Calculate the proportion of the cycle completed (0 to 1 and back to 0)
    cycle_progress = (1 - np.cos(2 * np.pi * elapsed_time / duration)) / 2

    # Interpolate control input
    return min_control + cycle_progress * (max_control - min_control)

# Run the simulation
start_time = time.time()
while time.time() - start_time < duration:
    elapsed_time = time.time() - start_time

    # Calculate smooth control input
    control_input = smooth_control(elapsed_time, duration/2)  # Dividing by 2 for open-close in half duration

    # Actuate the gripper
    data.ctrl[-1] = control_input
    mujoco.mj_step(model, data)

    # Render the scene using Gymnasium's renderer
    renderer.render(render_mode="human")

    # Wait for the next time step
    time.sleep(time_step)
