import mujoco
import numpy as np
import time
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Path to the XML file for the gripper
xml_path = '../gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml'

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create a renderer object for rendering
renderer = MujocoRenderer(model, data)

# Simulation time step
time_step = model.opt.timestep

# Control parameters
max_control = 255  # Maximum control input for opening
min_control = 0    # Minimum control input for closing
duration = 60      # Total duration of the script in seconds
cycles = 5         # Number of open-close cycles
cycle_duration = duration / cycles

# Actuate the gripper
def actuate_gripper(control_input):
    print(data.ctrl)
    data.ctrl[-1] = control_input
    mujoco.mj_step(model, data)

# Run the simulation
start_time = time.time()
while time.time() - start_time < duration:
    elapsed_time = time.time() - start_time
    cycle_time = elapsed_time % cycle_duration

    # Determine if opening or closing phase
    if cycle_time < cycle_duration / 2:
        # Opening phase
        control_input = max_control
    else:
        # Closing phase
        control_input = min_control

    # Actuate the gripper
    actuate_gripper(control_input)

    # Render the scene using Gymnasium's renderer
    renderer.render(render_mode="human")


    # Wait for the next time step
    time.sleep(time_step)
