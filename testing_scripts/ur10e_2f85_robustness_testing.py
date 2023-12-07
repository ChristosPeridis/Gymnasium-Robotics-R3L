import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
import time

# Load the UR10e model from the provided XML file
model = mujoco.MjModel.from_xml_path('gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml')
data = mujoco.MjData(model)

# Create a renderer object for rendering
renderer = MujocoRenderer(model, data)

# Simulation parameters
duration = 60  # Duration of the simulation in seconds
time_step = 0.01  # Simulation time step
n_steps = int(duration / time_step)

# Control loop
for step in range(n_steps):
    # Example control input (random in this case)
    control_input = np.random.uniform(-1, 1, model.nu)
    
    # Apply control input to the robot
    data.ctrl[:] = control_input

    # Step the simulation
    mujoco.mj_step(model, data)

    # Render the scene
    renderer.render(render_mode="human")
    
    # Optionally, add a delay for real-time visualization
    time.sleep(time_step)


# Note: Ensure that your environment supports graphical rendering. 
# In headless environments, rendering may not work as expected.
