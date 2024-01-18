import mujoco
import numpy as np
import time
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Load the XML model
model = mujoco.MjModel.from_xml_path('mocap_model.xml')
data = mujoco.MjData(model)

# Initialize the simulator
mujoco.mj_forward(model, data)

# Create a renderer for the model
renderer = MujocoRenderer(model, data)

# Assuming the mocap body is the first one (index 0)
mocap_body_index = 0

# Render loop
while True:
    # Modify the position of the mocap body
    data.mocap_pos[mocap_body_index] = np.array([0.0, 0.0, np.abs(np.sin(time.time()))])

    # Step the simulation
    mujoco.mj_forward(model, data)

    # Render the scene
    renderer.render(render_mode="human")

    time.sleep(0.01)  # Small delay to make the simulation visible
