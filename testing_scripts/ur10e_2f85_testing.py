import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

# Load the model
model = mujoco.MjModel.from_xml_path("gymnasium_robotics/envs/assets/ur10e_gripper/scene.xml")

# Create a data object
data = mujoco.MjData(model)

# Simulation step
mujoco.mj_step(model, data)

# Rendering
renderer = MujocoRenderer(model, data)
while True:
    renderer.render(render_mode="human")
