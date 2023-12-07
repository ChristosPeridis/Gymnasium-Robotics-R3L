import mujoco
from mujoco.wrapper.mjbindings import mjlib
import os

# Load the model
model = mujoco.Physics.from_xml_path('/home/cocp5/Gymnasium-Robotics-R3L/gymnasium_robotics/envs/assets/ur10e_gripper/ur10e_2f85.xml')

# Create a viewer
viewer = mujoco.MjViewer(model)

# Run the viewer
while True:
    viewer.render()