from typing import Any
import mujoco
import numpy as np
from mujoco import MjModel, MjData, mjtObj
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium_robotics.utils import mujoco_utils  # Assuming mujoco_utils contains relevant functions for mocap body manipulation

def load_model_from_xml(xml_path: str) -> Any:
    with open(xml_path, 'r') as xml_file:
        xml_string = xml_file.read()
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path('../gymnasium_robotics/envs/assets/ur10e_gripper/scene-testing.xml')

    # Create data object for the model
    data = mujoco.MjData(model)

    # Create the Mujoco Renderer
    renderer = MujocoRenderer(model, data)

    # Simulation loop
    while True:
        # Step the simulation
        mujoco.mj_step(model, data)

        # Access and print mocap body's position and quaternion
        mocap_pos = data.mocap_pos
        mocap_quat = data.mocap_quat
        print(f'Mocap Position: {mocap_pos}, Mocap Quaternion: {mocap_quat}')

        # Render the simulation
        renderer.render(render_mode='human')

if __name__ == '__main__':
    main()
