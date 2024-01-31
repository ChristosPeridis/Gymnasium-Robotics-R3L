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

def move_mocap_body(model: MjModel, data: MjData, t: float) -> None:
    # Example motion: move the mocap body in a circular path in the XY plane
    radius = 0.5  # radius of the circle
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = 1  # constant height

    # Update position
    data.mocap_pos[0, :] = [x, y, z]
    # Optionally, you could also update the orientation (quaternion) if needed

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path('../gymnasium_robotics/envs/assets/ur10e_gripper/scene-testing.xml')

    # Create data object for the model
    data = mujoco.MjData(model)

    # Create the Mujoco Renderer
    renderer = MujocoRenderer(model, data)

    # Time variable for the motion
    t = 0
    dt = 0.01  # time step

    # Simulation loop
    while True:
        # Update mocap body position
        move_mocap_body(model, data, t)
        t += dt

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
