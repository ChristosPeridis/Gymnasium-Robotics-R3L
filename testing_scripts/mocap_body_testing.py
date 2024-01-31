
import mujoco
import numpy as np
import time
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium_robotics.utils import mujoco_utils as mjcu

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('../gymnasium_robotics/envs/assets/ur10e_gripper/scene-testing.xml')
data = mujoco.MjData(model)

# Create a renderer
renderer = MujocoRenderer(model, data)

# Set the position of the mocap body
def set_mocap_pos(data, pos):
        mjcu.set_body_pos(data, 'robot0:mocap', pos)
        mjcu.set_body_quat(data, 'robot0:mocap', np.array([1, 0, 0, 0]))

mujoco.mj_step(model, data)

print("Dtata Specification of the Mocap Body:\n\n", model.body('robot0:mocap'), "\n\n\n\n")
print("Currtent Position of the Mocap Body:", mjcu.get_body_pos(data, 'robot0:mocap'), "\n\n")
print("Currtent Orientation of the Mocap Body:", mjcu.get_body_quat(data, 'robot0:mocap'), "\n\n") 


mjcu.reset_mocap2body_xpos(model, data)


print("Dtata Specification of the Mocap Body:\n\n", model.body('robot0:mocap'), "\n\n\n\n")
print("Currtent Position of the Mocap Body:", mjcu.get_body_pos(data, 'robot0:mocap'), "\n\n")
print("Currtent Orientation of the Mocap Body:", mjcu.get_body_quat(data, 'robot0:mocap'), "\n\n") 

# Simulation loop
while True:
    # Update the simulator
    mujoco.mj_step(model, data)

    print("Dtata Specification of the Mocap Body:\n\n", model.body('robot0:mocap'), "\n\n\n\n")
    print("Currtent Position of the Mocap Body:", mjcu.get_body_pos(data, 'robot0:mocap'), "\n\n")
    print("Currtent Orientation of the Mocap Body:", mjcu.get_body_quat(data, 'robot0:mocap'), "\n\n") 


    # Render the simulation
    renderer.render(render_mode="human")
    
    # Every 3 seconds, change the position of the mocap body
    for pos in [np.array([x, y, z]) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2)]:
        print ("Setting mocap position to: ", pos, "\n\n")
        set_mocap_pos(data, pos)
        time.sleep(3)
