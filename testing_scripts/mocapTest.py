import mujoco
import numpy as np
import time
import os

# Load the XML model
model = mujoco.MjModel.from_xml_path('mocap_model.xml')
data = mujoco.MjData(model)

# Initialize the simulator
mujoco.mj_forward(model, data)

# Set up offscreen rendering with OSMesa
width, height = 640, 480
#os.environ['MUJOCO_GL'] = 'osmesa'
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
buffer = mujoco.mjr_makeBuffer(ctx, width, height)

# Assuming the mocap body is the first one (index 0)
mocap_body_index = 0

# Render loop
while True:
    # Modify the position of the mocap body
    data.mocap_pos[mocap_body_index] = np.array([0.0, 0.0, np.abs(np.sin(time.time()))])

    # Step the simulation
    mujoco.mj_forward(model, data)

    # Update and render the scene offscreen
    scn = mujoco.MjvScene(model, 0)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_renderOffscreen(scn, buffer, ctx)

    # Do something with the rendered image in buffer
    # For example, save it to a file or process it

    time.sleep(0.01)  # Small delay to make the simulation visible

mujoco.mjr_freeBuffer(buffer)
mujoco.mjr_freeContext(ctx)
