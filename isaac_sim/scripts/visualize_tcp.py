import os
import sys
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import omni.kit.commands
from pxr import UsdGeom
import numpy as np

world = World()

# Import URDF directly without using any of our custom env code to avoid import crashes
urdf_path = "/home/ozan/Downloads/so100_ws/isaac_sim/urdf/so_100_dual_foldnet.urdf"

# Execute URDF Import Command
from isaacsim.asset.importer.urdf import _urdf
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path="/World/so_100_dual"
)

# Place spheres at the TCP links
from isaacsim.core.api.objects.sphere import VisualSphere

sphere_l = VisualSphere(
    prim_path="/World/so_100_dual/left_gripper_tcp_link/debug_tcp_l",
    name="debug_tcp_l",
    radius=0.015,
    color=np.array([1.0, 0.0, 0.0]) # Red
)

sphere_r = VisualSphere(
    prim_path="/World/so_100_dual/right_gripper_tcp_link/debug_tcp_r",
    name="debug_tcp_r",
    radius=0.015,
    color=np.array([0.0, 0.0, 1.0]) # Blue
)

print("\n" + "="*50)
print("TCP NOKTALARI GORSELLESTIRILIYOR...")
print("Sol TCP = Kirmizi Top")
print("Sag TCP = Mavi Top")
print("Toplar gripper'in neresindeyse IK orayi kontrol ediyor demektir!")
print("="*50 + "\n")

# Need to reset world to initialize objects
world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
