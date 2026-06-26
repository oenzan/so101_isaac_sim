"""
Debug: Check arm reach and workspace
"""
import sys, os
import numpy as np
import torch

foldnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code/src"))
sys.path.insert(0, foldnet_src)
batch_urdf_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code/external/batch_urdf/src"))
sys.path.insert(0, batch_urdf_src)
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code"))

from unittest.mock import MagicMock
sys.modules['pyflex'] = MagicMock()

import batch_urdf
import garmentds.common.utils as utils

urdf_path = "isaac_sim/urdf/so_100_dual_foldnet.urdf"
urdf = batch_urdf.URDF(batch_size=1, urdf_path=urdf_path, dtype=torch.float32, device="cpu", mesh_dir="")

# No base rotation (identity)
urdf.update_base_link_transformation("base_link", urdf.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))

# URDF link lengths from joints:
# Shoulder_Rotation at origin, shoulder offset
# From URDF: left_Base → left_Shoulder_Rotation: xyz="0 -0.0452 0.0165"
# left_Shoulder_Rotation → left_Upper_Arm: xyz="0 0.1025 0.0306"
# left_Upper_Arm → left_Lower_Arm: xyz="0 0.11257 0.028"
# left_Lower_Arm → left_Wrist_Pitch_Roll: xyz="0 0.0052 0.1349"
# left_Wrist_Pitch_Roll → left_Fixed_Gripper: xyz="0 -0.0601 0"
# left_Fixed_Gripper → left_tcp: xyz="0.0101 0.0 0"
print("=" * 60)
print("ARM LINK LENGTHS (from URDF)")
print("=" * 60)

links = [
    ("Base → Shoulder_Rotation", [0, -0.0452, 0.0165]),
    ("Shoulder_Rotation → Upper_Arm", [0, 0.1025, 0.0306]),
    ("Upper_Arm → Lower_Arm", [0, 0.11257, 0.028]),
    ("Lower_Arm → Wrist", [0, 0.0052, 0.1349]),
    ("Wrist → Fixed_Gripper", [0, -0.0601, 0]),
    ("Fixed_Gripper → TCP", [0.0101, 0.0, 0]),
]

total = 0
for name, xyz in links:
    length = np.linalg.norm(xyz)
    total += length
    print(f"  {name}: {length*100:.1f} cm  (xyz={xyz})")

print(f"\n  Total arm reach (straight): {total*100:.1f} cm")
print(f"  Left base offset from center: -0.3 m (30 cm to the left)")

# Check various poses to see where the TCP can reach
print("\n" + "=" * 60)
print("WORKSPACE EXPLORATION")
print("=" * 60)

# Set up with 180° Z rotation
urdf.update_base_link_transformation("base_link", urdf.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

# Try different shoulder rotation values
for sr in [0.0, 0.5, 1.0, -0.5, -1.0]:
    for sp in [0.0, 0.5, 1.0, 1.5]:
        for el in [0.0, -0.5, -1.0, -1.5]:
            urdf.update_cfg(urdf.cfg_f2t({
                "left_Shoulder_Rotation": sr, "left_Shoulder_Pitch": sp,
                "left_Elbow": el, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
                "left_Gripper": 0.0,
                "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
                "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
                "right_Gripper": 0.0,
            }))
            tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
            # Check if near the target zone (x ~ 0, y ~ 0.15, z ~ 0.02)
            dist = np.linalg.norm(tcp_l - np.array([0.05, 0.15, 0.02]))
            if dist < 0.15:
                print(f"  SR={sr:.1f} SP={sp:.1f} EL={el:.1f} → TCP=({tcp_l[0]:.3f}, {tcp_l[1]:.3f}, {tcp_l[2]:.3f}) dist={dist:.3f}")

# Also check what the FULL reach boundaries look like
print("\n" + "=" * 60)
print("TCP REACH ENVELOPE (left arm, z ~ 0)")
print("=" * 60)
min_xyz = np.array([np.inf, np.inf, np.inf])
max_xyz = np.array([-np.inf, -np.inf, -np.inf])
low_z_points = []

for sr in np.linspace(-3.0, 3.0, 20):
    for sp in np.linspace(-3.14, 3.14, 20):
        for el in np.linspace(-3.14, 3.14, 10):
            urdf.update_cfg(urdf.cfg_f2t({
                "left_Shoulder_Rotation": sr, "left_Shoulder_Pitch": sp,
                "left_Elbow": el, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
                "left_Gripper": 0.0,
                "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
                "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
                "right_Gripper": 0.0,
            }))
            tcp = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
            min_xyz = np.minimum(min_xyz, tcp)
            max_xyz = np.maximum(max_xyz, tcp)
            if abs(tcp[2]) < 0.05:  # near z=0
                low_z_points.append(tcp.copy())

print(f"  Min XYZ: ({min_xyz[0]:.3f}, {min_xyz[1]:.3f}, {min_xyz[2]:.3f})")
print(f"  Max XYZ: ({max_xyz[0]:.3f}, {max_xyz[1]:.3f}, {max_xyz[2]:.3f})")
print(f"  Points near z=0: {len(low_z_points)}")
if low_z_points:
    low_z = np.array(low_z_points)
    print(f"  Low-Z range: X=({low_z[:,0].min():.3f}, {low_z[:,0].max():.3f}) Y=({low_z[:,1].min():.3f}, {low_z[:,1].max():.3f})")
