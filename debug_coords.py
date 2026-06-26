"""
Debug script: trace what coordinates go where.
Run with: /path/to/isaac-sim-python.sh debug_coords.py
"""
import sys, os
import numpy as np
import torch

# Setup paths
foldnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code/src"))
sys.path.insert(0, foldnet_src)
batch_urdf_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code/external/batch_urdf/src"))
sys.path.insert(0, batch_urdf_src)
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoldNet_code"))

from unittest.mock import MagicMock
sys.modules['pyflex'] = MagicMock()

import batch_urdf
import garmentds.common.utils as utils

# ---- 1. Check quaternion format ----
from batch_urdf.utils import quaternion_matrix

print("=" * 60)
print("1. QUATERNION FORMAT CHECK")
print("=" * 60)
# batch_urdf says "real part first" = [w, x, y, z]
# Identity: w=1, x=0, y=0, z=0 → [1,0,0,0]
q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
print(f"Identity [1,0,0,0] → \n{quaternion_matrix(q_identity)}")

# 180° around Z: w=0, x=0, y=0, z=1
q_z180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
print(f"\n180° Z [0,0,0,1] → \n{quaternion_matrix(q_z180)}")
# Expected: [[-1,0,0],[0,-1,0],[0,0,1]]

# ---- 2. Load URDF and check FK ----
print("\n" + "=" * 60)
print("2. FORWARD KINEMATICS CHECK")
print("=" * 60)

urdf_path = "isaac_sim/urdf/so_100_dual_foldnet.urdf"
urdf = batch_urdf.URDF(
    batch_size=1, urdf_path=urdf_path, dtype=torch.float32,
    device="cpu", mesh_dir=""
)

# Case A: No base rotation (identity)
base_pos_identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # pos + [w,x,y,z] identity
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_identity]))
urdf.update_cfg(urdf.cfg_f2t({
    "left_Shoulder_Rotation": 0.0, "left_Shoulder_Pitch": 0.0,
    "left_Elbow": 0.0, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.0,
    "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
    "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
    "right_Gripper": 0.0,
}))

tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
tcp_r = utils.torch_to_numpy(urdf.link_transform_map["right_gripper_tcp_link"])[0, :3, 3]
base = utils.torch_to_numpy(urdf.link_transform_map["base_link"])[0]
print(f"\nCase A: Identity base (no rotation)")
print(f"  base_link transform:\n{base}")
print(f"  Left TCP:  ({tcp_l[0]:.4f}, {tcp_l[1]:.4f}, {tcp_l[2]:.4f})")
print(f"  Right TCP: ({tcp_r[0]:.4f}, {tcp_r[1]:.4f}, {tcp_r[2]:.4f})")

# Case B: 180° Z rotation with [0,0,0, 0,0,0,1]
base_pos_z180 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # pos + [w,x,y,z] = 180° Z
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_z180]))

tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
tcp_r = utils.torch_to_numpy(urdf.link_transform_map["right_gripper_tcp_link"])[0, :3, 3]
base = utils.torch_to_numpy(urdf.link_transform_map["base_link"])[0]
print(f"\nCase B: 180° Z rotation [0,0,0,1]")
print(f"  base_link transform:\n{base}")
print(f"  Left TCP:  ({tcp_l[0]:.4f}, {tcp_l[1]:.4f}, {tcp_l[2]:.4f})")
print(f"  Right TCP: ({tcp_r[0]:.4f}, {tcp_r[1]:.4f}, {tcp_r[2]:.4f})")

# Case C: NO rotation at all (current code has [0,0,0, 0,0,0,1])
# Let's also try identity to see what "natural" direction is
base_pos_id = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_id]))

tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
tcp_r = utils.torch_to_numpy(urdf.link_transform_map["right_gripper_tcp_link"])[0, :3, 3]
left_base = utils.torch_to_numpy(urdf.link_transform_map["left_base_link"])[0, :3, 3]
right_base = utils.torch_to_numpy(urdf.link_transform_map["right_base_link"])[0, :3, 3]
print(f"\nCase C: Identity [1,0,0,0]")
print(f"  Left base:  ({left_base[0]:.4f}, {left_base[1]:.4f}, {left_base[2]:.4f})")
print(f"  Right base: ({right_base[0]:.4f}, {right_base[1]:.4f}, {right_base[2]:.4f})")
print(f"  Left TCP:   ({tcp_l[0]:.4f}, {tcp_l[1]:.4f}, {tcp_l[2]:.4f})")
print(f"  Right TCP:  ({tcp_r[0]:.4f}, {tcp_r[1]:.4f}, {tcp_r[2]:.4f})")

# ---- 3. Check what FoldNet policy targets look like ----
print("\n" + "=" * 60)
print("3. FOLDNET POLICY TARGET COORDINATES")
print("=" * 60)
print(f"  PICKER_Z (grasp height above table) = 0.02 m")
print(f"  CLOTH_CENTER = (0.0, 0.15, 0.0)")
print(f"  So typical target = (some_x, ~0.15, 0.02)")

# ---- 4. Isaac Sim placement ----
print("\n" + "=" * 60)
print("4. ISAAC SIM PLACEMENT")
print("=" * 60)
TABLE_HEIGHT = 0.75
print(f"  Robot base placed at: (0, 0, {TABLE_HEIGHT}) + 180° Z rotation")
print(f"  Cloth placed at: (0, 0.15, {TABLE_HEIGHT + 0.005})")
print(f"  Marker target = FoldNet xyz + (0, 0, {TABLE_HEIGHT})")

# ---- 5. The KEY question: IK target vs Isaac Sim ----
print("\n" + "=" * 60)
print("5. COORDINATE FRAME ANALYSIS")
print("=" * 60)

# In BatchURDF, base is at (0,0,0) with some rotation
# In Isaac Sim, base is at (0,0,0.75) with 180° Z rotation
# FoldNet targets are relative to TABLE_TOP (z=0)
# So if IK solves for target (0, 0.15, 0.02) in BatchURDF frame,
# the resulting joint angles, when applied in Isaac Sim, should produce
# TCP at (0, 0.15, 0.02) RELATIVE to the robot base.
# But in Isaac Sim, the base is at z=0.75, so TCP would be at
# (0, 0.15, 0.02) + (0, 0, 0.75) = (0, 0.15, 0.77) in world coords.
# The cloth is at z=0.755 in Isaac Sim world coords.
# So 0.77 - 0.755 = 0.015 m above cloth. That's close!

# BUT the robot is also ROTATED 180° in Isaac Sim.
# With identity base in BatchURDF, arms face -Y (URDF natural direction)
# With 180° Z, arms face +Y (towards cloth)
# In Isaac Sim, robot is also rotated 180° Z, so arms face +Y too.
# So the rotation should match... IF we use the right quaternion.

print("  If base_pos uses 180° Z rotation:")
base_pos_z180 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_z180]))
tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
tcp_r = utils.torch_to_numpy(urdf.link_transform_map["right_gripper_tcp_link"])[0, :3, 3]
print(f"  Left TCP direction from base: Y={tcp_l[1]:.4f} (positive = towards cloth)")
print(f"  Right TCP direction from base: Y={tcp_r[1]:.4f}")
print(f"  → If Y is NEGATIVE, robot faces WRONG direction!")

print("\n  If base_pos uses identity (no rotation):")
base_pos_id = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_id]))
tcp_l = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
tcp_r = utils.torch_to_numpy(urdf.link_transform_map["right_gripper_tcp_link"])[0, :3, 3]
print(f"  Left TCP direction from base: Y={tcp_l[1]:.4f}")
print(f"  Right TCP direction from base: Y={tcp_r[1]:.4f}")
print(f"  → If Y is NEGATIVE, IK needs 180° rotation to face cloth")

# ---- 6. IK test: solve for a known target and check ----
print("\n" + "=" * 60)
print("6. IK SOLVE TEST")
print("=" * 60)
# Use 180° Z rotation
base_pos_z180 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
urdf.update_base_link_transformation("base_link", urdf.tensor([base_pos_z180]))
urdf.update_cfg(urdf.cfg_f2t({
    "left_Shoulder_Rotation": 0.0, "left_Shoulder_Pitch": 0.0,
    "left_Elbow": 0.0, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.0,
    "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
    "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
    "right_Gripper": 0.0,
}))

# Target: typical cloth point
target = np.array([0.05, 0.15, 0.02])
print(f"  IK target: {target}")

tcp_str = "left_gripper_tcp_link"
fix_joints = [j for j in urdf.cfg.keys() if j not in 
    ["left_Shoulder_Rotation", "left_Shoulder_Pitch", "left_Elbow", "left_Wrist_Pitch", "left_Wrist_Roll"]]

xyz_t = urdf.tensor(target.reshape(1, 3))

def loss_func(link_transform_map):
    curr_xyz = link_transform_map[tcp_str][:, :3, 3]
    return torch.sum(torch.square(curr_xyz - xyz_t), dim=1)

pose = np.eye(4)
pose[:3, 3] = target
pose = urdf.tensor(pose)
B = 1
mask_t = urdf.tensor([0.,0.,0.,1., 0.,0.,0.,1., 0.,0.,0.,1., 0.,0.,0.,0.])

def err_func(link_transform_map):
    curr_mat4 = link_transform_map[tcp_str].view(B, 16, 16)
    curr_mat16 = curr_mat4[:, torch.arange(16), torch.arange(16)]
    return (curr_mat16 - pose.view(B, 16)) * mask_t

qpos, info = urdf.inverse_kinematics_optimize(
    err_func=err_func, loss_func=loss_func, init_cfg=urdf.cfg,
    fix_joint=fix_joints,
    max_iter=64, square_err_th=1e-4, lda=1e-3,
)

# Check result
tcp_result = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  IK result TCP: ({tcp_result[0]:.4f}, {tcp_result[1]:.4f}, {tcp_result[2]:.4f})")
print(f"  IK error: {np.linalg.norm(tcp_result - target):.6f} m")
print(f"  IK converged at iter: {info['iter_idx']}")
print(f"  IK solved qpos: {urdf.cfg_t2f(qpos)}")
