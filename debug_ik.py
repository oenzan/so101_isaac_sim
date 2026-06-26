"""
Debug: Proper IK test with better parameters
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

# 180° Z rotation
urdf.update_base_link_transformation("base_link", urdf.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

# Start from a good initial config (the one that got close in the workspace test)
init_qpos = {
    "left_Shoulder_Rotation": 1.0, "left_Shoulder_Pitch": 1.5,
    "left_Elbow": -1.5, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.0,
    "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
    "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
    "right_Gripper": 0.0,
}
urdf.update_cfg(urdf.cfg_f2t(init_qpos))

tcp_before = utils.torch_to_numpy(urdf.link_transform_map["left_gripper_tcp_link"])[0, :3, 3]
print(f"TCP before IK: ({tcp_before[0]:.4f}, {tcp_before[1]:.4f}, {tcp_before[2]:.4f})")

target = np.array([0.05, 0.15, 0.02])
print(f"IK target: {target}")

tcp_str = "left_gripper_tcp_link"
fix_joints = [j for j in urdf.cfg.keys() if j not in 
    ["left_Shoulder_Rotation", "left_Shoulder_Pitch", "left_Elbow", "left_Wrist_Pitch", "left_Wrist_Roll"]]

xyz_t = urdf.tensor(target.reshape(1, 3))
B = 1

# Simple loss - only position
def loss_func(link_transform_map):
    curr_xyz = link_transform_map[tcp_str][:, :3, 3]
    return torch.sum(torch.square(curr_xyz - xyz_t), dim=1)

pose = np.eye(4)
pose[:3, 3] = target
pose = urdf.tensor(pose)
mask_t = urdf.tensor([0.,0.,0.,1., 0.,0.,0.,1., 0.,0.,0.,1., 0.,0.,0.,0.])

def err_func(link_transform_map):
    curr_mat4 = link_transform_map[tcp_str].view(B, 16, 16)
    curr_mat16 = curr_mat4[:, torch.arange(16), torch.arange(16)]
    return (curr_mat16 - pose.view(B, 16)) * mask_t

# Test 1: Original params
print("\n--- Test 1: max_iter=64, lda=1e-3, th=1e-4 ---")
urdf.update_cfg(urdf.cfg_f2t(init_qpos))
qpos, info = urdf.inverse_kinematics_optimize(
    err_func=err_func, loss_func=loss_func, init_cfg=urdf.cfg,
    fix_joint=fix_joints,
    max_iter=64, square_err_th=1e-4, lda=1e-3,
)
tcp = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  TCP: ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})")
print(f"  Error: {np.linalg.norm(tcp - target):.4f} m")
print(f"  Iter: {info['iter_idx']}")

# Test 2: More iterations, tighter threshold
print("\n--- Test 2: max_iter=500, lda=1e-3, th=1e-8 ---")
urdf.update_cfg(urdf.cfg_f2t(init_qpos))
qpos, info = urdf.inverse_kinematics_optimize(
    err_func=err_func, loss_func=loss_func, init_cfg=urdf.cfg,
    fix_joint=fix_joints,
    max_iter=500, square_err_th=1e-8, lda=1e-3,
)
tcp = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  TCP: ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})")
print(f"  Error: {np.linalg.norm(tcp - target):.4f} m")
print(f"  Iter: {info['iter_idx']}")

# Test 3: Smaller lambda (more aggressive)
print("\n--- Test 3: max_iter=500, lda=1e-5, th=1e-8 ---")
urdf.update_cfg(urdf.cfg_f2t(init_qpos))
qpos, info = urdf.inverse_kinematics_optimize(
    err_func=err_func, loss_func=loss_func, init_cfg=urdf.cfg,
    fix_joint=fix_joints,
    max_iter=500, square_err_th=1e-8, lda=1e-5,
)
tcp = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  TCP: ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})")
print(f"  Error: {np.linalg.norm(tcp - target):.4f} m")
print(f"  Iter: {info['iter_idx']}")
print(f"  Solved qpos: {urdf.cfg_t2f(qpos)}")

# Test 4: Start from zeros
print("\n--- Test 4: Start from ZEROS, max_iter=500, lda=1e-3, th=1e-8 ---")
zero_qpos = {k: 0.0 for k in init_qpos}
urdf.update_cfg(urdf.cfg_f2t(zero_qpos))
tcp_zero = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  TCP at zeros: ({tcp_zero[0]:.4f}, {tcp_zero[1]:.4f}, {tcp_zero[2]:.4f})")
qpos, info = urdf.inverse_kinematics_optimize(
    err_func=err_func, loss_func=loss_func, init_cfg=urdf.cfg,
    fix_joint=fix_joints,
    max_iter=500, square_err_th=1e-8, lda=1e-3,
)
tcp = utils.torch_to_numpy(urdf.link_transform_map[tcp_str])[0, :3, 3]
print(f"  TCP: ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})")
print(f"  Error: {np.linalg.norm(tcp - target):.4f} m")
print(f"  Iter: {info['iter_idx']}")
print(f"  Solved qpos: {urdf.cfg_t2f(qpos)}")

# ---- Check err_func values ----
print("\n--- Checking err_func at solution ---")
err = err_func(urdf.link_transform_map)
print(f"  err_func output: {err}")
sq_err = torch.sum(err**2)
print(f"  square_err: {sq_err.item()}")
print(f"  threshold: 1e-4 = {1e-4}")
print(f"  → converged? {sq_err.item() < 1e-4}")
