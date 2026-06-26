"""
SO-100 FoldEnv: FoldNet-based cloth folding environment for the bimanual SO-100 robot.

Usage:
    conda activate foldnet
    export CUDA_VISIBLE_DEVICES=0
    python isaac_sim/scripts/so100_foldenv.py --cloth tshirt_sp --num_trajs 1
"""

import os, sys, json, copy, argparse, pathlib, math
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from collections import deque

import numpy as np
import torch

# PyFlex must be importable before garmentds
_PYFLEX_LIBS = os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src/pyflex/libs")
sys.path.insert(0, os.path.abspath(_PYFLEX_LIBS))
os.environ["PYFLEX_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src/pyflex/PyFlex"))
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code"))

import pyflex
import garmentds.common.utils as utils
from garmentds.foldenv.fold_env import FoldEnv as FoldEnvOrig, FoldEnvCfg, RobotCfg, FoldEnvState, Picker
from garmentds.foldenv.fold_env import load_mesh_raw_sim
from garmentds.foldenv.fold_env import RenderProcess


# ==============================================================================
# Null render process (no Blender / Isaac needed)
# ==============================================================================
class RenderProcessNull(RenderProcess):
    def __init__(self, **kwargs):
        pass
    def send_message(self, msg):
        pass
    def sync(self):
        pass
    def join(self):
        pass
    def terminate(self):
        pass


# ==============================================================================
# SO-100 Robot (subclass without legs / torso logic)
# ==============================================================================
class RobotSO100:
    def __init__(self, picker_l: Picker, picker_r: Picker, cfg: RobotCfg):
        self._cfg = copy.deepcopy(cfg)
        import batch_urdf
        self._urdf = batch_urdf.URDF(
            batch_size=1, urdf_path=cfg.urdf_path, dtype=torch.float32,
            device=cfg.device, mesh_dir=cfg.mesh_dir,
        )
        self._urdf.update_base_link_transformation(cfg.base_link, self._urdf.tensor([cfg.base_pos]))
        self._urdf.update_cfg(self._urdf.cfg_f2t(cfg.init_qpos))
        self._tensor = self._urdf.tensor
        self._t2f = self._urdf.cfg_t2f
        self._f2t = self._urdf.cfg_f2t
        self._picker = {"left": picker_l, "right": picker_r}
        self._waypoints_qpos = deque()
        self._waypoints_picker = deque()
        self._ik_fail_count = 0

    @property
    def urdf(self):
        return self._urdf
    @property
    def picker(self):
        return self._picker
    @property
    def ik_fail_count(self):
        return self._ik_fail_count
    @property
    def waypoints_qpos(self):
        return self._waypoints_qpos

    def modify_gripper_qpos(self, qpos, hand, action):
        cfg = self._cfg
        for j in dict(left=cfg.gripper_l_joints, right=cfg.gripper_r_joints)[hand]:
            qpos[j] = (
                (action - Picker.OPEN) * cfg.gripper_close_val +
                (Picker.CLOSE - action) * cfg.gripper_open_val
            ) / (Picker.CLOSE - Picker.OPEN)
        return qpos

    def _set_gripper_qpos(self, hand, action):
        action = np.clip(action, 0., 1.)
        qpos = self._t2f(self._urdf.cfg)
        qpos = self.modify_gripper_qpos(qpos, hand, action)
        self._urdf.update_cfg(self._f2t(qpos))

    @staticmethod
    def _xyz_avg(xyz_l, xyz_r):
        xyz = (xyz_l + xyz_r) / 2.
        def max_abs_avg(x, y):
            if (x * y) > 0.:
                return max(abs(x), abs(y)) * np.sign(x)
            else:
                return x + y
        xyz[1] = max_abs_avg(xyz_l[1], xyz_r[1])
        return xyz

    def set_target_xyz(self, steps, xyz_l=None, xyz_r=None):
        cfg = self._cfg
        self._waypoints_qpos.clear()
        curr_qpos = self._t2f(self._urdf.cfg)

        if xyz_l is None and xyz_r is None:
            targ_qpos = {k: v for k, v in curr_qpos.items()}
        else:
            if xyz_l is None:
                xyz_l = utils.torch_to_numpy(self._urdf.link_transform_map[cfg.tcp_l])[0, :3, 3]
            if xyz_r is None:
                xyz_r = utils.torch_to_numpy(self._urdf.link_transform_map[cfg.tcp_r])[0, :3, 3]
            targ_qpos = {k: v for k, v in curr_qpos.items()}

            ik_init_cfg = self._f2t(targ_qpos)
            for hand in ["left", "right"]:
                # SWAP: FoldNet's "left" target goes to URDF's "right" arm and vice versa.
                # Because the robot is rotated 180° in Isaac Sim, URDF "left" (X=-0.3)
                # appears on the RIGHT side visually, and URDF "right" (X=+0.3) on the LEFT.
                xyz = dict(left=xyz_r, right=xyz_l)[hand]
                tcp_str = dict(left=cfg.tcp_l, right=cfg.tcp_r)[hand]
                arm_joints = dict(left=cfg.arm_l_joints, right=cfg.arm_r_joints)[hand]
                fix_joints = [j for j in self._urdf.cfg.keys() if j not in arm_joints]

                # We want the tip to reach xyz.
                # If we enforce the full 4x4 matrix (orientation + position), the arm will twist
                # into self-collision to maintain its starting orientation perfectly.
                # So we use a custom err_func that strongly penalizes position error, 
                # but only lightly penalizes orientation error to keep the arm from flipping wildly.
                curr_tcp_tf = utils.torch_to_numpy(self._urdf.link_transform_map[tcp_str])[0]
                target_mat4 = curr_tcp_tf.copy()
                target_mat4[:3, 3] = xyz
                target_mat4_t = self._tensor(target_mat4) # (4, 4)
                
                # Mask: 1.0 for translation (idx 3, 7, 11), 0.05 for rotation to guide it but not force it
                mask_t = self._tensor([
                    0.05, 0.05, 0.05, 1.0, 
                    0.05, 0.05, 0.05, 1.0, 
                    0.05, 0.05, 0.05, 1.0, 
                    0.0,  0.0,  0.0,  0.0
                ])
                B = 1

                def err_func(link_transform_map):
                    curr_mat4 = link_transform_map[tcp_str].view(B, 16, 16)
                    curr_mat16 = curr_mat4[:, torch.arange(16), torch.arange(16)]
                    return (curr_mat16 - target_mat4_t.view(B, 16)) * mask_t

                xyz_t = self._tensor(xyz.reshape(1, 3))
                def loss_func(link_transform_map):
                    curr_xyz = link_transform_map[tcp_str][:, :3, 3]
                    return torch.sum(torch.square(curr_xyz - xyz_t), dim=1)

                qpos, info = self._urdf.inverse_kinematics_optimize(
                    err_func=err_func, loss_func=loss_func, init_cfg=ik_init_cfg,
                    fix_joint=fix_joints,
                    max_iter=200, square_err_th=1e-6, lda=1e-4,
                )
                
                # Check position error purely for debugging/stats
                result_pos = utils.torch_to_numpy(self._urdf._forward_cfg(qpos, update=False)[tcp_str])[0, :3, 3]
                pos_err = np.linalg.norm(result_pos - xyz)
                if pos_err > 0.03:
                    self._ik_fail_count += 1

                qpos = self._t2f(qpos)
                for j in arm_joints:
                    targ_qpos[j] = qpos[j]

                # Update the init_cfg for the next arm to include the newly solved joints
                ik_init_cfg = self._f2t(targ_qpos)

        for i in range(steps):
            self._waypoints_qpos.append({
                k: curr_qpos[k] + (targ_qpos[k] - curr_qpos[k]) / steps * (i + 1)
                for k in targ_qpos.keys()
            })

    def set_target_picker(self, steps, picker_l=None, picker_r=None):
        self._waypoints_picker.clear()
        # SWAP: FoldNet's "left" picker controls URDF's "right" arm and vice versa
        curr_l = self._picker["left"].val_float
        curr_r = self._picker["right"].val_float
        targ_l = curr_l if picker_r is None else picker_r  # FoldNet right → URDF left
        targ_r = curr_r if picker_l is None else picker_l  # FoldNet left → URDF right
        for i in range(steps):
            self._waypoints_picker.append(dict(
                left=curr_l + (targ_l - curr_l) / steps * (i + 1),
                right=curr_r + (targ_r - curr_r) / steps * (i + 1)
            ))

    def step(self):
        if len(self._waypoints_qpos) > 0:
            wp = self._waypoints_qpos.popleft()
            self.set_qpos(wp, False)
        if len(self._waypoints_picker) > 0:
            wp = self._waypoints_picker.popleft()
            self.set_picker(left=wp["left"], right=wp["right"])
        for hand in ["left", "right"]:
            tcp_str = dict(left=self._cfg.tcp_l, right=self._cfg.tcp_r)[hand]
            tcp_tf = utils.torch_to_numpy(self._urdf.link_transform_map[tcp_str])[0, :, :]
            self._picker[hand].step(tcp_tf)

    def set_qpos(self, qpos, exclude_gripper_joints):
        if not exclude_gripper_joints:
            self._urdf.update_cfg(self._f2t(qpos))
        else:
            curr_qpos = self._t2f(self._urdf.cfg)
            exclude = set(self._cfg.gripper_l_joints + self._cfg.gripper_r_joints)
            next_qpos = {k: v if k not in exclude else curr_qpos[k] for k, v in qpos.items()}
            self._urdf.update_cfg(self._f2t(next_qpos))
        for hand in ["left", "right"]:
            tcp_str = dict(left=self._cfg.tcp_l, right=self._cfg.tcp_r)[hand]
            tcp_tf = utils.torch_to_numpy(self._urdf.link_transform_map[tcp_str])[0, :, :]
            self._picker[hand].set_tf(tcp_tf)

    def set_picker(self, left=None, right=None):
        if left is not None:
            self._picker["left"].set_action(left)
            self._set_gripper_qpos("left", left)
        if right is not None:
            self._picker["right"].set_action(right)
            self._set_gripper_qpos("right", right)

    def get_qpos(self):
        return self._t2f(self._urdf.cfg)

    def get_base_pose(self):
        return utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.base_link][0, ...])

    def get_tcp_xyz(self):
        # SWAP: match the left/right swap in set_target_xyz
        return {
            "left": utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.tcp_r])[0, :3, 3],
            "right": utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.tcp_l])[0, :3, 3],
        }

    def get_gripper_state(self):
        # SWAP: match the left/right swap
        return {"left": self._picker["right"].val_float, "right": self._picker["left"].val_float}

    def get_gripper_state_int(self):
        # SWAP: match the left/right swap
        return {"left": self._picker["right"].val, "right": self._picker["left"].val}

    def reset(self):
        self.set_qpos(self._cfg.init_qpos, False)
        self.set_picker(Picker.OPEN, Picker.OPEN)
        self._waypoints_qpos.clear()
        self._waypoints_picker.clear()
        self._ik_fail_count = 0


# ==============================================================================
# SO-100 FoldEnv
# ==============================================================================
class FoldEnvSO100(FoldEnvOrig):
    def __init__(self, cfg: FoldEnvCfg):
        cfg = copy.deepcopy(cfg)
        self._cfg = cfg
        self._state = FoldEnvState()
        self._domain_randomize(cfg)
        self._load_cloth(cfg)
        self._init_pyflex(cfg)
        self._init_env(cfg)
        self._init_cloth(cfg)
        self._init_robot_so100(cfg)
        self._init_cache()
        self._render_process = RenderProcessNull()

    def _init_robot_so100(self, cfg: FoldEnvCfg):
        self._robot = RobotSO100(self._new_picker("left"), self._new_picker("right"), cfg.robot_cfg)


def make_so100_robot_cfg() -> RobotCfg:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return RobotCfg(
        urdf_path=f"{base}/isaac_sim/urdf/so_100_dual_foldnet.urdf",
        mesh_dir=f"{base}/src/SO-100-arm/models/so_100_arm_5dof/meshes",
        device="cuda:0",
        arm_l_joints=["left_Shoulder_Rotation", "left_Shoulder_Pitch", "left_Elbow", "left_Wrist_Pitch", "left_Wrist_Roll"],
        tcp_l="left_gripper_tcp_link",
        arm_r_joints=["right_Shoulder_Rotation", "right_Shoulder_Pitch", "right_Elbow", "right_Wrist_Pitch", "right_Wrist_Roll"],
        tcp_r="right_gripper_tcp_link",
        base_link="base_link",
        base_pos=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # Identity - IK works in FoldNet's native frame
        gripper_l_joints=["left_Gripper"],
        gripper_r_joints=["right_Gripper"],
        gripper_open_val=0.7,
        gripper_close_val=0.0,
        leg_joints=[],
        init_qpos={
            "left_Shoulder_Rotation": 0.0, "left_Shoulder_Pitch": 0.0,
            "left_Elbow": 0.0, "left_Wrist_Pitch": 0.0, "left_Wrist_Roll": 0.0,
            "left_Gripper": 0.0,
            "right_Shoulder_Rotation": 0.0, "right_Shoulder_Pitch": 0.0,
            "right_Elbow": 0.0, "right_Wrist_Pitch": 0.0, "right_Wrist_Roll": 0.0,
            "right_Gripper": 0.0,
        },
        ik_init_qpos={},
        ik_move_leg=False,
        ik_kwargs=dict(max_iter=64, square_err_th=1e-4, lda=1e-3),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate SO-100 cloth folding dataset")
    parser.add_argument("--cloth", default="tshirt_sp", help="garment category (tshirt_sp, trousers, vest, etc.)")
    parser.add_argument("--variant", type=int, default=0, help="garment variant index")
    parser.add_argument("--out", default="/tmp/so100_fold_dataset", help="output directory")
    parser.add_argument("--num_trajs", type=int, default=1, help="number of trajectories")
    parser.add_argument("--headless", action="store_true", default=True, help="headless mode")
    args = parser.parse_args()

    cloth_dir = f"/home/ozan/Downloads/so100_ws/foldnet_garments/{args.cloth}_{args.variant}"
    cloth_path = os.path.join(cloth_dir, "mesh.obj")
    if not os.path.exists(cloth_path):
        print(f"Cloth not found: {cloth_path}")
        return 1

    robot_cfg = make_so100_robot_cfg()
    env_cfg = FoldEnvCfg(
        cloth_obj_path=cloth_path,
        cloth_scale=0.5,
        render=False,
        render_mode=[],
        render_process_num=1,
        robot_cfg=robot_cfg,
    )

    env = FoldEnvSO100(env_cfg)
    env.reset()
    print(f"Folding env ready. Cloth: {args.cloth}_{args.variant}")
    print(f"Robot joints: {list(env._robot.get_qpos().keys())}")

    tcp = env._robot.get_tcp_xyz()
    print(f"TCP: L({tcp['left'][0]:.3f}, {tcp['left'][1]:.3f}, {tcp['left'][2]:.3f}) "
          f"R({tcp['right'][0]:.3f}, {tcp['right'][1]:.3f}, {tcp['right'][2]:.3f})")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
