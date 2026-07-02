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

SO100_SAFE_TRAVEL_Z = 0.16
SO100_LIFTED_MOVE_XY_THRESHOLD = 0.08
SO100_IK_POSITION_FAIL_M = 0.03
SO100_INIT_GRIPPER_CLOSED_RATIO = 0.70


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
        self._waypoints_qpos.clear()
        curr_qpos = self._t2f(self._urdf.cfg)

        if xyz_l is None and xyz_r is None:
            self._append_qpos_interpolation(curr_qpos, curr_qpos, steps)
        else:
            if xyz_l is None:
                xyz_l = self.get_tcp_xyz()["left"]
            if xyz_r is None:
                xyz_r = self.get_tcp_xyz()["right"]

            waypoint_pairs = self._make_cartesian_waypoints(
                self.get_tcp_xyz()["left"], self.get_tcp_xyz()["right"], xyz_l, xyz_r
            )
            segment_count = len(waypoint_pairs)
            base_steps = max(1, steps // segment_count)
            remainder = max(0, steps - base_steps * segment_count)
            segment_start_qpos = curr_qpos
            for seg_idx, (wp_l, wp_r) in enumerate(waypoint_pairs):
                targ_qpos = self._solve_target_qpos(segment_start_qpos, wp_l, wp_r)
                steps_this_segment = base_steps + (1 if seg_idx < remainder else 0)
                self._append_qpos_interpolation(segment_start_qpos, targ_qpos, steps_this_segment)
                segment_start_qpos = targ_qpos

    def _make_cartesian_waypoints(self, curr_l, curr_r, targ_l, targ_r):
        curr_l, curr_r = np.array(curr_l, dtype=float), np.array(curr_r, dtype=float)
        targ_l, targ_r = np.array(targ_l, dtype=float), np.array(targ_r, dtype=float)
        max_xy_dist = max(
            np.linalg.norm((targ_l - curr_l)[:2]),
            np.linalg.norm((targ_r - curr_r)[:2]),
        )
        if max_xy_dist < SO100_LIFTED_MOVE_XY_THRESHOLD:
            return [(targ_l, targ_r)]

        safe_z = max(SO100_SAFE_TRAVEL_Z, curr_l[2], curr_r[2], targ_l[2], targ_r[2])
        curr_l_high, curr_r_high = curr_l.copy(), curr_r.copy()
        targ_l_high, targ_r_high = targ_l.copy(), targ_r.copy()
        curr_l_high[2] = curr_r_high[2] = safe_z
        targ_l_high[2] = targ_r_high[2] = safe_z

        waypoints = []
        if curr_l[2] < safe_z - 1e-4 or curr_r[2] < safe_z - 1e-4:
            waypoints.append((curr_l_high, curr_r_high))
        waypoints.append((targ_l_high, targ_r_high))
        if targ_l[2] < safe_z - 1e-4 or targ_r[2] < safe_z - 1e-4:
            waypoints.append((targ_l, targ_r))
        return waypoints

    def _solve_target_qpos(self, start_qpos, xyz_l, xyz_r):
        cfg = self._cfg
        targ_qpos = {k: v for k, v in start_qpos.items()}
        ik_init_cfg = self._f2t(targ_qpos)
        for hand in ["left", "right"]:
            xyz = dict(left=xyz_l, right=xyz_r)[hand]
            tcp_str = dict(left=cfg.tcp_l, right=cfg.tcp_r)[hand]
            arm_joints = dict(left=cfg.arm_l_joints, right=cfg.arm_r_joints)[hand]
            fix_joints = [j for j in self._urdf.cfg.keys() if j not in arm_joints]

            start_tf_map = self._urdf._forward_cfg(self._f2t(targ_qpos), update=False)
            curr_tcp_tf = utils.torch_to_numpy(start_tf_map[tcp_str])[0]
            target_mat4 = curr_tcp_tf.copy()
            target_mat4[:3, 3] = xyz
            target_mat4_t = self._tensor(target_mat4)
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

            ik_kw = self._cfg.ik_kwargs or {}
            qpos, _ = self._urdf.inverse_kinematics_optimize(
                err_func=err_func, loss_func=loss_func, init_cfg=ik_init_cfg,
                fix_joint=fix_joints,
                max_iter=ik_kw.get("max_iter", 100),
                square_err_th=ik_kw.get("square_err_th", 1e-5),
                lda=ik_kw.get("lda", 1e-4),
            )

            result_pos = utils.torch_to_numpy(self._urdf._forward_cfg(qpos, update=False)[tcp_str])[0, :3, 3]
            if np.linalg.norm(result_pos - xyz) > SO100_IK_POSITION_FAIL_M:
                self._ik_fail_count += 1

            qpos = self._t2f(qpos)
            for j in arm_joints:
                targ_qpos[j] = qpos[j]
            ik_init_cfg = self._f2t(targ_qpos)

        return targ_qpos

    def _append_qpos_interpolation(self, start_qpos, targ_qpos, steps):
        steps = max(1, int(steps))
        for i in range(steps):
            self._waypoints_qpos.append({
                k: start_qpos[k] + (targ_qpos[k] - start_qpos[k]) / steps * (i + 1)
                for k in targ_qpos.keys()
            })

    def set_target_picker(self, steps, picker_l=None, picker_r=None):
        self._waypoints_picker.clear()
        curr_l = self._picker["left"].val_float
        curr_r = self._picker["right"].val_float
        targ_l = curr_l if picker_l is None else picker_l
        targ_r = curr_r if picker_r is None else picker_r
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
        return {
            "left": utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.tcp_l])[0, :3, 3],
            "right": utils.torch_to_numpy(self._urdf.link_transform_map[self._cfg.tcp_r])[0, :3, 3],
        }

    def get_gripper_state(self):
        return {"left": self._picker["left"].val_float, "right": self._picker["right"].val_float}

    def get_gripper_state_int(self):
        return {"left": self._picker["left"].val, "right": self._picker["right"].val}

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
    init_gripper_open_qpos = 1.51 + SO100_INIT_GRIPPER_CLOSED_RATIO * (-0.18 - 1.51)
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
        # Keep the "open" pose partially closed so the jaws don't flare into
        # the table while approaching. CLOSE still maps to full pinch.
        gripper_open_val=init_gripper_open_qpos,
        gripper_close_val=-0.18,
        leg_joints=[],
        init_qpos={
            # A mild inward/downward ready pose gives IK an initial TCP frame
            # that looks toward the cloth instead of straight down.
            "left_Shoulder_Rotation": -0.30, "left_Shoulder_Pitch": 0.10,
            "left_Elbow": 0.30, "left_Wrist_Pitch": 0.10, "left_Wrist_Roll": 0.0,
            "left_Gripper": init_gripper_open_qpos,
            "right_Shoulder_Rotation": 0.30, "right_Shoulder_Pitch": 0.10,
            "right_Elbow": 0.30, "right_Wrist_Pitch": 0.10, "right_Wrist_Roll": 0.0,
            "right_Gripper": init_gripper_open_qpos,
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
