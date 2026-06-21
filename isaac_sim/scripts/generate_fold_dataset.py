"""
Generate cloth folding dataset with SO-100 bimanual robot.
One category per invocation (PyFlex limitation).

Usage:
    conda activate foldnet
    export CUDA_VISIBLE_DEVICES=0
    python isaac_sim/scripts/generate_fold_dataset.py \
        --category tshirt_sp --variant 0 --num_trajs 5 --out /tmp/so100_folds
"""

import os, sys, copy, argparse
import numpy as np

_PYFLEX_LIBS = os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src/pyflex/libs")
sys.path.insert(0, os.path.abspath(_PYFLEX_LIBS))
os.environ["PYFLEX_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src/pyflex/PyFlex"))
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code"))

import pyflex
import garmentds.common.utils as utils
from garmentds.foldenv.fold_env import FoldEnvCfg, load_mesh_raw_sim
from so100_foldenv import FoldEnvSO100, make_so100_robot_cfg

GARMENT_ARMS = {
    "tshirt_sp": ("l_shoulder", "r_shoulder"),
    "tshirt": ("l_shoulder", "r_shoulder"),
    "trousers": ("l_corner", "r_corner"),
    "vest": ("l_shoulder", "r_shoulder"),
    "vest_close": ("l_shoulder", "r_shoulder"),
    "shirt": ("l_shoulder", "r_shoulder"),
    "shirt_close": ("l_shoulder", "r_shoulder"),
    "hooded": ("l_shoulder", "r_shoulder"),
    "hooded_close": ("l_shoulder", "r_shoulder"),
}

GARMENT_FOLD_TARGET = {
    "tshirt_sp": "spine_bottom_f",
    "tshirt": "spine_bottom_f",
    "trousers": "crotch",
    "vest": "spine_bottom_b",
    "vest_close": "spine_bottom_f",
    "shirt": "spine_bottom_b",
    "shirt_close": "spine_bottom_f",
    "hooded": "spine_bottom_b",
    "hooded_close": "spine_bottom_b",
}


def get_kp_position(env, kp_name):
    vert_sim = env._get_cloth_xyzm()[:, :3]
    kp_idx = env._keypoint_idx[kp_name]
    vert_raw = vert_sim[env._vert_ren_to_sim]
    return vert_raw[kp_idx].copy()


def generate_demonstration(env, garment, output_dir, traj_id=0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    traj_dir = os.path.join(output_dir, f"traj_{traj_id:04d}")
    os.makedirs(os.path.join(traj_dir, "state"), exist_ok=True)
    os.makedirs(os.path.join(traj_dir, "action"), exist_ok=True)

    env.reset()
    env.perfect_init_cloth(
        rot_z_deg=float(np.random.uniform(-15, 15)),
        flip_y=bool(np.random.rand() < 0.5),
    )

    kp_l, kp_r = GARMENT_ARMS[garment]
    kp_fold = GARMENT_FOLD_TARGET[garment]

    cloth_center = env._get_cloth_xyzm()[:, :3].mean(axis=0)
    grasp_l = get_kp_position(env, kp_l)
    grasp_r = get_kp_position(env, kp_r)

    if grasp_l[0] > grasp_r[0]:
        grasp_l, grasp_r = grasp_r, grasp_l

    if grasp_l[0] > grasp_r[0]:
        grasp_l, grasp_r = grasp_r, grasp_l

    if kp_fold in env._keypoint_idx:
        fold_anchor = get_kp_position(env, kp_fold)
        fold_center = (fold_anchor + cloth_center) / 2.0
    else:
        fold_center = cloth_center

    lift_offset = np.array([0.0, 0.0, 0.06])
    pre_l = grasp_l + lift_offset
    pre_r = grasp_r + lift_offset

    print(f"    center={np.round(cloth_center,3)}  "
          f"grasp L={np.round(grasp_l,3)}  R={np.round(grasp_r,3)}")

    initial_mesh = env.get_raw_mesh_curr()
    if initial_mesh is not None:
        initial_mesh.export(os.path.join(traj_dir, "initial_mesh.obj"))

    fold_spread = 0.03
    actions = [
        {"xyz_l": pre_l, "xyz_r": pre_r,
         "picker_l": 0.0, "picker_r": 0.0, "phase": "approach"},
        {"xyz_l": grasp_l, "xyz_r": grasp_r,
         "picker_l": 0.0, "picker_r": 0.0, "phase": "descend"},
        {"xyz_l": grasp_l, "xyz_r": grasp_r,
         "picker_l": 1.0, "picker_r": 1.0, "phase": "grasp"},
        {"xyz_l": pre_l, "xyz_r": pre_r,
         "picker_l": 1.0, "picker_r": 1.0, "phase": "lift"},
        {"xyz_l": fold_center + np.array([-fold_spread, -0.04, 0.02]),
         "xyz_r": fold_center + np.array([+fold_spread, -0.04, 0.02]),
         "picker_l": 1.0, "picker_r": 1.0, "phase": "fold"},
        {"xyz_l": fold_center + np.array([-fold_spread, -0.04, 0.02]),
         "xyz_r": fold_center + np.array([+fold_spread, -0.04, 0.02]),
         "picker_l": 0.0, "picker_r": 0.0, "phase": "release"},
    ]

    step_idx = 0
    for act in actions:
        env.step(xyz_l=act["xyz_l"], xyz_r=act["xyz_r"],
                 picker_l=act["picker_l"], picker_r=act["picker_r"])
        for _ in range(env._cfg.n_substep):
            tcp = env._robot.get_tcp_xyz()
            gripper = env._robot.get_gripper_state()
            state = {
                "step": step_idx, "phase": act["phase"],
                "tcp_l": tcp["left"].tolist(),
                "tcp_r": tcp["right"].tolist(),
                "gripper_l": float(gripper["left"]),
                "gripper_r": float(gripper["right"]),
                "qpos": {k: float(v) for k, v in env._robot.get_qpos().items()},
            }
            utils.dump_json(os.path.join(traj_dir, "state", f"{step_idx}.json"), state)
            action = {
                "step": step_idx, "phase": act["phase"],
                "tcp_l_target": [float(x) for x in act["xyz_l"]],
                "tcp_r_target": [float(x) for x in act["xyz_r"]],
                "picker_l": act["picker_l"], "picker_r": act["picker_r"],
            }
            utils.dump_json(os.path.join(traj_dir, "action", f"{step_idx}.json"), action)
            step_idx += 1

    for _ in range(20):
        pyflex.step()

    final_mesh = env.get_raw_mesh_curr()
    if final_mesh is not None:
        final_mesh.export(os.path.join(traj_dir, "final_mesh.obj"))

    meta = {
        "traj_id": traj_id, "garment": garment,
        "num_steps": step_idx,
        "ik_fail_count": env._robot.ik_fail_count,
        "cloth_path": env._cfg.cloth_obj_path,
        "grasp_keypoints": [kp_l, kp_r],
        "fold_keypoint": kp_fold if kp_fold in env._keypoint_idx else None,
    }
    utils.dump_json(os.path.join(traj_dir, "meta.json"), meta)
    return traj_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="tshirt_sp")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--num_trajs", type=int, default=5)
    parser.add_argument("--out", default="/tmp/so100_fold_dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cloth_path = f"/home/ozan/Downloads/so100_ws/foldnet_garments/{args.category}_{args.variant}/mesh.obj"
    if not os.path.exists(cloth_path):
        print(f"ERROR: Cloth not found: {cloth_path}")
        return 1

    print(f"Category: {args.category}_{args.variant}")
    print(f"Cloth: {cloth_path}")

    _, _, keypoint_idx = load_mesh_raw_sim(cloth_path, 0.5)
    kp_l, kp_r = GARMENT_ARMS[args.category]
    for kp in [kp_l, kp_r]:
        if kp not in keypoint_idx:
            print(f"ERROR: Keypoint '{kp}' not found in {args.category}")
            print(f"  Available: {list(keypoint_idx.keys())}")
            return 1

    robot_cfg = make_so100_robot_cfg()
    env_cfg = FoldEnvCfg(
        cloth_obj_path=cloth_path, cloth_scale=0.5,
        render=False, render_mode=[], render_process_num=1,
        robot_cfg=robot_cfg,
    )

    out_dir = os.path.join(args.out, f"{args.category}_{args.variant}")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    env = FoldEnvSO100(env_cfg)

    success = 0
    for traj_id in range(args.num_trajs):
        try:
            traj_dir = generate_demonstration(
                env, args.category, out_dir, traj_id,
                seed=rng.randint(0, 2**31))
            success += 1
        except Exception as e:
            print(f"  FAILED traj {traj_id}: {e}")
            import traceback
            traceback.print_exc()

    env.close()
    print(f"\nDone: {success}/{args.num_trajs} OK in {out_dir}")
    return 0 if success == args.num_trajs else 1


if __name__ == "__main__":
    sys.exit(main())
