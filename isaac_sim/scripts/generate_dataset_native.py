import os
import sys
import argparse
import copy
import json
import sys
import os
import numpy as np
from unittest.mock import MagicMock
sys.modules['pyflex'] = MagicMock()

foldnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src"))
if foldnet_src not in sys.path:
    sys.path.insert(0, foldnet_src)
batch_urdf_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/external/batch_urdf/src"))
if batch_urdf_src not in sys.path:
    sys.path.insert(0, batch_urdf_src)
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code"))

import config
from native_isaac_foldenv import FoldEnvIsaacSimNative
from so100_foldenv import make_so100_robot_cfg
from garmentds.foldenv.fold_env import FoldEnvCfg
from garmentds.foldenv.policy.state.tshirt import FoldStateTShirtPolicy, FoldStateTShirtPolicyCfg

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth", default="tshirt_sp")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--repo-id", default="ozan/so100_fold_native")
    args = parser.parse_args()

    cloth_dir = f"/home/ozan/Downloads/so100_ws/foldnet_garments/{args.cloth}_{args.variant}"
    cloth_path = os.path.join(cloth_dir, "mesh.obj")
    
    robot_cfg = make_so100_robot_cfg()
    env_cfg = FoldEnvCfg(
        cloth_obj_path=cloth_path,
        cloth_scale=0.5,
        n_substep=10, # 10 Isaac Sim steps per policy action (was 30 → too slow visually)
        render=True, # We want to render in Isaac Sim natively
        render_mode=["mesh"],
        render_process_num=1,
        robot_cfg=robot_cfg,
    )
    # Diagnostic sphere radius for compute_grasp_vertices (not the PhysX attachment radius).
    # TCP now descends to cloth level (z_grasp≈0.02, cloth z≈0.007 → gap≈1.3 cm).
    # Keep z-axis at 0.05 for tolerance; x/y cover the sleeve width.
    env_cfg.grasp_threshold["left"] = np.array([0.05, 0.03, 0.05], dtype=np.float32)
    env_cfg.grasp_threshold["right"] = np.array([0.05, 0.03, 0.05], dtype=np.float32)
    
    print("Initializing Native Isaac Sim Environment...")
    env = FoldEnvIsaacSimNative(env_cfg)
    env.reset()
    
    # Initialize LeRobot dataset
    features = {
        "observation.images.wrist_left": {
            "dtype": "video",
            "shape": (config.WRIST_CAM_RESOLUTION[1], config.WRIST_CAM_RESOLUTION[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_right": {
            "dtype": "video",
            "shape": (config.WRIST_CAM_RESOLUTION[1], config.WRIST_CAM_RESOLUTION[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (config.OVERHEAD_CAM_RESOLUTION[1], config.OVERHEAD_CAM_RESOLUTION[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["qpos"],
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["qpos"],
        },
    }
    
    dataset_root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    import shutil
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
        
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        features=features,
        root=dataset_root,
    )

    print("Adding wrist cameras...")
    import cameras
    wrist_cams = cameras.add_wrist_cameras(
        env._stage, config, config.ROBOT_PRIM_PATH,
        make_sensors=True,
    )

    print("Adding overhead camera...")
    overhead_cam = cameras.add_overhead_camera(
        env._stage, config, config.ROBOT_PRIM_PATH,
        make_sensors=True,
    )
    for cam in wrist_cams:
        if cam.sensor is not None:
            cam.sensor.initialize()
            
    if overhead_cam.sensor is not None:
        overhead_cam.sensor.initialize()
            
    print("Initializing FoldStateTShirtPolicy...")
    tcp_init = env.get_tcp_xyz()
    init_xyz_l = np.array(tcp_init["left"], dtype=np.float32)
    init_xyz_r = np.array(tcp_init["right"], dtype=np.float32)
    init_xyz_l[2] = init_xyz_r[2] = max(init_xyz_l[2], init_xyz_r[2], 0.18)
    policy_cfg = FoldStateTShirtPolicyCfg(
        cloth_scale=env._cfg.cloth_scale,
        init_xyz_l=init_xyz_l,
        init_xyz_r=init_xyz_r,
    )
    # skip_rotate=True: skip rotation/alignment stages (stages 1-4) and jump directly
    # to fold_step=0 (fold sleeves) then fold_step=1 (fold body).
    # Rotate stages drag cloth across the table at rotate_z_move=0.02 m which causes
    # the SO-100 jaw to hit the table surface. Skip for now; enable once attachment
    # and collision are verified stable.
    policy_cfg.skip_rotate = True

    # Grasp/put z: keep the TCP safely above the table. In practice the USD
    # gripper geometry and IK/drive error can sit lower than the TCP link, so
    # too-small values inject table-contact jitter directly into the cloth.
    min_grasp_z = float(os.environ.get("ISAAC_POLICY_MIN_GRASP_Z", "0.035"))
    # Transit z: keep 0.12 m clearance so the arm doesn't drag over the cloth edge.
    for attr_name in [
        "rotate_z_grasp", "rotate_z_put",
        "align_z_grasp",  "align_z_put",
        "fold1_z_grasp",  "fold1_z_put",
        "fold2_z_grasp",  "fold2_z_put",
    ]:
        setattr(policy_cfg, attr_name, max(getattr(policy_cfg, attr_name), min_grasp_z))
    for attr_name in [
        "rotate_z_move",
        "align_z_move",
        "fold1_z_move",
        "fold2_z_move",
    ]:
        setattr(policy_cfg, attr_name, max(getattr(policy_cfg, attr_name), 0.12))
    print(
        "SO-100 policy init targets: "
        f"L=({init_xyz_l[0]:.3f}, {init_xyz_l[1]:.3f}, {init_xyz_l[2]:.3f}), "
        f"R=({init_xyz_r[0]:.3f}, {init_xyz_r[1]:.3f}, {init_xyz_r[2]:.3f})"
    )
    print(
        f"Policy z heights (FoldNet frame): "
        f"min_grasp_z={min_grasp_z:.3f}  "
        f"fold1_grasp={policy_cfg.fold1_z_grasp:.3f}  fold1_move={policy_cfg.fold1_z_move:.3f}  "
        f"fold2_grasp={policy_cfg.fold2_z_grasp:.3f}  fold2_move={policy_cfg.fold2_z_move:.3f}"
    )
    policy = FoldStateTShirtPolicy(policy_cfg, env)
    
    print("Creating IK target markers...")
    from omni.isaac.core.objects import VisualSphere
    marker_l = VisualSphere(prim_path="/World/marker_l", radius=0.015, color=np.array([1.0, 0.0, 0.0]))
    marker_r = VisualSphere(prim_path="/World/marker_r", radius=0.015, color=np.array([0.0, 0.0, 1.0]))
    
    print("Starting simulation loop...")
    if os.environ.get("ISAAC_HEADLESS", "1") == "0":
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(eye=[0.0, -0.5, 1.2], target=[0.0, 0.15, 0.0], camera_prim_path="/OmniverseKit_Persp")
        except Exception as e:
            print(f"Could not set camera view: {e}")
            
    step_idx = 0
    max_steps = 200
    while step_idx < max_steps:
        tcp = env.get_tcp_xyz()
        gripper = env.get_gripper_state()
        state_dict = {
            "tcp_xyz": {"left": tcp["left"].tolist() if isinstance(tcp["left"], np.ndarray) else tcp["left"], 
                        "right": tcp["right"].tolist() if isinstance(tcp["right"], np.ndarray) else tcp["right"]},
            "gripper_state": {"left": gripper["left"], "right": gripper["right"]},
        }
        
        action = policy.get_action()
        if action is None:
            print("Policy finished.")
            break
            
        action_env = action.asdict_to_env()
        xyz_l = action_env.get("xyz_l")
        xyz_r = action_env.get("xyz_r")
        if xyz_l is not None:
            xyz_l[2] = max(float(xyz_l[2]), min_grasp_z)
        if xyz_r is not None:
            xyz_r[2] = max(float(xyz_r[2]), min_grasp_z)
        
        # Debug: print targets and IK fail count
        ik_fails_before = env._robot.ik_fail_count
        print(f"\n--- Step {step_idx} ---")
        # FoldNet frame → Isaac Sim world: negate X,Y (180° Z rotation), add TABLE_HEIGHT to Z
        if xyz_l is not None:
            marker_pos_l = [-xyz_l[0], -xyz_l[1], xyz_l[2] + config.TABLE_HEIGHT]
            marker_l.set_world_pose(position=marker_pos_l)
            print(f"  Target L (FoldNet): ({xyz_l[0]:.3f}, {xyz_l[1]:.3f}, {xyz_l[2]:.3f}) → Marker (Isaac): ({marker_pos_l[0]:.3f}, {marker_pos_l[1]:.3f}, {marker_pos_l[2]:.3f})")
        if xyz_r is not None:
            marker_pos_r = [-xyz_r[0], -xyz_r[1], xyz_r[2] + config.TABLE_HEIGHT]
            marker_r.set_world_pose(position=marker_pos_r)
            print(f"  Target R (FoldNet): ({xyz_r[0]:.3f}, {xyz_r[1]:.3f}, {xyz_r[2]:.3f}) → Marker (Isaac): ({marker_pos_r[0]:.3f}, {marker_pos_r[1]:.3f}, {marker_pos_r[2]:.3f})")
            
        env.step(**action_env)
        
        ik_fails_after = env._robot.ik_fail_count
        if ik_fails_after > ik_fails_before:
            print(f"  ⚠️  IK FAILED on this step! ({ik_fails_after - ik_fails_before} new failures)")
        
        # Print where TCP actually ended up  
        tcp = env.get_tcp_xyz()
        print(f"  TCP L (BatchURDF): ({tcp['left'][0]:.3f}, {tcp['left'][1]:.3f}, {tcp['left'][2]:.3f})")
        print(f"  TCP R (BatchURDF): ({tcp['right'][0]:.3f}, {tcp['right'][1]:.3f}, {tcp['right'][2]:.3f})")
        
        # Get rendered images
        rgba_l = wrist_cams[0].sensor.get_rgba()
        rgba_r = wrist_cams[1].sensor.get_rgba()
        rgba_o = overhead_cam.sensor.get_rgba()

        img_l = rgba_l[:, :, :3] if rgba_l is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        img_r = rgba_r[:, :, :3] if rgba_r is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        img_o = rgba_o[:, :, :3] if rgba_o is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # Get robot joint positions
        dof_names = list(env._isaac_robot.dof_names)
        qpos_dict = env._robot.get_qpos()
        current_qpos = np.array([qpos_dict.get(n, 0.0) for n in dof_names], dtype=np.float32)

        action_dict = policy.delta_action(action).asdict_to_save()
        # In a real setup, we want action to be next_qpos, but for now we just use current qpos for test
        target_action = current_qpos

        frame_data = {
            "observation.state": current_qpos,
            "observation.images.wrist_left": img_l,
            "observation.images.wrist_right": img_r,
            "observation.images.overhead": img_o,
            "action": target_action,
            "task": "fold_cloth"
        }
        dataset.add_frame(frame_data)
        
        step_idx += 1
        
        if os.environ.get("ISAAC_HEADLESS", "1") == "0":
            import time
            time.sleep(1.0 / 30.0) # Slow down to ~30 FPS for visual debugging
        
    print("Saving episode...")
    import time
    time.sleep(5)
    dataset.save_episode()
    # Finalize the dataset
    print("Finalizing dataset...")
    dataset.finalize()
    print(f"Dataset fully finalized at {dataset.root}")

if __name__ == "__main__":
    main()
    from native_isaac_foldenv import sim_app
    
    if os.environ.get("ISAAC_HEADLESS", "1") == "0":
        print("Simulation finished. You can now inspect the scene.")
        print("Close the Isaac Sim window or press Ctrl+C in terminal to exit.")
        try:
            while sim_app.is_running():
                sim_app.update()
        except KeyboardInterrupt:
            pass
            
    sim_app.close()
