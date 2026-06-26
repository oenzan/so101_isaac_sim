#!/usr/bin/env python3
"""
render_dataset.py
=================
Plays back a generated dataset in Isaac Sim and records RGB video and Depth arrays.
"""

import argparse
import os
import glob
import json
import numpy as np
import cv2
import config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True, help="Path to a specific garment variant dataset dir, e.g. /tmp/so100_fold_dataset/tshirt_sp_0")
parser.add_argument("--headless", action="store_true", default=True, help="run without viewport")
parser.add_argument("--fps", type=int, default=30)
args, _ = parser.parse_known_args()

sim_app = config.get_simulation_app(headless=args.headless)

import omni.usd
from pxr import UsdGeom, UsdLux, Gf

World = config.get_world_cls()

try:
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.api.objects import FixedCuboid
except ImportError:
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import FixedCuboid

import garment_loader
import cameras

def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    physx = world.get_physics_context()
    physx.enable_gpu_dynamics(True)

    world.scene.add_default_ground_plane()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(900.0)

    # Table
    tx, ty, tz = config.TABLE_SIZE
    table_center_z = config.TABLE_HEIGHT - tz / 2.0
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.0, 0.2, table_center_z]),
        scale=np.array([tx, ty, tz]),
        color=np.array([0.45, 0.30, 0.20]),
    )

    # Robot
    if config.USD_PATH.exists():
        add_reference_to_stage(str(config.USD_PATH), config.ROBOT_PRIM_PATH)
    else:
        print("[render] Error: Robot USD not found. Run import_urdf_to_usd.py first.")
        return

    robot_xform = UsdGeom.Xformable(stage.GetPrimAtPath(config.ROBOT_PRIM_PATH))
    robot_xform.ClearXformOpOrder()
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, config.TABLE_HEIGHT))

    robot = Articulation(prim_path=config.ROBOT_PRIM_PATH, name="so_100_dual")
    world.scene.add(robot)

    wrist_cams = cameras.add_wrist_cameras(
        stage, config, config.ROBOT_PRIM_PATH,
        make_sensors=True,
    )

    world.reset()
    if not robot.handles_initialized:
        robot.initialize()

    for cam in wrist_cams:
        if cam.sensor is not None:
            cam.sensor.initialize()
            print(f"[render] Sensor initialized for {cam.name}")

    # Process Trajectories
    traj_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, "traj_*")))
    print(f"[render] Found {len(traj_dirs)} trajectories in {args.dataset_dir}")

    for traj_dir in traj_dirs:
        print(f"\n[render] Processing {traj_dir} ...")
        meta_path = os.path.join(traj_dir, "meta.json")
        if not os.path.exists(meta_path):
            print(f"[render] meta.json missing, skipping.")
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        state_files = sorted(glob.glob(os.path.join(traj_dir, "state", "*.json")), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if len(state_files) == 0:
            continue

        out_rgb_left = os.path.join(traj_dir, "rgb_left.mp4")
        out_rgb_right = os.path.join(traj_dir, "rgb_right.mp4")
        out_depth = os.path.join(traj_dir, "depth.npz")

        # Get resolution from config for VideoWriter
        w, h = config.WRIST_CAM_RESOLUTION
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_l = cv2.VideoWriter(out_rgb_left, fourcc, args.fps, (w, h))
        writer_r = cv2.VideoWriter(out_rgb_right, fourcc, args.fps, (w, h))
        
        depths_l = []
        depths_r = []

        # We will not load the cloth for pure kinematic replay of the robot to capture camera views, 
        # because simulating cloth perfectly in sync requires more logic. 
        # But wait! If there is no cloth, the cameras will just see the table and empty space.
        # We MUST load the cloth!
        # Clear previous cloth if any
        cloth_prim_path = "/World/Cloth/garmentMesh"
        surface_prim_path = "/World/Cloth/surfaceDeformable"
        if stage.GetPrimAtPath(cloth_prim_path).IsValid():
            stage.RemovePrim(cloth_prim_path)
        if stage.GetPrimAtPath(surface_prim_path).IsValid():
            stage.RemovePrim("/World/Cloth")

        # Load cloth
        cat = meta["garment"].rsplit("_", 1)[0]
        profile = config.get_garment_profile(cat)
        cx, cy, _ = config.CLOTH_CENTER
        cloth_center = (cx, cy, config.TABLE_HEIGHT + 0.005)
        
        garment_name = os.path.basename(args.dataset_dir) # e.g. tshirt_sp_0
        garment_loader.load_garment(
            stage=stage,
            scene_path="/physicsScene",
            root_path="/World/Cloth",
            garment_name=garment_name,
            garment_dir=str(config.GARMENT_DIR),
            scale=config.GARMENT_SCALE,
            center=cloth_center,
            particle_contact_offset=config.GARMENT_PARTICLE_CONTACT_OFFSET,
            profile=profile,
            mass=config.GARMENT_MASS,
            friction=config.GARMENT_FRICTION,
            backend="particle"
        )
        
        # Settle cloth briefly
        for _ in range(10):
            world.step(render=False)

        # Replay
        for state_file in state_files:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Extract qpos
            if "qpos" in state:
                qpos = state["qpos"]
                dof_names = list(robot.dof_names)
                targets = np.array([qpos.get(n, 0.0) for n in dof_names])
                robot.set_joint_positions(targets)
            else:
                # older format might just have tcp_xyz, ignore if so
                continue

            # Step physics and render
            world.step(render=True)

            # Capture
            rgba_l = wrist_cams[0].sensor.get_rgba()
            rgba_r = wrist_cams[1].sensor.get_rgba()
            
            # Convert to BGR for cv2
            if rgba_l is not None and rgba_r is not None:
                bgr_l = cv2.cvtColor(rgba_l[:, :, :3], cv2.COLOR_RGB2BGR)
                bgr_r = cv2.cvtColor(rgba_r[:, :, :3], cv2.COLOR_RGB2BGR)
                writer_l.write(bgr_l)
                writer_r.write(bgr_r)
            
            depth_l = wrist_cams[0].sensor.get_depth()
            depth_r = wrist_cams[1].sensor.get_depth()
            if depth_l is not None:
                depths_l.append(depth_l)
                depths_r.append(depth_r)

        writer_l.release()
        writer_r.release()
        
        # Save depths as compressed npz
        if len(depths_l) > 0:
            np.savez_compressed(out_depth, left=np.array(depths_l), right=np.array(depths_r))
        
        print(f"[render] Saved {out_rgb_left}, {out_rgb_right}, and {out_depth}")

if __name__ == "__main__":
    main()
    sim_app.close()
