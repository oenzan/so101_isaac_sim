#!/usr/bin/env python3
"""
setup_scene.py
==============
Build the full cloth-folding workspace and run it:

    ground plane  +  dome light  +  table  +  dual SO-101 robot  +  cloth

This is the script you run day-to-day. It loads the robot from the USD produced
by `import_urdf_to_usd.py` (or imports the URDF on the fly if the USD is
missing), enables GPU physics (needed for particle cloth), drives both arms to
a "ready" pose, and then steps the simulation in an interactive viewport.

Run with Isaac Sim's python:
    ./python.sh /path/to/so101_isaac_sim/isaac_sim/scripts/setup_scene.py
Optional flags:
    --headless     run without a window (e.g. on a server)
    --steps N      run N physics steps then exit (default: run until window closed)
"""

import argparse
import config

# --------------------------------------------------------------------------- #
# Parse args BEFORE booting the app (boot order matters).
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="run without a viewport")
parser.add_argument("--steps", type=int, default=0,
                    help="number of physics steps before exit (0 = run forever)")
parser.add_argument("--capture-cameras", action="store_true",
                    help="also create Isaac Camera sensors on the wrists so frames "
                         "can be read in code (adds render overhead)")
args, _ = parser.parse_known_args()

# 1) Boot Isaac Sim first.
sim_app = config.get_simulation_app(headless=args.headless)

# 2) Now import the rest of the API (cross-version shims kept local + simple).
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdLux, Gf

World = config.get_world_cls()

try:                                                   # Isaac Sim >= 4.5
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:                                    # Isaac Sim <= 4.2
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.core.utils.types import ArticulationAction

import cloth_utils
import cameras
import motors

try:                                                   # >= 4.5
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:                                    # <= 4.2
    from omni.isaac.core.utils.extensions import enable_extension


PHYSICS_SCENE_PATH = "/physicsScene"


def enable_physics_ui():
    """
    The minimal standalone Kit app doesn't load the PhysX UI extensions, so the
    Physics Inspector / Physics menu actions are missing (the log shows
    'Could not find action ... show_physics_inspector'). Enabling these brings
    the inspector and the physics authoring toolbar into the viewport.
    """
    for ext in ("omni.physx.ui", "omni.physx.supportui", "omni.physx.demos"):
        try:
            enable_extension(ext)
        except Exception as exc:
            print(f"[scene] could not enable {ext}: {exc}")


def main():
    # Bring in the PhysX UI / inspector (GUI only).
    if not args.headless:
        enable_physics_ui()

    # --- World + GPU physics -------------------------------------------------
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Particle cloth needs GPU dynamics + GPU broadphase.
    physx = world.get_physics_context()
    physx.enable_gpu_dynamics(True)
    try:
        physx.set_broadphase_type("GPU")
    except Exception:
        pass

    # --- Ground + light ------------------------------------------------------
    world.scene.add_default_ground_plane()
    # A dome light so the scene is evenly lit.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(900.0)

    # --- Table ---------------------------------------------------------------
    # Top surface sits at z = TABLE_HEIGHT; slab centre is half a thickness below.
    tx, ty, tz = config.TABLE_SIZE
    table_center_z = config.TABLE_HEIGHT - tz / 2.0
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.0, 0.2, table_center_z]),
        scale=np.array([tx, ty, tz]),
        color=np.array([0.45, 0.30, 0.20]),
    )

    # --- Robot ---------------------------------------------------------------
    # Prefer the pre-imported USD; fall back to importing the URDF in-process.
    if config.USD_PATH.exists():
        add_reference_to_stage(str(config.USD_PATH), config.ROBOT_PRIM_PATH)
    else:
        print("[scene] USD not found, importing URDF on the fly "
              "(run import_urdf_to_usd.py to cache it).")
        _import_urdf_inplace()

    # Lift the whole robot so its `world` frame rests on the table top.
    robot_xform = UsdGeom.Xformable(stage.GetPrimAtPath(config.ROBOT_PRIM_PATH))
    robot_xform.ClearXformOpOrder()
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, config.TABLE_HEIGHT))

    robot = Articulation(prim_path=config.ROBOT_PRIM_PATH, name="so_100_dual")
    world.scene.add(robot)

    # --- Wrist cameras -------------------------------------------------------
    # One 32x32 UVC camera per arm, mounted on the wrist-roll link (see
    # cameras.py). Spawn the prims now, before reset, so any sensors can be
    # initialised afterwards.
    wrist_cams = cameras.add_wrist_cameras(
        stage, config, config.ROBOT_PRIM_PATH,
        make_sensors=args.capture_cameras,
    )

    # --- Cloth ---------------------------------------------------------------
    # Place the cloth on the table top, just above the surface so it settles.
    cx, cy, _ = config.CLOTH_CENTER
    cloth_center = (cx, cy, config.TABLE_HEIGHT + 0.02)
    cloth_utils.add_cloth(
        stage=stage,
        scene_path=PHYSICS_SCENE_PATH,
        root_path="/World/Cloth",
        center=cloth_center,
        side=config.CLOTH_SIZE,
        resolution=config.CLOTH_RESOLUTION,
    )

    # --- Initialise + drive to ready pose -----------------------------------
    # world.reset() builds the physics sim view and initialises every object
    # registered in world.scene (including the robot), so dof handles are valid
    # afterwards. We call initialize() defensively in case this articulation was
    # not auto-initialised on a given Isaac build.
    world.reset()
    if not robot.handles_initialized:
        robot.initialize()

    # Make the joints behave like the real Feetech STS3215 servos (gains, torque
    # ceiling, speed ceiling). Must run after init so the physics handles exist.
    motors.build_default_model(config).apply(robot)

    # Initialise the wrist-camera sensors now that physics/render are live.
    if args.capture_cameras:
        for cam in wrist_cams:
            if cam.sensor is not None:
                try:
                    cam.sensor.initialize()
                except Exception as exc:
                    print(f"[camera] sensor init failed for {cam.name}: {exc}")

    dof_names = list(robot.dof_names)
    targets = np.array([config.READY_POSE.get(n, 0.0) for n in dof_names])

    # Teleport to the ready pose, then set the position-drive target ONCE.
    # A PhysX position drive is persistent: once the target is set, the drive
    # holds the joint there every step on its own. We must NOT re-send the
    # target every frame -- doing so fights the Physics Inspector (the joint
    # snaps back / vibrates) and crashes if the sim view is torn down by a GUI
    # Stop. Setting it once lets you scrub joints in the inspector freely.
    robot.set_joint_positions(targets)
    # Make the ready pose the default so a GUI Stop -> Play returns to it.
    try:
        robot.set_joints_default_state(positions=targets)
    except Exception:
        pass
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=targets)
    )
    print(f"[scene] {len(dof_names)} DOF: {dof_names}")
    print("[scene] ready. Drives hold the pose; use the Physics Inspector to "
          "scrub joints. Stepping simulation...")

    # --- Run loop ------------------------------------------------------------
    # The loop only advances the sim + renders. It does NOT re-command joints,
    # so the Physics Inspector stays in control of the targets.
    step = 0
    while sim_app.is_running():
        world.step(render=not args.headless)
        step += 1
        if args.steps and step >= args.steps:
            break


def _import_urdf_inplace():
    """Fallback: run the URDF importer directly into the live stage."""
    import omni.kit.commands
    try:
        from isaacsim.core.utils.extensions import enable_extension
    except ImportError:
        from omni.isaac.core.utils.extensions import enable_extension
    for ext in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
        try:
            enable_extension(ext)
            break
        except Exception:
            continue
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = True
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.distance_scale = 1.0
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(config.URDF_PATH),
        import_config=import_config,
        dest_path="",                      # import into current stage
    )


if __name__ == "__main__":
    main()
    sim_app.close()
