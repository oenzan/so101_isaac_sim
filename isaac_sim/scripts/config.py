"""
config.py
=========
Shared constants and a few cross-version import shims used by the Isaac Sim
scripts in this folder.

Isaac Sim renamed its Python packages from `omni.isaac.*` (<= 4.2) to
`isaacsim.*` (>= 4.5). Rather than hard-code one, the helpers below try the new
namespace first and fall back to the old one, so the scripts run on either.
"""

import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / "isaac_sim" / "urdf" / "so_100_dual.urdf"
USD_DIR = REPO_ROOT / "isaac_sim" / "usd"
USD_PATH = USD_DIR / "so_100_dual.usd"           # robot-only USD (importer output)

# Prim path under which the robot articulation is created in the stage.
ROBOT_PRIM_PATH = "/World/so_100_dual"

# The 12 actuated joints, in a stable order (left arm then right arm).
ARM_JOINTS = [
    "Shoulder_Rotation", "Shoulder_Pitch", "Elbow",
    "Wrist_Pitch", "Wrist_Roll", "Gripper",
]
LEFT_JOINTS = [f"left_{j}" for j in ARM_JOINTS]
RIGHT_JOINTS = [f"right_{j}" for j in ARM_JOINTS]
ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS

# Default "ready" pose (radians).
#
# Every joint is at 0 EXCEPT Shoulder_Rotation, which is swivelled 180deg.
# Why this exact pose:
#   * At q=0 the arm is the original CAD assembly, so it is guaranteed
#     collision-free (the grippers do NOT fold into the base).
#   * Shoulder_Rotation turns about the *vertical* axis. At q=0 the arm at zero
#     points along -y (away from the table); swivelling it 180deg makes both
#     arms face +y, over the cloth/table workspace.
#   * 0 and +/-pi are the two angles where the joint-axis sign cannot change the
#     result, so this pose is robust to how the URDF->USD importer orients axes.
# Verified with fk_check.py: grippers end ~0.15 m above the table top, in front
# of the bases. Tune individual joints live in the GUI, then copy values here.
READY_POSE = {
    "left_Shoulder_Rotation": math.pi,
    "left_Shoulder_Pitch": 0.0,
    "left_Elbow": 0.0,
    "left_Wrist_Pitch": 0.0,
    "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.0,
    "right_Shoulder_Rotation": math.pi,
    "right_Shoulder_Pitch": 0.0,
    "right_Elbow": 0.0,
    "right_Wrist_Pitch": 0.0,
    "right_Wrist_Roll": 0.0,
    "right_Gripper": 0.0,
}

# --------------------------------------------------------------------------- #
# Scene geometry (metres). Robot `world` frame is at z=0; we mount it on a table.
# --------------------------------------------------------------------------- #
TABLE_TOP_Z = 0.0          # robot bases sit on the table top (== world frame z)
TABLE_HEIGHT = 0.75        # table top height above the ground plane
TABLE_SIZE = (1.2, 0.8, 0.05)   # x, y, thickness of the table top slab

# Cloth: a flat square laid on the table, centred in front of the arms.
CLOTH_CENTER = (0.0, 0.25, 0.0)   # relative to world frame (on the table top)
CLOTH_SIZE = 0.30                 # side length (m)
CLOTH_RESOLUTION = 40             # particles per side (higher = finer cloth)

# --------------------------------------------------------------------------- #
# Actuators: Feetech STS3215 bus servo (the motor used on every SO-101 joint).
# --------------------------------------------------------------------------- #
# Datasheet figures for the 12 V follower configuration. The STS3215 is a smart
# *position-controlled* serial servo, so in simulation we model it as a PD joint
# drive (stiffness Kp, damping Kd) capped by the motor's torque and speed limits.
#   * stall torque  ~30 kg.cm  -> ~2.9 N.m   (hard torque ceiling = drive maxForce)
#   * no-load speed ~0.222 s/60deg @ 12 V    -> ~4.7 rad/s (joint velocity limit)
# Kp/Kd are control gains (not on the datasheet); these values make the joints
# hold position firmly yet compliantly, like the real servo. Tune freely.
MOTOR_STALL_TORQUE_NM = 2.9        # N.m  -> joint effort limit / drive max force
MOTOR_NO_LOAD_SPEED_RAD_S = 4.7    # rad/s -> joint velocity limit
ARM_DRIVE_STIFFNESS = 17.8         # N.m/rad  (position-loop P gain, arm joints)
ARM_DRIVE_DAMPING = 0.6            # N.m.s/rad
GRIPPER_DRIVE_STIFFNESS = 8.0      # N.m/rad  (lighter; the jaw carries little load)
GRIPPER_DRIVE_DAMPING = 0.3        # N.m.s/rad
# Substring that identifies the gripper DOFs (so they get the lighter gains).
GRIPPER_JOINT_TAG = "Gripper"

# --------------------------------------------------------------------------- #
# Wrist camera: TheRobotStudio "Wrist_Cam_Mount_32x32_UVC_Module".
# https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Wrist_Cam_Mount_32x32_UVC_Module
# The mount REPLACES the wrist-roll part, so the camera rides on the link that
# the wrist-roll joint drives -> our `Fixed_Gripper` link. It looks forward along
# the gripper's grasp/approach direction (the -Y axis of that link).
# --------------------------------------------------------------------------- #
WRIST_CAM_PARENT_LINK = "Fixed_Gripper"      # per-arm link the camera is parented to
WRIST_CAM_RESOLUTION = (640, 480)            # recommended capture size in the README
WRIST_CAM_HFOV_DEG = 70.0                    # FOV not given in README; sensible default
# Local pose of the camera on the wrist-roll piece (metres / radians).
# Translation: small standoff behind/above the jaw. RPY: -90deg about X aims the
# camera's view axis (-Z) along the link's -Y (the gripper approach direction).
WRIST_CAM_TRANSLATION = (0.0, 0.0, 0.045)
WRIST_CAM_RPY = (-math.pi / 2.0, 0.0, 0.0)
WRIST_CAM_CLIPPING = (0.005, 100.0)          # near/far planes (m)


# --------------------------------------------------------------------------- #
# Cross-version import helpers
# --------------------------------------------------------------------------- #
def get_simulation_app(headless: bool = False):
    """Return a started SimulationApp (must be the very first Isaac call)."""
    try:
        from isaacsim import SimulationApp           # Isaac Sim >= 4.5
    except ImportError:
        from omni.isaac.kit import SimulationApp     # Isaac Sim <= 4.2
    return SimulationApp({"headless": headless})


def get_world_cls():
    """Return the World class across Isaac versions."""
    try:
        from isaacsim.core.api import World          # >= 4.5
    except ImportError:
        from omni.isaac.core import World            # <= 4.2
    return World
