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
    "left_Shoulder_Rotation": 0.0,
    "left_Shoulder_Pitch": 0.0,
    "left_Elbow": 0.0,
    "left_Wrist_Pitch": 0.0,
    "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.0,
    "right_Shoulder_Rotation": 0.0,
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
CLOTH_CENTER = (0.0, 0.0, 0.0)   # raw mesh center (FoldNet native frame)
CLOTH_SIZE = 0.30                 # side length (m)
CLOTH_RESOLUTION = 40             # particles per side (higher = finer cloth)

# --------------------------------------------------------------------------- #
# Garment loading (FoldNet format)
# --------------------------------------------------------------------------- #
GARMENT_DIR = REPO_ROOT / "foldnet_garments"
DEFAULT_GARMENT = "tshirt_sp_0"
GARMENT_SCALE = 0.35               # FoldNet default cloth scale scaled down for SO-100 reach
GARMENT_CENTER = (0.0, 0.0, 0.0)  # raw mesh center (FoldNet native frame)
GARMENT_MASS = 0.05                # kg

# Garment physics profiles keyed by category prefix.
# Stiffness values are reduced vs the square cloth (1e4 / 80 / 80) because
# garment meshes have 3-5× more triangles per area → more springs → stiffer
# net behaviour. Profiles below are estimates; tune by running:
#   ./python.sh setup_scene.py --garment tshirt_sp_0 --garment-profile medium
#
# Damping is raised (0.2 → 0.35+) to suppress crumpling when the dense mesh
# hits the table. Particle contact offset should be ~0.012 for garments
# (vs 0.006 for the square cloth) so sleeve edges grip the table better.
GARMENT_PROFILES = {
    # Particle cloth spring stiffness in N/m.
    # Original FoldNet PyFlex used ~5000 N/m but that cloth was small (0.5× scale).
    # At 0.35× scale the mesh is coarser → fewer springs → each spring must be
    # stiffer to give the same net fabric feel. 40 000 N/m gives cotton-like stiffness.
    #                 stretch  bend   shear  damping  self_collision
    "light":   (4e4,   200,   200,   0.28,  True),   # tshirt_sp, vest, vest_close
    "medium":  (3e4,   180,   180,   0.25,  True),   # tshirt, shirt, shirt_close
    "heavy":   (2.5e4, 150,   150,   0.30,  True),   # hooded, hooded_close
    "trousers":(5e4,   150,   200,   0.20,  True),   # trousers (low bend for leg fold)
}
GARMENT_PARTICLE_CONTACT_OFFSET = 0.006  # same as square cloth; friction material keeps it planted
GARMENT_FRICTION = 0.8                     # particle-cloth friction against table/self

# Map category name → profile key (matched by prefix)
GARMENT_PROFILE_MAP = {
    "tshirt_sp": "light",
    "vest": "light",
    "vest_close": "light",
    "tshirt": "medium",
    "shirt": "medium",
    "shirt_close": "medium",
    "hooded": "heavy",
    "hooded_close": "heavy",
    "trousers": "trousers",
}

def get_garment_profile(category: str) -> dict:
    """Return physics parameters dict for a garment category."""
    key = GARMENT_PROFILE_MAP.get(category, "medium")
    s, b, sh, d, sc = GARMENT_PROFILES[key]
    return {
        "stretch": s, "bend": b, "shear": sh,
        "damping": d, "self_collision": sc,
    }

# --------------------------------------------------------------------------- #
# Actuators: Feetech STS3215 bus servo (the motor used on every SO-101 joint).
# --------------------------------------------------------------------------- #
# The STS3215 is a smart *position-controlled* serial servo with a 1:345 gearbox,
# so it tracks its target STIFFLY (it does not visibly sag under the arm's weight)
# but its output is bounded by the motor's stall torque and no-load speed. We model
# it as a PD joint drive whose realism comes from two datasheet CLAMPS:
#   * stall torque  ~30 kg.cm  -> ~2.9 N.m   (hard torque ceiling = drive maxForce)
#   * no-load speed ~0.222 s/60deg @ 12 V    -> ~4.7 rad/s (joint velocity limit)
# The stiffness/damping (Kp/Kd) are the *controller* gains, not datasheet numbers.
# They are kept HIGH so position tracking is tight like the real digital servo;
# the 2.9 N.m torque clamp is what limits the force to a realistic value. (An
# earlier version used a low Kp ~ 18, which let gravity droop every joint ~2 deg
# and made the whole arm sag to the table.)
MOTOR_STALL_TORQUE_NM = 2.9        # N.m  -> joint effort limit / drive max force
MOTOR_NO_LOAD_SPEED_RAD_S = 4.7    # rad/s -> joint velocity limit
ARM_DRIVE_STIFFNESS = 2000.0       # N.m/rad  (tight position tracking, arm joints)
ARM_DRIVE_DAMPING = 100.0          # N.m.s/rad (well damped, no oscillation)
GRIPPER_DRIVE_STIFFNESS = 1400.0   # N.m/rad  (close decisively enough to pinch cloth)
GRIPPER_DRIVE_DAMPING = 80.0       # N.m.s/rad
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
# In the default SO-100 FK, local +Y on `Fixed_Gripper` points upward in world,
# so shifting along +Y keeps the wrist camera on the air side of the tool
# without changing robot motion or trajectory generation.
WRIST_CAM_TRANSLATION = (0.0, 0.03, 0.0)
WRIST_CAM_RPY = (-math.pi / 2.0, 0.0, 0.0)
WRIST_CAM_CLIPPING = (0.005, 100.0)          # near/far planes (m)

# Overhead camera: mounted on the base or above the workspace looking down
OVERHEAD_CAM_PARENT_LINK = "world"          # parent to the root of the robot
OVERHEAD_CAM_RESOLUTION = (640, 480)
OVERHEAD_CAM_HFOV_DEG = 70.0
# Look straight down at the workspace from above.
# The table is at Z=0.75, garment is at Y=0.25.
# Placing the camera at Z=1.4 (~65cm above table), looking straight down (RPY=0,0,0).
OVERHEAD_CAM_TRANSLATION = (0.0, -0.1, 1.3)
OVERHEAD_CAM_RPY = (0.567, 0.0, 0.0)
OVERHEAD_CAM_CLIPPING = (0.01, 100.0)


# --------------------------------------------------------------------------- #
# Cross-version import helpers
# --------------------------------------------------------------------------- #
def get_simulation_app(headless: bool = False):
    """Return a started SimulationApp (must be the very first Isaac call)."""
    try:
        from isaacsim import SimulationApp           # Isaac Sim >= 4.5
    except ImportError:
        from omni.isaac.kit import SimulationApp     # Isaac Sim <= 4.2
    return SimulationApp({
        "headless": headless,
        "physics_gpu": 0,
        "active_gpu": 0
    })


def get_world_cls():
    """Return the World class across Isaac versions."""
    try:
        from isaacsim.core.api import World          # >= 4.5
    except ImportError:
        from omni.isaac.core import World            # <= 4.2
    return World
