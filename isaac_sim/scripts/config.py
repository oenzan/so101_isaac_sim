"""
config.py
=========
Shared constants and a few cross-version import shims used by the Isaac Sim
scripts in this folder.

Isaac Sim renamed its Python packages from `omni.isaac.*` (<= 4.2) to
`isaacsim.*` (>= 4.5). Rather than hard-code one, the helpers below try the new
namespace first and fall back to the old one, so the scripts run on either.
"""

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

# A relaxed "ready" pose (radians) used as the simulation's default.
# Order matches ALL_JOINTS. Shoulder_Pitch/Elbow lifted so the arms stand up
# and the grippers hover over the table instead of lying flat.
READY_POSE = {
    "left_Shoulder_Rotation": 0.0,
    "left_Shoulder_Pitch": -0.6,
    "left_Elbow": 1.0,
    "left_Wrist_Pitch": 0.6,
    "left_Wrist_Roll": 0.0,
    "left_Gripper": 0.6,
    "right_Shoulder_Rotation": 0.0,
    "right_Shoulder_Pitch": -0.6,
    "right_Elbow": 1.0,
    "right_Wrist_Pitch": 0.6,
    "right_Wrist_Roll": 0.0,
    "right_Gripper": 0.6,
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
