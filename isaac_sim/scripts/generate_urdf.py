#!/usr/bin/env python3
"""
generate_urdf.py
================
Build a clean, self-contained URDF for the *dual* SO-101 (SO-100) robot that
Isaac Sim's URDF importer can consume directly.

Why we regenerate instead of using `robot_description.txt`:
  * The original description references meshes via `package://so_100_arm/...`,
    which only resolves inside a ROS workspace. Isaac Sim resolves mesh paths
    relative to the URDF file, so we emit *relative* file paths instead.
  * The original joints are `continuous` with **no limits / effort / velocity**.
    A physics simulator needs sane joint limits and drive parameters, so we add
    them here.
  * The two arms are identical except for a name prefix and the base location,
    so we describe ONE arm once and stamp it out twice (DRY).

Run with any normal Python 3 (no Isaac needed):
    python3 isaac_sim/scripts/generate_urdf.py
Output:
    isaac_sim/urdf/so_100_dual.urdf
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]          # .../so101_isaac_sim
URDF_OUT = REPO_ROOT / "isaac_sim" / "urdf" / "so_100_dual.urdf"

# Mesh location, expressed *relative to the URDF file* so the URDF stays portable.
# URDF lives in isaac_sim/urdf/ ; meshes live in src/SO-100-arm/models/.../meshes/
MESH_DIR_REL = "../../src/SO-100-arm/models/so_100_arm_5dof/meshes"

# --------------------------------------------------------------------------- #
# Per-link inertial + mesh data (identical for both arms).
# Values copied verbatim from robot_description.txt.
# --------------------------------------------------------------------------- #
LINKS = {
    "Base": dict(
        mesh="Base.STL",
        com=(-2.45960666746703e-07, 0.0311418169687909, 0.0175746661003382),
        mass=0.193184127927598,
        inertia=(1.37030709467877e-04, 2.10136126944992e-08, 4.24087422551286e-09,
                 1.69089551209259e-04, 2.26514711036514e-05, 1.45097720857224e-04),
    ),
    "Shoulder_Rotation_Pitch": dict(
        mesh="Shoulder_Rotation_Pitch.STL",
        com=(-9.07886224712597e-05, 0.0590971820568318, 0.031089016892169),
        mass=0.119226314127197,
        inertia=(5.90408775624429e-05, 4.90800532852998e-07, -5.90451772654387e-08,
                 3.21498601038881e-05, -4.58026206663885e-06, 5.86058514263952e-05),
    ),
    "Upper_Arm": dict(
        mesh="Upper_Arm.STL",
        com=(-1.7205170190925e-05, 0.0701802156327694, 0.00310545118155671),
        mass=0.162409284599177,
        inertia=(1.67153146617081e-04, 1.03902689187701e-06, -1.20161820645189e-08,
                 7.01946992214245e-05, 2.11884806298698e-06, 2.13280241160769e-04),
    ),
    "Lower_Arm": dict(
        mesh="Lower_Arm.STL",
        com=(-0.00339603710186651, 0.00137796353960074, 0.0768006751156044),
        mass=0.147967774582291,
        inertia=(1.05333995841409e-04, 1.73059237226499e-07, -1.1720305455211e-05,
                 1.38766654485212e-04, 1.77429964684103e-06, 5.08741652515214e-05),
    ),
    "Wrist_Pitch_Roll": dict(
        mesh="Wrist_Pitch_Roll.STL",
        com=(-0.00852653127372418, -0.0352278997897927, -2.34622481569413e-05),
        mass=0.066132067097723,
        inertia=(1.95717492443445e-05, -6.62714374412293e-07, 5.20089016442066e-09,
                 2.38028417569933e-05, 4.09549055863776e-08, 3.4540143384536e-05),
    ),
    "Fixed_Gripper": dict(
        mesh="Fixed_Gripper.STL",
        com=(0.00552376906426563, -0.0280167153359021, 0.000483582592841092),
        mass=0.0929859131176897,
        inertia=(4.3328249304211e-05, 7.09654328670947e-06, 5.99838530879484e-07,
                 3.04451747368212e-05, -1.58743247545413e-07, 5.02460913506734e-05),
    ),
    "Moving_Jaw": dict(
        mesh="Moving_Jaw.STL",
        com=(-0.00161744605468241, -0.0303472584046471, 0.000449645961853651),
        mass=0.0202443794940372,
        inertia=(1.10911325081525e-05, -5.35076503033314e-07, -9.46105662101403e-09,
                 3.03576451001973e-06, -1.71146075110632e-07, 8.9916083370498e-06),
    ),
}

# --------------------------------------------------------------------------- #
# Per-joint kinematics (identical for both arms).
# Each entry: name, parent link, child link, xyz, rpy, axis.
# The original joints were `continuous`; we convert them to `revolute` and add
# limits below so the simulation behaves sensibly.
# --------------------------------------------------------------------------- #
JOINTS = [
    # name              parent                    child                     xyz                          rpy                  axis
    ("Shoulder_Rotation", "Base",                   "Shoulder_Rotation_Pitch", (0, -0.0452, 0.0165),       (1.5708, 0, 0),      (0, 1, 0)),
    ("Shoulder_Pitch",    "Shoulder_Rotation_Pitch", "Upper_Arm",             (0, 0.1025, 0.0306),        (0, 0, 0),           (1, 0, 0)),
    ("Elbow",             "Upper_Arm",              "Lower_Arm",              (0, 0.11257, 0.028),        (0, 0, 0),           (1, 0, 0)),
    ("Wrist_Pitch",       "Lower_Arm",              "Wrist_Pitch_Roll",       (0, 0.0052, 0.1349),        (0, 0, 0),           (1, 0, 0)),
    ("Wrist_Roll",        "Wrist_Pitch_Roll",       "Fixed_Gripper",          (0, -0.0601, 0),            (0, 0, 0),           (0, 1, 0)),
    ("Gripper",           "Fixed_Gripper",          "Moving_Jaw",             (-0.0202, -0.0244, 0),      (3.1416, 0, 3.1416), (0, 0, 1)),
]

# Joint limits (radians). The arm servos (Feetech STS3215) can swing ~±pi;
# the gripper has a small opening range. Effort/velocity are conservative and
# can be tuned. These are deliberately explicit so they are easy to adjust.
JOINT_LIMITS = {
    # Base swivel: a little past +/-pi so the 180deg ready pose sits comfortably
    # inside the limit instead of on the hard stop.
    "Shoulder_Rotation": dict(lower=-3.3, upper=3.3, effort=5.0, velocity=2.0),
    "Shoulder_Pitch":    dict(lower=-3.14159, upper=3.14159, effort=5.0, velocity=2.0),
    "Elbow":             dict(lower=-3.14159, upper=3.14159, effort=5.0, velocity=2.0),
    "Wrist_Pitch":       dict(lower=-3.14159, upper=3.14159, effort=5.0, velocity=2.0),
    "Wrist_Roll":        dict(lower=-3.14159, upper=3.14159, effort=5.0, velocity=2.0),
    "Gripper":           dict(lower=-0.2,     upper=1.7,     effort=5.0, velocity=2.0),
}

# Where each arm's fixed base sits in the world frame (matches robot_description.txt).
ARMS = {
    "left":  (-0.3, 0.0, 0.0),
    "right": (0.3, 0.0, 0.0),
}


# --------------------------------------------------------------------------- #
# Small XML emit helpers
# --------------------------------------------------------------------------- #
def _xyz(t):
    return f"{t[0]:g} {t[1]:g} {t[2]:g}"


def link_xml(prefix, link_name, data):
    """Emit a full <link> with inertial, visual and collision (shared mesh)."""
    cx, cy, cz = data["com"]
    ixx, ixy, ixz, iyy, iyz, izz = data["inertia"]
    mesh = f"{MESH_DIR_REL}/{data['mesh']}"
    return f"""  <link name="{prefix}_{link_name}">
    <inertial>
      <origin xyz="{cx:g} {cy:g} {cz:g}" rpy="0 0 0"/>
      <mass value="{data['mass']:g}"/>
      <inertia ixx="{ixx:g}" ixy="{ixy:g}" ixz="{ixz:g}" iyy="{iyy:g}" iyz="{iyz:g}" izz="{izz:g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh}"/></geometry>
      <material name="so101_grey"><color rgba="0.79 0.82 0.93 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh}"/></geometry>
    </collision>
  </link>
"""


def joint_xml(prefix, jname, parent, child, xyz, rpy, axis):
    """Emit a revolute <joint> with limits."""
    lim = JOINT_LIMITS[jname]
    return f"""  <joint name="{prefix}_{jname}" type="revolute">
    <origin xyz="{_xyz(xyz)}" rpy="{_xyz(rpy)}"/>
    <parent link="{prefix}_{parent}"/>
    <child link="{prefix}_{child}"/>
    <axis xyz="{_xyz(axis)}"/>
    <limit lower="{lim['lower']:g}" upper="{lim['upper']:g}" effort="{lim['effort']:g}" velocity="{lim['velocity']:g}"/>
  </joint>
"""


def build_arm(prefix, base_xyz):
    """Build one complete arm: empty base_link -> Base -> ... -> Moving_Jaw."""
    out = []
    # An empty mounting link (mirrors the original `<prefix>_base_link`).
    out.append(f'  <link name="{prefix}_base_link"/>\n')
    # Fixed weld from world to this arm's base_link at its world position.
    out.append(f"""  <joint name="world_to_{prefix}_base" type="fixed">
    <origin xyz="{_xyz(base_xyz)}" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="{prefix}_base_link"/>
  </joint>
""")
    # Fixed weld base_link -> Base (the first meshed link).
    out.append(f"""  <joint name="{prefix}_base_link_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_base_link"/>
    <child link="{prefix}_Base"/>
  </joint>
""")
    # All meshed links.
    for link_name, data in LINKS.items():
        out.append(link_xml(prefix, link_name, data))
    # All actuated joints.
    for (jname, parent, child, xyz, rpy, axis) in JOINTS:
        out.append(joint_xml(prefix, jname, parent, child, xyz, rpy, axis))
    return "".join(out)


def main():
    parts = [
        '<?xml version="1.0"?>\n',
        '<!-- Auto-generated by isaac_sim/scripts/generate_urdf.py -->\n',
        '<!-- Dual SO-101 (SO-100) bimanual robot, prepared for Isaac Sim. -->\n',
        '<robot name="so_100_dual">\n',
        '  <link name="world"/>\n',
    ]
    for prefix, base_xyz in ARMS.items():
        parts.append(f"\n  <!-- ============== {prefix.upper()} ARM ============== -->\n")
        parts.append(build_arm(prefix, base_xyz))
    parts.append("</robot>\n")

    URDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    URDF_OUT.write_text("".join(parts))
    print(f"[generate_urdf] wrote {URDF_OUT}")
    print(f"[generate_urdf] meshes referenced at: {MESH_DIR_REL}/")


if __name__ == "__main__":
    main()
