#!/usr/bin/env python3
"""
fk_check.py  (dev tool, plain Python + numpy)
=============================================
Forward kinematics for ONE SO-101 arm, used to pick a sensible "ready" pose
*before* running Isaac. We print the world position of every link frame so we
can see whether the gripper is up over the table or folded into the base.

Frames/values are taken straight from isaac_sim/urdf/so_100_dual.urdf.
"""

import numpy as np

# Robot `world` frame is lifted onto the table top in setup_scene.py.
TABLE_HEIGHT = 0.75
# Left arm base offset (right is +0.3). We analyse the left arm.
BASE_XYZ = np.array([-0.3, 0.0, 0.0])


def rpy_to_R(r, p, y):
    """URDF fixed-axis XYZ: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = rpy_to_R(*rpy)
    M[:3, 3] = xyz
    return M


def axis_rot(axis, q):
    """4x4 rotation of angle q about a unit axis (Rodrigues)."""
    a = np.array(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = np.cos(q), np.sin(q), 1 - np.cos(q)
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M


# (name, xyz, rpy, axis) for each actuated joint, in chain order.
CHAIN = [
    ("Shoulder_Rotation", (0, -0.0452, 0.0165), (1.5708, 0, 0), (0, 1, 0)),
    ("Shoulder_Pitch",    (0, 0.1025, 0.0306),  (0, 0, 0),      (1, 0, 0)),
    ("Elbow",             (0, 0.11257, 0.028),  (0, 0, 0),      (1, 0, 0)),
    ("Wrist_Pitch",       (0, 0.0052, 0.1349),  (0, 0, 0),      (1, 0, 0)),
    ("Wrist_Roll",        (0, -0.0601, 0),      (0, 0, 0),      (0, 1, 0)),
    ("Gripper",           (-0.0202, -0.0244, 0), (3.1416, 0, 3.1416), (0, 0, 1)),
]


def fk(q):
    """q: dict joint->angle. Returns dict joint-child-frame -> world xyz."""
    # world -> base_link (fixed) -> Base (fixed). Base origin is on the table top.
    M = np.eye(4)
    M[:3, 3] = np.array([0, 0, TABLE_HEIGHT]) + BASE_XYZ   # world frame + base offset
    out = {"Base": M[:3, 3].copy()}
    for name, xyz, rpy, axis in CHAIN:
        M = M @ T(xyz, rpy) @ axis_rot(axis, q.get(name, 0.0))
        out[name] = M[:3, 3].copy()
    return out


def show(label, q):
    pts = fk(q)
    print(f"\n=== {label} ===")
    base_z = pts["Base"][2]
    for k, v in pts.items():
        tag = ""
        if k in ("Gripper",) and v[2] < base_z + 0.02:
            tag = "  <-- gripper at/below base height!"
        print(f"  {k:20s} world=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}){tag}")
    print(f"  gripper height above table: {pts['Gripper'][2] - TABLE_HEIGHT:+.3f} m")


if __name__ == "__main__":
    zero = {n: 0.0 for n, *_ in CHAIN}
    show("ALL ZERO", zero)

    old = dict(Shoulder_Rotation=0.0, Shoulder_Pitch=-0.6, Elbow=1.0,
               Wrist_Pitch=0.6, Wrist_Roll=0.0, Gripper=0.6)
    show("OLD READY_POSE (folds into base)", old)

    # SAFE pose: zero everything (== CAD assembly, collision-free) but swivel
    # the shoulder 180deg so the arm faces +y (toward the cloth/table).
    # At exactly 0 or +/-pi the joint-axis SIGN cannot change the result, so this
    # pose is robust to any URDF->USD axis convention.
    safe = {n: 0.0 for n, *_ in CHAIN}
    safe["Shoulder_Rotation"] = np.pi
    show("SAFE (shoulder swivel pi, faces +y)", safe)
