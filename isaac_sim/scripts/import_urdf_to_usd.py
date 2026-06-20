#!/usr/bin/env python3
"""
import_urdf_to_usd.py
=====================
Convert the generated dual-arm URDF into a USD asset using Isaac Sim's URDF
importer. Run this ONCE; afterwards `setup_scene.py` just references the USD,
which loads much faster than re-importing the URDF every time.

Run with Isaac Sim's python:
    ./python.sh /path/to/so101_isaac_sim/isaac_sim/scripts/import_urdf_to_usd.py
(or `isaacsim` / `omni_python` depending on your install)

Output:
    isaac_sim/usd/so_100_dual.usd
"""

import config

# 1) Boot Isaac Sim FIRST. Nothing from omni/isaacsim may be imported before
#    SimulationApp is constructed, so we do it at the very top.
sim_app = config.get_simulation_app(headless=True)

# 2) Now that the app is up, the rest of the Omniverse API is importable.
import omni.kit.commands
try:
    from isaacsim.core.utils.extensions import enable_extension      # >= 4.5
except ImportError:
    from omni.isaac.core.utils.extensions import enable_extension    # <= 4.2

# The URDF importer ships as an extension under different ids across versions.
for ext in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
    try:
        enable_extension(ext)
        break
    except Exception:
        continue


def main():
    urdf_path = str(config.URDF_PATH)
    config.USD_DIR.mkdir(parents=True, exist_ok=True)
    usd_path = str(config.USD_PATH)

    print(f"[import] URDF in : {urdf_path}")
    print(f"[import] USD out : {usd_path}")

    # 3) Build an import configuration. These settings matter for a robot we
    #    intend to actuate and use for contact-rich (cloth) manipulation.
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = True      # collapse welded links -> fewer bodies
    import_config.fix_base = True                # `world` link is anchored (table-mounted)
    import_config.make_default_prim = True
    import_config.import_inertia_tensor = True   # use the inertials from the URDF
    import_config.distance_scale = 1.0           # URDF is already in metres
    import_config.self_collision = False         # off by default; enable if arms clip
    # Give every revolute joint a position drive so we can command joint targets.
    try:
        from isaacsim.asset.importer.urdf import _urdf as urdf_iface
        import_config.default_drive_type = urdf_iface.UrdfJointTargetType.JOINT_DRIVE_POSITION
    except Exception:
        pass  # older builds default to position drive anyway
    # Baseline drive gains == the Feetech STS3215 model (see config.py / motors.py).
    # Previously these were 1e4 / 1e3, i.e. an unrealistically stiff "perfect"
    # motor; the STS3215 values make the USD behave like the real servo even
    # before motors.py refines per-joint gains at runtime.
    import_config.default_drive_strength = config.ARM_DRIVE_STIFFNESS       # Kp [N.m/rad]
    import_config.default_position_drive_damping = config.ARM_DRIVE_DAMPING  # Kd [N.m.s/rad]

    # 4) Parse + import + write the USD in one command.
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=usd_path,
    )
    print(f"[import] done. articulation root prim: {prim_path}")


if __name__ == "__main__":
    main()
    sim_app.close()
