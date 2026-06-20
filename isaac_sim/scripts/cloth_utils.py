"""
cloth_utils.py
==============
Helpers to drop a simulated piece of cloth into the stage using PhysX
*particle cloth*. Particle cloth is the right primitive for a folding task:
it bends, drapes and self-collides like real fabric.

NOTE: particle cloth requires GPU PhysX. `setup_scene.py` enables that on the
physics scene before these helpers are called.

Everything here is written against `pxr` (USD) + `omni.physx.scripts`, which are
stable across Isaac Sim 4.2 / 4.5 / 5.0.
"""

from pxr import Gf, UsdGeom, Sdf
from omni.physx.scripts import particleUtils, physicsUtils


def _build_grid_mesh(stage, mesh_path, center, side, resolution):
    """
    Create a flat NxN UsdGeom.Mesh square (in the XY plane) centred at `center`.
    This is the geometry that the particle-cloth solver will animate.
    """
    n = resolution
    half = side / 2.0
    step = side / (n - 1)

    points = []
    for j in range(n):
        for i in range(n):
            x = center[0] - half + i * step
            y = center[1] - half + j * step
            z = center[2]
            points.append(Gf.Vec3f(x, y, z))

    # Two triangles per grid cell.
    indices = []
    counts = []
    for j in range(n - 1):
        for i in range(n - 1):
            v0 = j * n + i
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            indices += [v0, v2, v1]   # triangle 1
            indices += [v1, v2, v3]   # triangle 2
            counts += [3, 3]

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(mesh_path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def add_cloth(stage, scene_path, root_path, center, side, resolution,
              particle_contact_offset=0.006):
    """
    Add a particle-cloth square to the stage.

    stage          : Usd.Stage
    scene_path     : Sdf path of the PhysicsScene (owns the particle system)
    root_path      : where to create the cloth prims, e.g. "/World/Cloth"
    center         : (x, y, z) world position of the cloth centre
    side           : side length in metres
    resolution     : particles per side (more = finer, heavier sim)
    """
    system_path = Sdf.Path(f"{root_path}/particleSystem")
    mesh_path = f"{root_path}/clothMesh"

    # 1) A particle system holds shared solver settings for everything made of
    #    particles. Offsets define how far particles "feel" each other / contact.
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=system_path,
        contact_offset=particle_contact_offset * 1.5,
        rest_offset=particle_contact_offset,
        particle_contact_offset=particle_contact_offset,
        solid_rest_offset=particle_contact_offset,
        fluid_rest_offset=0.0,
        simulation_owner=Sdf.Path(scene_path),
    )

    # 2) The cloth geometry.
    _build_grid_mesh(stage, mesh_path, center, side, resolution)

    # 3) Turn that mesh into simulated cloth. Stiffness values tuned for a
    #    light, easily-foldable fabric; raise them for stiffer material.
    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=Sdf.Path(mesh_path),
        dynamic_mesh_path=None,
        particle_system_path=system_path,
        spring_stretch_stiffness=1.0e4,
        spring_bend_stiffness=80.0,
        spring_shear_stiffness=80.0,
        spring_damping=0.2,
        self_collision=True,
        self_collision_filter=True,
        particle_group=0,
    )

    # 4) Give the whole cloth a small total mass so it drapes naturally.
    from pxr import UsdPhysics
    mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(Sdf.Path(mesh_path)))
    mass_api.CreateMassAttr(0.05)   # ~50 g of fabric

    print(f"[cloth] added {resolution}x{resolution} particle cloth at {center}")
    return mesh_path
