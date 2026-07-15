"""
garment_loader.py
=================
Load a FoldNet garment mesh (.obj + mesh_info.json) into the Isaac Sim stage as a
PhysX particle cloth.

The mesh is a flat 2D pattern lying in the XY plane, centred at the origin, with
Z ~ ±0.02–0.03 m (the fabric half-thickness). Callers specify a desired world
position (typically on the table top) and an optional scale factor.

Keypoints from mesh_info.json are returned as a dict mapping semantic names
(e.g. "l_shoulder", "spine_bottom_f") to lists of vertex indices in the mesh.
"""

import json
import numpy as np
from pathlib import Path

from pxr import Gf, UsdGeom, Sdf, UsdPhysics, UsdShade, PhysxSchema
from omni.physx.scripts import deformableUtils, particleUtils, physicsUtils


def _parse_obj(path):
    vertices = []
    face_vertices = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                # Face can be "f v1 v2 v3" or "f v1/t1 v2/t2 v3/t3" or
                # "f v1//n1 v2//n2 v3//n3" – always extract the vertex index.
                idxs = []
                for p in parts[1:]:
                    idxs.append(int(p.split("/")[0]) - 1)  # OBJ is 1-indexed
                face_vertices.append(idxs)
    return np.array(vertices, dtype=np.float32), face_vertices


def _set_attr_if_present(prim, name, value):
    attr = prim.GetAttribute(name)
    if attr.IsValid():
        attr.Set(value)


def _define_mesh(stage, mesh_path, vertices, face_vertices):
    face_counts = [len(f) for f in face_vertices]
    face_indices = [idx for f in face_vertices for idx in f]

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(mesh_path))
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def _add_particle_cloth(stage, scene_path, root_path, mesh_path, profile,
                        particle_contact_offset, mass, friction,
                        solid_rest_offset=None):
    system_path = Sdf.Path(f"{root_path}/particleSystem")
    solver_position_iterations = 16
    # Detection radius (particle_contact_offset) and pushed-apart distance
    # (solid_rest_offset) are decoupled: two contacting cloth layers settle at
    # 2*solid_rest_offset apart, so this must stay below the mesh edge length
    # or stacked layers get inflated and slide around.
    if solid_rest_offset is None:
        solid_rest_offset = particle_contact_offset
    solid_rest_offset = min(solid_rest_offset, particle_contact_offset * 0.99)
    prim = stage.GetPrimAtPath(system_path)
    if not prim.IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=system_path,
            contact_offset=particle_contact_offset * 1.5,
            rest_offset=solid_rest_offset,
            particle_contact_offset=particle_contact_offset,
            solid_rest_offset=solid_rest_offset,
            fluid_rest_offset=0.0,
            solver_position_iterations=solver_position_iterations,
            simulation_owner=Sdf.Path(scene_path),
        )
        prim = stage.GetPrimAtPath(system_path)

    if prim.IsValid():
        particle_system = PhysxSchema.PhysxParticleSystem(prim)
        attr = particle_system.GetSolverPositionIterationCountAttr()
        if not attr.IsValid():
            attr = particle_system.CreateSolverPositionIterationCountAttr()
        attr.Set(solver_position_iterations)

    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=Sdf.Path(mesh_path),
        dynamic_mesh_path=None,
        particle_system_path=system_path,
        spring_stretch_stiffness=profile["stretch"],
        spring_bend_stiffness=profile["bend"],
        spring_shear_stiffness=profile["shear"],
        spring_damping=profile["damping"],
        self_collision=profile["self_collision"],
        self_collision_filter=profile["self_collision"],
        particle_group=0,
    )

    material_path = f"{root_path}/clothMaterial"
    # particle_friction_scale multiplies friction for particle-particle
    # contacts only (cloth-on-cloth); cloth-table friction is unaffected.
    # At 1.0 a pressed fold lying on the body slides back open (FoldDiag
    # back% 29->51% in 60f), because the elastic bend springs at the fold
    # line beat the layer-on-layer friction. Real cotton-on-cotton holds.
    import os
    particle_friction_scale = float(
        os.environ.get("ISAAC_GARMENT_PARTICLE_FRICTION_SCALE", "") or 1.0
    )
    particleUtils.add_pbd_particle_material(
        stage=stage,
        path=material_path,
        friction=friction,
        particle_friction_scale=particle_friction_scale,
    )
    binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(Sdf.Path(mesh_path)))
    binding.Bind(UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(material_path))))

    mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(Sdf.Path(mesh_path)))
    mass_api.CreateMassAttr(mass)


def _add_surface_deformable(stage, scene_path, root_path, mesh_path, profile,
                            particle_contact_offset, mass, friction):
    import carb
    import omni.physx.bindings._physx as physx_settings_bindings

    carb.settings.get_settings().set_bool(
        physx_settings_bindings.SETTING_ENABLE_DEFORMABLE_BETA, True
    )

    scene_prim = stage.GetPrimAtPath(Sdf.Path(scene_path))
    if scene_prim.IsValid():
        scene_prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(262144)
        # Ensure GPU dynamics is ON — deformable bodies require it
        scene_api.CreateEnableGPUDynamicsAttr(True)
        scene_api.CreateBroadphaseTypeAttr("MBP")

    deformable_root = Sdf.Path(mesh_path).GetParentPath()
    sim_mesh_path = deformable_root.AppendChild("simMesh")
    deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=deformable_root,
        simulation_mesh_path=sim_mesh_path,
        cooking_src_mesh_path=Sdf.Path(mesh_path),
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )

    root_prim = stage.GetPrimAtPath(deformable_root)
    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    _set_attr_if_present(root_prim, "physxDeformableBody:selfCollision", False)
    _set_attr_if_present(root_prim, "physxDeformableBody:enableSpeculativeCCD", False)
    _set_attr_if_present(root_prim, "physxDeformableBody:solverPositionIterationCount", 16)
    _set_attr_if_present(root_prim, "physxDeformableBody:collisionPairUpdateFrequency", 4)
    _set_attr_if_present(root_prim, "physxDeformableBody:collisionIterationMultiplier", 4)
    _set_attr_if_present(root_prim, "physxDeformableBody:maxLinearVelocity", particle_contact_offset * 3.0 * 120.0)

    sim_mesh_prim = stage.GetPrimAtPath(sim_mesh_path)
    if sim_mesh_prim.IsValid():
        sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
        collision_api = PhysxSchema.PhysxCollisionAPI(sim_mesh_prim)
        collision_api.GetRestOffsetAttr().Set(particle_contact_offset)
        collision_api.GetContactOffsetAttr().Set(particle_contact_offset * 3.0)

    material_path = f"{root_path}/surfaceDeformableMaterial"
    deformableUtils.add_surface_deformable_material(
        stage,
        material_path,
        density=mass,
        static_friction=friction,
        dynamic_friction=friction,
        surface_thickness=particle_contact_offset,
        surface_stretch_stiffness=profile["stretch"],
        surface_shear_stiffness=profile["shear"],
        surface_bend_stiffness=profile["bend"],
    )
    physicsUtils.add_physics_material_to_prim(stage, root_prim, material_path)

    material_prim = stage.GetPrimAtPath(Sdf.Path(material_path))
    if material_prim.IsValid():
        material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
        _set_attr_if_present(material_prim, "physxDeformableMaterial:elasticityDamping", profile["damping"])
        _set_attr_if_present(material_prim, "physxDeformableMaterial:bendDamping", profile["damping"])


def load_garment(
    stage,
    scene_path,
    root_path,
    garment_name,
    garment_dir,
    scale=0.5,
    center=(0.0, 0.25, 0.77),
    particle_contact_offset=0.008,
    solid_rest_offset=None,
    profile=None,
    mass=0.05,
    friction=0.8,
    backend="particle",
    flip_xy=False,
):
    """
    Load a FoldNet garment as PhysX particle cloth or surface deformable.

    Parameters
    ----------
    stage : Usd.Stage
    scene_path : str or Sdf.Path
        Path of the PhysicsScene (owns the particle system).
    root_path : str
        Prim path prefix, e.g. "/World/Garment".
    garment_name : str
        Category_variant, e.g. "tshirt_sp_0".
    garment_dir : str or Path
        Base directory containing ``<garment_name>/mesh.obj`` etc.
    scale : float
        Scale factor applied to vertex positions (FoldNet uses 0.5).
    center : (x, y, z)
        World position of the garment centre after scaling.
    particle_contact_offset : float
        PhysX particle contact offset (collision detection radius).
    solid_rest_offset : float or None
        Particle rest offset; stacked cloth layers settle at
        2*solid_rest_offset apart. Keep at or below ~0.5x the mesh edge
        length to avoid inflated, self-sliding folds. None = same as
        particle_contact_offset (legacy behaviour).
    profile : dict or None
        Physics profile dict with keys ``stretch``, ``bend``, ``shear``,
        ``damping``, ``self_collision``. If None, uses light defaults.
    mass : float
        Total cloth mass in kg.
    friction : float
        Cloth/table friction coefficient.
    backend : "particle" or "surface"
        Physics backend. "surface" uses Isaac Sim's beta Surface Deformable body.

    Returns
    -------
    mesh_path : str
        USD prim path of the cloth mesh.
    keypoint_idx : dict
        ``{name: [vertex_index, ...]}`` from mesh_info.json.
    boundary_idx : dict
        ``{name: [vertex_index, ...]}`` boundary vertex sets.
    vert_info : list[str]
        Per-vertex region label.
    n_vertices : int
        Number of vertices in the mesh.
    """
    garment_dir = Path(garment_dir)
    obj_path = garment_dir / garment_name / "mesh.obj"
    info_path = garment_dir / garment_name / "mesh_info.json"

    if not obj_path.exists():
        raise FileNotFoundError(f"Garment OBJ not found: {obj_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Garment info not found: {info_path}")

    if profile is None:
        profile = {"stretch": 5e3, "bend": 60, "shear": 60,
                   "damping": 0.2, "self_collision": True}

    backend = str(backend).lower()
    if backend not in {"particle", "surface"}:
        raise ValueError(f"Unsupported garment backend: {backend}")

    # ---- Load mesh data ------------------------------------------------------
    vertices_raw, face_vertices = _parse_obj(str(obj_path))
    if len(vertices_raw) == 0:
        raise ValueError(f"Empty mesh: {obj_path}")

    # Apply scale
    vertices = vertices_raw * scale

    # Surface deformables work best with a very thin skin. FoldNet garments have
    # front/back layers with centimeters of local Z thickness; keeping all of it
    # cooks a bulky shell, while flattening to exactly Z=0 makes layers overlap.
    # Compress the thickness to a small nonzero gap to avoid z-fighting/jitter.
    if backend == "surface":
        max_abs_z = float(np.max(np.abs(vertices[:, 2])))
        if max_abs_z > 0.0:
            target_half_thickness = min(particle_contact_offset * 0.15, 0.001)
            vertices[:, 2] *= target_half_thickness / max_abs_z

    # Negate X and Y to match robot's 180° Z rotation in Isaac Sim
    if flip_xy:
        vertices[:, 0] *= -1.0
        vertices[:, 1] *= -1.0

    # Translate to target center
    center = np.asarray(center, dtype=np.float32)
    vertices += center

    # ---- Load metadata -------------------------------------------------------
    with open(info_path) as f:
        info = json.load(f)

    keypoint_idx = info.get("triangulation", {}).get("keypoint_idx", {})
    boundary_idx = info.get("triangulation", {}).get("boundary_idx", {})
    vert_info = info.get("triangulation", {}).get("vert_info", [])

    # The mesh_info keypoint vertex indices are 0-indexed (PyFlex convention).
    # No adjustment needed.

    # ---- Create UsdGeom.Mesh -------------------------------------------------
    if backend == "surface":
        UsdGeom.Xform.Define(stage, Sdf.Path(f"{root_path}/surfaceDeformable"))
        mesh_path = f"{root_path}/surfaceDeformable/mesh"
    else:
        mesh_path = f"{root_path}/garmentMesh"
    _define_mesh(stage, mesh_path, vertices, face_vertices)

    # ---- Register with selected physics backend ------------------------------
    if backend == "particle":
        _add_particle_cloth(
            stage, scene_path, root_path, mesh_path, profile,
            particle_contact_offset, mass, friction,
            solid_rest_offset=solid_rest_offset,
        )
    else:
        _add_surface_deformable(
            stage, scene_path, root_path, mesh_path, profile,
            particle_contact_offset, mass, friction,
        )

    n_verts = len(vertices)
    cat = garment_name.rsplit("_", 1)[0]  # e.g. tshirt_sp_0 → tshirt_sp
    print(f"[garment] loaded '{garment_name}': {n_verts} verts, "
          f"{len(face_vertices)} faces, {len(keypoint_idx)} keypoints, "
          f"backend={backend}, "
          f"profile=({profile['stretch']:.0f}/{profile['bend']:.0f}/"
          f"{profile['shear']:.0f}/{profile['damping']:.1f}), "
          f"at {center}")

    return mesh_path, keypoint_idx, boundary_idx, vert_info, n_verts
