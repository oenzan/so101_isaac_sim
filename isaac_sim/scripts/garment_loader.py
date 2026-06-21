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

from pxr import Gf, UsdGeom, Sdf, UsdPhysics
from omni.physx.scripts import particleUtils


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


def load_garment(
    stage,
    scene_path,
    root_path,
    garment_name,
    garment_dir,
    scale=0.5,
    center=(0.0, 0.25, 0.77),
    particle_contact_offset=0.012,
    profile=None,
    mass=0.05,
):
    """
    Load a FoldNet garment as PhysX particle cloth.

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
        PhysX particle contact offset.
    profile : dict or None
        Physics profile dict with keys ``stretch``, ``bend``, ``shear``,
        ``damping``, ``self_collision``. If None, uses light defaults.
    mass : float
        Total cloth mass in kg.

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

    # ---- Load mesh data ------------------------------------------------------
    vertices_raw, face_vertices = _parse_obj(str(obj_path))
    if len(vertices_raw) == 0:
        raise ValueError(f"Empty mesh: {obj_path}")

    # Apply scale
    vertices = vertices_raw * scale

    # Translate to target center
    center = np.asarray(center, dtype=np.float32)
    vertices += center

    # Build flat face lists for USD
    face_counts = [len(f) for f in face_vertices]
    face_indices = [idx for f in face_vertices for idx in f]

    # ---- Load metadata -------------------------------------------------------
    with open(info_path) as f:
        info = json.load(f)

    keypoint_idx = info.get("triangulation", {}).get("keypoint_idx", {})
    boundary_idx = info.get("triangulation", {}).get("boundary_idx", {})
    vert_info = info.get("triangulation", {}).get("vert_info", [])

    # The mesh_info keypoint vertex indices are 0-indexed (PyFlex convention).
    # No adjustment needed.

    # ---- Create particle system (if not already present) ----------------------
    system_path = Sdf.Path(f"{root_path}/particleSystem")
    prim = stage.GetPrimAtPath(system_path)
    if not prim.IsValid():
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

    # ---- Create UsdGeom.Mesh -------------------------------------------------
    mesh_path = f"{root_path}/garmentMesh"
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(mesh_path))

    # Convert numpy to Gf.Vec3f list
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateDoubleSidedAttr(True)

    # ---- Register as PhysX particle cloth ------------------------------------
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

    # ---- Set mass ------------------------------------------------------------
    mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(Sdf.Path(mesh_path)))
    mass_api.CreateMassAttr(mass)

    n_verts = len(vertices)
    cat = garment_name.rsplit("_", 1)[0]  # e.g. tshirt_sp_0 → tshirt_sp
    print(f"[garment] loaded '{garment_name}': {n_verts} verts, "
          f"{len(face_vertices)} faces, {len(keypoint_idx)} keypoints, "
          f"profile=({profile['stretch']:.0f}/{profile['bend']:.0f}/"
          f"{profile['shear']:.0f}/{profile['damping']:.1f}), "
          f"at {center}")

    return mesh_path, keypoint_idx, boundary_idx, vert_info, n_verts
