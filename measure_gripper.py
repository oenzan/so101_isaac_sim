import trimesh
import numpy as np

for part in ['Fixed_Gripper.STL', 'Moving_Jaw.STL']:
    mesh = trimesh.load(f'/home/ozan/Downloads/so100_ws/src/SO-100-arm/models/so_100_arm_5dof/meshes/{part}')
    bounds = mesh.bounds
    print(f"{part}:")
    print(f"  Min: {bounds[0]}")
    print(f"  Max: {bounds[1]}")
    print(f"  Size: {bounds[1] - bounds[0]}")
    print(f"  Center: {mesh.centroid}")
    print()
