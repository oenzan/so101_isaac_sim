"""Check raw mesh bounds vs cloth placement - no numpy needed"""
import json

# Parse OBJ
vertices = []
with open("foldnet_garments/tshirt_sp_0/mesh.obj") as f:
    for line in f:
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]) * 0.5, float(parts[2]) * 0.5, float(parts[3]) * 0.5])

xs = [v[0] for v in vertices]
ys = [v[1] for v in vertices]
zs = [v[2] for v in vertices]

print(f"Raw mesh (scaled by 0.5), {len(vertices)} vertices:")
print(f"  X range: [{min(xs):.3f}, {max(xs):.3f}]  center={sum(xs)/len(xs):.3f}")
print(f"  Y range: [{min(ys):.3f}, {max(ys):.3f}]  center={sum(ys)/len(ys):.3f}")
print(f"  Z range: [{min(zs):.3f}, {max(zs):.3f}]  center={sum(zs)/len(zs):.3f}")

print(f"\nIsaac Sim cloth placed at: (0.0, 0.15, 0.755)")
print(f"FoldNet expects cloth at raw mesh center: ({sum(xs)/len(xs):.3f}, {sum(ys)/len(ys):.3f}, 0)")

# Load keypoints
with open("foldnet_garments/tshirt_sp_0/mesh_info.json") as f:
    info = json.load(f)
kp = info.get("triangulation", {}).get("keypoint_idx", {})
print(f"\nKeypoints ({len(kp)} total):")
for name, indices in sorted(kp.items()):
    if isinstance(indices, list) and len(indices) > 0:
        idx = indices[0]
        pos = vertices[idx]
        print(f"  {name:25s}: vertex {idx:5d} -> ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
