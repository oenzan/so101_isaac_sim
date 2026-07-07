import os
import sys
import copy
import json
import importlib.util
import numpy as np
import torch
import glob
from collections import deque

# Mock pyflex before importing any garmentds modules
import sys
import os
from unittest.mock import MagicMock
sys.modules['pyflex'] = MagicMock()

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.py")
_CONFIG_SPEC = importlib.util.spec_from_file_location("isaac_sim_scripts_config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_CONFIG_SPEC)
assert _CONFIG_SPEC.loader is not None
_CONFIG_SPEC.loader.exec_module(config)

foldnet_base_dir = os.environ.get(
    "FOLDNET_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code")),
)
foldnet_src = os.path.join(foldnet_base_dir, "src")
if foldnet_src not in sys.path:
    sys.path.insert(0, foldnet_src)
batch_urdf_src = os.path.join(foldnet_base_dir, "external", "batch_urdf", "src")
if batch_urdf_src not in sys.path:
    sys.path.insert(0, batch_urdf_src)
os.environ["FOLDNET_BASE_DIR"] = foldnet_base_dir

sim_app = config.get_simulation_app(headless=os.environ.get("ISAAC_HEADLESS", "1") == "1")

import omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf, PhysxSchema, UsdPhysics, Sdf

try:                                                   # Isaac Sim >= 4.5
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:                                    # Isaac Sim <= 4.2
    from omni.isaac.core.utils.extensions import enable_extension

for ext in ("isaacsim.core.api", "omni.isaac.core"):
    try:
        enable_extension(ext)
        break
    except Exception:
        continue

try:                                                   # Isaac Sim >= 4.5
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.api.objects import FixedCuboid
except ImportError:                                    # Isaac Sim <= 4.2
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import FixedCuboid
from omni.physx.scripts import physicsUtils

import garment_loader
from so100_foldenv import RobotSO100, make_so100_robot_cfg, RenderProcessNull
from garmentds.foldenv.fold_env import FoldEnv, FoldEnvCfg, FoldEnvState, Picker
import garmentds.common.utils as utils


def _parse_enabled_hands(env_var: str, default=("left", "right")):
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return set(default)

    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if "none" in tokens:
        return set()
    if "all" in tokens or "both" in tokens:
        return {"left", "right"}
    return {token for token in tokens if token in {"left", "right"}}


def _parse_float_env(env_var: str, default: float) -> float:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"  [env] invalid {env_var}={raw!r}; using default {default}")
        return float(default)


def _parse_int_env(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print(f"  [env] invalid {env_var}={raw!r}; using default {default}")
        return int(default)


def _parse_choice_env(env_var: str, allowed: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in allowed:
        return value
    print(f"  [env] invalid {env_var}={raw!r}; using default {default}")
    return default


def _parse_bool_env(env_var: str, default: bool) -> bool:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    print(f"  [env] invalid {env_var}={raw!r}; using default {default}")
    return bool(default)


def _parse_vec3_env(env_var: str, default) -> np.ndarray:
    raw = os.environ.get(env_var)
    default_arr = np.asarray(default, dtype=np.float32)
    if raw is None or raw.strip() == "":
        return default_arr.astype(np.float32, copy=True)
    try:
        tokens = raw.replace(",", " ").split()
        values = [float(token) for token in tokens]
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError
        return np.asarray(values, dtype=np.float32)
    except ValueError:
        print(f"  [env] invalid {env_var}={raw!r}; using default {default_arr.tolist()}")
        return default_arr.astype(np.float32, copy=True)


class _SurfaceDeformableClothView:
    """Adapts a physics-tensor DeformableBodyView to the classic ParticleClothView
    interface (get/set_positions, get/set_velocities, max_particles_per_cloth,
    count) so existing cloth-view call sites work unmodified on Isaac Sim/PhysX
    builds where classic particle cloth (and create_particle_cloth_view) has been
    removed in favor of surface deformable bodies.

    get_masses/set_masses are intentionally NOT implemented here: deformable-body
    mass comes from material density, not a per-node settable scalar. Call sites
    already guard with hasattr(cloth_view, "get_masses") before using it, so
    omitting it here makes that mass-freeze trick skip gracefully.
    """

    def __init__(self, deformable_body_view):
        self._view = deformable_body_view

    @property
    def count(self):
        return self._view.count

    @property
    def max_particles_per_cloth(self):
        return self._view.max_simulation_nodes_per_body

    def get_positions(self):
        # Shape (count, max_nodes, 3) -- callers reshape by total element
        # count (N*3), so the extra explicit "nodes" dim is transparent.
        return self._view.get_simulation_nodal_positions()

    def set_positions(self, data, indices):
        n = self._view.max_simulation_nodes_per_body
        data_3d = data.reshape(data.shape[0], n, 3)
        indices_2d = indices.reshape(-1, 1)
        self._view.set_simulation_nodal_positions(data_3d, indices_2d)

    def get_velocities(self):
        return self._view.get_simulation_nodal_velocities()

    def set_velocities(self, data, indices):
        n = self._view.max_simulation_nodes_per_body
        data_3d = data.reshape(data.shape[0], n, 3)
        indices_2d = indices.reshape(-1, 1)
        self._view.set_simulation_nodal_velocities(data_3d, indices_2d)


class IsaacSimPicker(Picker):
    def __init__(self, env, name, grasp_threshold, squeeze_factor):
        super().__init__(env, name, grasp_threshold, squeeze_factor)
        self.attachment_path = f"/World/Attachments/{name}_attachment"
        self.block_attachment_path = f"/World/Attachments/{name}_block_attachment"
        self.block_path = f"/World/Attachments/{name}_attachment_block"
        mode = getattr(env, "_grasp_mode", "kinematic")
        upper_name = name.upper()
        self._picker_tcp_offset = _parse_vec3_env(
            f"ISAAC_{upper_name}_PICKER_TCP_OFFSET",
            _parse_vec3_env("ISAAC_PICKER_TCP_OFFSET", [0.0, 0.0, 0.0]),
        )
        self._squeeze_factor = _parse_vec3_env(
            f"ISAAC_{upper_name}_GRASP_SQUEEZE_FACTOR",
            _parse_vec3_env("ISAAC_GRASP_SQUEEZE_FACTOR", self._squeeze_factor),
        )
        attachment_default = ("left", "right") if mode in {"attachment", "handover"} else ()
        kinematic_default = ("left", "right") if mode in {"kinematic", "handover"} else ()
        block_default = ("left", "right") if mode == "block_attachment" else ()
        attachment_hands = _parse_enabled_hands("ISAAC_ATTACHMENT_HANDS", default=attachment_default)
        kinematic_hands = _parse_enabled_hands("ISAAC_KINEMATIC_GRASP_HANDS", default=kinematic_default)
        block_hands = _parse_enabled_hands("ISAAC_BLOCK_ATTACHMENT_HANDS", default=block_default)
        self._attachment_enabled = name in attachment_hands
        self._kinematic_grasp_enabled = name in kinematic_hands
        self._block_attachment_enabled = name in block_hands
        self._attachment_filter_distance = max(
            config.GARMENT_PARTICLE_CONTACT_OFFSET * 2.5,
            _parse_float_env("ISAAC_ATTACHMENT_FILTER_DISTANCE", 0.03),
        )
        self._attachment_filter_distance_max = max(
            self._attachment_filter_distance,
            _parse_float_env("ISAAC_ATTACHMENT_FILTER_DISTANCE_MAX", 0.06),
        )
        self._grasp_selection_shape = _parse_choice_env(
            "ISAAC_GRASP_SELECTION_SHAPE", ("box", "ellipsoid", "sphere"), "box"
        )
        self._max_grasp_vertices = max(0, _parse_int_env("ISAAC_MAX_GRASP_VERTICES", 24))
        self._sleeve_edge_grasp = _parse_bool_env("ISAAC_SLEEVE_EDGE_GRASP", False)
        self._sleeve_edge_verts_per_edge = max(
            0, _parse_int_env("ISAAC_SLEEVE_EDGE_GRASP_VERTS_PER_EDGE", 1)
        )
        self._sleeve_edge_activation_radius = max(
            0.0, _parse_float_env("ISAAC_SLEEVE_EDGE_GRASP_ACTIVATION_RADIUS", 0.07)
        )
        self._sleeve_edge_radius = max(
            0.0, _parse_float_env("ISAAC_SLEEVE_EDGE_GRASP_RADIUS", 0.025)
        )
        self._attachment_release_steps = max(
            0, _parse_int_env("ISAAC_ATTACHMENT_RELEASE_STEPS", 4)
        )
        # Frames of local velocity damping around released particles so stored
        # squeeze/constraint energy dissipates instead of launching the cloth.
        self._release_damp_frames = max(
            0, _parse_int_env("ISAAC_RELEASE_DAMP_FRAMES", 0)
        )
        self._release_damp_frames_left = 0
        self._released_vid = None
        # <=1 disables the grasp-time mass increase entirely (PhysX 5.1 warns
        # that changing particle cloth masses mid-simulation is unsupported).
        self._kinematic_mass_scale = max(
            0.0, _parse_float_env("ISAAC_KINEMATIC_MASS_SCALE", 1.0e6)
        )
        self._block_kinematic_freeze_rotation = _parse_bool_env(
            "ISAAC_BLOCK_KINEMATIC_FREEZE_ROTATION", False
        )
        self._kinematic_position_blend = float(
            np.clip(_parse_float_env("ISAAC_KINEMATIC_POSITION_BLEND", 0.5), 0.05, 1.0)
        )
        self._kinematic_neighbor_radius = max(
            0.0, _parse_float_env("ISAAC_KINEMATIC_NEIGHBOR_RADIUS", 0.04)
        )
        self._kinematic_neighbor_velocity_scale = float(
            np.clip(_parse_float_env("ISAAC_KINEMATIC_NEIGHBOR_VEL_SCALE", 0.35), 0.0, 1.0)
        )
        self._block_attachment_size = max(
            config.GARMENT_PARTICLE_CONTACT_OFFSET * 2.0,
            _parse_float_env("ISAAC_BLOCK_ATTACHMENT_SIZE", 0.02),
        )
        # "cube" or "sphere"; size is the edge length / diameter respectively.
        self._block_attachment_shape = (
            os.environ.get("ISAAC_BLOCK_ATTACHMENT_SHAPE", "cube").strip().lower()
        )
        self._block_attachment_overlap = max(
            config.GARMENT_PARTICLE_CONTACT_OFFSET,
            _parse_float_env("ISAAC_BLOCK_ATTACHMENT_OVERLAP", 0.02),
        )
        self._block_attachment_mass = max(
            0.001, _parse_float_env("ISAAC_BLOCK_ATTACHMENT_MASS", 1000.0)
        )
        self._block_attachment_follow_blend = float(
            np.clip(_parse_float_env("ISAAC_BLOCK_ATTACHMENT_BLEND", 1.0), 0.05, 1.0)
        )
        self._block_attachment_visible = _parse_bool_env("ISAAC_BLOCK_ATTACHMENT_VISIBLE", False)
        self._block_attachment_collision = _parse_bool_env("ISAAC_BLOCK_ATTACHMENT_COLLISION", False)
        self._block_attachment_auto = _parse_bool_env("ISAAC_BLOCK_ATTACHMENT_AUTO", False)
        self._block_attachment_use_physx = _parse_bool_env("ISAAC_BLOCK_ATTACHMENT_USE_PHYSX", False)
        self._block_kinematic_blend = float(
            np.clip(_parse_float_env("ISAAC_BLOCK_KINEMATIC_BLEND", 0.18), 0.05, 1.0)
        )
        self._block_kinematic_max_step = max(
            0.0, _parse_float_env("ISAAC_BLOCK_KINEMATIC_MAX_STEP", 0.008)
        )
        self._block_kinematic_catchup_step = max(
            self._block_kinematic_max_step,
            _parse_float_env("ISAAC_BLOCK_KINEMATIC_CATCHUP_STEP", 0.020),
        )
        self._block_kinematic_catchup_error = max(
            0.0, _parse_float_env("ISAAC_BLOCK_KINEMATIC_CATCHUP_ERROR", 0.025)
        )
        self._block_kinematic_poststep_only = _parse_bool_env(
            "ISAAC_BLOCK_KINEMATIC_POSTSTEP_ONLY", True
        )
        self._block_target_blend = float(
            np.clip(_parse_float_env("ISAAC_BLOCK_TARGET_BLEND", 0.35), 0.02, 1.0)
        )
        self._block_target_deadband = max(
            0.0, _parse_float_env("ISAAC_BLOCK_TARGET_DEADBAND", 0.001)
        )
        self._block_diagnostic_snap = _parse_bool_env("ISAAC_BLOCK_DIAGNOSTIC_SNAP", False)
        self._block_debug_marker_visible = _parse_bool_env(
            "ISAAC_BLOCK_DEBUG_MARKER", self._block_attachment_enabled
        )
        self._block_debug_marker_size = max(
            0.002, _parse_float_env("ISAAC_BLOCK_DEBUG_MARKER_SIZE", 0.018)
        )
        self._attachment_handover_frames_left = 0
        self._attachment_is_active = False
        self._block_attachment_is_active = False
        self._block_translate_op = None
        self._block_marker_path = f"/World/Attachments/{name}_grasp_marker"
        self._block_target_marker_path = f"/World/Attachments/{name}_grasp_target_marker"
        self._block_tcp_marker_path = f"/World/Attachments/{name}_picker_tcp_marker"
        self._block_marker_translate_op = None
        self._block_target_marker_translate_op = None
        self._block_tcp_marker_translate_op = None
        self._block_last_pos = None
        self._block_smoothed_particle_target = None
        self._block_debug_frames_left = 0
        self._block_attach_diag_frames = 0
        self._block_park_pos = np.array([0.0, 0.0, config.TABLE_HEIGHT + 0.5], dtype=np.float32)
        if self._block_attachment_enabled:
            self._ensure_block_prim()
        print(
            f"  [Picker:{self._name}] debug config:"
            f" mode={mode},"
            f" attachment={'on' if self._attachment_enabled else 'off'},"
            f" kinematic={'on' if self._kinematic_grasp_enabled else 'off'},"
            f" block_attachment={'on' if self._block_attachment_enabled else 'off'},"
            f" attach_filter={self._attachment_filter_distance:.3f}-{self._attachment_filter_distance_max:.3f}m,"
            f" grasp_shape={self._grasp_selection_shape},"
            f" tcp_offset={np.round(self._picker_tcp_offset, 4)},"
            f" squeeze={np.round(self._squeeze_factor, 3)},"
            f" max_grasp_verts={self._max_grasp_vertices if self._max_grasp_vertices > 0 else 'all'},"
            f" sleeve_edge={'on' if self._sleeve_edge_grasp else 'off'},"
            f" sleeve_edge_verts_per_edge={self._sleeve_edge_verts_per_edge},"
            f" sleeve_edge_activation={self._sleeve_edge_activation_radius:.3f},"
            f" sleeve_edge_radius={self._sleeve_edge_radius:.3f},"
            f" attach_release_steps={self._attachment_release_steps},"
            f" kinematic_mass_scale={self._kinematic_mass_scale:.1e},"
            f" kinematic_blend={self._kinematic_position_blend:.2f},"
            f" neighbor_radius={self._kinematic_neighbor_radius:.3f},"
            f" neighbor_vel_scale={self._kinematic_neighbor_velocity_scale:.2f},"
            f" block_size={self._block_attachment_size:.3f},"
            f" block_overlap={self._block_attachment_overlap:.3f},"
            f" block_blend={self._block_attachment_follow_blend:.2f},"
            f" block_collision={'on' if self._block_attachment_collision else 'off'},"
            f" block_auto={'on' if self._block_attachment_auto else 'off'},"
            f" block_physx={'on' if self._block_attachment_use_physx else 'off'},"
            f" block_kinematic_blend={self._block_kinematic_blend:.2f},"
            f" block_kinematic_max_step={self._block_kinematic_max_step:.3f},"
            f" block_catchup_step={self._block_kinematic_catchup_step:.3f},"
            f" block_catchup_error={self._block_kinematic_catchup_error:.3f},"
            f" block_poststep_only={'on' if self._block_kinematic_poststep_only else 'off'},"
            f" block_freeze_rot={'on' if self._block_kinematic_freeze_rotation else 'off'},"
            f" block_target_blend={self._block_target_blend:.2f},"
            f" block_target_deadband={self._block_target_deadband:.4f},"
            f" block_diag_snap={'on' if self._block_diagnostic_snap else 'off'},"
            f" block_marker={'on' if self._block_debug_marker_visible else 'off'}"
        )

    def _select_sleeve_edge_vertices(self, tf: np.ndarray, xyz_foldnet: np.ndarray):
        """Semantic sleeve grasp: pinch BOTH cloth layers (front + back) directly
        under the TCP so the whole sleeve thickness is carried from where the
        gripper actually closes. Sleeve keypoints are only used to confirm the
        TCP is near a sleeve (activation); the selection itself is TCP-centred.
        Returns a vertex index array, or None to fall back to shape selection."""
        kp_full = getattr(self._env, "_keypoint_idx_full", None)
        vert_info = getattr(self._env, "_vert_info", None)
        if not kp_full or not vert_info:
            print(f"  [SleeveEdgeGrasp:{self._name}] no keypoint/vert_info data on env; falling back to shape selection")
            return None
        tcp_xy = tf[:3, 3][:2]
        best = None
        for side in ("l", "r"):
            top = kp_full.get(f"{side}_sleeve_top")
            bottom = kp_full.get(f"{side}_sleeve_bottom")
            if not top or not bottom:
                continue
            mid = (xyz_foldnet[top[0]] + xyz_foldnet[bottom[0]]) / 2.0
            dist = float(np.linalg.norm(mid[:2] - tcp_xy))
            if best is None or dist < best[1]:
                best = (side, dist)
        if best is None:
            print(f"  [SleeveEdgeGrasp:{self._name}] no sleeve_top/sleeve_bottom keypoints; falling back to shape selection")
            return None
        side, mid_dist = best
        if mid_dist > self._sleeve_edge_activation_radius:
            print(
                f"  [SleeveEdgeGrasp:{self._name}] nearest sleeve '{side}' midpoint {mid_dist:.3f}m"
                f" > activation radius {self._sleeve_edge_activation_radius:.3f}m; falling back to shape selection"
            )
            return None
        # vert_info labels: 'front'/'*f' = front layer, 'back'/'*b' = back layer.
        labels = vert_info
        n = min(len(labels), xyz_foldnet.shape[0])
        dxy = np.linalg.norm(xyz_foldnet[:n, :2] - tcp_xy, axis=1)
        selected = []
        layer_msgs = []
        per_layer = max(1, self._sleeve_edge_verts_per_edge)
        for layer_name, matcher in (
            ("front", lambda l: l == "front" or l.endswith("f")),
            ("back", lambda l: l == "back" or l.endswith("b")),
        ):
            layer_ids = np.array([i for i in range(n) if matcher(labels[i])], dtype=int)
            if layer_ids.size == 0:
                layer_msgs.append(f"{layer_name}: none-in-mesh")
                continue
            order = layer_ids[np.argsort(dxy[layer_ids])]
            within = order[dxy[order] <= self._sleeve_edge_radius]
            if within.size == 0:
                nearest = order[0]
                if dxy[nearest] > self._sleeve_edge_activation_radius:
                    layer_msgs.append(f"{layer_name}: nearest {dxy[nearest]:.3f}m too far")
                    continue
                # Guarantee at least one particle per layer even outside radius.
                within = order[:1]
                layer_msgs.append(f"{layer_name}: n=1 d={dxy[nearest]:.3f}m (beyond radius)")
            else:
                within = within[:per_layer]
                layer_msgs.append(f"{layer_name}: n={within.size} d={dxy[within[0]]:.3f}m")
            selected.extend(int(i) for i in within)
        if not selected:
            print(f"  [SleeveEdgeGrasp:{self._name}] no layer particles near TCP; falling back to shape selection")
            return None
        vid = np.array(sorted(set(selected)), dtype=int)
        if 0 < self._max_grasp_vertices < vid.shape[0]:
            print(
                f"  [SleeveEdgeGrasp:{self._name}] ISAAC_MAX_GRASP_VERTICES={self._max_grasp_vertices}"
                f" raised to {vid.shape[0]} to keep both cloth layers grasped"
            )
        print(
            f"  [SleeveEdgeGrasp:{self._name}] side={side} tcp->sleeve_mid={mid_dist:.3f}m"
            f" verts={vid.shape[0]} [{'; '.join(layer_msgs)}]"
        )
        return vid

    def compute_grasp_vertices(self, tf: np.ndarray):
        # Cloth positions are in Isaac Sim world frame (table at z=z_ref).
        # Convert to FoldNet frame: negate X,Y and subtract z_ref from z.
        xyz_isaac = self._env._get_cloth_xyz()
        if xyz_isaac is None or xyz_isaac.size == 0:
            print(f"  [Picker:{self._name}] WARNING: cloth points empty/None")
            return np.array([], dtype=int), np.zeros((0, 3)), np.zeros(0)

        z_ref = getattr(self._env, "_z_ref", config.TABLE_HEIGHT)
        xyz_foldnet = xyz_isaac.copy()
        xyz_foldnet[:, 0] *= -1
        xyz_foldnet[:, 1] *= -1
        xyz_foldnet[:, 2] -= z_ref

        # Diagnostic: print cloth z range so we know if cloth fell through the table
        z_min, z_max = xyz_foldnet[:, 2].min(), xyz_foldnet[:, 2].max()

        # Convert candidate particles into the current TCP frame. Selection is
        # local to the gripper tip so the capture window can stay narrow on the
        # axis where sleeve/torso points are easy to accidentally include.
        xyz_picker_frame = (
            np.concatenate([xyz_foldnet, np.ones((xyz_foldnet.shape[0], 1), dtype=np.float32)], axis=1)
            @ np.linalg.inv(tf).T
        )[:, :3]
        sleeve_vid = (
            self._select_sleeve_edge_vertices(tf, xyz_foldnet)
            if self._sleeve_edge_grasp else None
        )
        if sleeve_vid is not None:
            # Semantic selection: keep every edge anchor, bypass the max-verts cap.
            vid = sleeve_vid
            threshold_label = (
                f"sleeve_edge(act={self._sleeve_edge_activation_radius:.3f}m,"
                f" r={self._sleeve_edge_radius:.3f}m)"
            )
            candidate_count = int(vid.shape[0])
        else:
            threshold = np.maximum(np.asarray(self._grasp_threshold, dtype=np.float32), 1e-6)
            normalized = xyz_picker_frame / threshold
            if self._grasp_selection_shape == "sphere":
                score = np.linalg.norm(xyz_picker_frame, axis=1)
                radius = float(np.max(threshold))
                vid = np.where(score < radius)[0]
                threshold_label = f"r={radius:.2f}m"
            elif self._grasp_selection_shape == "ellipsoid":
                score = np.linalg.norm(normalized, axis=1)
                vid = np.where(score < 1.0)[0]
                threshold_label = f"ellipsoid={np.round(threshold, 3)}m"
            else:
                score = np.linalg.norm(normalized, axis=1)
                vid = np.where(np.all(np.abs(xyz_picker_frame) < threshold, axis=1))[0]
                threshold_label = f"box={np.round(threshold, 3)}m"
            candidate_count = int(vid.shape[0])
            if self._max_grasp_vertices > 0 and candidate_count > self._max_grasp_vertices:
                order = np.argsort(score[vid])[:self._max_grasp_vertices]
                vid = vid[order]

        n = xyz_foldnet.shape[0]
        mass_inv = np.ones(n, dtype=np.float32)
        if vid.shape[0] > 0:
            selected_center = np.mean(xyz_foldnet[vid], axis=0)
            selected_spread = np.max(xyz_foldnet[vid], axis=0) - np.min(xyz_foldnet[vid], axis=0)
            selected_dist = np.linalg.norm(selected_center - tf[:3, 3])
            selected_msg = (
                f" center={np.round(selected_center, 3)}"
                f" spread={np.round(selected_spread, 3)}"
                f" center_dist={selected_dist:.3f}m"
            )
        else:
            selected_msg = ""
        print(f"  [Picker:{self._name}] {vid.shape[0]}/{candidate_count}/{n} verts within {threshold_label} "
              f"of TCP {np.round(tf[:3, 3], 3)}{selected_msg} | cloth z=[{z_min:.3f},{z_max:.3f}]")
        return vid, xyz_picker_frame, mass_inv

    def set_action(self, action: float):
        self._prev_val = self._val
        self._val = self.OPEN if self.is_open_action(action) else self.CLOSE
        self._val_float = float(action)

        if (self._prev_val, self._val) == (self.OPEN, self.CLOSE):
            vid, rel, mass_inv = self.compute_grasp_vertices(self._tf)
            xyz_isaac = self._env._get_cloth_xyz()
            block_initial_pos_isaac = (
                np.mean(xyz_isaac[vid, :], axis=0).astype(np.float32)
                if xyz_isaac is not None and vid.shape[0] > 0 else np.zeros(3, dtype=np.float32)
            )
            if self._block_attachment_use_physx:
                # Real PhysX attachment: the weld region is the block's own
                # collision volume, so the block sits exactly at the TCP (the
                # gripper's contact point) instead of the selected-vertex
                # centroid, and keeps zero offset while tracking the TCP.
                block_initial_pos_isaac = self._tcp_position_isaac()
            block_particle_offsets_isaac = (
                (xyz_isaac[vid, :] - block_initial_pos_isaac).astype(np.float32)
                if xyz_isaac is not None and vid.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
            )
            self._grasp = dict(
                vid=vid,
                xyz_offset=rel[vid, :] if vid.shape[0] > 0 else np.zeros((0, 3)),
                block_xyz_offset=(
                    np.zeros(3, dtype=np.float32)
                    if self._block_attachment_use_physx
                    else np.mean(rel[vid, :], axis=0).astype(np.float32)
                    if vid.shape[0] > 0 else np.zeros(3, dtype=np.float32)
                ),
                block_initial_pos_isaac=block_initial_pos_isaac,
                block_particle_offsets_isaac=block_particle_offsets_isaac,
                old_mass_inv=mass_inv[vid] if vid.shape[0] > 0 else np.zeros(0),
                old_masses=None,
                grasp_rot=self._tf[:3, :3].copy(),
            )
            self._block_debug_frames_left = 20
            if vid.shape[0] > 0:
                offset_mean = np.mean(self._grasp["xyz_offset"], axis=0)
                offset_norm = float(np.linalg.norm(offset_mean))
                print(
                    f"  [Picker:{self._name}] grasp debug:"
                    f" tcp_foldnet={np.round(self._tf[:3, 3], 4)},"
                    f" tcp_isaac={np.round(self._tcp_position_isaac(), 4)},"
                    f" offset_mean={np.round(offset_mean, 4)},"
                    f" offset_norm={offset_norm:.4f}m,"
                    f" squeeze={np.round(self._squeeze_factor, 3)}"
                )
            self._freeze_grasped_particles()
            self._env._update_runtime_cloth_damping()
            if self._block_attachment_enabled:
                if vid.shape[0] > 0:
                    self._create_block_attachment()
                else:
                    print(f"  [Picker:{self._name}] block attachment skipped: no nearby cloth verts")
            elif self._attachment_enabled:
                # PhysX attachment uses filterDistance to select the actual cloth
                # vertices. Our proximity check above is only diagnostic.
                if vid.shape[0] > 0:
                    self._create_attachment()
                    self._attachment_handover_frames_left = self._attachment_release_steps
                else:
                    print(f"  [Picker:{self._name}] attachment skipped: no nearby cloth verts")
            else:
                print(f"  [Picker:{self._name}] attachment disabled by ISAAC_ATTACHMENT_HANDS")
        elif (self._prev_val, self._val) == (self.CLOSE, self.OPEN):
            if (
                self._release_damp_frames > 0
                and self._grasp is not None
                and self._grasp["vid"].shape[0] > 0
            ):
                self._released_vid = self._grasp["vid"].copy()
                self._release_damp_frames_left = self._release_damp_frames
            self._restore_grasped_particles()
            self._destroy_attachment()
            self._destroy_block_attachment()
            self._grasp = None
            self._env._update_runtime_cloth_damping()

    def _apply_release_damping(self):
        """For a few frames after release, strongly damp velocities around the
        released particles so stored squeeze/constraint energy dissipates
        locally instead of throwing the sleeve back."""
        if self._release_damp_frames_left <= 0 or self._released_vid is None:
            return
        cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None or not (
            hasattr(cloth_view, "get_velocities") and hasattr(cloth_view, "set_velocities")
        ):
            self._release_damp_frames_left = 0
            return
        try:
            N = cloth_view.max_particles_per_cloth
            raw = cloth_view.get_positions()
            flat = self._env._physics_data_to_numpy(raw).reshape(N, 3)
            raw_vel = cloth_view.get_velocities()
            vel = self._env._physics_data_to_numpy(raw_vel).reshape(N, 3).copy()
            anchors = flat[self._released_vid]
            dist = np.min(
                np.linalg.norm(flat[:, None, :] - anchors[None, :, :], axis=2), axis=1
            )
            radius = max(self._kinematic_neighbor_radius, 0.04)
            mask = dist < radius
            vel[mask] *= 0.15
            vel_out = self._env._physics_positions_to_backend(vel.reshape(1, N * 3), raw_vel)
            indices = self._env._physics_indices_to_backend([0], raw_vel)
            cloth_view.set_velocities(vel_out, indices)
        except Exception as e:
            print(f"  [ReleaseDamp:{self._name}] skipped: {e}")
            self._release_damp_frames_left = 0
            return
        self._release_damp_frames_left -= 1
        if self._release_damp_frames_left == 0:
            self._released_vid = None

    def _freeze_grasped_particles(self):
        if self._block_attachment_enabled and self._block_attachment_use_physx:
            return
        if (
            not self._kinematic_grasp_enabled
            and not (self._block_attachment_enabled and not self._block_attachment_use_physx)
        ) or self._grasp is None:
            return
        vid = self._grasp["vid"]
        if vid.shape[0] == 0:
            return
        if self._kinematic_mass_scale <= 1.0:
            print(
                f"  [KinematicGrasp:{self._name}] mass freeze disabled"
                " (ISAAC_KINEMATIC_MASS_SCALE<=1)"
            )
            return
        cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None or not hasattr(cloth_view, "get_masses") or not hasattr(cloth_view, "set_masses"):
            return
        try:
            raw = cloth_view.get_masses()
            masses = self._env._physics_data_to_numpy(raw).reshape(1, -1).copy()
            self._grasp["old_masses"] = masses[0, vid].copy()
            masses[0, vid] = masses[0, vid] * self._kinematic_mass_scale
            out = self._env._physics_array_to_backend(masses, raw, dtype=np.float32)
            indices = self._env._physics_indices_to_backend([0], raw)
            cloth_view.set_masses(out, indices)
            print(
                f"  [KinematicGrasp:{self._name}] increased masses for {vid.shape[0]} particles"
            )
        except Exception as e:
            print(f"  [KinematicGrasp:{self._name}] mass freeze skipped: {e}")

    def _restore_grasped_particles(self):
        if self._grasp is None:
            return
        vid = self._grasp["vid"]
        old_masses = self._grasp.get("old_masses")
        if vid.shape[0] == 0 or old_masses is None:
            return
        cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None or not hasattr(cloth_view, "get_masses") or not hasattr(cloth_view, "set_masses"):
            return
        try:
            raw = cloth_view.get_masses()
            masses = self._env._physics_data_to_numpy(raw).reshape(1, -1).copy()
            masses[0, vid] = old_masses.astype(np.float32)
            out = self._env._physics_array_to_backend(masses, raw, dtype=np.float32)
            indices = self._env._physics_indices_to_backend([0], raw)
            cloth_view.set_masses(out, indices)
        except Exception as e:
            print(f"  [KinematicGrasp:{self._name}] mass restore skipped: {e}")

    def _get_attachment_filter_distance(self) -> float:
        filter_distance = self._attachment_filter_distance
        if self._grasp is not None and self._grasp["xyz_offset"].size > 0:
            squeezed = self._grasp["xyz_offset"] * self._squeeze_factor
            local_radius = float(np.max(np.linalg.norm(squeezed, axis=1)))
            filter_distance = max(
                filter_distance,
                local_radius + config.GARMENT_PARTICLE_CONTACT_OFFSET * 2.0,
            )
        return min(filter_distance, self._attachment_filter_distance_max)

    def _create_attachment(self):
        stage = omni.usd.get_context().get_stage()
        gripper_path = self._find_prim_path_by_suffix(f"{self._name}_Fixed_Gripper")
        cloth_path = getattr(self._env, "_cloth_attachment_prim_path", "/World/Cloth/garmentMesh")

        if gripper_path is None:
            print(f"  Warning: cannot create grasp attachment, missing {self._name}_Fixed_Gripper prim")
            return
        if not stage.GetPrimAtPath(cloth_path).IsValid():
            print(f"  Warning: cannot create grasp attachment, missing cloth prim {cloth_path}")
            return

        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, self.attachment_path)
        attachment.GetActor0Rel().SetTargets([Sdf.Path(gripper_path)])
        attachment.GetActor1Rel().SetTargets([Sdf.Path(cloth_path)])
        # Keep the attachment local to the pinch patch; a wide search radius turns
        # the cloth into a rigid welded island and excites long-range ringing.
        filter_distance = self._get_attachment_filter_distance()
        attachment_prim = stage.GetPrimAtPath(self.attachment_path)
        attachment_prim.CreateAttribute("physxPhysicsAttachment:filterType",
                                        Sdf.ValueTypeNames.Int).Set(0)
        attachment_prim.CreateAttribute("physxPhysicsAttachment:filterDistance",
                                        Sdf.ValueTypeNames.Float).Set(filter_distance)
        self._attachment_is_active = True
        print(
            f"  [Picker:{self._name}] attachment created: {gripper_path} ↔ {cloth_path}"
            f" (filterDistance={filter_distance:.3f} m)"
        )

    def _find_prim_path_by_suffix(self, suffix: str):
        stage = omni.usd.get_context().get_stage()
        direct = f"{config.ROBOT_PRIM_PATH}/{suffix}"
        if stage.GetPrimAtPath(direct).IsValid():
            return direct
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(config.ROBOT_PRIM_PATH) and path.endswith(f"/{suffix}"):
                return path
        return None

    def _destroy_attachment(self):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(self.attachment_path).IsValid():
            stage.RemovePrim(self.attachment_path)
        self._attachment_is_active = False
        self._attachment_handover_frames_left = 0

    def _block_target_position_isaac(self):
        if self._grasp is None:
            return None
        if self._block_last_pos is None:
            initial_pos = self._grasp.get("block_initial_pos_isaac")
            if initial_pos is not None:
                return initial_pos.astype(np.float32)
        offset = self._grasp.get("block_xyz_offset")
        if offset is None:
            offset = np.zeros(3, dtype=np.float32)
        pos_foldnet = (np.append(offset.astype(np.float32), 1.0) @ self._grasp_tf().T)[:3]
        z_ref = getattr(self._env, "_z_ref", config.TABLE_HEIGHT)
        pos_isaac = pos_foldnet.astype(np.float32)
        pos_isaac[0] *= -1.0
        pos_isaac[1] *= -1.0
        pos_isaac[2] += z_ref
        return pos_isaac

    def _tcp_position_isaac(self):
        pos_foldnet = self._tf[:3, 3].astype(np.float32)
        z_ref = getattr(self._env, "_z_ref", config.TABLE_HEIGHT)
        pos_isaac = pos_foldnet.copy()
        pos_isaac[0] *= -1.0
        pos_isaac[1] *= -1.0
        pos_isaac[2] += z_ref
        return pos_isaac

    def _set_block_pose_usd(self, pos):
        if (
            self._block_attachment_use_physx
            and getattr(self._env, "_pt_sim_view", None) is not None
        ):
            # GPU pipeline: teleport through the tensor API; the USD write below
            # is kept for rendering and the attachment parser.
            self._set_block_pose_view(pos)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        if self._block_translate_op is None:
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    self._block_translate_op = op
                    break
            if self._block_translate_op is None:
                self._block_translate_op = xformable.AddTranslateOp()
        self._block_translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _set_block_visibility(self, visible: bool):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if prim.IsValid():
            visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(visibility)

    def _marker_path_and_op(self, marker_kind):
        if marker_kind == "target":
            return self._block_target_marker_path, self._block_target_marker_translate_op
        if marker_kind == "tcp":
            return self._block_tcp_marker_path, self._block_tcp_marker_translate_op
        return self._block_marker_path, self._block_marker_translate_op

    def _set_marker_op(self, marker_kind, translate_op):
        if marker_kind == "target":
            self._block_target_marker_translate_op = translate_op
        elif marker_kind == "tcp":
            self._block_tcp_marker_translate_op = translate_op
        else:
            self._block_marker_translate_op = translate_op

    def _set_marker_pose_usd(self, pos, target_marker=False, marker_kind=None):
        stage = omni.usd.get_context().get_stage()
        if marker_kind is None:
            marker_kind = "target" if target_marker else "grasp"
        path, translate_op = self._marker_path_and_op(marker_kind)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        if translate_op is None:
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            self._set_marker_op(marker_kind, translate_op)
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _set_marker_visibility(self, visible: bool, target_marker=False, marker_kind=None):
        stage = omni.usd.get_context().get_stage()
        if marker_kind is None:
            marker_kind = "target" if target_marker else "grasp"
        path, _ = self._marker_path_and_op(marker_kind)
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(visibility)

    def _ensure_block_marker_prim(self, target_marker=False, marker_kind=None):
        stage = omni.usd.get_context().get_stage()
        if marker_kind is None:
            marker_kind = "target" if target_marker else "grasp"
        path, _ = self._marker_path_and_op(marker_kind)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            sphere = UsdGeom.Sphere.Define(stage, path)
            radius_scale = {"grasp": 1.0, "target": 0.75, "tcp": 0.55}.get(marker_kind, 1.0)
            radius = self._block_debug_marker_size * radius_scale
            sphere.CreateRadiusAttr(float(radius))
            color = {
                "grasp": Gf.Vec3f(0.0, 1.0, 0.1),
                "target": Gf.Vec3f(0.0, 0.8, 1.0),
                "tcp": Gf.Vec3f(1.0, 0.0, 1.0),
            }.get(marker_kind, Gf.Vec3f(0.0, 1.0, 0.1))
            sphere.CreateDisplayColorAttr().Set([color])
            prim = sphere.GetPrim()
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            self._set_marker_op(marker_kind, translate_op)
        else:
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    self._set_marker_op(marker_kind, op)
                    break
        self._set_marker_pose_usd(self._block_park_pos, marker_kind=marker_kind)
        self._set_marker_visibility(False, marker_kind=marker_kind)

    def _set_block_collision_enabled(self, enabled: bool):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if not prim.IsValid():
            return
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr.IsValid():
            attr.Set(bool(enabled))
        else:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(bool(enabled))

    def _ensure_block_prim(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if not prim.IsValid():
            if self._block_attachment_shape == "sphere":
                shape = UsdGeom.Sphere.Define(stage, self.block_path)
                shape.CreateRadiusAttr(0.5)
            else:
                shape = UsdGeom.Cube.Define(stage, self.block_path)
                shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            prim = shape.GetPrim()
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            self._block_translate_op = xformable.AddTranslateOp()
            scale_op = xformable.AddScaleOp()
            scale_op.Set(
                Gf.Vec3f(
                    self._block_attachment_size,
                    self._block_attachment_size,
                    self._block_attachment_size,
                )
            )
            UsdPhysics.CollisionAPI.Apply(prim)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            # GarmentLab recipe: PhysX attachments need a dynamic (non-kinematic)
            # rigid driven by velocity; the fake-grasp path keeps it kinematic.
            rigid_body_api.CreateKinematicEnabledAttr().Set(
                not self._block_attachment_use_physx
            )
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(float(self._block_attachment_mass))
            physx_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_body_api.CreateDisableGravityAttr(True)
        else:
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    self._block_translate_op = op
                    break

        self._set_block_pose_usd(self._block_park_pos)
        self._set_block_collision_enabled(False)
        self._set_block_visibility(False)
        if self._block_debug_marker_visible:
            self._ensure_block_marker_prim()
            self._ensure_block_marker_prim(target_marker=True)
            self._ensure_block_marker_prim(marker_kind="tcp")

    def _get_block_pos_usd(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                value = op.Get()
                if value is not None:
                    return np.array([value[0], value[1], value[2]], dtype=np.float32)
        return None

    def _move_attachment_block(self, apply_kinematic=True):
        if not self._block_attachment_is_active:
            return
        target = self._block_target_position_isaac()
        if target is None:
            return

        dt = 1.0 / 120.0
        try:
            dt = float(self._env._world.get_physics_dt())
        except Exception:
            pass

        if self._block_attachment_use_physx:
            # GarmentLab-style drive: the block is a dynamic rigid pulled toward
            # the gripper by velocity commands; the attachment constraint drags
            # the cloth inside the solver (no position teleports).
            cur = self._get_block_pos_view()
            if cur is None:
                cur = self._get_block_pos_usd()
            if cur is None:
                cur = self._block_last_pos if self._block_last_pos is not None else target
            velocity = ((target - cur) / (max(dt, 1e-6) * 3.0)).astype(np.float32)
            self._set_block_velocity_usd(velocity)
            self._block_last_pos = cur.astype(np.float32)
            if self._block_attach_diag_frames > 0:
                self._block_attach_diag_frames -= 1
                if self._block_attach_diag_frames == 0:
                    self._log_block_attachment_points()
            return

        if self._block_last_pos is None:
            pos = target
        else:
            alpha = self._block_attachment_follow_blend
            pos = self._block_last_pos * (1.0 - alpha) + target * alpha

        velocity = np.zeros(3, dtype=np.float32)
        if self._block_last_pos is not None and dt > 0.0:
            velocity = ((pos - self._block_last_pos) / dt).astype(np.float32)

        self._set_block_pose_usd(pos)
        self._set_block_velocity_usd(velocity)
        self._block_last_pos = pos.astype(np.float32)
        if apply_kinematic:
            self._apply_block_kinematic_grasp()

    def _smooth_block_particle_target(self, target):
        target = target.astype(np.float32, copy=False)
        prev = self._block_smoothed_particle_target
        if prev is None or prev.shape != target.shape:
            self._block_smoothed_particle_target = target.copy()
            return target

        if self._block_target_deadband > 0.0:
            delta_norm = np.linalg.norm(target - prev, axis=1, keepdims=True)
            target = np.where(delta_norm < self._block_target_deadband, prev, target)

        alpha = self._block_target_blend
        smoothed = prev * (1.0 - alpha) + target * alpha
        self._block_smoothed_particle_target = smoothed.astype(np.float32, copy=False)
        return self._block_smoothed_particle_target

    def _apply_block_kinematic_grasp(self):
        if self._grasp is None or self._block_last_pos is None:
            return
        vid = self._grasp["vid"]
        if vid.shape[0] == 0:
            return

        cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None and not getattr(self, '_block_view_retry_done', False):
            self._env._try_init_cloth_view()
            self._block_view_retry_done = True
            cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None:
            if not getattr(self, '_warned_no_block_view', False):
                print(
                    f"  [BlockKinematic:{self._name}] WARNING: cloth_view is None — "
                    "block kinematic grasp disabled"
                )
                self._warned_no_block_view = True
            return
        self._warned_no_block_view = False

        try:
            N = cloth_view.max_particles_per_cloth
            raw = cloth_view.get_positions()
            flat = self._env._physics_data_to_numpy(raw).reshape(N, 3).copy()

            offsets_picker = self._grasp["xyz_offset"] * self._squeeze_factor
            target_foldnet = (
                np.concatenate(
                    [offsets_picker, np.ones((offsets_picker.shape[0], 1), dtype=np.float32)],
                    axis=1,
                )
                @ self._grasp_tf().T
            )[:, :3]

            z_ref = getattr(self._env, "_z_ref", config.TABLE_HEIGHT)
            target = target_foldnet.astype(np.float32)
            target[:, 0] *= -1.0
            target[:, 1] *= -1.0
            target[:, 2] += z_ref
            target = self._smooth_block_particle_target(target)
            current_before = flat[vid].copy()
            if self._block_diagnostic_snap:
                alpha = 1.0
                desired = target.copy()
            else:
                alpha = self._block_kinematic_blend
                desired = flat[vid] * (1.0 - alpha) + target * alpha
            if self._block_kinematic_max_step > 0.0 and not self._block_diagnostic_snap:
                delta = desired - flat[vid]
                step_norm = np.linalg.norm(delta, axis=1, keepdims=True)
                if (
                    self._block_kinematic_catchup_error > 0.0
                    and self._block_kinematic_catchup_step > self._block_kinematic_max_step
                ):
                    # Catchup must be gated by the raw tracking error, not the
                    # blended step (which is already scaled down by alpha).
                    err_norm = np.linalg.norm(target - flat[vid], axis=1, keepdims=True)
                    catchup_t = np.clip(
                        (err_norm - self._block_kinematic_catchup_error)
                        / max(self._block_kinematic_catchup_error, 1e-6),
                        0.0,
                        1.0,
                    )
                    max_step = (
                        self._block_kinematic_max_step * (1.0 - catchup_t)
                        + self._block_kinematic_catchup_step * catchup_t
                    )
                else:
                    max_step = self._block_kinematic_max_step
                step_scale = np.minimum(
                    1.0,
                    max_step / np.maximum(step_norm, 1e-6),
                )
                desired = flat[vid] + delta * step_scale
            flat[vid] = desired
            if self._block_debug_marker_visible:
                self._ensure_block_marker_prim()
                self._ensure_block_marker_prim(target_marker=True)
                self._ensure_block_marker_prim(marker_kind="tcp")
                self._set_marker_pose_usd(np.mean(desired, axis=0))
                self._set_marker_pose_usd(np.mean(target, axis=0), target_marker=True)
                self._set_marker_pose_usd(self._tcp_position_isaac(), marker_kind="tcp")
                self._set_marker_visibility(True)
                self._set_marker_visibility(True, target_marker=True)
                self._set_marker_visibility(True, marker_kind="tcp")
            if self._block_debug_frames_left > 0:
                current_mean = np.mean(current_before, axis=0)
                desired_mean = np.mean(desired, axis=0)
                target_mean = np.mean(target, axis=0)
                tcp_isaac = self._tcp_position_isaac()
                print(
                    f"  [BlockKinematic:{self._name}] track debug:"
                    f" current={np.round(current_mean, 4)},"
                    f" desired={np.round(desired_mean, 4)},"
                    f" target={np.round(target_mean, 4)},"
                    f" tcp={np.round(tcp_isaac, 4)},"
                    f" curr_target_err={np.linalg.norm(current_mean - target_mean):.4f}m,"
                    f" target_tcp_err={np.linalg.norm(target_mean - tcp_isaac):.4f}m,"
                    f" max_step={self._block_kinematic_max_step:.4f},"
                    f" catchup_step={self._block_kinematic_catchup_step:.4f},"
                    f" blend={alpha:.2f},"
                    f" diag_snap={'on' if self._block_diagnostic_snap else 'off'}"
                )
                self._block_debug_frames_left -= 1

            out = self._env._physics_positions_to_backend(flat.reshape(1, N * 3), raw)
            indices = self._env._physics_indices_to_backend([0], raw)
            cloth_view.set_positions(out, indices)
            if self._block_debug_frames_left > 0:
                try:
                    verify_raw = cloth_view.get_positions()
                    verify = self._env._physics_data_to_numpy(verify_raw).reshape(N, 3)
                    verify_mean = np.mean(verify[vid], axis=0)
                    target_mean = np.mean(target, axis=0)
                    print(
                        f"  [BlockKinematic:{self._name}] set verify:"
                        f" verify={np.round(verify_mean, 4)},"
                        f" verify_target_err={np.linalg.norm(verify_mean - target_mean):.4f}m"
                    )
                except Exception as verify_exc:
                    print(f"  [BlockKinematic:{self._name}] set verify skipped: {verify_exc}")

            if hasattr(cloth_view, "get_velocities") and hasattr(cloth_view, "set_velocities"):
                raw_vel = cloth_view.get_velocities()
                vel = self._env._physics_data_to_numpy(raw_vel).reshape(N, 3).copy()
                vel[vid] = 0.0
                if self._kinematic_neighbor_radius > 0.0:
                    dist_to_grasp = np.min(
                        np.linalg.norm(flat[:, None, :] - flat[vid][None, :, :], axis=2),
                        axis=1,
                    )
                    neighbor_mask = dist_to_grasp < self._kinematic_neighbor_radius
                    vel[neighbor_mask] *= self._kinematic_neighbor_velocity_scale
                    vel[vid] = 0.0
                vel_out = self._env._physics_positions_to_backend(vel.reshape(1, N * 3), raw_vel)
                cloth_view.set_velocities(vel_out, indices)
            if not getattr(self, '_logged_block_grasp_ok', False):
                print(
                    f"  [BlockKinematic:{self._name}] first set_positions OK "
                    f"(N={N}, moving {vid.shape[0]} particles,"
                    f" blend={self._block_kinematic_blend:.2f},"
                    f" max_step={self._block_kinematic_max_step:.3f},"
                    f" target_blend={self._block_target_blend:.2f},"
                    f" deadband={self._block_target_deadband:.4f})"
                )
                self._logged_block_grasp_ok = True
        except Exception as e:
            print(f"  [BlockKinematic:{self._name}] ERROR: {e}")

    def _get_block_rigid_view(self):
        """Tensor-API view of the block rigid body. Required on the GPU pipeline
        (eENABLE_DIRECT_GPU_API): CPU/USD velocity or pose writes are illegal."""
        if getattr(self, "_block_rigid_view", None) is not None:
            return self._block_rigid_view
        sim_view = getattr(self._env, "_pt_sim_view", None)
        if sim_view is None:
            self._env._try_init_cloth_view()
            sim_view = getattr(self._env, "_pt_sim_view", None)
        if sim_view is None:
            return None
        try:
            self._block_rigid_view = sim_view.create_rigid_body_view(self.block_path)
            print(f"  [Picker:{self._name}] block rigid view OK: {self.block_path}")
        except Exception as e:
            print(f"  [Picker:{self._name}] block rigid view FAILED: {e}")
            self._block_rigid_view = None
        return self._block_rigid_view

    def _set_block_velocity_view(self, velocity) -> bool:
        view = self._get_block_rigid_view()
        if view is None:
            return False
        try:
            like = view.get_velocities()
            vel6 = np.zeros((1, 6), dtype=np.float32)
            vel6[0, :3] = np.asarray(velocity, dtype=np.float32)
            data = self._env._physics_positions_to_backend(vel6, like)
            indices = self._env._physics_indices_to_backend([0], like)
            view.set_velocities(data, indices)
            return True
        except Exception as e:
            print(f"  [Picker:{self._name}] block view set_velocities failed: {e}")
            return False

    def _set_block_pose_view(self, pos) -> bool:
        view = self._get_block_rigid_view()
        if view is None:
            return False
        try:
            like = view.get_transforms()
            tf7 = np.zeros((1, 7), dtype=np.float32)
            tf7[0, :3] = np.asarray(pos, dtype=np.float32)
            tf7[0, 6] = 1.0  # identity quaternion, xyzw order
            data = self._env._physics_positions_to_backend(tf7, like)
            indices = self._env._physics_indices_to_backend([0], like)
            view.set_transforms(data, indices)
            return True
        except Exception as e:
            print(f"  [Picker:{self._name}] block view set_transforms failed: {e}")
            return False

    def _get_block_pos_view(self):
        view = self._get_block_rigid_view()
        if view is None:
            return None
        try:
            tf = self._env._physics_data_to_numpy(view.get_transforms()).reshape(-1)
            return tf[:3].astype(np.float32)
        except Exception:
            return None

    def _set_block_velocity_usd(self, velocity):
        if self._block_attachment_use_physx:
            # GPU pipeline: USD velocity writes raise PhysX errors; tensor API only.
            self._set_block_velocity_view(velocity)
            return
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_path)
        if not prim.IsValid():
            return
        prim.CreateAttribute("physics:velocity", Sdf.ValueTypeNames.Vector3f).Set(
            Gf.Vec3f(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        )
        prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f).Set(
            Gf.Vec3f(0.0, 0.0, 0.0)
        )

    def _create_block_attachment(self):
        stage = omni.usd.get_context().get_stage()
        cloth_path = getattr(self._env, "_cloth_attachment_prim_path", "/World/Cloth/garmentMesh")
        if not stage.GetPrimAtPath(cloth_path).IsValid():
            print(f"  Warning: cannot create block attachment, missing cloth prim {cloth_path}")
            return

        pos = self._block_target_position_isaac()
        if pos is None:
            print(f"  [Picker:{self._name}] block attachment skipped: no target position")
            return

        self._destroy_block_attachment()
        self._ensure_block_prim()
        self._set_block_pose_usd(pos)
        self._set_block_visibility(self._block_attachment_visible)
        block_prim = stage.GetPrimAtPath(self.block_path)
        if not block_prim.IsValid():
            print(f"  [Picker:{self._name}] block attachment skipped: block prim was not created")
            return

        if not self._block_attachment_use_physx:
            self._set_block_collision_enabled(False)
            self._block_attachment_is_active = True
            self._block_last_pos = pos.astype(np.float32)
            self._block_smoothed_particle_target = None
            if self._block_debug_marker_visible:
                self._ensure_block_marker_prim()
                self._ensure_block_marker_prim(target_marker=True)
                self._ensure_block_marker_prim(marker_kind="tcp")
                self._set_marker_visibility(True)
                self._set_marker_visibility(True, target_marker=True)
                self._set_marker_visibility(True, marker_kind="tcp")
            self._apply_block_kinematic_grasp()
            print(
                f"  [Picker:{self._name}] block kinematic grasp created: {self.block_path}"
                f" (moving {self._grasp['vid'].shape[0]} particles,"
                f" pos={np.round(pos, 3)}, blend={self._block_kinematic_blend:.2f},"
                f" poststep_only={'on' if self._block_kinematic_poststep_only else 'off'},"
                f" marker={'on' if self._block_debug_marker_visible else 'off'})"
            )
            return

        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, self.block_attachment_path)
        attachment_prim = attachment.GetPrim()
        if self._block_attachment_auto:
            # GarmentLab-style auto attachment. In our particle-cloth GPU setup this
            # can be too broad, so the safer default below uses explicit filtering.
            attachment.GetActor0Rel().SetTargets([Sdf.Path(cloth_path)])
            attachment.GetActor1Rel().SetTargets([Sdf.Path(self.block_path)])
            auto_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_prim)
            auto_api.CreateDeformableVertexOverlapOffsetAttr(defaultValue=self._block_attachment_overlap)
            auto_api.CreateCollisionFilteringOffsetAttr(defaultValue=self._block_attachment_overlap)
        else:
            # Filtered rigid<->cloth attachment: keep the capture radius tiny so
            # PhysX cannot weld a large island of the shirt to the block.
            attachment.GetActor0Rel().SetTargets([Sdf.Path(self.block_path)])
            attachment.GetActor1Rel().SetTargets([Sdf.Path(cloth_path)])
            attachment_prim.CreateAttribute(
                "physxPhysicsAttachment:filterType",
                Sdf.ValueTypeNames.Int,
            ).Set(0)
            attachment_prim.CreateAttribute(
                "physxPhysicsAttachment:filterDistance",
                Sdf.ValueTypeNames.Float,
            ).Set(float(self._block_attachment_overlap))
        # Collision stays enabled (auto attachment captures vertices from the
        # block's collision geometry) but must never push the cloth or table.
        filtered = UsdPhysics.FilteredPairsAPI.Apply(block_prim)
        pairs_rel = filtered.CreateFilteredPairsRel()
        for filtered_path in (
            cloth_path,
            "/World/Cloth/particleSystem",
            "/World/Table",
            "/World/TableSurface",
        ):
            if stage.GetPrimAtPath(filtered_path).IsValid():
                pairs_rel.AddTarget(Sdf.Path(filtered_path))
        self._set_block_collision_enabled(self._block_attachment_collision)
        self._block_attachment_is_active = True
        self._block_last_pos = pos.astype(np.float32)
        # Log the auto-computed attachment points a few frames later, once the
        # physics parser has processed the new attachment prim.
        self._block_attach_diag_frames = 8
        tcp = self._tcp_position_isaac()
        print(
            f"  [Picker:{self._name}] block attachment created: {cloth_path} <-> {self.block_path}"
            f" (size={self._block_attachment_size:.3f}m, overlap={self._block_attachment_overlap:.3f}m,"
            f" collision={'on' if self._block_attachment_collision else 'off'},"
            f" auto={'on' if self._block_attachment_auto else 'off'},"
            f" pos={np.round(pos, 3)}, tcp={np.round(tcp, 3)},"
            f" tcp_dist={np.linalg.norm(pos - tcp):.3f}m)"
        )

    def _log_block_attachment_points(self):
        """Diagnostic: report the attachment points PhysX auto-computed, so we
        can see how many cloth vertices got welded and over what extent."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.block_attachment_path)
        if not prim.IsValid():
            print(f"  [Picker:{self._name}] attach diag: attachment prim missing")
            return
        attachment = PhysxSchema.PhysxPhysicsAttachment(prim)
        pts = attachment.GetPoints0Attr().Get()
        label = "points0"
        if not pts:
            pts = attachment.GetPoints1Attr().Get()
            label = "points1"
        blk = self._get_block_pos_view()
        if blk is None:
            blk = self._get_block_pos_usd()
        tcp = self._tcp_position_isaac()
        pose_msg = (
            f" block={np.round(blk, 3)} tcp={np.round(tcp, 3)}"
            f" block_tcp_dist={np.linalg.norm(blk - tcp):.3f}m"
            if blk is not None else ""
        )
        if not pts:
            print(
                f"  [Picker:{self._name}] attach diag: no attachment points authored"
                f" on the prim (auto points may stay PhysX-internal).{pose_msg}"
            )
            return
        arr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        center = arr.mean(axis=0)
        extent = arr.max(axis=0) - arr.min(axis=0)
        print(
            f"  [Picker:{self._name}] attach diag: {label} n={arr.shape[0]}"
            f" extent={np.round(extent, 3)} center={np.round(center, 3)}"
            f" (actor-local coords).{pose_msg}"
        )

    def _destroy_block_attachment(self):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(self.block_attachment_path).IsValid():
            stage.RemovePrim(self.block_attachment_path)
        if stage.GetPrimAtPath(self.block_path).IsValid():
            self._set_block_pose_usd(self._block_park_pos)
            self._set_block_velocity_usd(np.zeros(3, dtype=np.float32))
            self._set_block_collision_enabled(False)
            self._set_block_visibility(False)
        if stage.GetPrimAtPath(self._block_marker_path).IsValid():
            self._set_marker_pose_usd(self._block_park_pos)
            self._set_marker_visibility(False)
        if stage.GetPrimAtPath(self._block_target_marker_path).IsValid():
            self._set_marker_pose_usd(self._block_park_pos, target_marker=True)
            self._set_marker_visibility(False, target_marker=True)
        if stage.GetPrimAtPath(self._block_tcp_marker_path).IsValid():
            self._set_marker_pose_usd(self._block_park_pos, marker_kind="tcp")
            self._set_marker_visibility(False, marker_kind="tcp")
        self._block_attachment_is_active = False
        self._block_last_pos = None
        self._block_smoothed_particle_target = None

    def update_attachment_handover(self):
        if not self._attachment_is_active:
            return
        if not self._kinematic_grasp_enabled:
            return
        if self._attachment_release_steps <= 0:
            return
        if self._attachment_handover_frames_left > 0:
            self._attachment_handover_frames_left -= 1
        if self._attachment_handover_frames_left == 0:
            print(f"  [Picker:{self._name}] attachment handover -> kinematic grasp")
            self._destroy_attachment()

    def _should_apply_kinematic_grasp(self) -> bool:
        if not self._kinematic_grasp_enabled:
            return False
        if self._grasp is None or self._grasp["vid"].shape[0] == 0:
            return False
        if self._attachment_is_active:
            return False
        if self._block_attachment_is_active:
            return False
        return True

    def _apply_picker_tcp_offset(self, tf):
        tf_offset = np.asarray(tf, dtype=np.float32).copy()
        if np.any(np.abs(self._picker_tcp_offset) > 1e-7):
            tf_offset[:3, 3] = (
                np.append(self._picker_tcp_offset.astype(np.float32), 1.0)
                @ tf_offset.T
            )[:3]
        return tf_offset

    def set_tf(self, tf):
        self._tf[...] = self._apply_picker_tcp_offset(tf)

    def _grasp_tf(self) -> np.ndarray:
        """TCP transform used to drive grasped particles. With
        ISAAC_BLOCK_KINEMATIC_FREEZE_ROTATION=1 the grasp-time orientation is
        kept (translation still tracks the TCP) so wrist rotation during
        transport is not injected as torque into the pinched cloth."""
        if not self._block_kinematic_freeze_rotation or self._grasp is None:
            return self._tf
        rot = self._grasp.get("grasp_rot")
        if rot is None:
            return self._tf
        tf = self._tf.copy()
        tf[:3, :3] = rot
        return tf

    def step(self, tf=None):
        if tf is not None:
            self.set_tf(tf)
        if self._val == self.CLOSE:
            apply_block_grasp = (
                self._block_attachment_use_physx
                or not self._block_kinematic_poststep_only
            )
            self._move_attachment_block(apply_kinematic=apply_block_grasp)
        # Kinematic grasp: forcibly move grasped cloth particles to follow TCP.
        # This is the reliable alternative to PhysX attachment (which depends on
        # filterDistance matching the Fixed_Gripper geometry extent).
        if self._val == self.CLOSE and self._should_apply_kinematic_grasp():
            self._apply_kinematic_grasp()

    def _apply_kinematic_grasp(self):
        vid = self._grasp["vid"]
        if vid.shape[0] == 0:
            return
        offsets_picker = self._grasp["xyz_offset"] * self._squeeze_factor

        # Picker local frame -> current FoldNet world frame
        new_pos_foldnet = (
            np.concatenate(
                [offsets_picker, np.ones((offsets_picker.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            @ self._grasp_tf().T
        )[:, :3]

        # FoldNet → Isaac world: negate x,y, add z_ref
        z_ref = getattr(self._env, "_z_ref", config.TABLE_HEIGHT)
        new_pos_isaac = new_pos_foldnet.copy()
        new_pos_isaac[:, 0] *= -1
        new_pos_isaac[:, 1] *= -1
        new_pos_isaac[:, 2] += z_ref

        cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None and not getattr(self, '_view_retry_done', False):
            # Simulation has been running for many steps now — retry view creation.
            self._env._try_init_cloth_view()
            self._view_retry_done = True
            cloth_view = getattr(self._env, "_cloth_physics_view", None)
        if cloth_view is None:
            if not getattr(self, '_warned_no_view', False):
                print(f"  [KinematicGrasp:{self._name}] WARNING: cloth_view is None — "
                      "kinematic grasp disabled (tensor API unavailable)")
                self._warned_no_view = True
            return
        self._warned_no_view = False
        try:
            # get_positions() → (count, max_particles * 3) flat float32 array.
            # set_positions(data, indices) — data shape (selected_count, max_particles*3),
            #   indices = uint32 array of which cloths to update.
            N = cloth_view.max_particles_per_cloth
            raw = cloth_view.get_positions()
            flat = self._env._physics_data_to_numpy(raw).reshape(N, 3).copy()

            target = new_pos_isaac.astype(np.float32)
            alpha = self._kinematic_position_blend
            flat[vid] = flat[vid] * (1.0 - alpha) + target * alpha

            out = self._env._physics_positions_to_backend(flat.reshape(1, N * 3), raw)
            indices = self._env._physics_indices_to_backend([0], raw)
            cloth_view.set_positions(out, indices)

            # Best-effort damping: if the tensor view exposes particle velocities,
            # zero the grasped vertices so the solver does not keep injecting
            # spring energy into the same patch after we kinematically move it.
            if hasattr(cloth_view, "get_velocities") and hasattr(cloth_view, "set_velocities"):
                raw_vel = cloth_view.get_velocities()
                vel = self._env._physics_data_to_numpy(raw_vel).reshape(N, 3).copy()
                vel[vid] = 0.0
                if self._kinematic_neighbor_radius > 0.0 and vid.shape[0] > 0:
                    # Locally damp the patch around the grasp so the cloth does not
                    # ring like a plucked membrane after each kinematic correction.
                    dist_to_grasp = np.min(
                        np.linalg.norm(flat[:, None, :] - flat[vid][None, :, :], axis=2),
                        axis=1,
                    )
                    neighbor_mask = dist_to_grasp < self._kinematic_neighbor_radius
                    vel[neighbor_mask] *= self._kinematic_neighbor_velocity_scale
                    vel[vid] = 0.0
                vel_out = self._env._physics_positions_to_backend(vel.reshape(1, N * 3), raw_vel)
                cloth_view.set_velocities(vel_out, indices)
            if not getattr(self, '_logged_grasp_ok', False):
                print(f"  [KinematicGrasp:{self._name}] first set_positions OK "
                      f"(N={N}, moving {vid.shape[0]} particles)")
                self._logged_grasp_ok = True
        except Exception as e:
            print(f"  [KinematicGrasp:{self._name}] ERROR: {e}")

    def current_grasp_nothing(self) -> bool:
        return self._val == self.CLOSE and (
            self._grasp is None or self._grasp["vid"].shape == (0,)
        )

class FoldEnvIsaacSimNative(FoldEnv):
    def __init__(self, cfg: FoldEnvCfg):
        cfg = copy.deepcopy(cfg)
        self._cfg = cfg
        self._state = FoldEnvState()
        self._grasp_mode = _parse_choice_env(
            "ISAAC_GRASP_MODE",
            ("kinematic", "attachment", "handover", "jaw_pinch", "block_attachment"),
            "kinematic",
        )
        self._jaw_collisions_enabled = self._grasp_mode == "jaw_pinch"
        self._disable_gripper_collisions_enabled = _parse_bool_env(
            "ISAAC_DISABLE_GRIPPER_COLLISIONS",
            self._grasp_mode in {"kinematic", "block_attachment", "attachment", "handover"},
        )
        self._poststep_kinematic_correction = _parse_bool_env(
            "ISAAC_POSTSTEP_KINEMATIC",
            self._grasp_mode in {"attachment", "handover"},
        )
        self._grasp_damping_scale = max(
            1.0, _parse_float_env("ISAAC_GRASP_DAMPING_SCALE", 2.0)
        )
        self._cloth_vel_damp = float(
            np.clip(_parse_float_env("ISAAC_CLOTH_VEL_DAMP", 1.0), 0.0, 1.0)
        )
        # Hard per-particle speed cap (m/s); 0 disables. Kills solver ejection
        # spikes (single particles shooting away from kinematic grasp stress).
        self._cloth_max_vel = max(0.0, _parse_float_env("ISAAC_CLOTH_MAX_VEL", 0.0))
        self._cloth_vel_damp_view_retry_done = False
        print(
            f"[GraspMode] mode={self._grasp_mode},"
            f" jaw_collisions={'on' if self._jaw_collisions_enabled else 'off'},"
            f" gripper_collisions={'off' if self._disable_gripper_collisions_enabled else 'on'},"
            f" poststep_kinematic={'on' if self._poststep_kinematic_correction else 'off'},"
            f" grasp_damping_scale={self._grasp_damping_scale:.2f},"
            f" cloth_vel_damp={self._cloth_vel_damp:.2f},"
            f" cloth_max_vel={self._cloth_max_vel:.2f}"
        )
        
        # Override PyFlex init with Isaac Sim init
        self._init_isaac_sim()
        self._load_cloth(cfg)
        self._init_env(cfg)
        self._init_cloth(cfg)
        self._init_robot_isaac(cfg)
        self._init_cloth_isaac(cfg)
        self._init_cache()
    def _init_pyflex(self, cfg):
        # Override to prevent pyflex initialization
        pass

    def _init_cloth(self, cfg):
        # We still need self._vert_ren_to_sim mapping for trimesh wrapper to work
        tm_mesh_raw_rest, tm_mesh_sim_rest = self._tm_mesh_raw_rest, self._tm_mesh_sim_rest
        vert_xyz_to_idx = {}
        for i, v in enumerate(tm_mesh_sim_rest.vertices):
            vert_xyz_to_idx[tuple(v)] = i
        vert_ren_to_sim = []
        for i, v in enumerate(tm_mesh_raw_rest.vertices):
            vert_ren_to_sim.append(vert_xyz_to_idx[tuple(v)])
        self._vert_ren_to_sim = np.array(vert_ren_to_sim)
        self._cloth_xyz_init = tm_mesh_sim_rest.vertices + np.array([0., 0., 0.3])
        # Skip pyflex.add_cloth_mesh

    def _init_isaac_sim(self):
        # Force GPU dynamics via carb settings BEFORE anything creates the physics scene
        import carb
        carb.settings.get_settings().set_bool("/physics/enableGPUDynamics", True)
        carb.settings.get_settings().set_bool("/physics/updateParticlesToGpu", True)
        carb.settings.get_settings().set_bool("/physics/useGpu", True)
        carb.settings.get_settings().set_bool("/physics/updateToGpu", True)

        self._world = config.get_world_cls()(
            stage_units_in_meters=1.0,
            physics_dt=1.0/120.0,
            rendering_dt=1.0/30.0,
            backend="torch",
            device="cuda:0",
            sim_params={"use_gpu": True}
        )
        self._stage = omni.usd.get_context().get_stage()

        # The World automatically creates /physicsScene. Ensure it has GPU dynamics enabled.
        physx_ctx = self._world.get_physics_context()
        physx_ctx.enable_gpu_dynamics(True)
        physx_ctx.set_broadphase_type("GPU")
        
        # Double check the scene prim
        from pxr import PhysxSchema
        scene_prim = self._stage.GetPrimAtPath("/physicsScene")
        if scene_prim.IsValid():
            api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            api.CreateEnableGPUDynamicsAttr(True)
            api.GetBroadphaseTypeAttr().Set("GPU")

        # Ground plane
        self._world.scene.add_ground_plane()

        dome = UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome.CreateIntensityAttr(900.0)

        # Visual table (FixedCuboid uses mesh collision — GPU particle cloth
        # passes through mesh shapes, so this is visual only).
        tx, ty, tz = config.TABLE_SIZE
        table_center_z = config.TABLE_HEIGHT - tz / 2.0
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.0, 0.0, table_center_z]),
            scale=np.array([tx, ty, tz]),
            color=np.array([0.45, 0.30, 0.20]),
        )

        # Infinite collision plane at the table surface for particle cloth.
        # UsdGeom.Plane (infinite) is the only shape reliably detected by the GPU
        # particle broadphase — UsdGeom.Cube/FixedCuboid use mesh collision, which
        # GPU particles miss. Moving-jaw collision is disabled (see _disable_jaw_collisions)
        # so the jaw no longer hits this plane when closing.
        # Use World.scene.add_ground_plane with z_position — the exact same mechanism
        # as the working z=0 ground plane. physicsUtils.add_ground_plane and manual
        # UsdGeom.Plane approaches both failed (GPU particle broadphase ignores them).
        # world.scene.add_ground_plane registers the plane via Isaac Sim's scene manager
        # which ensures PhysX picks it up during world.reset().
        self._world.scene.add_ground_plane(
            z_position=config.TABLE_HEIGHT,
            name="table_surface",
            prim_path="/World/TableSurface",                # MUST differ from /World/groundPlane (z=0 plane)
            size=1500.0,
            color=np.array([0.45, 0.30, 0.20]),            # brown to match visual table
        )

        # Attachments container
        UsdGeom.Xform.Define(self._stage, "/World/Attachments")

    def _new_picker(self, name: str):
        picker = IsaacSimPicker(self, name, grasp_threshold=self._cfg.grasp_threshold[name], squeeze_factor=self._cfg.grasp_squeeze_factor[name])
        if name in self._picker_dict:
            raise ValueError(f"Picker with name {name} already exists")
        self._picker_dict[name] = picker
        return picker

    def _init_robot_isaac(self, cfg: FoldEnvCfg):
        self._robot = RobotSO100(self._new_picker("left"), self._new_picker("right"), cfg.robot_cfg)
        
        if config.USD_PATH.exists():
            add_reference_to_stage(str(config.USD_PATH), config.ROBOT_PRIM_PATH)
        else:
            raise FileNotFoundError("Robot USD not found.")

        robot_xform = UsdGeom.Xformable(self._stage.GetPrimAtPath(config.ROBOT_PRIM_PATH))
        robot_xform.ClearXformOpOrder()
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, config.TABLE_HEIGHT))
        robot_xform.AddRotateZOp().Set(180.0)

        self._isaac_robot = Articulation(prim_path=config.ROBOT_PRIM_PATH, name="so_100_dual")
        self._world.scene.add(self._isaac_robot)

    def _init_cloth_isaac(self, cfg: FoldEnvCfg):
        # We need the garment name (e.g. tshirt_sp_0) to load it via garment_loader
        # We can extract it from the cloth_obj_path
        cloth_dir = os.path.dirname(cfg.cloth_obj_path)
        garment_name = os.path.basename(cloth_dir)
        base_dir = os.path.dirname(cloth_dir)
        
        cat = garment_name.rsplit("_", 1)[0]
        profile = dict(config.get_garment_profile(cat))
        garment_mass = config.GARMENT_MASS
        garment_friction = config.GARMENT_FRICTION
        overrides = []
        for key, env_name in (
            ("stretch", "ISAAC_GARMENT_STRETCH"),
            ("bend", "ISAAC_GARMENT_BEND"),
            ("shear", "ISAAC_GARMENT_SHEAR"),
            ("damping", "ISAAC_GARMENT_DAMPING"),
        ):
            if os.environ.get(env_name, "").strip():
                profile[key] = _parse_float_env(env_name, profile[key])
                overrides.append(env_name)
        if os.environ.get("ISAAC_GARMENT_MASS", "").strip():
            garment_mass = max(1e-4, _parse_float_env("ISAAC_GARMENT_MASS", garment_mass))
            overrides.append("ISAAC_GARMENT_MASS")
        if os.environ.get("ISAAC_GARMENT_FRICTION", "").strip():
            garment_friction = max(0.0, _parse_float_env("ISAAC_GARMENT_FRICTION", garment_friction))
            overrides.append("ISAAC_GARMENT_FRICTION")
        self._cloth_base_damping = float(profile["damping"])
        print(
            f"  [GarmentPhysics] profile={config.GARMENT_PROFILE_MAP.get(cat, 'medium')}"
            f" stretch={profile['stretch']:.0f} bend={profile['bend']:.0f}"
            f" shear={profile['shear']:.0f} damping={profile['damping']:.2f}"
            f" mass={garment_mass:.3f}kg friction={garment_friction:.2f}"
            f" (env overrides: {', '.join(overrides) if overrides else 'none'})"
        )
        
        # The robot in Isaac Sim is rotated 180° around Z, so the cloth mesh
        # (in FoldNet's native frame) needs its X and Y negated to match.
        # FoldNet garment meshes have z ≈ ±0.02–0.03 m (front/back layers);
        # at scale=0.35 that becomes ±0.007–0.011 m.  With center=TABLE_HEIGHT+0.005
        # the bottom vertices start at ~0.744 m — 6 mm BELOW the plane at 0.75 m.
        # PhysX resolves that interpenetration by pushing particles downward → cloth
        # falls through.  Raise center by 0.03 m so the lowest vertex is at least
        # TABLE_HEIGHT+0.019 m, safely above the plane before the first physics tick.
        cloth_center = (0.0, 0.0, config.TABLE_HEIGHT + 0.03)
        
        (
            mesh_path, keypoint_idx, self._boundary_idx, self._vert_info, self._n_vertices,
            self._cloth_backend, self._cloth_sim_prim_path, self._cloth_material_path,
        ) = garment_loader.load_garment(
            stage=self._stage,
            scene_path="/physicsScene",
            root_path="/World/Cloth",
            garment_name=garment_name,
            garment_dir=base_dir,
            scale=config.GARMENT_SCALE,
            center=cloth_center,
            particle_contact_offset=config.GARMENT_PARTICLE_CONTACT_OFFSET,
            profile=profile,
            mass=garment_mass,
            friction=garment_friction,
            # PBD particles — reliable grasping, matches FoldNet PyFlex origin.
            # load_garment() auto-downgrades to "surface" if this Isaac Sim/PhysX
            # build has removed classic particle cloth (see garment_loader.py).
            backend="particle",
            flip_xy=True,  # Negate X and Y to match robot's 180° Z rotation
        )
        if self._cloth_backend != "particle":
            print(
                f"  [GarmentPhysics] using fallback backend={self._cloth_backend!r} "
                f"(sim_prim={self._cloth_sim_prim_path})"
            )
        self._keypoint_idx = {
            name: int(indices[0]) if isinstance(indices, (list, tuple)) else int(indices)
            for name, indices in keypoint_idx.items()
        }
        # Full per-keypoint index lists (front+back layer) for semantic grasping.
        self._keypoint_idx_full = {
            name: [int(i) for i in indices] if isinstance(indices, (list, tuple)) else [int(indices)]
            for name, indices in keypoint_idx.items()
        }
        self._cloth_attachment_prim_path = mesh_path
        # Force GPU dynamics on ALL physics scenes right before reset
        from pxr import PhysxSchema, UsdPhysics
        gpu_scenes = []
        for prim in self._stage.Traverse():
            # Check by API, by type, and by prim type name
            is_scene = (
                prim.HasAPI(PhysxSchema.PhysxSceneAPI) or 
                prim.IsA(UsdPhysics.Scene) or
                prim.GetTypeName() == "PhysicsScene"
            )
            if is_scene:
                prim_path = str(prim.GetPath())
                if prim_path == "/physicsScene":
                    # Our main scene - enable GPU dynamics
                    api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    api.CreateEnableGPUDynamicsAttr(True)
                    api.GetBroadphaseTypeAttr().Set("GPU")
                    gpu_scenes.append(prim_path)
                else:
                    # Extra scene from robot USD etc - also enable GPU dynamics
                    api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    api.CreateEnableGPUDynamicsAttr(True)
                    api.GetBroadphaseTypeAttr().Set("GPU")
                    gpu_scenes.append(prim_path)
        print(f"  GPU dynamics forced on {len(gpu_scenes)} scene(s): {gpu_scenes}")

        # Also force via World physics context as a fallback
        try:
            physx_ctx = self._world.get_physics_context()
            physx_ctx.enable_gpu_dynamics(True)
            physx_ctx.set_broadphase_type("GPU")
        except Exception as e:
            print(f"  Warning: could not set physics context: {e}")

        self._world.reset()
        if not self._isaac_robot.handles_initialized:
            self._isaac_robot.initialize()

        # In kinematic / attachment modes we keep moving-jaw collision off because
        # grasping is handled artificially and jaw-table interference adds noise.
        # In jaw_pinch mode we leave jaw collision enabled so the cloth can be
        # trapped physically between the fixed and moving gripper surfaces.
        if self._jaw_collisions_enabled:
            print("  [jaw] collisions left enabled for jaw_pinch mode")
        elif self._disable_gripper_collisions_enabled:
            self._disable_gripper_collisions()
        else:
            self._disable_jaw_collisions()

        # Re-apply GPU dynamics after reset, just in case World.reset() wiped it!
        physx_ctx = self._world.get_physics_context()
        physx_ctx.enable_gpu_dynamics(True)
        physx_ctx.set_broadphase_type("GPU")
        for prim in self._stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")

        self._cloth_mesh_prim = self._stage.GetPrimAtPath(mesh_path)

        # cloth_physics_view is created AFTER debug warmup steps so the GPU
        # particle buffer is populated before we try to access it.
        self._cloth_physics_view = None

        self._debug_cloth_physics()   # steps 5 frames → GPU buffer is now live

        # Now create the tensor view — GPU data is available after the first steps.
        self._try_init_cloth_view()

        self._cloth_xyz_init = self._get_cloth_xyz()
        # Derive table-surface z from the settled cloth: lowest vertex ≈ table top + contact_offset.
        # Subtract contact_offset so z_ref = true table surface, not cloth bottom surface.
        self._z_ref = float(np.min(self._cloth_xyz_init[:, 2])) - config.GARMENT_PARTICLE_CONTACT_OFFSET

    def _set_runtime_cloth_damping(self, damping: float):
        if getattr(self, "_cloth_backend", "particle") == "surface":
            # Surface deformable bodies have no PhysxAutoParticleClothAPI (removed in
            # this PhysX build along with classic particle cloth) -- damping instead
            # lives on the deformable material prim, mirroring garment_loader.py's
            # _add_surface_deformable().
            material_path = getattr(self, "_cloth_material_path", None)
            material_prim = self._stage.GetPrimAtPath(material_path) if material_path else None
            if material_prim is None or not material_prim.IsValid():
                return
            garment_loader._set_attr_if_present(
                material_prim, "physxDeformableMaterial:elasticityDamping", float(damping)
            )
            garment_loader._set_attr_if_present(
                material_prim, "physxDeformableMaterial:bendDamping", float(damping)
            )
            return
        cloth_prim = self._stage.GetPrimAtPath(
            getattr(self, "_cloth_sim_prim_path", "/World/Cloth/garmentMesh")
        )
        if not cloth_prim.IsValid():
            return
        api = (
            PhysxSchema.PhysxAutoParticleClothAPI(cloth_prim)
            if cloth_prim.HasAPI(PhysxSchema.PhysxAutoParticleClothAPI)
            else PhysxSchema.PhysxAutoParticleClothAPI.Apply(cloth_prim)
        )
        attr = api.GetSpringDampingAttr()
        if not attr.IsValid():
            attr = api.CreateSpringDampingAttr()
        attr.Set(float(damping))

    def _update_runtime_cloth_damping(self):
        base = float(getattr(self, "_cloth_base_damping", 0.3))
        boosted = base * self._grasp_damping_scale
        any_grasp = any(
            (
                getattr(picker, "_kinematic_grasp_enabled", False)
                or getattr(picker, "_block_attachment_enabled", False)
            )
            and getattr(picker, "_grasp", None) is not None
            and picker._grasp["vid"].shape[0] > 0
            for picker in getattr(self._robot, "_picker", {}).values()
        )
        try:
            self._set_runtime_cloth_damping(boosted if any_grasp else base)
        except Exception as e:
            print(f"  [cloth] runtime damping update skipped: {e}")

    def _try_init_cloth_view(self):
        """Create the cloth tensor view. Must be called after at least one world.step()."""
        import omni.physics.tensors as pt
        from pxr import UsdUtils
        sim_prim_path = getattr(self, "_cloth_sim_prim_path", "/World/Cloth/garmentMesh")
        use_surface = getattr(self, "_cloth_backend", "particle") == "surface"
        # create_simulation_view()'s default stage_id=-1 fails to resolve on this
        # build ("Failed to get a valid attached USD stage id from PhysX
        # simulation") -- pass the stage id explicitly, matching NVIDIA's own
        # tensor-API demos (e.g. FrankaDeformableDemo.on_tensor_start).
        stage_id = UsdUtils.StageCache.Get().GetId(self._stage).ToLongInt()
        for backend in ("torch", "warp", "numpy"):
            try:
                sim_view = pt.create_simulation_view(backend, stage_id)
                sim_view.set_subspace_roots("/")
                if use_surface:
                    cloth_view = _SurfaceDeformableClothView(
                        sim_view.create_surface_deformable_body_view(sim_prim_path)
                    )
                else:
                    cloth_view = sim_view.create_particle_cloth_view(sim_prim_path)
                self._pt_sim_view = sim_view
                self._cloth_physics_view = cloth_view
                self._cloth_view_backend = backend
                print(f"  [ClothView] OK (backend={backend}): "
                      f"count={cloth_view.count}  "
                      f"max_particles={cloth_view.max_particles_per_cloth}")
                n_verts = getattr(self, "_n_vertices", None)
                if n_verts is not None and cloth_view.max_particles_per_cloth != n_verts:
                    print(
                        f"  [ClothView] WARNING: view reports "
                        f"{cloth_view.max_particles_per_cloth} nodes but the authored "
                        f"mesh has {n_verts} vertices -- PhysX cooking likely welded/"
                        f"reordered vertices, so keypoint-index-based grasping may "
                        f"reference the wrong nodes on this backend."
                    )
                return
            except Exception as e:
                print(f"  [ClothView] backend={backend} FAILED: {e}")
        print("  [ClothView] ALL backends failed — kinematic grasp disabled")
        self._cloth_physics_view = None

    def _debug_cloth_physics(self):
        """Diagnose why cloth may fall through the table surface."""
        print("\n========== CLOTH PHYSICS DEBUG ==========")

        # 1. TableSurface plane — GroundPlane creates root Xform at /World/TableSurface
        #    and the actual collision UsdGeom.Plane at /World/TableSurface/geom
        plane_prim = self._stage.GetPrimAtPath("/World/TableSurface")
        print(f"  /World/TableSurface  valid={plane_prim.IsValid()}")
        if plane_prim.IsValid():
            print(f"    CollisionAPI    : {plane_prim.HasAPI(UsdPhysics.CollisionAPI)}")
            print(f"    PhysxCollisionAPI: {plane_prim.HasAPI(PhysxSchema.PhysxCollisionAPI)}")
            for attr in plane_prim.GetAttributes():
                print(f"    {attr.GetName()} = {attr.Get()}")
            # Traverse all children — collision is at /World/TableSurface/geom
            print("    Children:")
            for child in Usd.PrimRange(plane_prim):
                if child == plane_prim:
                    continue
                cpath = str(child.GetPath())
                print(f"      {cpath}  type={child.GetTypeName()}"
                      f"  col={child.HasAPI(UsdPhysics.CollisionAPI)}"
                      f"  physx={child.HasAPI(PhysxSchema.PhysxCollisionAPI)}")

        # 2. Particle system
        ps_prim = self._stage.GetPrimAtPath("/World/Cloth/particleSystem")
        print(f"  /World/Cloth/particleSystem  valid={ps_prim.IsValid()}")
        if ps_prim.IsValid():
            for attr in ps_prim.GetAttributes():
                print(f"    {attr.GetName()} = {attr.Get()}")

        # 3. All physics scenes
        print("  Physics scenes:")
        for prim in self._stage.Traverse():
            if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                print(f"    {prim.GetPath()}")
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    api = PhysxSchema.PhysxSceneAPI(prim)
                    print(f"      gpuDynamics={api.GetEnableGPUDynamicsAttr().Get()}")
                    print(f"      broadphase ={api.GetBroadphaseTypeAttr().Get()}")

        # 4. Cloth z at t=0 — show both USD points and tensor API to diagnose mismatch
        usd_pts = self._cloth_mesh_prim.GetAttribute("points").Get()
        if usd_pts is not None:
            usd_z = np.array(usd_pts, dtype=np.float32).reshape(-1, 3)[:, 2]
            print(f"  USD points z at t=0:   [{usd_z.min():.4f}, {usd_z.max():.4f}]"
                  f"  (rest pos, local; expect ~±0.010)")
        xyz = self._get_cloth_xyz()
        if xyz is not None and xyz.size > 0:
            print(f"  _get_cloth_xyz z t=0:  [{xyz[:,2].min():.4f}, {xyz[:,2].max():.4f}]"
                  f"  (expect ≈{config.TABLE_HEIGHT:.3f} if tensor API works)")
        else:
            print("  Cloth positions unavailable at t=0")

        # 5. Step 5 frames and watch cloth z to see if/when it falls
        print("  Stepping 5 frames (watching cloth z vs table):")
        for i in range(5):
            self._world.step(render=True)
            xyz = self._get_cloth_xyz()
            if xyz is not None and xyz.size > 0:
                zlo, zhi = xyz[:,2].min(), xyz[:,2].max()
                on_table = zlo >= config.TABLE_HEIGHT - 0.02
                print(f"    frame {i+1}: z=[{zlo:.4f}, {zhi:.4f}]"
                      f"  {'OK (on table)' if on_table else 'FALLING THROUGH!'}")
            else:
                print(f"    frame {i+1}: no cloth data")

        print("=========================================\n")

    def _disable_jaw_collisions(self):
        """Disable PhysX collision on Moving_Jaw links for both arms."""
        self._disable_robot_collisions_by_tags(("Moving_Jaw",), label="jaw")

    def _disable_gripper_collisions(self):
        """Disable PhysX collision on all gripper fingertip + wrist links for fake grasp modes."""
        self._disable_robot_collisions_by_tags(
            ("Moving_Jaw", "Fixed_Gripper", "Wrist_Pitch_Roll"), label="gripper"
        )

    def _disable_robot_collisions_by_tags(self, tags, label: str):
        disabled = []
        prims_found = []
        for prim in self._stage.Traverse():
            path = str(prim.GetPath())
            if not any(tag in path for tag in tags):
                continue
            prims_found.append(path)
            has_col     = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_physx   = prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            has_mesh    = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            # Disable via physics:collisionEnabled (works regardless of which API is applied)
            col_attr = prim.GetAttribute("physics:collisionEnabled")
            if col_attr.IsValid():
                col_attr.Set(False)
                disabled.append(path)
            elif has_col or has_physx or has_mesh:
                # Attribute not yet created — apply CollisionAPI and disable
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
                disabled.append(path)
            print(f"  [{label}] {path}  col={has_col} physx={has_physx} mesh={has_mesh}")

        print(f"  {label.capitalize()} collisions: found {len(prims_found)} prims, "
              f"disabled {len(disabled)}: {disabled}")

    def _isaac_to_foldnet(self, xyz_isaac: np.ndarray) -> np.ndarray:
        # Isaac world → FoldNet policy frame.
        # flip_xy=True (garment_loader) negated x,y when placing cloth, so Isaac x = -FoldNet x.
        # Inverse: negate x,y to recover FoldNet frame.  z_ref is derived at init from settled
        # cloth min-z (no need to hardcode TABLE_HEIGHT here).
        xyz = xyz_isaac.copy()
        if xyz.ndim == 3 and xyz.shape[0] == 1:
            xyz = xyz[0]
        xyz[:, 0] *= -1
        xyz[:, 1] *= -1
        xyz[:, 2] -= self._z_ref
        return xyz

    def get_raw_mesh_curr(self):
        import trimesh
        xyz_sim = self._isaac_to_foldnet(self._get_cloth_xyz())
        tm_mesh_raw = copy.deepcopy(self._tm_mesh_raw)
        tm_mesh_raw.vertices = xyz_sim[self._vert_ren_to_sim]
        return tm_mesh_raw

    def get_raw_mesh_rest(self):
        return copy.deepcopy(self._tm_mesh_raw_rest)

    def get_keypoint_idx(self):
        return getattr(self, '_keypoint_idx', {})

    def get_tcp_xyz(self):
        return self._robot.get_tcp_xyz()

    def get_gripper_state(self):
        return self._robot.get_gripper_state()

    def _physics_data_to_numpy(self, data):
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return np.asarray(data.numpy(), dtype=np.float32)
        return np.asarray(data, dtype=np.float32)

    def _physics_array_to_backend(self, data, like, dtype=np.float32):
        backend = getattr(self, "_cloth_view_backend", None)
        arr = np.ascontiguousarray(data, dtype=dtype)
        if backend == "torch" or hasattr(like, "detach"):
            device = like.device if hasattr(like, "device") else "cuda:0"
            torch_dtype = torch.float32 if arr.dtype == np.float32 else torch.int64
            return torch.as_tensor(arr, dtype=torch_dtype, device=device)
        if backend == "warp":
            import warp as wp
            device = str(getattr(like, "device", "cuda:0"))
            wp_dtype = wp.float32 if arr.dtype == np.float32 else wp.int32
            return wp.array(arr, dtype=wp_dtype, device=device)
        return arr

    def _physics_positions_to_backend(self, data, like):
        return self._physics_array_to_backend(data, like, dtype=np.float32)

    def _physics_indices_to_backend(self, data, like):
        backend = getattr(self, "_cloth_view_backend", None)
        arr = np.ascontiguousarray(data, dtype=np.int32)
        if backend == "torch" or hasattr(like, "detach"):
            device = like.device if hasattr(like, "device") else "cuda:0"
            return torch.as_tensor(arr, dtype=torch.long, device=device)
        if backend == "warp":
            import warp as wp
            device = str(getattr(like, "device", "cuda:0"))
            return wp.array(arr, dtype=wp.int32, device=device)
        return arr

    def _get_cloth_xyz(self):
        # Prefer physics tensor API — reads actual world positions from PhysX GPU buffer.
        # USD `points` stores only REST positions (local z≈0), not current physics positions.
        if getattr(self, "_cloth_physics_view", None) is not None:
            try:
                positions = self._cloth_physics_view.get_positions()
                # API returns (count, max_particles * 3) flat float32 array.
                # Reshape to (N, 3) for particle-level access.
                N = self._cloth_physics_view.max_particles_per_cloth
                xyz = self._physics_data_to_numpy(positions).reshape(N, 3)
                if xyz.shape[0] > 0:
                    return xyz
            except Exception as e:
                print(f"  [ClothXYZ] tensor API failed: {e} — falling back to USD points")
                # fall through to USD fallback

        # Fallback: USD points (rest positions, z≈0 local).
        # Add TABLE_HEIGHT so proximity check works for settled cloth on table.
        points = self._cloth_mesh_prim.GetAttribute("points").Get()
        if points is None:
            return np.zeros((0, 3), dtype=np.float32)
        xyz = np.array(points, dtype=np.float32)
        if xyz.ndim == 1:
            xyz = xyz.reshape(-1, 3)
        if xyz.ndim == 3 and xyz.shape[0] == 1:
            xyz = xyz[0]
        xyz[:, 2] += config.TABLE_HEIGHT   # rest z≈0 → world z≈TABLE_HEIGHT
        return xyz

    def _get_cloth_xyzm(self):
        xyz = self._get_cloth_xyz()
        m = np.ones((xyz.shape[0], 1), dtype=np.float32)
        return np.concatenate([xyz, m], axis=1)

    def _set_cloth_xyz(self, xyz: np.ndarray):
        # Cannot easily teleport cloth in Isaac Sim. 
        # For initialization, we should let it settle naturally or re-create it.
        pass

    def _set_cloth_vel(self, vel: np.ndarray):
        pass
    
    def _get_cloth_vel(self):
        return np.zeros_like(self._get_cloth_xyz())

    def _damp_cloth_velocities(self):
        damp = float(getattr(self, "_cloth_vel_damp", 1.0))
        max_vel = float(getattr(self, "_cloth_max_vel", 0.0))
        if damp >= 1.0 and max_vel <= 0.0:
            return

        cloth_view = getattr(self, "_cloth_physics_view", None)
        if cloth_view is None and not getattr(self, "_cloth_vel_damp_view_retry_done", False):
            self._try_init_cloth_view()
            self._cloth_vel_damp_view_retry_done = True
            cloth_view = getattr(self, "_cloth_physics_view", None)
        if cloth_view is None:
            return
        if not (hasattr(cloth_view, "get_velocities") and hasattr(cloth_view, "set_velocities")):
            return

        try:
            N = cloth_view.max_particles_per_cloth
            raw_vel = cloth_view.get_velocities()
            vel = self._physics_data_to_numpy(raw_vel).reshape(N, 3).copy()
            vel *= damp
            if max_vel > 0.0:
                speed = np.linalg.norm(vel, axis=1, keepdims=True)
                vel *= np.minimum(1.0, max_vel / np.maximum(speed, 1e-6))

            out = self._physics_positions_to_backend(vel.reshape(1, N * 3), raw_vel)
            indices = self._physics_indices_to_backend([0], raw_vel)
            cloth_view.set_velocities(out, indices)
        except Exception:
            return

    def _step_pyflex(self):
        # Replace PyFlex step with Isaac Sim step
        # Apply qpos from self._robot to self._isaac_robot
        dof_names = list(self._isaac_robot.dof_names)
        qpos_dict = self._robot.get_qpos()
        targets = np.array([qpos_dict.get(n, 0.0) for n in dof_names], dtype=np.float32)
        
        if not hasattr(self, '_drives_set'):
            ctrl = self._isaac_robot.get_articulation_controller()
            n_dof = self._isaac_robot.num_dof
            stiffness = np.full(n_dof, config.ARM_DRIVE_STIFFNESS, dtype=np.float32)
            damping = np.full(n_dof, config.ARM_DRIVE_DAMPING, dtype=np.float32)
            # Use lighter gains for gripper joints
            for i, name in enumerate(dof_names):
                if config.GRIPPER_JOINT_TAG in name:
                    stiffness[i] = config.GRIPPER_DRIVE_STIFFNESS
                    damping[i] = config.GRIPPER_DRIVE_DAMPING
            backend_utils = ctrl._articulation_view._backend_utils
            device = ctrl._articulation_view._device
            stiffness = backend_utils.convert(stiffness, device=device, dtype="float32")
            damping = backend_utils.convert(damping, device=device, dtype="float32")
            ctrl.set_gains(kps=stiffness, kds=damping)
            self._drives_set = True
            self._step_count = 0
            
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        self._isaac_robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=targets))
        self._world.step(render=True)

        # After every physics tick: re-apply kinematic grasp to prevent PBD drift.
        # PBD gravity + spring forces move grasped particles away from the gripper;
        # correcting after each tick keeps them rigidly attached.
        if self._poststep_kinematic_correction:
            for picker in self._robot._picker.values():
                picker.update_attachment_handover()
                picker._move_attachment_block()
                if picker._val == picker.CLOSE and picker._should_apply_kinematic_grasp():
                    picker._apply_kinematic_grasp()
        else:
            for picker in self._robot._picker.values():
                picker.update_attachment_handover()
                picker._move_attachment_block()

        for picker in self._robot._picker.values():
            picker._apply_release_damping()

        self._damp_cloth_velocities()

        self._step_count = getattr(self, '_step_count', 0) + 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            actual = self._isaac_robot.get_joint_positions()
            print(f"[step {self._step_count}] target={targets[:3]}... actual={actual[:3] if actual is not None else 'N/A'}...")

    def _step_render(self, *args, **kwargs):
        # We handle rendering natively in Isaac Sim
        pass

if __name__ == "__main__":
    robot_cfg = make_so100_robot_cfg()
    env_cfg = FoldEnvCfg(
        cloth_obj_path=str(config.GARMENT_DIR / "tshirt_sp_0" / "mesh.obj"),
        cloth_scale=0.5,
        render=False,
        render_mode=[],
        render_process_num=1,
        robot_cfg=robot_cfg,
    )
    env = FoldEnvIsaacSimNative(env_cfg)
    
    print("Environment successfully initialized with Isaac Sim Native!")
    for i in range(10):
        env._step_pyflex()
    
    xyz = env._get_cloth_xyz()
    print("Cloth Vertices shape:", xyz.shape)
    sim_app.close()
