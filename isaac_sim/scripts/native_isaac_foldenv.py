import os
import sys
import copy
import json
import numpy as np
import torch
import glob
from collections import deque

import config

# Mock pyflex before importing any garmentds modules
import sys
import os
from unittest.mock import MagicMock
sys.modules['pyflex'] = MagicMock()

foldnet_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/src"))
if foldnet_src not in sys.path:
    sys.path.insert(0, foldnet_src)
batch_urdf_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code/external/batch_urdf/src"))
if batch_urdf_src not in sys.path:
    sys.path.insert(0, batch_urdf_src)
os.environ["FOLDNET_BASE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FoldNet_code"))

sim_app = config.get_simulation_app(headless=os.environ.get("ISAAC_HEADLESS", "1") == "1")

import omni.usd
from pxr import UsdGeom, UsdLux, Gf, PhysxSchema, UsdPhysics, Sdf
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import FixedCuboid
from omni.physx.scripts import physicsUtils

import garment_loader
from so100_foldenv import RobotSO100, make_so100_robot_cfg, RenderProcessNull
from garmentds.foldenv.fold_env import FoldEnv, FoldEnvCfg, FoldEnvState, Picker
import garmentds.common.utils as utils

class IsaacSimPicker(Picker):
    def __init__(self, env, name, grasp_threshold, squeeze_factor):
        super().__init__(env, name, grasp_threshold, squeeze_factor)
        self.attachment_path = f"/World/Attachments/{name}_attachment"

    def set_action(self, action: float):
        self._prev_val = self._val
        self._val = self.OPEN if self.is_open_action(action) else self.CLOSE
        self._val_float = float(action)

        if (self._prev_val, self._val) == (self.OPEN, self.CLOSE):
            self._create_attachment()
        elif (self._prev_val, self._val) == (self.CLOSE, self.OPEN):
            self._destroy_attachment()

    def _create_attachment(self):
        stage = omni.usd.get_context().get_stage()
        gripper_path = f"{config.ROBOT_PRIM_PATH}/{self._name}_Fixed_Gripper" 
        cloth_path = "/World/Cloth/surfaceDeformable"
        
        # We can dynamically create the PhysX Attachment between the two paths
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, self.attachment_path)
        attachment.GetActor0Rel().SetTargets([Sdf.Path(gripper_path)])
        attachment.GetActor1Rel().SetTargets([Sdf.Path(cloth_path)])

    def _destroy_attachment(self):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(self.attachment_path).IsValid():
            stage.RemovePrim(self.attachment_path)

    def current_grasp_nothing(self) -> bool:
        # In Isaac Sim we don't track grasped vertices the same way, assume false for now
        return False

class FoldEnvIsaacSimNative(FoldEnv):
    def __init__(self, cfg: FoldEnvCfg):
        cfg = copy.deepcopy(cfg)
        self._cfg = cfg
        self._state = FoldEnvState()
        
        # Override PyFlex init with Isaac Sim init
        self._init_isaac_sim()
        self._load_cloth(cfg)
        self._init_env(cfg)
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
        carb.settings.get_settings().set_bool("/physics/updateParticlesToGpu", True)
        carb.settings.get_settings().set_bool("/physics/useGpu", True)
        carb.settings.get_settings().set_bool("/physics/updateToGpu", True)

        self._world = config.get_world_cls()(
            stage_units_in_meters=1.0,
            physics_dt=1.0/120.0,
            rendering_dt=1.0/30.0,
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

        # Table
        tx, ty, tz = config.TABLE_SIZE
        table_center_z = config.TABLE_HEIGHT - tz / 2.0
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.0, 0.0, table_center_z]),
            scale=np.array([tx, ty, tz]),
            color=np.array([0.45, 0.30, 0.20]),
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
        profile = config.get_garment_profile(cat)
        
        # The robot in Isaac Sim is rotated 180° around Z, so the cloth mesh
        # (in FoldNet's native frame) needs its X and Y negated to match.
        # We set center=(0, 0, TABLE_HEIGHT+0.005) and flip XY in the loader.
        cloth_center = (0.0, 0.0, config.TABLE_HEIGHT + 0.005)
        
        garment_loader.load_garment(
            stage=self._stage,
            scene_path="/physicsScene",
            root_path="/World/Cloth",
            garment_name=garment_name,
            garment_dir=base_dir,
            scale=config.GARMENT_SCALE,
            center=cloth_center,
            particle_contact_offset=config.GARMENT_PARTICLE_CONTACT_OFFSET,
            profile=profile,
            mass=config.GARMENT_MASS,
            friction=config.GARMENT_FRICTION,
            backend="surface",
            flip_xy=True,  # Negate X and Y to match robot's 180° Z rotation
        )
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
            
        # Re-apply GPU dynamics after reset, just in case World.reset() wiped it!
        physx_ctx = self._world.get_physics_context()
        physx_ctx.enable_gpu_dynamics(True)
        physx_ctx.set_broadphase_type("GPU")
        for prim in self._stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")

        self._cloth_mesh_prim = self._stage.GetPrimAtPath("/World/Cloth/surfaceDeformable/mesh")
        self._cloth_xyz_init = self._get_cloth_xyz()

    def get_raw_mesh_curr(self):
        import trimesh
        xyz = self._get_cloth_xyz()
        return trimesh.Trimesh(vertices=xyz, faces=self._tm_mesh_raw_rest.faces if hasattr(self, '_tm_mesh_raw_rest') else [])

    def get_raw_mesh_rest(self):
        import trimesh
        return trimesh.Trimesh(vertices=self._cloth_xyz_init, faces=self._tm_mesh_raw_rest.faces if hasattr(self, '_tm_mesh_raw_rest') else [])

    def get_keypoint_idx(self):
        return getattr(self, '_keypoint_idx', {})

    def get_tcp_xyz(self):
        return self._robot.get_tcp_xyz()

    def get_gripper_state(self):
        return {"left": 1.0, "right": 1.0}


    def _get_cloth_xyz(self):
        # Read vertices from Surface Deformable
        points = self._cloth_mesh_prim.GetAttribute("points").Get()
        return np.array(points, dtype=np.float32)

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

    def _step_pyflex(self):
        # Replace PyFlex step with Isaac Sim step
        # Apply qpos from self._robot to self._isaac_robot
        dof_names = list(self._isaac_robot.dof_names)
        qpos_dict = self._robot.get_qpos()
        targets = np.array([qpos_dict.get(n, 0.0) for n in dof_names], dtype=np.float32)
        
        if not hasattr(self, '_drives_set'):
            ctrl = self._isaac_robot.get_articulation_controller()
            n_dof = self._isaac_robot.num_dof
            stiffness = np.full(n_dof, config.ARM_DRIVE_STIFFNESS)
            damping = np.full(n_dof, config.ARM_DRIVE_DAMPING)
            # Use lighter gains for gripper joints
            for i, name in enumerate(dof_names):
                if config.GRIPPER_JOINT_TAG in name:
                    stiffness[i] = config.GRIPPER_DRIVE_STIFFNESS
                    damping[i] = config.GRIPPER_DRIVE_DAMPING
            ctrl.set_gains(kps=stiffness, kds=damping)
            self._drives_set = True
            self._step_count = 0
            
        from omni.isaac.core.utils.types import ArticulationAction
        self._isaac_robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=targets))
        self._world.step(render=True)
        
        self._step_count = getattr(self, '_step_count', 0) + 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            actual = self._isaac_robot.get_joint_positions()
            print(f"[step {self._step_count}] target={targets[:3]}... actual={actual[:3] if actual is not None else 'N/A'}...")
        self._world.step(render=True)

    def _step_render(self, *args, **kwargs):
        # We handle rendering natively in Isaac Sim
        pass

if __name__ == "__main__":
    robot_cfg = make_so100_robot_cfg()
    env_cfg = FoldEnvCfg(
        cloth_obj_path="/home/ozan/Downloads/so100_ws/foldnet_garments/tshirt_sp_0/mesh.obj",
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
