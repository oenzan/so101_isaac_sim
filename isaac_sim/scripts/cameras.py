"""
cameras.py
==========
Wrist cameras for the SO-101, matching TheRobotStudio
"Wrist_Cam_Mount_32x32_UVC_Module":

    https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Wrist_Cam_Mount_32x32_UVC_Module

In hardware the mount *replaces the wrist-roll part* and screws a 32x32 mm USB
UVC camera (>=720p/30fps, run at 640x480) onto it, so the camera spins with the
wrist-roll joint and looks forward along the gripper. In our URDF the body driven
by the wrist-roll joint is the `Fixed_Gripper` link, so that is where we parent
the camera.

Two layers are provided:
  * `WristCamera.spawn()` creates the USD `Camera` prim (always) -- it is visible
    in the stage, can be selected as the active viewport camera, and moves with
    the gripper.
  * `WristCamera.make_sensor()` optionally wraps it in an Isaac `Camera` sensor
    for reading RGB/Depth frames in code (for a folding policy / dataset).
"""

import math

from pxr import Gf, UsdGeom, Sdf


def _hfov_to_aperture(hfov_deg: float, focal_length: float) -> float:
    """USD camera: horizontalAperture = 2 * f * tan(HFOV / 2)."""
    return 2.0 * focal_length * math.tan(math.radians(hfov_deg) / 2.0)


class WristCamera:
    """A single wrist-mounted UVC camera."""

    # USD's default focal length (mm). We keep it and solve the aperture from the
    # desired field of view, which is the cleaner knob to expose.
    FOCAL_LENGTH_MM = 18.147

    # Physical size of the 32x32 UVC module (metres), used for the visible marker.
    BODY_SIZE = (0.032, 0.032, 0.018)

    def __init__(self, name, parent_link_path, translation, rpy,
                 resolution, hfov_deg, clipping, show_body=True):
        self.name = name
        self.parent_link_path = parent_link_path
        self.prim_path = f"{parent_link_path}/{name}"
        self.translation = translation
        self.rpy = rpy
        self.resolution = resolution
        self.hfov_deg = hfov_deg
        self.clipping = clipping
        self.show_body = show_body          # draw a visible camera body box
        self._sensor = None

    # --- USD prim ---------------------------------------------------------- #
    def spawn(self, stage):
        """Create the UsdGeom.Camera prim under the gripper link."""
        cam = UsdGeom.Camera.Define(stage, Sdf.Path(self.prim_path))

        # Local transform on the wrist-roll piece.
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*self.translation))
        rx, ry, rz = (math.degrees(a) for a in self.rpy)
        # XYZ euler, matching the (roll, pitch, yaw) order used elsewhere.
        xform.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

        # Intrinsics: drive the field of view through the aperture.
        h_ap = _hfov_to_aperture(self.hfov_deg, self.FOCAL_LENGTH_MM)
        w, h = self.resolution
        cam.CreateFocalLengthAttr(self.FOCAL_LENGTH_MM)
        cam.CreateHorizontalApertureAttr(h_ap)
        cam.CreateVerticalApertureAttr(h_ap * float(h) / float(w))
        cam.CreateClippingRangeAttr(Gf.Vec2f(*self.clipping))

        # A USD Camera prim is invisible (only a frustum gizmo when selected), so
        # draw a small box at the same pose to make the module visible on the
        # wrist. Oriented like the camera, so it also hints the view direction.
        if self.show_body:
            self._spawn_body_marker(stage)

        print(f"[camera] spawned {self.prim_path} "
              f"({w}x{h}, HFOV={self.hfov_deg} deg)")
        return self

    def _spawn_body_marker(self, stage):
        """Visible 32x32 mm box representing the UVC module, at the camera pose."""
        path = f"{self.parent_link_path}/{self.name}_body"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)                       # unit cube; scaled below
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.06)])  # near-black
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()                          # order: T * R * S
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.translation))
        rx, ry, rz = (math.degrees(a) for a in self.rpy)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
        xf.AddScaleOp().Set(Gf.Vec3f(*self.BODY_SIZE))
        return cube

    # --- optional Isaac sensor (frame capture) ----------------------------- #
    def make_sensor(self):
        """
        Wrap the prim in an Isaac `Camera` sensor so frames can be read with
        `get_rgba()` / `get_depth()`. Must be called after the prim is spawned;
        the returned sensor still needs `.initialize()` after world.reset().
        Returns None if the Camera API is unavailable.
        """
        try:                                            # Isaac Sim >= 4.5
            from isaacsim.sensors.camera import Camera
        except ImportError:
            try:                                        # Isaac Sim <= 4.2
                from omni.isaac.sensor import Camera
            except ImportError:
                print("[camera] Camera sensor API not available; "
                      "prim still usable as a viewport camera.")
                return None
        self._sensor = Camera(prim_path=self.prim_path, resolution=self.resolution)
        return self._sensor

    @property
    def sensor(self):
        return self._sensor


def add_wrist_cameras(stage, cfg, robot_prim_path, arm_prefixes=("left", "right"),
                      make_sensors=False):
    """
    Spawn a wrist camera on each arm.

    cfg            : the config module (for camera parameters)
    robot_prim_path: e.g. "/World/so_100_dual"
    make_sensors   : also create Isaac Camera sensors (initialise them after reset)
    Returns the list of WristCamera objects.
    """
    cams = []
    for prefix in arm_prefixes:
        parent = f"{robot_prim_path}/{prefix}_{cfg.WRIST_CAM_PARENT_LINK}"
        if not stage.GetPrimAtPath(Sdf.Path(parent)).IsValid():
            print(f"[camera] WARNING: parent link {parent} not found; "
                  f"skipping {prefix} wrist camera. Check WRIST_CAM_PARENT_LINK.")
            continue
        cam = WristCamera(
            name=f"{prefix}_wrist_cam",
            parent_link_path=parent,
            translation=cfg.WRIST_CAM_TRANSLATION,
            rpy=cfg.WRIST_CAM_RPY,
            resolution=cfg.WRIST_CAM_RESOLUTION,
            hfov_deg=cfg.WRIST_CAM_HFOV_DEG,
            clipping=cfg.WRIST_CAM_CLIPPING,
        ).spawn(stage)
        if make_sensors:
            cam.make_sensor()
        cams.append(cam)
    return cams
