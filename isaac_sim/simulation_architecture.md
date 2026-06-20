# Simulation Architecture — Dual SO-101 in Isaac Sim

This document explains how the bimanual SO-101 cloth-folding simulation is put
together: the build pipeline, every script's responsibility, and the design
decisions behind the robot model, the motor model, the wrist cameras and the
cloth. It is the "why" companion to the run instructions in `README.md`.

---

## 1. Goal

Stand up two SO-101 (SO-100) 6-DOF arms in NVIDIA Isaac Sim, mounted on a table,
facing a piece of simulated fabric, with:

- **physics that behaves like the real hardware** (Feetech STS3215 servos), and
- **a wrist camera on each arm** matching TheRobotStudio 32×32 UVC mount,

so the scene can be used to develop bimanual **cloth-folding** behaviours.

---

## 2. Build pipeline (3 stages)

```
  robot_description.txt (ROS dual URDF, package:// meshes, no limits)
            │
            │  generate_urdf.py      (plain Python — no Isaac needed)
            ▼
  urdf/so_100_dual.urdf  (relative mesh paths, real joint limits = STS3215)
            │
            │  import_urdf_to_usd.py (Isaac URDF importer, position drives)
            ▼
  usd/so_100_dual.usd    (articulation: bodies, joints, drives, inertia)
            │
            │  setup_scene.py        (assemble + simulate)
            ▼
  Live stage: ground + light + table + 2 arms + 2 wrist cams + cloth
              with the STS3215 motor model applied at runtime.
```

Why three stages instead of importing the URDF every run:

- **Generation is decoupled from Isaac.** `generate_urdf.py` is pure Python, so
  the robot description (limits, mesh paths, structure) can be edited and
  validated without launching the simulator.
- **Import is a one-off, expensive step.** Converting URDF→USD cooks meshes and
  builds the articulation. Caching it as USD makes day-to-day runs fast.
- **Scene assembly is what you iterate on.** `setup_scene.py` only references the
  cached USD and adds the world around it.

---

## 3. Files and responsibilities

| File | Layer | Responsibility |
|------|-------|----------------|
| `scripts/generate_urdf.py` | offline | Emit a clean dual-arm URDF from one arm template (DRY). Adds joint limits + STS3215 effort/velocity, relative mesh paths. |
| `scripts/config.py` | shared | Single source of truth: paths, joint names, ready pose, scene sizes, **motor params**, **camera params**, version shims. |
| `scripts/import_urdf_to_usd.py` | offline | Run Isaac's URDF importer with settings tuned for actuation + contact (position drives, STS3215 baseline gains). |
| `scripts/motors.py` | runtime | OOP actuator model (`FeetechSTS3215`, `SO101MotorModel`) applied to the articulation. |
| `scripts/cameras.py` | runtime | OOP wrist camera (`WristCamera`) + `add_wrist_cameras()` helper. |
| `scripts/cloth_utils.py` | runtime | Build a PhysX particle-cloth square. |
| `scripts/setup_scene.py` | runtime | Orchestrate everything: world, GPU physics, table, robot, cameras, motors, cloth, ready pose, run loop. |
| `scripts/fk_check.py` | dev tool | Forward kinematics (numpy) to choose/verify poses without launching Isaac. |

---

## 4. Robot model

### Kinematics
Two identical arms welded to a shared `world` link: left base at `x = -0.3 m`,
right base at `x = +0.3 m`. Each arm chain:

```
Base → Shoulder_Rotation → Shoulder_Pitch → Elbow → Wrist_Pitch → Wrist_Roll → Gripper(Moving_Jaw)
```

`generate_urdf.py` describes **one** arm (links + joints) and stamps it out for
`left_`/`right_`, so the two arms can never drift out of sync.

### Joint conversion
The source description used limit-less `continuous` joints. We convert them to
`revolute` with explicit limits because a physics solver needs both position
limits and actuator limits. Limits live in `JOINT_LIMITS` in `generate_urdf.py`.

### Ready pose (collision-free by construction)
`READY_POSE` keeps every joint at 0 — which is the original CAD assembly and
therefore self-collision-free — and only swivels `Shoulder_Rotation` by 180°.
That joint turns about the **vertical** axis, so the swivel just rotates each
otherwise-zero arm to face +y (the table/cloth) while staying collision-free.
0 and ±π are also the only angles where a flipped joint-axis sign can't change
the result, so the pose is robust to importer axis conventions. Verified with
`fk_check.py` (grippers settle ≈0.15 m above the table).

The pose is applied **once** after `world.reset()` (PhysX position drives are
persistent — they hold the target on their own) and stored as the articulation's
default state. We deliberately do **not** re-command it every frame, so the
Physics Inspector can take over the targets without fighting the script.

---

## 5. Motor model (the "physics like the real SO-101")

Every SO-101 joint is a **Feetech STS3215** smart serial servo. It is a
*position-controlled* actuator bounded by a fixed stall torque and no-load speed.
We reproduce that, rather than an idealised infinitely-strong motor, in two
complementary places:

**a) In the USD/URDF (static baseline)**
- `generate_urdf.py` writes each joint's `effort = 2.9 N·m` (stall torque, 12 V)
  and `velocity = 4.7 rad/s` (no-load speed). The importer maps these to the
  drive's **max force** and the joint's **max velocity**.
- `import_urdf_to_usd.py` sets the default drive **stiffness/damping** to the
  STS3215 control gains (so even a freshly loaded USD behaves like the servo).

**b) At runtime (authoritative, per-joint)** — `motors.py`
- `FeetechSTS3215` is a dataclass of the four numbers that define one servo:
  `stall_torque`, `no_load_speed`, `stiffness (Kp)`, `damping (Kd)`.
- `SO101MotorModel` holds two specs — one for the arm joints, a lighter-gain one
  for the gripper jaw — and, given the live articulation, builds per-DOF arrays
  and calls `set_gains`, `set_max_efforts`, and the max-velocity setter.

So the simulated joint is a **PD position controller** whose output torque is
clamped to the motor's stall torque and whose speed is clamped to the no-load
speed — exactly the envelope of the real servo.

```
target ──► [ Kp·(θ_target−θ) − Kd·θ̇ ] ──► clamp to ±2.9 N·m ──► joint torque
                                                       (speed clamped to 4.7 rad/s)
```

Default values (12 V follower; all in `config.py`, easy to retune):

| | arm joints | gripper |
|---|---|---|
| stall torque | 2.9 N·m | 2.9 N·m |
| no-load speed | 4.7 rad/s | 4.7 rad/s |
| stiffness Kp | 17.8 N·m/rad | 8.0 N·m/rad |
| damping Kd | 0.6 N·m·s/rad | 0.3 N·m·s/rad |

> Note: gains (Kp/Kd) are not on the datasheet — they are control parameters
> chosen so the joints hold position firmly yet compliantly, like the servo.
> The torque/speed limits *are* datasheet figures and are what make the dynamics
> match. For the 7.4 V build, lower the stall torque (~1.6 N·m).

---

## 6. Wrist cameras

Hardware reference:
[TheRobotStudio Wrist_Cam_Mount_32x32_UVC_Module](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/Wrist_Cam_Mount_32x32_UVC_Module).
It is a 32×32 mm USB/UVC module (≥720p/30 fps, run at 640×480) and the mount
**replaces the wrist-roll part**.

Design choices in `cameras.py`:

- **Parent link = `Fixed_Gripper`.** Because the mount replaces the wrist-roll
  piece, the camera must spin with the wrist-roll joint. In our URDF the body
  that joint drives is `Fixed_Gripper`, so each camera is parented to
  `/World/so_100_dual/{left,right}_Fixed_Gripper` and moves with the gripper.
- **Orientation = look along the gripper approach.** The moving jaw sits toward
  the link's −Y, so the grasp/approach direction is −Y. USD cameras look down
  their local −Z, so a −90° rotation about X aims the camera along −Y.
- **Intrinsics from FOV.** The README doesn't give a field of view, so we expose
  `WRIST_CAM_HFOV_DEG` (default 70°) and solve the USD `horizontalAperture` from
  it; vertical aperture follows the 640×480 aspect ratio.

Two usage layers:

1. `WristCamera.spawn()` always creates the USD `Camera` prim — visible in the
   stage and selectable as the active viewport camera (great for eyeballing the
   wrist view).
2. `WristCamera.make_sensor()` (enabled with `setup_scene.py --capture-cameras`)
   wraps the prim in an Isaac `Camera` sensor so frames can be read in code
   (`get_rgba()`, `get_depth()`) for a folding policy or dataset. It is off by
   default because render products add overhead.

> The mount translation/rotation are sensible defaults, not CAD-exact (the README
> ships STLs, not offsets). They are single constants in `config.py`; drop in
> measured values to match a specific build precisely.

---

## 7. Cloth

`cloth_utils.add_cloth()` builds a flat N×N triangle mesh and turns it into PhysX
**particle cloth** (`add_physx_particle_system` + `add_physx_particle_cloth`).
Particle cloth bends, drapes and self-collides like fabric, which is what a
folding task needs. Stiffness (stretch/bend/shear), damping and resolution are
parameters; lower bend stiffness = floppier, easier-to-fold cloth.

Particle cloth requires **GPU PhysX**, which is why `setup_scene.py` enables GPU
dynamics + GPU broadphase on the physics scene.

> Grasping cloth with the gripper is the genuinely hard part of folding and is
> *not* solved here: particle cloth does not stick to a rigid jaw by default. The
> next step is a PhysX *particle attachment* between a jaw and the nearest cloth
> particles (or high-friction contact). This scene provides the robot + cloth +
> physics + cameras foundation to build that on.

---

## 8. Coordinate frames & placement

- Ground plane at `z = 0`.
- Table top surface at `z = TABLE_HEIGHT` (0.75 m); the slab is centred half a
  thickness below.
- The robot's `world` frame is translated to `z = TABLE_HEIGHT` so the arm bases
  rest on the table top; bases at `x = ±0.3`.
- Cloth laid just above the table top, in +y, within reach of both grippers in
  the ready pose.

---

## 9. Cross-version compatibility

Isaac Sim renamed packages from `omni.isaac.*` (≤4.2) to `isaacsim.*` (≥4.5).
All imports use try/new-then-fallback-old, and feature calls that vary across
builds (max-velocity setter, Camera sensor, default-state) are wrapped so the
scripts degrade gracefully rather than crash. Developed/validated against Isaac
Sim 5.1.

---

## 10. How to extend

- **Different fabric size/stiffness** → `CLOTH_*` in `config.py`.
- **Retune the servos** → motor table in `config.py` (`MOTOR_*`, `*_DRIVE_*`).
- **Camera placement / FOV / resolution** → `WRIST_CAM_*` in `config.py`.
- **New ready pose** → scrub joints in the Physics Inspector, read the angles,
  or use `fk_check.py`, then update `READY_POSE`.
- **Read camera frames** → run with `--capture-cameras` and call
  `cam.sensor.get_rgba()` on the objects returned by `add_wrist_cameras`.
```
