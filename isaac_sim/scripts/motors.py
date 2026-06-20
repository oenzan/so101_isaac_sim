"""
motors.py
=========
Actuator model for the SO-101 arms.

Every joint of the SO-101 is driven by a **Feetech STS3215** smart bus servo.
That servo runs an internal *position* controller and is limited by a fixed
stall torque and no-load speed. The faithful way to reproduce that in PhysX is:

    target position  ->  PD drive (stiffness Kp, damping Kd)  ->  torque,
    with the torque clamped to the motor's stall torque and the joint speed
    clamped to the motor's no-load speed.

This module wraps those parameters in small classes and applies them to the
imported articulation, so the simulated joints behave like the real servos
instead of like idealised, infinitely-strong motors.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class FeetechSTS3215:
    """
    Physical + control model of a single Feetech STS3215 servo.

    stall_torque   : maximum torque the servo can exert            [N.m]
    no_load_speed  : maximum joint speed                            [rad/s]
    stiffness (Kp) : position-loop proportional gain               [N.m/rad]
    damping  (Kd)  : velocity damping of the position loop         [N.m.s/rad]
    """
    stall_torque: float
    no_load_speed: float
    stiffness: float
    damping: float


class SO101MotorModel:
    """
    Applies STS3215 characteristics to a dual-arm articulation.

    The arm joints and the gripper joint share the same servo hardware but are
    tuned with different control gains (the gripper carries far less load), so
    two `FeetechSTS3215` specs are supplied.
    """

    def __init__(self, arm_servo: FeetechSTS3215, gripper_servo: FeetechSTS3215,
                 gripper_tag: str = "Gripper"):
        self._arm = arm_servo
        self._gripper = gripper_servo
        self._gripper_tag = gripper_tag

    def _is_gripper(self, dof_name: str) -> bool:
        return self._gripper_tag in dof_name

    def _spec_for(self, dof_name: str) -> FeetechSTS3215:
        return self._gripper if self._is_gripper(dof_name) else self._arm

    def _per_dof_arrays(self, dof_names: List[str]):
        """Build aligned Kp / Kd / max-torque / max-speed arrays for each DOF."""
        kp, kd, max_tau, max_vel = [], [], [], []
        for name in dof_names:
            spec = self._spec_for(name)
            kp.append(spec.stiffness)
            kd.append(spec.damping)
            max_tau.append(spec.stall_torque)
            max_vel.append(spec.no_load_speed)
        return (np.array(kp, dtype=np.float32),
                np.array(kd, dtype=np.float32),
                np.array(max_tau, dtype=np.float32),
                np.array(max_vel, dtype=np.float32))

    def apply(self, articulation) -> None:
        """
        Push the motor model onto an *initialised* articulation
        (call after world.reset() so the physics handles exist).
        """
        dof_names = list(articulation.dof_names)
        kp, kd, max_tau, max_vel = self._per_dof_arrays(dof_names)

        controller = articulation.get_articulation_controller()

        # PD gains: how stiffly the servo tracks its target position.
        try:
            controller.set_gains(kps=kp, kds=kd)
        except Exception as exc:
            print(f"[motors] set_gains failed: {exc}")

        # Torque ceiling: the joint can never exert more than the stall torque.
        try:
            controller.set_max_efforts(values=max_tau)
        except Exception as exc:
            print(f"[motors] set_max_efforts failed: {exc}")

        # Speed ceiling: clamp joint velocity to the servo's no-load speed.
        # (API name varies across Isaac builds; try the common ones.)
        for setter in ("set_max_joint_velocities", "set_max_velocities"):
            fn = getattr(articulation, setter, None)
            if fn is not None:
                try:
                    fn(max_vel)
                    break
                except Exception as exc:
                    print(f"[motors] {setter} failed: {exc}")

        print(f"[motors] applied STS3215 model to {len(dof_names)} joints "
              f"(arm Kp={self._arm.stiffness}, grip Kp={self._gripper.stiffness}, "
              f"max torque={self._arm.stall_torque} N.m)")


def build_default_model(cfg) -> SO101MotorModel:
    """Construct the motor model from values in config.py."""
    arm = FeetechSTS3215(
        stall_torque=cfg.MOTOR_STALL_TORQUE_NM,
        no_load_speed=cfg.MOTOR_NO_LOAD_SPEED_RAD_S,
        stiffness=cfg.ARM_DRIVE_STIFFNESS,
        damping=cfg.ARM_DRIVE_DAMPING,
    )
    gripper = FeetechSTS3215(
        stall_torque=cfg.MOTOR_STALL_TORQUE_NM,
        no_load_speed=cfg.MOTOR_NO_LOAD_SPEED_RAD_S,
        stiffness=cfg.GRIPPER_DRIVE_STIFFNESS,
        damping=cfg.GRIPPER_DRIVE_DAMPING,
    )
    return SO101MotorModel(arm, gripper, gripper_tag=cfg.GRIPPER_JOINT_TAG)
