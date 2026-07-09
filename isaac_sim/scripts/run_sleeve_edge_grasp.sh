#!/usr/bin/env bash
# Run native Isaac Sim dataset generation with sleeve dual-edge grasp
# (top+bottom, front+back layers) and softened garment physics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export ISAAC_HEADLESS=0
export ISAAC_GRASP_MODE=block_attachment
export ISAAC_BLOCK_ATTACHMENT_HANDS=right
export ISAAC_ATTACHMENT_HANDS=none
export ISAAC_KINEMATIC_GRASP_HANDS=none
# PhysX attachment, GarmentLab recipe: dynamic block (mass 1000, no gravity),
# auto attachment (PhysxAutoAttachmentAPI computes the attachment points),
# collision enabled for vertex capture but pair-filtered vs cloth/table,
# block driven by velocity commands (no position teleports).
export ISAAC_BLOCK_ATTACHMENT_USE_PHYSX=1
export ISAAC_BLOCK_ATTACHMENT_AUTO=1
export ISAAC_BLOCK_ATTACHMENT_COLLISION=1
# Sphere of 1.5cm diameter placed exactly at the TCP (matches the gripper's
# ~1.5cm contact surface). Capture radius = size/2 + overlap = 0.0075+0.01.
export ISAAC_BLOCK_ATTACHMENT_SHAPE=sphere
export ISAAC_BLOCK_ATTACHMENT_SIZE=0.015
export ISAAC_BLOCK_ATTACHMENT_OVERLAP=0.01
export ISAAC_BLOCK_ATTACHMENT_MASS=1000
export ISAAC_KINEMATIC_MASS_SCALE=1
export ISAAC_DISABLE_GRIPPER_COLLISIONS=1
# 0.015: let the policy's own grasp z (PICKER_Z=0.02) through. The old 0.028
# floor kept the TCP ~2cm above the sleeve (cloth z=[0.006,0.016]), so the
# PhysX attachment yanked the cloth up to the sphere at grasp (visible hop).
# Jaw-table contact is a non-issue: ISAAC_DISABLE_GRIPPER_COLLISIONS=1.
export ISAAC_POLICY_MIN_GRASP_Z=0.015
export ISAAC_GRASP_SELECTION_SHAPE=box
export ISAAC_MAX_GRASP_VERTICES=2

export ISAAC_SLEEVE_EDGE_GRASP=1
export ISAAC_SLEEVE_EDGE_GRASP_VERTS_PER_EDGE=3
export ISAAC_SLEEVE_EDGE_GRASP_ACTIVATION_RADIUS=0.06
export ISAAC_SLEEVE_EDGE_GRASP_RADIUS=0.02

# Cloth motion diagnostics: [ClothDiag] report every N frames + spike alarms.
export ISAAC_CLOTH_DIAG=1
export ISAAC_CLOTH_DIAG_EVERY=30
export ISAAC_CLOTH_DIAG_SPIKE_VEL=0.25

# Diagnostic: show the red attachment cube so we can see where PhysX welds.
export ISAAC_BLOCK_ATTACHMENT_VISIBLE=1
export ISAAC_BLOCK_DEBUG_MARKER=1
export ISAAC_BLOCK_DIAGNOSTIC_SNAP=0
export ISAAC_RIGHT_GRASP_SQUEEZE_FACTOR=1.0,1.0,1.0
export ISAAC_RELEASE_DAMP_FRAMES=12
# Press-then-release: FoldNet opens the gripper at z~0.04 (4.6cm in the air),
# so the un-anchored fold arch unrolls back (FoldDiag: back +17%->+103% while
# airborne, then immobile once flat). Keep the attachment, drive the block
# down to table+HEIGHT, hold FRAMES, then detach. FRAMES=0 disables (A/B).
export ISAAC_RELEASE_PRESS_FRAMES=15
export ISAAC_RELEASE_PRESS_HEIGHT=0.010
export ISAAC_RELEASE_PRESS_TIMEOUT=60
export ISAAC_BLOCK_KINEMATIC_POSTSTEP_ONLY=1
export ISAAC_BLOCK_KINEMATIC_BLEND=0.35
export ISAAC_BLOCK_KINEMATIC_MAX_STEP=0.006
export ISAAC_BLOCK_KINEMATIC_CATCHUP_STEP=0.010
export ISAAC_BLOCK_KINEMATIC_CATCHUP_ERROR=0.012
export ISAAC_BLOCK_TARGET_BLEND=0.40
export ISAAC_BLOCK_TARGET_DEADBAND=0.002

# Mesh mean edge length is ~4.8mm at 0.35 scale. solid_rest_offset must stay
# well below that: stacked layers settle at 2*solid_rest_offset apart, and the
# old value (=contact offset, 0.006) forced folded layers 12mm apart, making
# the shirt inflate and slide around by itself after the sleeve was placed.
# 0.0025 -> layers rest ~5mm apart; contact offset 0.006 keeps detection wide.
export ISAAC_GARMENT_CONTACT_OFFSET=0.006
export ISAAC_GARMENT_SOLID_REST_OFFSET=0.0025

export ISAAC_GARMENT_STRETCH=8000
export ISAAC_GARMENT_BEND=35
export ISAAC_GARMENT_SHEAR=50
export ISAAC_GARMENT_DAMPING=0.3
export ISAAC_GARMENT_FRICTION=1.0
export ISAAC_GARMENT_MASS=0.09
export ISAAC_CLOTH_VEL_DAMP=0.95
export ISAAC_CLOTH_MAX_VEL=0.5
export ISAAC_KINEMATIC_NEIGHBOR_RADIUS=0.025
export ISAAC_KINEMATIC_NEIGHBOR_VEL_SCALE=0.05

/home/ozan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    isaac_sim/scripts/generate_dataset_native.py "$@"
