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
# On Isaac Sim 6.0+ (which removed PhysxPhysicsAttachment) deformable
# attachments are parse-time only -- runtime insertion/updates are ignored, so
# a grasp-time weld is impossible. The code auto-falls back to the kinematic
# block grasp there; the rigid ISAAC_BLOCK_KINEMATIC_* values below make that
# path emulate the weld (they are unused on builds with the real attachment).
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
export ISAAC_POLICY_MIN_GRASP_Z=0.028
export ISAAC_GRASP_SELECTION_SHAPE=box
# 6 held vertices (matches the 3-per-edge sleeve grasp below) gives the rigid
# kinematic weld a wide enough pinch to carry the fabric without tearing free.
export ISAAC_MAX_GRASP_VERTICES=6

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
# Rigid kinematic weld emulation (only used where PhysX attachments are
# unavailable, i.e. Isaac Sim 6.0+): full blend, generous per-step travel and
# tight catchup so grasped vertices track the block like a weld instead of
# lagging behind and slipping off (the old soft values, blend=0.35 /
# max_step=0.006, lost the cloth mid-carry). Verified: sleeve lifts 5cm and
# stays held through the whole fold1 carry.
export ISAAC_BLOCK_KINEMATIC_POSTSTEP_ONLY=1
export ISAAC_BLOCK_KINEMATIC_BLEND=1.0
export ISAAC_BLOCK_KINEMATIC_MAX_STEP=0.05
export ISAAC_BLOCK_KINEMATIC_CATCHUP_STEP=0.05
export ISAAC_BLOCK_KINEMATIC_CATCHUP_ERROR=0.005
export ISAAC_BLOCK_TARGET_BLEND=1.0
export ISAAC_BLOCK_TARGET_DEADBAND=0.0

# Mesh mean edge length is ~4.8mm at 0.35 scale (Ozan, feature_clocth_stability).
# solid_rest_offset must stay well below that: stacked layers settle at
# 2*solid_rest_offset apart, and the old value (=contact offset, 0.006) forced
# folded layers 12mm apart, making the shirt inflate and slide around by
# itself after the sleeve was placed. Applies to both backends -- see
# garment_loader.py's _add_particle_cloth/_add_surface_deformable.
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
export ISAAC_CLOTH_MAX_VEL=1.0
# Wider neighbor drag so fabric around the rigidly-held vertices follows the
# carry instead of stretching (kinematic path only).
export ISAAC_KINEMATIC_NEIGHBOR_RADIUS=0.04
export ISAAC_KINEMATIC_NEIGHBOR_VEL_SCALE=0.2

resolve_isaac_python() {
    if [[ -n "${ISAAC_SIM_PYTHON:-}" && -x "${ISAAC_SIM_PYTHON}" ]]; then
        printf '%s\n' "${ISAAC_SIM_PYTHON}"
        return 0
    fi

    local candidate
    for candidate in \
        "$HOME/isaacsim/python.sh" \
        "$HOME/isaac-sim/python.sh" \
        "$HOME/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh" \
        "$HOME/.local/share/ov/pkg"/isaac-sim*/python.sh
    do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ERROR: set ISAAC_SIM_PYTHON to the Isaac Sim python.sh path" >&2
    echo "       or install Isaac Sim in one of the standard locations." >&2
    exit 1
}

ISAAC_PYTHON="$(resolve_isaac_python)"
"${ISAAC_PYTHON}" \
    isaac_sim/scripts/generate_dataset_native.py "$@"
