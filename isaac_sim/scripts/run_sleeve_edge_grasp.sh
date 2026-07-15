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
# z-from-mesh (in the FoldNet policy, tshirt.py): grasp/put heights are read
# from the current cloth mesh once per fold stage — grasp at the layer
# mid-plane (~0.009), put at local cloth top + Z_PUT_OFFSET — instead of the
# blind PICKER_Z=0.02 constant that assumed FleX's teleporting point picker.
# Fixes the grasp-time hop (TCP used to stop 1-2cm above the sleeve, PhysX
# weld yanked the cloth up) and the mid-air release.
# Jaw-table contact is a non-issue: ISAAC_DISABLE_GRIPPER_COLLISIONS=1.
export ISAAC_POLICY_Z_FROM_MESH=1
# Policy rest-mesh rescale (2026-07-10): the FoldNet-side rest mesh is 2.01x
# the Isaac cloth (cloth_scale=0.5 sqrt-normalized vs GARMENT_SCALE=0.35
# direct multiply). fold2's put width comes from REST corner distance
# (tshirt.py xyd_3) -> puts landed 7-12cm outside the cloth (z_from_mesh
# nearest-K warnings: closest 0.068/0.118m) and outside the arm workspace;
# the garbage carry ripped the fold1 sleeve open (back% 28->142 at f=380-420)
# while fold2 itself achieved nothing (release 1cm from grasp). Expect the
# [PolicyRestRescale] line (factor ~0.50) and fold2 nearest-K warnings gone.
export ISAAC_POLICY_REST_MESH_RESCALE=1
export ISAAC_POLICY_Z_MESH_RADIUS=0.03
export ISAAC_POLICY_Z_GRASP_OFFSET=0.0
export ISAAC_POLICY_Z_PUT_OFFSET=0.005
export ISAAC_POLICY_Z_FLOOR=0.005
# Isaac-boundary snap (adjust_target_z) superseded by z-from-mesh; keep off.
export ISAAC_GRASP_Z_FROM_CLOTH=0
export ISAAC_PUT_Z_FROM_CLOTH=0
# Safety floor only — must stay below the mesh-derived grasp z (~0.009) or it
# would clamp it back up.
export ISAAC_POLICY_MIN_GRASP_Z=0.005
export ISAAC_GRASP_SELECTION_SHAPE=box
export ISAAC_MAX_GRASP_VERTICES=2

export ISAAC_SLEEVE_EDGE_GRASP=1
export ISAAC_SLEEVE_EDGE_GRASP_VERTS_PER_EDGE=3
export ISAAC_SLEEVE_EDGE_GRASP_ACTIVATION_RADIUS=0.06
export ISAAC_SLEEVE_EDGE_GRASP_RADIUS=0.02

# Cloth motion diagnostics: [ClothDiag] report every N frames + spike alarms.
export ISAAC_CLOTH_DIAG=1
# Temporarily 10 (normally 30): the fold1 result is destroyed inside the
# f≈350-390 fold2-approach window and the 30f cadence only caught two
# snapshots of it. Revert to 30 once the kick source is identified.
export ISAAC_CLOTH_DIAG_EVERY=10
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
# Tension relaxation before detach (2026-07-10, strain diag): the fold-back is
# elastic stretch recoil — at release the dragged corridor holds patch strain
# p95=7.7% (global baseline 4.0, max 141.9% at the weld) and it decays to
# baseline exactly while back% jumps to 34% in the first 29 frames; friction x3
# and bend /3.5 changed nothing. After the press hold, back the still-welded
# block up along the release->grasp line by this many meters so the corridor
# contracts under control, THEN detach. Free recoil moved the tip 3.4cm, so
# 0.02 stays inside the tension-relief regime. 0 disables (A/B). Watch the
# release line's strain% p95 (expect ~baseline) and the back% series.
export ISAAC_RELEASE_EASE_BACK=0.02
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
# Unroll A/B results (2026-07-10): back% series is nearly identical at
# BEND=35 and BEND=10, and at particleFrictionScale 1.0 and 3.0 — neither
# bend tension nor layer friction drives the fold-back. Current suspect:
# elastic recoil of fabric stretched during the drag (see FoldDiag strain%).
export ISAAC_GARMENT_BEND=35
export ISAAC_GARMENT_SHEAR=50
export ISAAC_GARMENT_DAMPING=0.3
export ISAAC_GARMENT_FRICTION=1.0
# Cloth-on-cloth friction multiplier (particle-particle only; table friction
# unaffected). At 1.0 the pressed sleeve fold slides back open on top of the
# body: FoldDiag back% 29->51% in 60f with coher 0.8-0.9 — elastic bend
# springs at the fold line beat layer friction. PBD springs have no plastic
# bending, so the layer grip must carry what fold-set does in real cotton.
# Effective value is visible in the ClothDiag material dump
# (particleFrictionScale=...). 1.0 (or empty) = old behavior, A/B.
export ISAAC_GARMENT_PARTICLE_FRICTION_SCALE=3.0
export ISAAC_GARMENT_MASS=0.09
export ISAAC_CLOTH_VEL_DAMP=0.95
export ISAAC_CLOTH_MAX_VEL=0.5
# Attachment-creation shock damping (2026-07-10): creating the fold2 PhysX
# attachment mid-sim re-parses the particle cloth (PhysX: "Changing particle
# cloth mesh ... is not supported") and kicks ~ALL particles at once
# (f=380: 4978/5050 moving >0.05, peaks 0.69 m/s) — the placed sleeve fold
# is shaken out to the flat rest shape in ~20 frames (back% 47 -> 155, bbox
# back to 0.341 full flat). Same signature at fold1's attach (f=124:
# 1681/5050) where the flat cloth had nothing to lose. Not the jaw: grippers
# hovered 2mm over the cloth at f=370 with v_mean=0.015. Damp all particle
# velocities to FACTOR per frame for FRAMES frames after each attachment
# creation until contacts re-form. FRAMES=0 disables (A/B).
export ISAAC_ATTACH_DAMP_FRAMES=25
export ISAAC_ATTACH_DAMP_FACTOR=0.4
export ISAAC_KINEMATIC_NEIGHBOR_RADIUS=0.025
export ISAAC_KINEMATIC_NEIGHBOR_VEL_SCALE=0.05

/home/ozan/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    isaac_sim/scripts/generate_dataset_native.py "$@"
