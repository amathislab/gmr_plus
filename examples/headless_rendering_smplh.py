"""
Example: Headless Rendering with SMPL-H Data

This example demonstrates how to use GMR for headless video rendering
with SMPL-H (AMASS) motion data. This is useful for batch processing
on servers without displays.

Requirements:
- Set MUJOCO_GL=egl environment variable for headless rendering
- SMPL-H body models in assets/body_models/smplh/
- AMASS motion data file

Usage:
    MUJOCO_GL=egl python examples/headless_rendering_smplh.py \
        --smplh_file <path_to_amass_data.npz> \
        --robot unitree_g1 \
        --num_frames 90
"""

import argparse
import pathlib
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplh_file,
    get_smplh_data_offline_fast,
)

# Parse arguments
parser = argparse.ArgumentParser(description="Headless rendering example with SMPL-H data")
parser.add_argument("--smplh_file", type=str, required=True, help="Path to AMASS SMPL-H data file (.npz)")
parser.add_argument("--robot", type=str, default="unitree_g1", help="Robot type")
parser.add_argument("--output_video", type=str, default="outputs/headless_smplh_example.mp4", help="Output video path")
parser.add_argument("--num_frames", type=int, default=90, help="Number of frames to process (default: 90 = 3 seconds)")
args = parser.parse_args()

# Configuration
REPO_ROOT = pathlib.Path(__file__).parent.parent
SMPLH_FOLDER = REPO_ROOT / "assets" / "body_models" / "smplh"
SMPLH_FILE = args.smplh_file
ROBOT = args.robot
OUTPUT_VIDEO = args.output_video
NUM_FRAMES = args.num_frames

print(f"[Headless Rendering Example]")
print(f"  Robot: {ROBOT}")
print(f"  Motion data: {SMPLH_FILE}")
print(f"  Output: {OUTPUT_VIDEO}")
print(f"  Frames: {NUM_FRAMES}")

# Load SMPL-H motion data
print("\n[1/4] Loading SMPL-H motion data...")
smplh_data, body_model, smplh_output, actual_human_height = load_smplh_file(
    SMPLH_FILE, SMPLH_FOLDER
)

# Align fps and limit frames
print("[2/4] Aligning FPS and preparing frames...")
tgt_fps = 30
smplh_data_frames, aligned_fps = get_smplh_data_offline_fast(
    smplh_data, body_model, smplh_output, tgt_fps=tgt_fps
)
smplh_data_frames = smplh_data_frames[:NUM_FRAMES]
print(f"  Processing {len(smplh_data_frames)} frames at {aligned_fps:.2f} fps")

# Initialize retargeting
print("[3/4] Initializing retargeting system...")
retarget = GMR(
    actual_human_height=actual_human_height,
    src_human="smplh",
    tgt_robot=ROBOT,
)

# Initialize viewer with headless video recording
robot_motion_viewer = RobotMotionViewer(
    robot_type=ROBOT,
    motion_fps=aligned_fps,
    transparent_robot=0,
    record_video=True,
    video_path=OUTPUT_VIDEO,
)

# Process and render frames
print("[4/4] Rendering video...")
for i, smplh_data in enumerate(smplh_data_frames):
    if i % 30 == 0:
        print(f"  Frame {i}/{len(smplh_data_frames)}")

    # Retarget human motion to robot
    qpos, _ = retarget.retarget(smplh_data, offset_to_ground=True)

    # Render frame to video
    robot_motion_viewer.step(
        root_pos=qpos[:3],
        root_rot=qpos[3:7],
        dof_pos=qpos[7:],
        human_motion_data=retarget.scaled_human_data,
        rate_limit=False,  # No rate limiting for batch processing
    )

# Finalize video
robot_motion_viewer.close()
print(f"\n✓ Done! Video saved to: {OUTPUT_VIDEO}")
