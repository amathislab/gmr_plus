"""Loader for SOMA-format BVH files.

This mirrors upstream GMR PR #169. It returns frames in the same contract as
the LAFAN1/Nokov BVH loader: per-frame world positions in meters and
scalar-first quaternions.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .lafan_vendor import utils as fk_utils

_ROT_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
_CHANNEL_RE = re.compile(r"\s*CHANNELS\s+(\d+)\s+(.+)")
_JOINT_RE = re.compile(r"\s*(ROOT|JOINT)\s+(\S+)")
_OFFSET_RE = re.compile(r"\s*OFFSET\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)")
_FRAMES_RE = re.compile(r"\s*Frames:\s+(\d+)")
_FRAME_TIME_RE = re.compile(r"\s*Frame Time:\s+([\d\.eE]+)")


def _parse_bvh(bvh_file: str | Path) -> dict:
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []

    stack: list[int] = []
    end_site = False
    in_motion = False
    frames_total: int | None = None
    frame_time: float | None = None
    motion_rows: list[list[float]] = []

    with Path(bvh_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not in_motion:
                if "MOTION" in line:
                    in_motion = True
                    continue

                joint_match = _JOINT_RE.match(line)
                if joint_match:
                    names.append(joint_match.group(2))
                    parents.append(stack[-1] if stack else -1)
                    offsets.append([0.0, 0.0, 0.0])
                    channels.append([])
                    stack.append(len(names) - 1)
                    continue

                if "End Site" in line:
                    end_site = True
                    continue
                if "{" in line:
                    continue
                if "}" in line:
                    if end_site:
                        end_site = False
                    else:
                        stack.pop()
                    continue

                offset_match = _OFFSET_RE.match(line)
                if offset_match and not end_site:
                    offsets[stack[-1]] = [float(value) for value in offset_match.groups()]
                    continue

                channel_match = _CHANNEL_RE.match(line)
                if channel_match:
                    channels[stack[-1]] = channel_match.group(2).split()
                    continue
            else:
                frame_match = _FRAMES_RE.match(line)
                if frame_match:
                    frames_total = int(frame_match.group(1))
                    continue

                time_match = _FRAME_TIME_RE.match(line)
                if time_match:
                    frame_time = float(time_match.group(1))
                    continue

                stripped = line.strip()
                if stripped:
                    motion_rows.append([float(value) for value in stripped.split()])

    if frame_time is None:
        raise ValueError(f"Could not find 'Frame Time' in BVH file: {bvh_file}")

    motion = np.asarray(motion_rows, dtype=np.float32)
    if frames_total is not None and motion.shape[0] != frames_total:
        motion = motion[:frames_total]

    return {
        "names": names,
        "parents": np.asarray(parents, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.float32),
        "channels": channels,
        "motion": motion,
        "frame_time": frame_time,
    }


def _extract_local_pose(parsed: dict) -> tuple[np.ndarray, np.ndarray]:
    names = parsed["names"]
    offsets = parsed["offsets"]
    motion = parsed["motion"]
    channels = parsed["channels"]

    num_frames = motion.shape[0]
    num_joints = len(names)
    local_pos = np.tile(offsets[None, :, :], (num_frames, 1, 1)).astype(np.float32)
    local_quat = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_quat[..., 0] = 1.0

    cursor = 0
    for joint_idx, joint_channels in enumerate(channels):
        if not joint_channels:
            continue

        joint_slice = motion[:, cursor : cursor + len(joint_channels)]
        cursor += len(joint_channels)

        rot_order = ""
        rot_values = []
        for col, channel in enumerate(joint_channels):
            if channel == "Xposition":
                local_pos[:, joint_idx, 0] = joint_slice[:, col]
            elif channel == "Yposition":
                local_pos[:, joint_idx, 1] = joint_slice[:, col]
            elif channel == "Zposition":
                local_pos[:, joint_idx, 2] = joint_slice[:, col]
            elif channel in _ROT_CHANNELS:
                rot_order += channel[0].upper()
                rot_values.append(joint_slice[:, col])
            else:
                raise ValueError(f"Unsupported BVH channel '{channel}' on joint '{names[joint_idx]}'")

        if rot_order:
            eulers_deg = np.stack(rot_values, axis=-1)
            quat_xyzw = R.from_euler(rot_order, eulers_deg, degrees=True).as_quat()
            local_quat[:, joint_idx, 0] = quat_xyzw[:, 3]
            local_quat[:, joint_idx, 1:] = quat_xyzw[:, :3]

    return local_pos, local_quat


def load_soma_bvh_file(bvh_file: str | Path) -> tuple[list[dict[str, list[np.ndarray]]], float]:
    parsed = _parse_bvh(bvh_file)
    names = parsed["names"]
    parents = parsed["parents"].tolist()

    local_pos, local_quat = _extract_local_pose(parsed)
    global_quat, global_pos = fk_utils.quat_fk(local_quat, local_pos, parents)

    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True).astype(np.float32)

    frames = []
    for frame_idx in range(global_pos.shape[0]):
        result = {}
        for joint_idx, bone_name in enumerate(names):
            orientation = fk_utils.quat_mul(rotation_quat, global_quat[frame_idx, joint_idx])
            position = global_pos[frame_idx, joint_idx] @ rotation_matrix.T / 100.0
            result[bone_name] = [position, orientation]
        frames.append(result)

    return frames, 1.70
