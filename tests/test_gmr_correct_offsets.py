"""
unit tests for GMR offset computation and application.

Based on understanding that:
1. Offsets are in ROBOT's T-pose local frame
2. Offsets rotate with ROBOT's target rotation (not SMPL's)
3. Offsets are CENTERED at roots (because root positions are synced separately)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_gmr_offset(
    smpl_pos: np.ndarray,
    smpl_root: np.ndarray,
    smpl_rot: R,
    robot_pos: np.ndarray,
    robot_root: np.ndarray,
    robot_rot: R,
) -> tuple[np.ndarray, R]:
    """
    Compute offset in GMR format (matches shape_fitting.py:434-440).

    Args:
        smpl_pos: SMPL joint position (global)
        smpl_root: SMPL root position (global)
        smpl_rot: SMPL joint rotation
        robot_pos: Robot joint position (global)
        robot_root: Robot root position (global)
        robot_rot: Robot joint rotation

    Returns:
        pos_offset: Position offset in robot's local frame (centered)
        rot_offset: Rotation offset
    """
    # Rotation offset (line 435)
    rot_offset = smpl_rot.inv() * robot_rot

    # Position offset (line 440) - centered at roots, in robot's local frame
    pos_offset_local = robot_rot.inv().apply(
        (robot_pos - robot_root) - (smpl_pos - smpl_root)
    )

    return pos_offset_local, rot_offset


def apply_gmr_offset(
    smpl_pos: np.ndarray,
    smpl_rot: R,
    pos_offset: np.ndarray,
    rot_offset: R,
) -> tuple[np.ndarray, R]:
    """
    Apply offset in GMR format (matches motion_retarget.py:370-378).

    Note: This assumes smpl_pos and roots are already aligned (by IK solver).

    Args:
        smpl_pos: SMPL joint position at runtime (global)
        smpl_rot: SMPL joint rotation at runtime
        pos_offset: Position offset (robot's local frame, centered)
        rot_offset: Rotation offset

    Returns:
        robot_pos: Robot target position (global)
        robot_rot: Robot target rotation
    """
    # Rotation (line 370)
    robot_rot = smpl_rot * rot_offset

    # Position (line 375) - rotate by robot's target rotation
    global_offset = robot_rot.apply(pos_offset)
    robot_pos = smpl_pos + global_offset

    return robot_pos, robot_rot


class TestGMROffsetLogic:
    """Test GMR offset computation and application logic."""

    def test_identity_with_centered_roots(self):
        """When SMPL and robot are identical (and centered), no offset."""
        # T-pose: Both at same position/rotation
        smpl_root_T = np.array([0.0, 0.0, 1.0])
        robot_root_T = np.array([0.0, 0.0, 1.0])  # Same root

        smpl_shoulder_T = np.array([-0.2, 0.0, 1.5])
        robot_shoulder_T = np.array([-0.2, 0.0, 1.5])  # Same shoulder

        smpl_rot_T = R.identity()
        robot_rot_T = R.identity()

        # Compute offset
        pos_offset, rot_offset = compute_gmr_offset(
            smpl_shoulder_T, smpl_root_T, smpl_rot_T,
            robot_shoulder_T, robot_root_T, robot_rot_T
        )

        # Should be zero
        np.testing.assert_allclose(pos_offset, [0, 0, 0], atol=1e-10)
        assert rot_offset.magnitude() < 1e-10

        # Runtime: Move and rotate
        smpl_root_t = np.array([1.0, 2.0, 1.5])  # Moved
        smpl_shoulder_t = smpl_root_t + (smpl_shoulder_T - smpl_root_T)  # Preserve relative
        smpl_rot_t = R.from_euler('xyz', [30, 45, 60], degrees=True)

        # Apply offset
        robot_shoulder_t, robot_rot_t = apply_gmr_offset(
            smpl_shoulder_t, smpl_rot_t, pos_offset, rot_offset
        )

        # Should match exactly
        np.testing.assert_allclose(robot_shoulder_t, smpl_shoulder_t, atol=1e-10)
        assert (robot_rot_t.inv() * smpl_rot_t).magnitude() < 1e-10


    def test_robot_shoulder_wider(self):
        """Robot has wider shoulders than SMPL."""
        # T-pose: Same roots
        smpl_root_T = np.array([0.0, 0.0, 1.0])
        robot_root_T = np.array([0.0, 0.0, 1.0])

        # SMPL shoulder at -20cm left
        smpl_shoulder_T = np.array([-0.2, 0.0, 1.5])

        # Robot shoulder at -25cm left (5cm wider)
        robot_shoulder_T = np.array([-0.25, 0.0, 1.5])

        smpl_rot_T = R.identity()
        robot_rot_T = R.identity()

        # Compute offset
        pos_offset, rot_offset = compute_gmr_offset(
            smpl_shoulder_T, smpl_root_T, smpl_rot_T,
            robot_shoulder_T, robot_root_T, robot_rot_T
        )

        # Offset in robot's local frame: [-0.05, 0, 0] (5cm left in robot's X)
        np.testing.assert_allclose(pos_offset, [-0.05, 0.0, 0.0], atol=1e-10)

        # Runtime: SMPL rotates 90° around Z
        smpl_root_t = np.array([0.0, 0.0, 1.0])  # Root stays
        smpl_rot_t = R.from_euler('xyz', [0, 0, 90], degrees=True)
        # SMPL shoulder at [0, -0.2, 1.5] (rotated)
        smpl_shoulder_t = np.array([0.0, -0.2, 1.5])

        # Apply offset
        robot_shoulder_t, robot_rot_t = apply_gmr_offset(
            smpl_shoulder_t, smpl_rot_t, pos_offset, rot_offset
        )

        # Robot rotation should match SMPL (90°)
        assert (robot_rot_t.inv() * smpl_rot_t).magnitude() < 1e-6

        # Robot shoulder position:
        # - global_offset = R(90°).apply([-0.05, 0, 0]) = [0, -0.05, 0]
        # - robot_pos = [0, -0.2, 1.5] + [0, -0.05, 0] = [0, -0.25, 1.5]
        expected_pos = np.array([0.0, -0.25, 1.5])
        np.testing.assert_allclose(robot_shoulder_t, expected_pos, atol=1e-6)

        print("✓ Robot maintains 5cm wider shoulders after rotation")


    def test_robot_rotated_relative_to_smpl(self):
        """Robot joint has different orientation than SMPL in T-pose."""
        # T-pose: Same positions
        smpl_root_T = np.array([0.0, 0.0, 1.0])
        robot_root_T = np.array([0.0, 0.0, 1.0])

        smpl_shoulder_T = np.array([-0.2, 0.0, 1.5])
        robot_shoulder_T = np.array([-0.2, 0.0, 1.5])

        # But different rotations
        smpl_rot_T = R.identity()
        robot_rot_T = R.from_euler('xyz', [0, 0, 30], degrees=True)  # 30° offset

        # Compute offset
        pos_offset, rot_offset = compute_gmr_offset(
            smpl_shoulder_T, smpl_root_T, smpl_rot_T,
            robot_shoulder_T, robot_root_T, robot_rot_T
        )

        # Position offset should be zero (same relative positions)
        np.testing.assert_allclose(pos_offset, [0, 0, 0], atol=1e-10)

        # Rotation offset should be 30°
        assert np.abs(rot_offset.magnitude() - np.radians(30)) < 1e-6

        # Runtime: SMPL rotates 60°
        smpl_rot_t = R.from_euler('xyz', [0, 0, 60], degrees=True)
        smpl_shoulder_t = smpl_shoulder_T  # Position unchanged for this test

        robot_shoulder_t, robot_rot_t = apply_gmr_offset(
            smpl_shoulder_t, smpl_rot_t, pos_offset, rot_offset
        )

        # Robot should be at 60° + 30° = 90°
        expected_rot = R.from_euler('xyz', [0, 0, 90], degrees=True)
        assert (robot_rot_t.inv() * expected_rot).magnitude() < 1e-6

        print("✓ Robot maintains 30° rotation offset")


    def test_combined_offset(self):
        """Combined position and rotation offset."""
        # T-pose
        smpl_root_T = np.array([0.0, 0.0, 1.0])
        robot_root_T = np.array([0.0, 0.0, 1.0])

        smpl_shoulder_T = np.array([-0.2, 0.0, 1.5])
        robot_shoulder_T = np.array([-0.25, 0.0, 1.5])  # 5cm wider

        smpl_rot_T = R.identity()
        robot_rot_T = R.from_euler('xyz', [0, 0, 30], degrees=True)  # 30° rotated

        # Compute offset
        pos_offset, rot_offset = compute_gmr_offset(
            smpl_shoulder_T, smpl_root_T, smpl_rot_T,
            robot_shoulder_T, robot_root_T, robot_rot_T
        )

        # Position offset in robot's T-pose frame (rotated 30°)
        # Global diff: [-0.05, 0, 0]
        # In robot's frame: robot_rot_T.inv().apply([-0.05, 0, 0])
        expected_pos_offset = robot_rot_T.inv().apply([-0.05, 0.0, 0.0])
        np.testing.assert_allclose(pos_offset, expected_pos_offset, atol=1e-10)

        # Runtime: SMPL at 60°, shoulder rotated to [0, -0.2, 1.5]
        smpl_rot_t = R.from_euler('xyz', [0, 0, 60], degrees=True)
        smpl_shoulder_t = np.array([0.0, -0.2, 1.5])

        robot_shoulder_t, robot_rot_t = apply_gmr_offset(
            smpl_shoulder_t, smpl_rot_t, pos_offset, rot_offset
        )

        # Robot rotation: 60° + 30° = 90°
        expected_rot = R.from_euler('xyz', [0, 0, 90], degrees=True)
        assert (robot_rot_t.inv() * expected_rot).magnitude() < 1e-6

        # Robot position: Offset rotates with robot's 90° rotation
        global_offset = expected_rot.apply(pos_offset)
        expected_pos = smpl_shoulder_t + global_offset
        np.testing.assert_allclose(robot_shoulder_t, expected_pos, atol=1e-6)

        print("✓ Combined offset works correctly")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
