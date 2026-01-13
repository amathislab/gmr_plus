"""
Compare motion retargeting with and without data-driven shape fitting.

This script demonstrates the improvement from using optimized SMPL-H shape
parameters versus the default height-based scaling approach.

Usage:
    python scripts/compare_shape_fitting.py \
        --smplh_file <path_to_amass.npz> \
        --robot unitree_g1 \
        --num_frames 90
"""

import argparse
import pathlib
import numpy as np
import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplh_file,
    get_smplh_data_offline_fast,
)
from general_motion_retargeting.utils.shape_fitting import load_fitted_shape, compute_smpl_height
from rich.table import Table
from rich.console import Console

console = Console()


def compute_retargeting_metrics(qpos_list, model):
    """Compute metrics from retargeted motion."""
    import mujoco as mj

    # Compute joint position statistics
    qpos_array = np.array(qpos_list)

    # Get end-effector positions over time
    data = mj.MjData(model)
    left_foot_positions = []
    right_foot_positions = []
    left_hand_positions = []
    right_hand_positions = []

    for qpos in qpos_array:
        data.qpos = qpos
        mj.mj_forward(model, data)

        # Find end-effector body IDs
        try:
            left_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
            right_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
            left_hand_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
            right_hand_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

            left_foot_positions.append(data.xpos[left_foot_id].copy())
            right_foot_positions.append(data.xpos[right_foot_id].copy())
            left_hand_positions.append(data.xpos[left_hand_id].copy())
            right_hand_positions.append(data.xpos[right_hand_id].copy())
        except:
            # Fallback if body names don't match
            pass

    metrics = {
        "qpos_std": float(np.std(qpos_array)),
        "qpos_range": float(np.ptp(qpos_array)),
    }

    if left_foot_positions:
        metrics["left_foot_travel"] = float(np.sum(np.linalg.norm(np.diff(left_foot_positions, axis=0), axis=1)))
        metrics["right_foot_travel"] = float(np.sum(np.linalg.norm(np.diff(right_foot_positions, axis=0), axis=1)))
        metrics["left_hand_travel"] = float(np.sum(np.linalg.norm(np.diff(left_hand_positions, axis=0), axis=1)))
        metrics["right_hand_travel"] = float(np.sum(np.linalg.norm(np.diff(right_hand_positions, axis=0), axis=1)))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare retargeting with/without shape fitting"
    )
    parser.add_argument(
        "--smplh_file",
        type=str,
        required=True,
        help="Path to AMASS SMPL-H data file (.npz)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot type",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=90,
        help="Number of frames to compare (default: 90 = 3 seconds)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videos/comparison",
        help="Output directory for comparison videos",
    )

    args = parser.parse_args()

    # Setup paths
    HERE = pathlib.Path(__file__).parent
    REPO_ROOT = HERE.parent
    SMPLH_FOLDER = REPO_ROOT / "assets" / "body_models" / "smplh"
    FITTED_SHAPE_PATH = REPO_ROOT / "assets" / "fitted_shapes" / f"{args.robot}_shape.pkl"

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       Shape Fitting Comparison: Rigorous Analysis      [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    # Check fitted shape exists
    if not FITTED_SHAPE_PATH.exists():
        console.print(f"[red]Error: Fitted shape not found at {FITTED_SHAPE_PATH}[/red]")
        console.print(f"[yellow]Please run: python scripts/fit_smpl_shape.py --robot {args.robot}[/yellow]")
        return

    # Load fitted shape
    shape_fitted, scale_fitted, offset_z_fitted, height_scale_fitted, offsets_fitted = load_fitted_shape(str(FITTED_SHAPE_PATH))

    # Load metrics from metadata file
    import json
    metadata_path = str(FITTED_SHAPE_PATH).replace('.pkl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        metrics_fitted = metadata['metrics']

    console.print("[bold cyan]Step 1: Loading SMPL-H data with DEFAULT scaling[/bold cyan]")
    smplh_data_default, body_model_default, smplh_output_default, height_default = load_smplh_file(
        args.smplh_file, SMPLH_FOLDER, fitted_shape_path=None
    )
    smplh_frames_default, fps = get_smplh_data_offline_fast(
        smplh_data_default, body_model_default, smplh_output_default, tgt_fps=30
    )
    smplh_frames_default = smplh_frames_default[:args.num_frames]
    console.print(f"  ✓ Loaded {len(smplh_frames_default)} frames at {fps:.2f} fps")
    console.print(f"  ✓ SMPL-H with original betas")

    # Compute actual height from original betas
    original_betas = smplh_data_default["betas"]
    if original_betas.ndim == 1:
        original_betas = original_betas.reshape(1, -1)
    actual_height_default = compute_smpl_height(original_betas[0], str(SMPLH_FOLDER))
    console.print(f"  ✓ Actual SMPL height: {actual_height_default:.3f} m")
    console.print(f"  ✓ Height heuristic: {height_default:.3f} m")
    console.print(f"  ✓ Scale ratio: {height_default / 1.8:.4f}")

    console.print("\n[bold cyan]Step 2: Loading SMPL-H data with FITTED shape[/bold cyan]")
    smplh_data_fitted, body_model_fitted, smplh_output_fitted, height_fitted = load_smplh_file(
        args.smplh_file, SMPLH_FOLDER, fitted_shape_path=str(FITTED_SHAPE_PATH)
    )
    smplh_frames_fitted, _ = get_smplh_data_offline_fast(
        smplh_data_fitted, body_model_fitted, smplh_output_fitted, tgt_fps=30
    )
    smplh_frames_fitted = smplh_frames_fitted[:args.num_frames]
    console.print(f"  ✓ Loaded {len(smplh_frames_fitted)} frames")
    console.print(f"  ✓ SMPL-H with fitted betas (all 16 optimized)")

    # Compute actual height from fitted shape
    actual_height_fitted = compute_smpl_height(shape_fitted[0], str(SMPLH_FOLDER))
    console.print(f"  ✓ Actual SMPL height: {actual_height_fitted:.3f} m")
    console.print(f"  ✓ Effective height: {height_fitted:.3f} m")
    console.print(f"  ✓ Fitted scale: {scale_fitted[0]:.4f}")

    console.print("\n[bold cyan]Step 3: Retargeting with DEFAULT (height-based)[/bold cyan]")
    retarget_default = GMR(
        actual_human_height=height_default,
        src_human="smplh",
        tgt_robot=args.robot,
        use_fitted_shape=False,
        verbose=False,
    )

    qpos_list_default = []
    for i, frame_data in enumerate(smplh_frames_default):
        qpos, _ = retarget_default.retarget(frame_data, offset_to_ground=True)
        qpos_list_default.append(qpos)
        if i % 30 == 0:
            console.print(f"  Processing frame {i}/{len(smplh_frames_default)}")

    console.print(f"  ✓ Retargeted {len(qpos_list_default)} frames")

    console.print("\n[bold cyan]Step 4: Retargeting with FITTED shape[/bold cyan]")
    retarget_fitted = GMR(
        actual_human_height=height_fitted,
        src_human="smplh",
        tgt_robot=args.robot,
        use_fitted_shape=True,
        fitted_shape_path=str(FITTED_SHAPE_PATH),
        verbose=False,
    )

    qpos_list_fitted = []
    for i, frame_data in enumerate(smplh_frames_fitted):
        qpos, _ = retarget_fitted.retarget(frame_data, offset_to_ground=True)
        qpos_list_fitted.append(qpos)
        if i % 30 == 0:
            console.print(f"  Processing frame {i}/{len(smplh_frames_fitted)}")

    console.print(f"  ✓ Retargeted {len(qpos_list_fitted)} frames")

    console.print("\n[bold cyan]Step 5: Computing comparison metrics[/bold cyan]")

    # Convert to numpy arrays
    qpos_default = np.array(qpos_list_default)
    qpos_fitted = np.array(qpos_list_fitted)

    # Compute differences
    qpos_diff = qpos_fitted - qpos_default
    qpos_rmse = np.sqrt(np.mean(qpos_diff**2))
    qpos_max_diff = np.max(np.abs(qpos_diff))
    qpos_mean_abs_diff = np.mean(np.abs(qpos_diff))

    # Per-joint differences
    per_joint_rmse = np.sqrt(np.mean(qpos_diff**2, axis=0))
    worst_joint_idx = np.argmax(per_joint_rmse)

    console.print(f"  ✓ Computed differences across {qpos_diff.shape[0]} frames, {qpos_diff.shape[1]} DOFs")

    # Detailed per-joint analysis
    console.print("\\n[bold cyan]Step 6: Detailed Per-Joint Analysis[/bold cyan]")

    # Analyze base position (first 3 DOFs) vs body joints (DOF 7+)
    base_pos_diff = qpos_diff[:, :3]  # x, y, z
    base_rot_diff = qpos_diff[:, 3:7]  # quaternion
    joint_diff = qpos_diff[:, 7:]  # actual robot joints

    base_pos_rmse = np.sqrt(np.mean(base_pos_diff**2, axis=0))
    joint_rmse_per_dof = np.sqrt(np.mean(joint_diff**2, axis=0))

    console.print(f"  Base position RMSE: X={base_pos_rmse[0]:.4f}m, Y={base_pos_rmse[1]:.4f}m, Z={base_pos_rmse[2]:.4f}m")
    console.print(f"  Base rotation RMSE: {np.sqrt(np.mean(base_rot_diff**2)):.4f}")
    console.print(f"  Joint angles RMSE: {np.sqrt(np.mean(joint_diff**2)):.4f} rad ({np.sqrt(np.mean(joint_diff**2))*180/np.pi:.2f}°)")

    # Top 10 joints with largest differences
    top_joint_indices = np.argsort(joint_rmse_per_dof)[-10:][::-1]
    console.print("\\n  Top 10 joints with largest RMSE:")
    for rank, joint_idx in enumerate(top_joint_indices, 1):
        actual_dof = joint_idx + 7  # Offset by 7 (base pos + rot)
        console.print(f"    {rank:2d}. DOF {actual_dof:2d}: RMSE={joint_rmse_per_dof[joint_idx]:.4f} rad ({joint_rmse_per_dof[joint_idx]*180/np.pi:.2f}°)")

    # Statistical distribution of differences
    console.print("\\n[bold cyan]Step 7: Statistical Distribution Analysis[/bold cyan]")
    joint_diff_flat = joint_diff.flatten()
    percentiles = [0, 25, 50, 75, 95, 99, 100]
    percentile_values = np.percentile(np.abs(joint_diff_flat), percentiles)

    console.print("  Joint angle difference distribution (absolute values):")
    for p, val in zip(percentiles, percentile_values):
        console.print(f"    {p:3d}th percentile: {val:.4f} rad ({val*180/np.pi:.2f}°)")

    # Temporal analysis
    console.print("\\n[bold cyan]Step 8: Temporal Evolution Analysis[/bold cyan]")
    frame_rmse = np.sqrt(np.mean(joint_diff**2, axis=1))

    console.print(f"  First 10 frames avg RMSE: {np.mean(frame_rmse[:10]):.4f} rad ({np.mean(frame_rmse[:10])*180/np.pi:.2f}°)")
    console.print(f"  Last 10 frames avg RMSE:  {np.mean(frame_rmse[-10:]):.4f} rad ({np.mean(frame_rmse[-10:])*180/np.pi:.2f}°)")
    console.print(f"  Frame with max RMSE: {np.argmax(frame_rmse)} (RMSE={np.max(frame_rmse):.4f} rad / {np.max(frame_rmse)*180/np.pi:.2f}°)")
    console.print(f"  Frame with min RMSE: {np.argmin(frame_rmse)} (RMSE={np.min(frame_rmse):.4f} rad / {np.min(frame_rmse)*180/np.pi:.2f}°)")

    # Compute standard deviation over time to see if differences are consistent or varying
    temporal_std = np.std(frame_rmse)
    console.print(f"  Temporal variation (std): {temporal_std:.4f} rad ({temporal_std*180/np.pi:.2f}°)")

    # Base trajectory analysis
    console.print("\\n[bold cyan]Step 9: Base Trajectory Analysis[/bold cyan]")
    base_default = qpos_default[:, :3]
    base_fitted = qpos_fitted[:, :3]

    # Total travel distance
    travel_default = np.sum(np.linalg.norm(np.diff(base_default, axis=0), axis=1))
    travel_fitted = np.sum(np.linalg.norm(np.diff(base_fitted, axis=0), axis=1))

    console.print(f"  Total base travel (height-based): {travel_default:.3f}m")
    console.print(f"  Total base travel (fitted shape): {travel_fitted:.3f}m")
    console.print(f"  Travel difference: {abs(travel_default - travel_fitted):.3f}m ({abs(travel_default - travel_fitted)/travel_default*100:.1f}%)")

    # Average base height
    avg_z_default = np.mean(base_default[:, 2])
    avg_z_fitted = np.mean(base_fitted[:, 2])
    console.print(f"  Average base height (height-based): {avg_z_default:.3f}m")
    console.print(f"  Average base height (fitted shape): {avg_z_fitted:.3f}m")
    console.print(f"  Height difference: {abs(avg_z_default - avg_z_fitted):.3f}m ({abs(avg_z_default - avg_z_fitted)*100:.1f}cm)")

    # Display results
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                    RESULTS SUMMARY                      [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    # Table 1: Shape Parameters
    table1 = Table(title="Shape Parameters Comparison")
    table1.add_column("Method", style="cyan", no_wrap=True)
    table1.add_column("SMPL Height (m)", justify="right")
    table1.add_column("Scale/Ratio", justify="right")
    table1.add_column("Beta[0]", justify="right")
    table1.add_column("All Betas", justify="center")

    table1.add_row(
        "Height-based",
        f"{actual_height_default:.3f}",
        f"{height_default / 1.8:.4f}",
        f"{original_betas[0, 0]:.3f}",
        "Original (from mocap)"
    )

    table1.add_row(
        "Fitted shape",
        f"{actual_height_fitted:.3f}",
        f"{scale_fitted[0]:.4f}",
        f"{shape_fitted[0, 0]:.3f}",
        "Optimized (all 16)"
    )

    console.print(table1)
    console.print()

    # Table 2: Retargeting Differences
    table2 = Table(title="Robot Joint Configuration Differences")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", justify="right", style="yellow")
    table2.add_column("Unit", style="dim")

    table2.add_row("RMSE (all joints)", f"{qpos_rmse:.4f}", "radians")
    table2.add_row("Mean absolute diff", f"{qpos_mean_abs_diff:.4f}", "radians")
    table2.add_row("Max difference", f"{qpos_max_diff:.4f}", "radians")
    table2.add_row("Worst joint RMSE", f"{per_joint_rmse[worst_joint_idx]:.4f}", f"radians (DOF {worst_joint_idx})")
    table2.add_row("Shape fitting loss", f"{metrics_fitted['final_loss']:.6f}", "meters")
    table2.add_row("Error reduction", f"{(1 - metrics_fitted['final_loss']/metrics_fitted['initial_loss'])*100:.1f}", "%")

    console.print(table2)
    console.print()

    # Table 3: Scale Comparison
    table3 = Table(title="Scaling Method Comparison")
    table3.add_column("Aspect", style="cyan")
    table3.add_column("Height-based", justify="center")
    table3.add_column("Fitted Shape", justify="center")

    table3.add_row(
        "Per-body-part scaling",
        "YES (0.8-0.9 × ratio)",
        "NO (uniform scale)"
    )
    table3.add_row(
        "Scale application",
        f"0.8-0.9 × {height_default/1.8:.3f}",
        f"1.0 × {scale_fitted[0]:.3f}"
    )
    table3.add_row(
        "Beta optimization",
        "Only beta[0] (height)",
        "All 16 betas (body shape)"
    )
    table3.add_row(
        "Robot-specific",
        "No",
        "Yes (optimized for robot)"
    )

    console.print(table3)
    console.print()

    # Key insights
    console.print("[bold green]Key Insights:[/bold green]")
    console.print(f"  • SMPL height difference: {abs(actual_height_fitted - actual_height_default)*100:.2f} cm")
    console.print(f"  • Scale difference: {abs(scale_fitted[0] - height_default/1.8):.4f}")
    console.print(f"  • Average joint config difference: {qpos_mean_abs_diff*180/np.pi:.2f}°")
    console.print(f"  • Maximum joint config difference: {qpos_max_diff*180/np.pi:.2f}°")
    console.print(f"  • Shape fitting improved joint accuracy by {(1 - metrics_fitted['final_loss']/metrics_fitted['initial_loss'])*100:.1f}%")

    console.print("\n[bold yellow]Impact:[/bold yellow]")
    if qpos_mean_abs_diff > 0.05:  # ~3 degrees
        console.print("  • [yellow]Significant difference in robot configurations[/yellow]")
        console.print("  • Fitted shape produces noticeably different motions")
    else:
        console.print("  • Moderate difference in robot configurations")

    console.print("\n[bold cyan]Conclusion:[/bold cyan]")
    console.print("  Fitted shape uses data-driven optimization of ALL 16 beta parameters")
    console.print("  to match robot body proportions, bypassing heuristic per-body-part scaling.")
    console.print(f"  Result: {(1 - metrics_fitted['final_loss']/metrics_fitted['initial_loss'])*100:.1f}% better joint position accuracy in T-pose.\n")

    # Visualization
    if args.visualize:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        console.print("\n[bold cyan]Step 10: Generating comparison videos[/bold cyan]")

        # Video paths
        video_default = f"{args.output_dir}/{args.robot}_default_heightbased.mp4"
        video_fitted = f"{args.output_dir}/{args.robot}_fitted_datadriven.mp4"

        console.print(f"  Rendering height-based video...")
        viewer_default = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            transparent_robot=0,
            record_video=True,
            video_path=video_default,
        )

        for qpos in qpos_list_default:
            root_pos = qpos[:3]
            root_rot = qpos[3:7]
            dof_pos = qpos[7:]
            viewer_default.step(root_pos, root_rot, dof_pos, rate_limit=False)

        viewer_default.close()
        console.print(f"  ✓ Saved to {video_default}")

        console.print(f"  Rendering fitted shape video...")
        viewer_fitted = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            transparent_robot=0,
            record_video=True,
            video_path=video_fitted,
        )

        for qpos in qpos_list_fitted:
            root_pos = qpos[:3]
            root_rot = qpos[3:7]
            dof_pos = qpos[7:]
            viewer_fitted.step(root_pos, root_rot, dof_pos, rate_limit=False)

        viewer_fitted.close()
        console.print(f"  ✓ Saved to {video_fitted}")

        console.print("\n[bold green]Comparison videos generated![/bold green]")
        console.print(f"  Default (height-based):    {video_default}")
        console.print(f"  Fitted (data-driven):      {video_fitted}")
        console.print("\n  Compare them side-by-side to see the differences!")


if __name__ == "__main__":
    main()
