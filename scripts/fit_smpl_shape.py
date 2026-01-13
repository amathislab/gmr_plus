"""
Fit SMPL-H shape parameters to match a specific robot's proportions.

This script optimizes SMPL-H beta parameters to minimize the difference
between SMPL joint positions and robot link positions in T-pose.

Usage:
    # Install torch first
    uv pip install -e '.[shape-fitting]'

    # Fit shape for a robot
    python scripts/fit_smpl_shape.py --robot unitree_g1 --iterations 1000

    # Compare with existing height-based method
    python scripts/fit_smpl_shape.py --robot unitree_g1 --compare
"""

import argparse
import pathlib
import json
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.utils.smpl import SMPLH_Parser
from general_motion_retargeting.utils.shape_fitting import (
    get_robot_tpose_targets,
    get_smpl_tpose_indices,
    fit_smpl_shape_to_robot,
    save_fitted_shape,
    load_fitted_shape,
    compare_scaling_methods,
    check_torch_availability,
    compute_alignment_offsets,
)
from rich import print
from rich.table import Table
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Fit SMPL-H shape to robot proportions"
    )
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        choices=list(ROBOT_XML_DICT.keys()),
        help="Robot type to fit",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for optimization",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default=None,
        help="Path to SMPL-H models (default: assets/body_models/smplh)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save fitted shape (default: assets/fitted_shapes/{robot}_shape.pkl)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fitted shape with height-based scaling",
    )

    args = parser.parse_args()

    # Check torch availability
    try:
        check_torch_availability()
    except ImportError as e:
        print(f"[red]Error: {e}[/red]")
        return

    # Setup paths
    HERE = pathlib.Path(__file__).parent
    smpl_model_path = args.smpl_model_path or (HERE.parent / "assets" / "body_models" / "smplh")
    save_path = args.save_path or (HERE.parent / "assets" / "fitted_shapes" / f"{args.robot}_shape.pkl")

    console.print(f"\n[bold cyan]Step 1: Loading robot and IK configuration[/bold cyan]")
    console.print(f"  Robot: [green]{args.robot}[/green]")

    # Load robot XML and IK config (use SMPL-H configs for shape fitting)
    robot_xml_path = str(ROBOT_XML_DICT[args.robot])
    ik_config_path = IK_CONFIG_DICT["smplh"][args.robot]

    with open(ik_config_path, 'r') as f:
        ik_config = json.load(f)

    console.print(f"  Robot XML: {robot_xml_path}")
    console.print(f"  IK Config: {ik_config_path}")

    console.print(f"\n[bold cyan]Step 2: Extracting robot T-pose targets[/bold cyan]")
    target_positions, smpl_joint_names, target_rotations, human_joint_names = get_robot_tpose_targets(
        robot_xml_path,
        args.robot,
        ik_config,
        include_rotations=True,
    )

    console.print(f"  Found {len(target_positions)} target joints")
    console.print(f"  Target joints: {', '.join(smpl_joint_names[:5])}...")

    console.print(f"\n[bold cyan]Step 3: Loading SMPL-H model[/bold cyan]")
    smpl_parser = SMPLH_Parser(
        model_path=str(smpl_model_path),
        gender="neutral",
        use_pca=False,
    )
    console.print(f"  SMPL-H model loaded from {smpl_model_path}")

    # Get SMPL joint indices
    smpl_joint_indices = get_smpl_tpose_indices(smpl_joint_names)

    console.print(f"\n[bold cyan]Step 4: Optimizing shape parameters[/bold cyan]")
    console.print(f"  Iterations: {args.iterations}")
    console.print(f"  Learning rate: {args.lr}")
    console.print(f"  Device: {args.device}\n")

    # Fit shape
    (shape, scale, smpl2robot_pos, smpl2robot_rot_mat,
     offset_z, height_scale, metrics) = fit_smpl_shape_to_robot(
        smpl_parser,
        target_positions,
        smpl_joint_indices,
        target_rotations=target_rotations,  # Pass rotations
        iterations=args.iterations,
        lr=args.lr,
        device=args.device,
        verbose=True,
        robot_type=args.robot,  # Robot-specific SMPL T-pose rotation
    )

    # Compute LOCAL frame offsets for GMR's offset_human_data() function
    # This converts global frame offsets to robot-local frame offsets compatible with JSON config format
    console.print(f"\n[bold cyan]Step 4.5: Computing LOCAL frame offsets for IK compatibility[/bold cyan]")
    local_offsets = compute_alignment_offsets(
        smpl_parser,
        shape,
        scale,
        smpl_joint_indices,
        human_joint_names,
        target_positions,
        target_rotations if target_rotations is not None else np.eye(3)[None].repeat(len(target_positions), axis=0),
        robot_type=args.robot,  # Robot-specific SMPL T-pose rotation
    )
    console.print(f"  ✓ Computed LOCAL frame offsets for {len(local_offsets['pos_offsets'])} joints")

    # Save results
    console.print(f"\n[bold cyan]Step 5: Saving fitted shape[/bold cyan]")
    save_fitted_shape(
        shape, scale,
        smpl2robot_pos, smpl2robot_rot_mat,
        offset_z, height_scale,
        metrics,
        str(save_path),
        human_joint_names=human_joint_names,
        musclemimic_format=True,
        local_offsets=local_offsets,  # Pass LOCAL offsets for GMR
    )

    # Print results
    console.print(f"\n[bold green]✓ Shape fitting complete![/bold green]")
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Final loss: {metrics['final_loss']:.6f} m")
    console.print(f"  Initial loss: {metrics['initial_loss']:.6f} m")
    console.print(f"  Improvement: {(1 - metrics['final_loss']/metrics['initial_loss'])*100:.1f}%")
    console.print(f"  Converged: {'Yes' if metrics['converged'] else 'No'}")
    console.print(f"  Global scale: {scale[0].item():.4f}")
    console.print(f"  Beta[0] (height): {shape[0, 0].item():.4f}")
    console.print(f"  offset_z: {offset_z:.4f} m")
    console.print(f"  height_scale: {height_scale:.4f}")
    console.print(f"  Learned offsets stored for {len(human_joint_names)} joints")

    # Comparison mode
    if args.compare:
        console.print(f"\n[bold cyan]Step 6: Comparing with GMR baseline (main branch)[/bold cyan]")

        # Load a sample AMASS motion to get original betas (for reference)
        sample_amass = "/media/data/share/AMASS/CMU/CMU/01/01_01_poses.npz"
        try:
            amass_data = np.load(sample_amass, allow_pickle=True)
            original_betas = amass_data['betas'][:16] if 'betas' in amass_data else np.zeros(16)
        except:
            original_betas = np.zeros(16)

        human_scale_table = ik_config.get("human_scale_table", {})

        # GMR baseline: actual_human_height=None means ratio=1.0 (same as main branch default)
        comparison = compare_scaling_methods(
            original_betas,
            shape.cpu().numpy(),
            scale[0].item(),
            str(smpl_model_path),
            ik_config.get("human_height_assumption", 1.8),
            human_scale_table,
            actual_human_height=None,  # main branch default
        )

        # Display comparison table
        table = Table(title="Scaling Method Comparison")
        table.add_column("Method", style="cyan")
        table.add_column("Height Ratio", justify="right")
        table.add_column("Pelvis Scale\n(effective)", justify="right")
        table.add_column("Beta[0]", justify="right")

        # GMR baseline: ratio=1.0 when actual_human_height=None
        gmr = comparison['gmr_baseline']
        table.add_row(
            "GMR Baseline",
            f"{gmr['height_ratio']:.4f}",
            f"{gmr['pelvis_base_scale']:.2f} × {gmr['height_ratio']:.2f} = {gmr['pelvis_effective_scale']:.4f}",
            f"{comparison['reference']['original_beta_0']:.4f}",
        )
        # Fitted: uniform scale applied to all joints
        table.add_row(
            "Fitted",
            f"-",
            f"{comparison['fitted']['scale']:.4f} (uniform)",
            f"{comparison['fitted']['beta_0']:.4f}",
        )

        console.print()
        console.print(table)

        # Show per-body scales from JSON
        if human_scale_table:
            console.print()
            console.print("[bold]GMR JSON per-body scales (ratio=1.0):[/bold]")
            for body, base_scale in sorted(human_scale_table.items()):
                console.print(f"  {body}: {base_scale:.2f}")

        console.print()
        console.print(f"[bold]Differences:[/bold]")
        console.print(f"  Fitted scale vs GMR pelvis: {comparison['difference']['fitted_vs_gmr_pelvis_scale']:.4f}")
        console.print(f"  Beta L2 norm: {comparison['difference']['beta_l2_norm']:.4f}")

    console.print(f"\n[bold green]Done![/bold green] Fitted shape saved to: {save_path}\n")


if __name__ == "__main__":
    main()
