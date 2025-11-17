import pathlib
from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:
    import torch
    from torch.autograd import Variable
    import joblib
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import mujoco as mj


# Mapping from GMR IK config joint names to SMPL-H bone names
GMR_TO_SMPLH_JOINT_MAP = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "left_foot": "L_Foot",
    "right_foot": "R_Foot",
    "spine1": "Spine1",
    "spine2": "Spine2",
    "spine3": "Spine3",
    "neck": "Neck",
    "head": "Head",
    "left_collar": "L_Collar",
    "right_collar": "R_Collar",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}

# T-pose configurations for different robots (joint angles in radians)
ROBOT_TPOSE_CONFIGS = {
    "unitree_g1": {
        "left_shoulder_roll_joint": np.pi / 2,
        "left_elbow_joint": np.pi / 2,
        "right_shoulder_roll_joint": -np.pi / 2,
        "right_elbow_joint": np.pi / 2,
    },
    "unitree_h1": {
        "left_shoulder_roll": np.pi / 2,
        "left_elbow": np.pi / 2,
        "right_shoulder_roll": -np.pi / 2,
        "right_elbow": np.pi / 2,
    },
}


def check_torch_availability():
    """Check if torch is available for shape fitting."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for shape fitting. Install with:\n"
            "  uv pip install -e '.[shape-fitting]'\n"
            "or:\n"
            "  pip install torch joblib"
        )


def set_robot_to_tpose(
    model,
    data,
    robot_type: str
) -> None:
    """
    Set robot to T-pose by modifying joint positions.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        robot_type: Robot type (e.g., 'unitree_g1')
    """

    # Get T-pose config for this robot
    tpose_config = ROBOT_TPOSE_CONFIGS.get(robot_type, {})

    # Reset all joints to zero first
    data.qpos[7:] = 0.0

    # Apply T-pose joint angles
    for joint_name, angle in tpose_config.items():
        try:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = angle
        except:
            print(f"Warning: Joint {joint_name} not found in robot model")
            continue

    # Forward kinematics
    mj.mj_forward(model, data)


def get_robot_tpose_targets(
    robot_xml_path: str,
    robot_type: str,
    ik_config: Dict,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract target joint positions from robot in T-pose.

    Args:
        robot_xml_path: Path to robot MuJoCo XML
        robot_type: Robot type (e.g., 'unitree_g1')
        ik_config: IK configuration with site-to-joint mappings

    Returns:
        target_positions: (N, 3) array of target joint positions
        smpl_joint_names: List of corresponding SMPL-H joint names
    """
    # Load robot model
    model = mj.MjModel.from_xml_path(robot_xml_path)
    data = mj.MjData(model)

    # Set robot to T-pose
    set_robot_to_tpose(model, data, robot_type)

    # Extract positions for tracked joints from ik_match_table1
    target_positions = []
    smpl_joint_names = []

    for robot_link, smpl_joint_info in ik_config["ik_match_table1"].items():
        ik_joint_name = smpl_joint_info[0]  # GMR IK config name (lowercase)

        # Convert to SMPL-H bone name (capitalized)
        smpl_joint_name = GMR_TO_SMPLH_JOINT_MAP.get(ik_joint_name)
        if smpl_joint_name is None:
            print(f"Warning: No mapping for IK joint {ik_joint_name}")
            continue

        # Get robot link position
        try:
            link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot_link)
            pos = data.xpos[link_id].copy()
            target_positions.append(pos)
            smpl_joint_names.append(smpl_joint_name)
        except Exception as e:
            print(f"Warning: Could not find body {robot_link}: {e}")
            continue

    return np.array(target_positions), smpl_joint_names


def get_smpl_tpose_indices(
    smpl_joint_names: List[str]
) -> np.ndarray:
    """
    Convert SMPL-H joint names to indices in SMPL-H joint array.

    Args:
        smpl_joint_names: List of SMPL-H joint names

    Returns:
        indices: Array of joint indices
    """
    from general_motion_retargeting.utils.smpl import SMPLH_JOINT_NAMES

    indices = []
    for name in smpl_joint_names:
        try:
            idx = SMPLH_JOINT_NAMES.index(name)
            indices.append(idx)
        except ValueError:
            print(f"Warning: SMPL-H joint {name} not found")
            continue

    return np.array(indices)


def fit_smpl_shape_to_robot(
    smpl_parser,
    target_positions: np.ndarray,
    smpl_joint_indices: np.ndarray,
    iterations: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Optimize SMPL-H shape parameters to match robot target positions.

    This implements the core optimization to minimize L2 distance between SMPL
    joint positions and robot targets.

    Args:
        smpl_parser: SMPL-H parser instance
        target_positions: (N, 3) target positions from robot in T-pose
        smpl_joint_indices: (N,) indices of SMPL joints to match
        iterations: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        device: PyTorch device ('cpu' or 'cuda')
        verbose: Print optimization progress

    Returns:
        optimized_shape: (1, 16) optimized beta parameters
        optimized_scale: (1,) global scale factor
        metrics: Dictionary of optimization metrics
    """
    check_torch_availability()

    device = torch.device(device)

    # Convert targets to torch
    target_positions = torch.from_numpy(target_positions).float().to(device)

    # Initialize optimization variables
    shape = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)

    # T-pose for SMPL-H: pelvis rotated to face forward
    pose_aa_tpose = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_tpose[:, 0] = sRot.from_euler("xyz", [np.pi/2, 0.0, np.pi/2], degrees=False).as_rotvec()  # Pelvis
    pose_aa_tpose = torch.from_numpy(pose_aa_tpose.reshape(-1, 156)).float().to(device)
    trans = torch.zeros([1, 3]).to(device)

    optimizer = torch.optim.Adam([shape, scale], lr=lr)

    losses = []

    if verbose:
        try:
            from tqdm import tqdm
            pbar = tqdm(range(iterations), desc="Fitting SMPL shape")
        except ImportError:
            pbar = range(iterations)
            print("Installing tqdm for progress bar: uv pip install tqdm")
    else:
        pbar = range(iterations)

    for iteration in pbar:
        # Forward pass through SMPL
        vertices, joints = smpl_parser.get_joints_verts(
            pose=pose_aa_tpose,
            th_betas=shape,
            th_trans=trans
        )

        # Extract relevant joints and convert to numpy if needed
        if isinstance(joints, np.ndarray):
            predicted_positions = torch.from_numpy(joints[0, smpl_joint_indices]).float().to(device)
        else:
            predicted_positions = joints[0, smpl_joint_indices]

        # Center at pelvis (index 0) and apply scale
        predicted_positions = (predicted_positions - predicted_positions[0:1]) * scale
        target_positions_centered = target_positions - target_positions[0:1]

        # Compute loss (L2 distance)
        loss = (predicted_positions - target_positions_centered).pow(2).sum(dim=-1).mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and hasattr(pbar, 'set_description'):
            if iteration % 100 == 0:
                pbar.set_description(f"Fitting SMPL shape - Loss: {loss.item():.6f}")

    metrics = {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "losses": losses,
        "iterations": iterations,
        "converged": losses[-1] < 0.01,  # Threshold in meters
    }

    return shape.detach(), scale.detach(), metrics


def save_fitted_shape(
    shape: torch.Tensor,
    scale: torch.Tensor,
    metrics: Dict,
    save_path: str,
):
    """Save fitted shape parameters to disk."""
    check_torch_availability()

    save_data = {
        "shape": shape.cpu().numpy(),
        "scale": scale.cpu().numpy(),
        "metrics": metrics,
    }

    # Create directory if needed
    save_dir = pathlib.Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(save_data, save_path)
    print(f"✓ Saved fitted shape to {save_path}")
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print(f"  Converged: {metrics['converged']}")


def load_fitted_shape(load_path: str) -> Tuple[np.ndarray, float, Dict]:
    """Load fitted shape parameters from disk."""
    check_torch_availability()

    data = joblib.load(load_path)
    return data["shape"], data["scale"], data["metrics"]


def compute_smpl_height(shape: np.ndarray, smplh_model_path: str) -> float:
    """Compute actual SMPL-H height from shape parameters in T-pose.

    Args:
        shape: SMPL-H shape parameters (16,)
        smplh_model_path: Path to SMPL-H model

    Returns:
        Height in meters (head top - lowest foot)
    """
    import torch
    from scipy.spatial.transform import Rotation as sRot
    from general_motion_retargeting.utils.smpl import SMPLH_Parser

    smpl_parser = SMPLH_Parser(
        model_path=smplh_model_path,
        gender="neutral",
        use_pca=False,
    )

    # T-pose
    pose_aa_tpose = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_tpose[:, 0] = sRot.from_euler('xyz', [np.pi/2, 0.0, np.pi/2], degrees=False).as_rotvec()
    pose_aa_tpose = torch.from_numpy(pose_aa_tpose.reshape(-1, 156)).float()
    trans = torch.zeros([1, 3])

    # Forward pass
    betas_torch = torch.from_numpy(shape).float().view(1, -1)
    _, joints = smpl_parser.get_joints_verts(pose=pose_aa_tpose, th_betas=betas_torch, th_trans=trans)

    # Height = top of head - lowest foot
    head_idx = 15  # Head joint in SMPL-H
    left_foot_idx = 10  # L_Foot
    right_foot_idx = 11  # R_Foot
    height = float(joints[0, head_idx, 2] - min(joints[0, left_foot_idx, 2], joints[0, right_foot_idx, 2]))

    return height


def compare_scaling_methods(
    original_betas: np.ndarray,
    fitted_shape: np.ndarray,
    fitted_scale: float,
    smplh_model_path: str,
    config_height_assumption: float = 1.8,
) -> Dict:
    """
    Compare original height-based scaling vs fitted shape parameters.

    Args:
        original_betas: Original shape parameters from mocap
        fitted_shape: Optimized shape parameters
        fitted_scale: Optimized global scale
        smplh_model_path: Path to SMPL-H model (required for computing actual heights)
        config_height_assumption: Height assumption in IK config

    Returns:
        Dictionary comparing both methods
    """
    # Compute ACTUAL heights from SMPL-H model (not heuristic!)
    original_height_unscaled = compute_smpl_height(original_betas, smplh_model_path)
    fitted_height_unscaled = compute_smpl_height(fitted_shape[0], smplh_model_path)

    height_ratio = original_height_unscaled / config_height_assumption

    # Compute final effective heights AFTER scaling
    # Original: uses height-based scaling (not directly applied to joints in old GMR)
    # Fitted: scale is applied directly to joints (new MuscleMimic-style approach)
    original_height_scaled = original_height_unscaled  # Not directly scaled, uses ratio in scale_table
    fitted_height_scaled = fitted_height_unscaled * fitted_scale

    comparison = {
        "original": {
            "height_unscaled": float(original_height_unscaled),
            "height_scaled": float(original_height_scaled),
            "scale_ratio": float(height_ratio),
            "beta_0": float(original_betas[0]),
            "method": "height-based linear scaling",
        },
        "fitted": {
            "height_unscaled": float(fitted_height_unscaled),
            "height_scaled": float(fitted_height_scaled),
            "scale": float(fitted_scale),
            "beta_0": float(fitted_shape[0, 0]),
            "shape_params": fitted_shape[0].tolist(),
            "method": "optimized ALL 16 betas + fitted scale",
        },
        "difference": {
            "height_diff_cm": abs(fitted_height_scaled - original_height_scaled) * 100,
            "scale_diff": abs(fitted_scale - height_ratio),
            "beta_l2_norm": float(np.linalg.norm(fitted_shape[0] - original_betas)),
        }
    }

    return comparison
