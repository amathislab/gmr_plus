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
    "myofullbody": {
        "shoulder_elv_r": np.pi / 2,
        "shoulder_elv_l": np.pi / 2,
    },
}

# SMPL T-pose pelvis rotation (Euler XYZ in radians) for each robot's coordinate system
# SMPL native: Y-up, arms along X axis
# After rotation: should align with robot's coordinate convention
SMPL_TPOSE_ROTATIONS = {
    # myofullbody: forward=+Y in SMPL native -> forward=+X in MuJoCo (Z-up)
    # Only need Y-up to Z-up conversion (π/2 about X)
    "myofullbody": [np.pi / 2, 0.0, 0.0],

    # unitree_g1: forward=+Y in SMPL native -> forward=+X in G1's frame
    # G1 uses different axis convention, needs extra yaw
    "unitree_g1": [np.pi / 2, 0.0, np.pi / 2],

    # unitree_h1: same as G1
    "unitree_h1": [np.pi / 2, 0.0, np.pi / 2],
}

# Default rotation for unknown robots
SMPL_TPOSE_DEFAULT = [np.pi / 2, 0.0, 0.0]


def get_smpl_tpose_rotation(robot_type: str) -> np.ndarray:
    """Get the SMPL T-pose pelvis rotation for a specific robot."""
    euler_xyz = SMPL_TPOSE_ROTATIONS.get(robot_type, SMPL_TPOSE_DEFAULT)
    return sRot.from_euler("xyz", euler_xyz, degrees=False).as_rotvec()


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
    include_rotations: bool = False,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[List[str]]]:
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

    target_positions = []
    target_rotations = []
    smpl_joint_names = []
    human_joint_names = []

    for robot_link, smpl_joint_info in ik_config["ik_match_table1"].items():
        ik_joint_name = smpl_joint_info[0]  # GMR IK config name (lowercase)

        smpl_joint_name = GMR_TO_SMPLH_JOINT_MAP.get(ik_joint_name)
        if smpl_joint_name is None:
            print(f"Warning: No mapping for IK joint {ik_joint_name}")
            continue

        try:
            link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot_link)
            pos = data.xpos[link_id].copy()
            target_positions.append(pos)
            smpl_joint_names.append(smpl_joint_name)
            human_joint_names.append(ik_joint_name)
            if include_rotations:
                rot_flat = data.xmat[link_id].copy()
                target_rotations.append(rot_flat.reshape(3, 3))
        except Exception as e:
            print(f"Warning: Could not find body {robot_link}: {e}")
            continue

    rotations = np.array(target_rotations) if include_rotations and target_rotations else None
    return np.array(target_positions), smpl_joint_names, rotations, human_joint_names


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
    target_rotations: Optional[np.ndarray] = None,
    iterations: int = 1000,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    robot_type: str = "myofullbody",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, float, float, Dict]:
    """
    Optimize SMPL-H shape parameters to match robot target positions.

    This implements the core optimization to minimize L2 distance between SMPL
    joint positions and robot targets.

    Args:
        smpl_parser: SMPL-H parser instance
        target_positions: (N, 3) target positions from robot in T-pose
        smpl_joint_indices: (N,) indices of SMPL joints to match
        target_rotations: (N, 3, 3) optional target rotation matrices from robot
        iterations: Number of optimization iterations
        lr: Learning rate for Adam optimizer
        device: PyTorch device ('cpu' or 'cuda')
        verbose: Print optimization progress
        robot_type: Robot type for selecting correct SMPL T-pose rotation

    Returns:
        optimized_shape: (1, 16) optimized beta parameters
        optimized_scale: (1,) global scale factor
        smpl2robot_pos: (N, 3) position offsets
        smpl2robot_rot_mat: (N, 3, 3) rotation offset matrices
        offset_z: float, vertical offset correction
        height_scale: float, height scaling factor
        metrics: Dictionary of optimization metrics
    """
    check_torch_availability()

    device = torch.device(device)

    # Convert targets to torch
    target_positions = torch.from_numpy(target_positions).float().to(device)

    # Initialize optimization variables
    shape = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)

    # T-pose for SMPL-H: pelvis rotated to align with robot's coordinate system
    pose_aa_tpose = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_tpose[:, 0] = get_smpl_tpose_rotation(robot_type)  # Robot-specific pelvis rotation
    pose_aa_tpose = torch.from_numpy(pose_aa_tpose.reshape(-1, 156)).float().to(device)
    trans = torch.zeros([1, 3]).to(device)

    optimizer = torch.optim.Adam([shape, scale], lr=lr)

    losses = []

    # Track heights for computing offset_z and height_scale
    init_feet_z = None
    init_head_z = None

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
        # Use get_joints_verts for optimization (guarantees gradients flow)
        vertices, joints = smpl_parser.get_joints_verts(
            pose=pose_aa_tpose,
            th_betas=shape,
            th_trans=trans
        )

        # Convert to tensor if needed
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints).float().to(device)

        # Track heights on first iteration
        if init_feet_z is None:
            # Indices: 10=L_Foot, 11=R_Foot, 15=Head in SMPL-H
            init_feet_z = min(
                joints[0, 10, 2].detach().cpu().item(),
                joints[0, 11, 2].detach().cpu().item()
            )
            init_head_z = joints[0, 15, 2].detach().cpu().item()

        # Extract relevant joints
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

    # After optimization, compute final parameters
    with torch.no_grad():
        # Use get_joints_verts for consistency with optimization loop
        vertices, joints = smpl_parser.get_joints_verts(
            pose=pose_aa_tpose,
            th_betas=shape,
            th_trans=trans
        )

        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints).float()

        # Final heights (using same method as initial heights)
        final_feet_z = min(joints[0, 10, 2].item(), joints[0, 11, 2].item())
        final_head_z = joints[0, 15, 2].item()

        # Now get transformations for rotation offsets (if available)
        # Always use joints for positions (more reliable); rotations from transforms if available
        if hasattr(smpl_parser, 'get_joint_transformations'):
            transforms = smpl_parser.get_joint_transformations(
                pose=pose_aa_tpose,
                th_betas=shape,
                th_trans=trans
            )
            global_rot = transforms[..., :3, :3]
        else:
            transforms = None
            global_rot = None  # No rotations available

        global_pos = joints  # positions from forward kinematics

        # Apply scale to positions
        root_pos = global_pos[:, 0:1, :]
        global_pos = (global_pos - root_pos) * scale + root_pos

        # Compute position offsets
        # Center by pelvis to match the optimization objective frame
        smpl_pos = global_pos[0, smpl_joint_indices].cpu().numpy()
        target_pos_np = target_positions.cpu().numpy()
        smpl2robot_pos = smpl_pos - target_pos_np
        root_delta = smpl2robot_pos[0].copy()  # pelvis assumed to be first
        smpl2robot_pos = smpl2robot_pos - root_delta

        # Compute rotation offsets (R_offset = R_smpl^T @ R_robot)
        if target_rotations is not None and global_rot is not None:
            smpl_rot = global_rot[0, smpl_joint_indices].cpu().numpy()  # (N, 3, 3)
            # R_offset = R_smpl^T @ R_robot
            smpl2robot_rot_mat = np.einsum('nij,njk->nik',
                                           smpl_rot.transpose(0, 2, 1),
                                           target_rotations)
        else:
            # No rotations available, use identity
            smpl2robot_rot_mat = np.eye(3)[None].repeat(len(smpl_joint_indices), axis=0)

        # Compute offset_z and height_scale
        offset_z = float(init_feet_z - final_feet_z) if init_feet_z is not None else 0.0
        height_scale = float((final_head_z - final_feet_z) / (init_head_z - init_feet_z)) if init_head_z is not None else 1.0

    metrics = {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "losses": losses,
        "iterations": iterations,
        "converged": losses[-1] < 0.01,  # Threshold in meters
        "offset_z": offset_z,
        "height_scale": height_scale,
    }

    return shape.detach(), scale.detach(), smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale, metrics


def compute_alignment_offsets(
    smpl_parser,
    shape: torch.Tensor,
    scale: torch.Tensor,
    smpl_joint_indices: np.ndarray,
    human_joint_names: List[str],
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    robot_type: str = "myofullbody",
) -> Dict:
    """Compute SMPL→robot offsets using full joint transforms.

    Extracts a global rotation offset (from pelvis) and per-joint LOCAL rotation
    offsets (relative to pelvis). This separates the coordinate frame difference
    (global) from per-joint alignment corrections (local).

    Rotation convention:
        - global_rot_offset: R_smpl_pelvis^T @ R_robot_pelvis (coordinate axis flip)
        - local_rot_offset[joint]: global_rot_offset^T @ (R_smpl[joint]^T @ R_robot[joint])
        - Application: R_new = R_smpl @ global_rot_offset @ local_rot_offset[joint]

    Position offset (robot local frame):
        pos_offset = R_robot^T @ (p_robot - p_smpl)  [centered at pelvis]

    All quaternions stored in wxyz format.
    """
    check_torch_availability()

    device = shape.device
    pose_aa_tpose = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_tpose[:, 0] = get_smpl_tpose_rotation(robot_type)  # Robot-specific pelvis rotation
    pose_aa_tpose = torch.from_numpy(pose_aa_tpose.reshape(-1, 156)).float().to(device)
    trans = torch.zeros([1, 3]).to(device)

    # Positions from joints (more trustworthy), rotations from transformations if available
    with torch.no_grad():
        vertices, joints = smpl_parser.get_joints_verts(
            pose=pose_aa_tpose,
            th_betas=shape,
            th_trans=trans
        )
        if hasattr(smpl_parser, 'get_joint_transformations'):
            transforms = smpl_parser.get_joint_transformations(
                pose_aa_tpose,
                th_betas=shape,
                th_trans=trans,
            )
            global_rot = transforms[..., :3, :3]
        else:
            global_rot = None

    global_pos = joints  # torch tensor
    root_pos = global_pos[:, 0:1, :]
    global_pos = (global_pos - root_pos) * float(scale.view(-1)[0]) + root_pos

    global_pos = global_pos.detach().cpu().numpy()[0]
    if global_rot is not None:
        global_rot = global_rot.detach().cpu().numpy()[0]
    else:
        global_rot = np.repeat(np.eye(3)[None], global_pos.shape[0], axis=0)

    # Find pelvis index for centering (fallback to first if missing)
    try:
        root_idx = human_joint_names.index("pelvis")
    except ValueError:
        root_idx = 0

    smpl_root = global_pos[smpl_joint_indices[root_idx]]
    robot_root = target_positions[root_idx]

    # Compute global rotation offset from pelvis
    pelvis_smpl_idx = smpl_joint_indices[root_idx]
    pelvis_smpl_rot = global_rot[pelvis_smpl_idx]
    pelvis_robot_rot = target_rotations[root_idx] if target_rotations is not None else np.eye(3)
    global_rot_offset_mat = pelvis_smpl_rot.T @ pelvis_robot_rot  # R_smpl^T @ R_robot

    # Convert global offset to quaternion (wxyz)
    global_quat_xyzw = sRot.from_matrix(global_rot_offset_mat).as_quat()
    global_rot_offset_wxyz = [
        float(global_quat_xyzw[3]),
        float(global_quat_xyzw[0]),
        float(global_quat_xyzw[1]),
        float(global_quat_xyzw[2]),
    ]

    offsets = {
        "pos_offsets": {},
        "rot_offsets": {},  # LOCAL offsets (relative to pelvis)
        "global_rot_offset": global_rot_offset_wxyz,  # Coordinate frame difference
        "human_joint_names": human_joint_names,
    }

    for idx, human_name in enumerate(human_joint_names):
        smpl_idx = smpl_joint_indices[idx]
        smpl_pos = global_pos[smpl_idx]
        smpl_rot = global_rot[smpl_idx]
        robot_pos = target_positions[idx]
        robot_rot = target_rotations[idx] if target_rotations is not None else np.eye(3)

        # Full rotation offset (global frame): R_smpl^T @ R_robot
        full_rot_offset_mat = smpl_rot.T @ robot_rot

        # LOCAL rotation offset: global^T @ full = (R_smpl_pelvis^T @ R_robot_pelvis)^T @ (R_smpl^T @ R_robot)
        # This removes the global coordinate frame difference, leaving only per-joint correction
        local_rot_offset_mat = global_rot_offset_mat.T @ full_rot_offset_mat

        # Convert to quaternion (wxyz)
        local_quat_xyzw = sRot.from_matrix(local_rot_offset_mat).as_quat()
        local_quat_wxyz = [
            float(local_quat_xyzw[3]),
            float(local_quat_xyzw[0]),
            float(local_quat_xyzw[1]),
            float(local_quat_xyzw[2]),
        ]

        # Position offset in robot local frame (centered at pelvis)
        pos_offset_local = robot_rot.T @ ((robot_pos - robot_root) - (smpl_pos - smpl_root))

        offsets["pos_offsets"][human_name] = [float(x) for x in pos_offset_local]
        offsets["rot_offsets"][human_name] = local_quat_wxyz

    return offsets


def save_fitted_shape(
    shape: torch.Tensor,
    scale: torch.Tensor,
    smpl2robot_pos: np.ndarray,
    smpl2robot_rot_mat: np.ndarray,
    offset_z: float,
    height_scale: float,
    metrics: Dict,
    save_path: str,
    human_joint_names: Optional[List[str]] = None,
    musclemimic_format: bool = True,
    local_offsets: Optional[Dict] = None,
):
    """Save fitted shape parameters in MuscleMimic-compatible tuple format.

    Args:
        shape: (1, 16) optimized beta parameters
        scale: (1,) global scale factor
        smpl2robot_pos: (N, 3) position offsets
        smpl2robot_rot_mat: (N, 3, 3) rotation offset matrices
        offset_z: vertical offset correction
        height_scale: height scaling factor
        metrics: optimization metrics
        save_path: path to save file
        human_joint_names: list of human joint names (for mapping offsets)
        musclemimic_format: if True, save as tuple; if False, save as dict
    """
    check_torch_availability()

    if musclemimic_format:
        # MuscleMimic tuple format: (shape, scale, pos, rot, offset_z, height_scale)
        save_data = (
            shape.detach().cpu(),
            scale.detach().cpu(),
            smpl2robot_pos,
            smpl2robot_rot_mat,
            offset_z,
            height_scale
        )
    else:
        # Old GMR dict format (for backward compatibility)
        save_data = {
            "shape": shape.cpu().numpy(),
            "scale": scale.cpu().numpy(),
            "smpl2robot_pos": smpl2robot_pos,
            "smpl2robot_rot_mat": smpl2robot_rot_mat,
            "offset_z": offset_z,
            "height_scale": height_scale,
            "metrics": metrics,
            "human_joint_names": human_joint_names,
        }

    # Create directory if needed
    save_dir = pathlib.Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(save_data, save_path)

    # Save metrics and joint names as JSON for analysis
    import json
    metadata_path = save_path.replace('.pkl', '_metadata.json')
    metadata = {
        "metrics": metrics,
        "human_joint_names": human_joint_names,
        "offset_z": offset_z,
        "height_scale": height_scale,
        "local_offsets": local_offsets if local_offsets else None,  # LOCAL frame offsets for GMR IK
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved fitted shape to {save_path}")
    print(f"  Format: {'MuscleMimic tuple' if musclemimic_format else 'GMR dict'}")
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print(f"  Converged: {metrics['converged']}")
    print(f"  offset_z: {offset_z:.4f} m")
    print(f"  height_scale: {height_scale:.4f}")
    print(f"  Metadata saved to {metadata_path}")


def load_fitted_shape(load_path: str) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[Dict]]:
    """Load fitted shape parameters from disk.

    Handles both MuscleMimic tuple format and old GMR dict format.

    Returns:
        shape: (1, 16) or (16,) shape parameters
        scale: (1,) or float scale factor
        offset_z: vertical offset
        height_scale: height scaling factor
        offsets: optional dict with per-joint offsets (GMR format only)
    """
    check_torch_availability()

    data = joblib.load(load_path)

    if isinstance(data, tuple):
        # MuscleMimic tuple format: (shape, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale)
        if len(data) == 6:
            shape, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale = data

            # Convert torch tensors to numpy if needed
            if hasattr(shape, 'numpy'):
                shape = shape.numpy()
            if hasattr(scale, 'numpy'):
                scale = scale.numpy()

            # Try to load LOCAL offsets from metadata (preferred for GMR IK)
            import json
            metadata_path = load_path.replace('.pkl', '_metadata.json')
            human_joint_names = None
            local_offsets = None
            try:
                if pathlib.Path(metadata_path).exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        human_joint_names = metadata.get("human_joint_names")
                        local_offsets = metadata.get("local_offsets")  # LOCAL frame offsets
            except Exception:
                pass  # Metadata file not found or invalid

            if human_joint_names is None:
                raise ValueError(
                    "Metadata file missing or does not contain human_joint_names. "
                    "Cannot load fitted shape without joint name mapping."
                )

            # Use LOCAL offsets if available (computed by compute_alignment_offsets)
            # These are compatible with GMR's offset_human_data() function
            if local_offsets is not None and "pos_offsets" in local_offsets and "rot_offsets" in local_offsets:
                offsets = local_offsets  # Already in correct format!
            else:
                # Fall back to converting global offsets (old format)
                print("[load_fitted_shape] Warning: using global (uncentered) offsets fallback; metadata missing local_offsets.")
                from scipy.spatial.transform import Rotation as R
                offsets = {
                    "pos_offsets": {
                        name: smpl2robot_pos[i].tolist() if hasattr(smpl2robot_pos[i], 'tolist') else smpl2robot_pos[i]
                        for i, name in enumerate(human_joint_names)
                    },
                    "rot_offsets": {
                        name: R.from_matrix(smpl2robot_rot_mat[i]).as_quat(scalar_first=True).tolist()
                        for i, name in enumerate(human_joint_names)
                    },
                    "human_joint_names": human_joint_names
                }

            return shape, scale, offset_z, height_scale, offsets
        else:
            raise ValueError(f"Unexpected tuple length: {len(data)}")

    elif isinstance(data, dict):
        # Old GMR dict format
        shape = data["shape"]
        scale = data["scale"]
        offset_z = data.get("offset_z", 0.0)
        height_scale = data.get("height_scale", 1.0)

        # Try to get offsets in various formats
        offsets = data.get("offsets")
        if offsets is None and "smpl2robot_pos" in data:
            offsets = {
                "smpl2robot_pos": data["smpl2robot_pos"],
                "smpl2robot_rot_mat": data.get("smpl2robot_rot_mat"),
            }

        return shape, scale, offset_z, height_scale, offsets

    else:
        raise ValueError(f"Unknown fitted shape format: {type(data)}")


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
    pose_aa_tpose[:, 0] = sRot.from_euler('xyz', [np.pi/2, 0.0, 0.0], degrees=False).as_rotvec()
    pose_aa_tpose = torch.from_numpy(pose_aa_tpose.reshape(-1, 156)).float()
    trans = torch.zeros([1, 3])

    # Forward pass
    betas_torch = torch.from_numpy(shape).float().view(1, -1)
    # Truncate betas if necessary (AMASS may have 300 dims, but model expects 16)
    if betas_torch.shape[1] > 16:
        betas_torch = betas_torch[:, :16]
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
