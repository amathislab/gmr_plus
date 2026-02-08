import numpy as np
import smplx
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d
from smplx import SMPLH as _SMPLH
import os

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.shape_fitting import load_fitted_shape
from general_motion_retargeting.utils.shape_fitting import load_fitted_shape


# SMPLH joint names (52 joints: 22 body + 30 hands)
SMPLH_BONE_ORDER_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder',
    'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3',
    'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3',
    'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
    'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3',
    'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3',
    'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
]

SMPLH_JOINT_NAMES = SMPLH_BONE_ORDER_NAMES


class SMPLH_Parser(_SMPLH):
    """SMPL-H parser with special handling for AMASS data with 16 betas.
    """

    def __init__(self, *args, **kwargs):
        super(SMPLH_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPLH_BONE_ORDER_NAMES
        self.zero_pose = torch.zeros(1, 156).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPLH_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Get joints and vertices from pose parameters.

        Args:
            pose: Pose tensor of shape (batch_size, 156)
            th_betas: Shape parameters (can be 10 or 16 betas)
            th_trans: Translation tensor

        Returns:
            vertices, joints
        """
        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        smpl_output = self.forward(
            body_pose=pose[:, 3:66],
            global_orient=pose[:, :3],
            left_hand_pose=pose[:, 66:111],
            right_hand_pose=pose[:, 111:156],
            betas=th_betas,
            transl=th_trans,
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints
        return vertices, joints

    def get_joint_transformations(self, pose, th_betas=None, th_trans=None):
        """Return per-joint 4x4 transforms in global frame for given pose/betas/trans."""
        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        batch_size = max(
            th_betas.shape[0] if th_betas is not None else 1,
            pose.shape[0],
        )

        global_orient = pose[:, :3]
        body_pose = pose[:, 3:66]
        left_hand_pose = pose[:, 66:111]
        right_hand_pose = pose[:, 111:156]

        global_orient = match_dim(global_orient, batch_size)
        body_pose = match_dim(body_pose, batch_size)
        left_hand_pose = match_dim(left_hand_pose, batch_size)
        right_hand_pose = match_dim(right_hand_pose, batch_size)
        betas = th_betas if th_betas is not None else match_dim(self.betas, batch_size)

        if self.use_pca:
            left_hand_pose = torch.einsum("bi,ij->bj", [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum("bi,ij->bj", [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1)
        full_pose += self.pose_mean

        device, dtype = betas.device, betas.dtype

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        J = vertices2joints(self.J_regressor, v_shaped)

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        posed_joints, transforms = batch_rigid_transform(rot_mats, J, self.parents, dtype=dtype)

        if th_trans is not None:
            transl = th_trans.view(batch_size, 3)
            T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            T[:, :3, 3] = transl
            transforms = torch.matmul(T[:, None, :, :], transforms)
            posed_joints = posed_joints + transl.unsqueeze(1)

        return transforms


def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data

def load_smplx_files_meta(folder_path, smplx_body_model_path, gender="neutral"):
    """
    Load SMPL-X data from a folder that contains subfolders like:
        smplx_mesh_betas/
        smplx_mesh_body_pose/
        smplx_mesh_global_orient/
        smplx_mesh_left_hand_pose/
        smplx_mesh_right_hand_pose/
        smplx_mesh_transl/

    and return the same outputs as `load_smplx_file`:
        smplx_data, body_model, smplx_output, human_height
    """

    # Map subfolder names -> keys you used in npz version
    folder_map = {
        "smplx_mesh_betas": "betas",
        "smplx_mesh_body_pose": "pose_body",
        "smplx_mesh_global_orient": "root_orient",
        "smplx_mesh_transl": "trans",
        "smplx_mesh_left_hand_pose": "left_hand_pose",
        "smplx_mesh_right_hand_pose": "right_hand_pose",
    }

    smplx_data = {}

    def _load_and_concat(subdir):
        """Load all .npy files in a subdir and concat along time."""
        arrays = []
        for f in sorted(os.listdir(subdir)):
            if f.endswith(".npy"):
                arrays.append(np.load(os.path.join(subdir, f)))
        if not arrays:
            return None
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

    # Load each attribute from its folder
    for subfolder, key in folder_map.items():
        path = os.path.join(folder_path, subfolder)
        if os.path.isdir(path):
            arr = _load_and_concat(path)
            if arr is not None:
                smplx_data[key] = arr

    # Basic sanity: pose_body must exist
    if "pose_body" not in smplx_data:
        raise ValueError(f"Could not find pose_body (smplx_mesh_body_pose) in {folder_path}")

    num_frames = smplx_data["pose_body"].shape[0]

    # If hands are missing, fill with zeros like your original function
    if "left_hand_pose" not in smplx_data:
        smplx_data["left_hand_pose"] = np.zeros((num_frames, 45), dtype=np.float32)
    if "right_hand_pose" not in smplx_data:
        smplx_data["right_hand_pose"] = np.zeros((num_frames, 45), dtype=np.float32)

    # Betas: if missing or per-frame, fall back to first frame
    if "betas" not in smplx_data:
        raise ValueError(f"Could not find betas (smplx_mesh_betas) in {folder_path}")
    betas = smplx_data["betas"]
    if betas.ndim > 1:
        # assume (T, n_betas) or (1, n_betas); use first row
        betas_single = betas[0]
    else:
        betas_single = betas
    smplx_data["betas"] = betas_single

    # Gender (no file, so we pass it in)
    smplx_data["gender"] = gender
    smplx_data["mocap_frame_rate"] = torch.tensor(30)
    smplx_data["trans"] = smplx_data["trans"][..., [2, 1, 0]]
    root_orient = smplx_data["root_orient"]      # shape (T, 3), axis-angle

    R_orig = R.from_rotvec(root_orient)          # (T,)
    R_fix = R.from_matrix(np.array([
        [0, 0, 1],    # new X = old Z (forward)
        [1, 0, 0],   # new Y = -old X (left)
        [0, 1, 0],    # new Z = old Y (up)
    ]))

    R_new = R_fix * R_orig
    smplx_data["root_orient"] = R_new.as_rotvec()

    # Create SMPL-X model
    body_model = smplx.create(
        smplx_body_model_path,
        model_type="smplx",
        gender=str(gender),
        use_pca=False,
        num_betas=300, 
    )

    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),   # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),        # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(),               # (N, 3)
        left_hand_pose=torch.tensor(smplx_data["left_hand_pose"]).float(),
        right_hand_pose=torch.tensor(smplx_data["right_hand_pose"]).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )

    if smplx_data["betas"].ndim == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height

def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data["betas"].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def load_smplh_file(smplh_file, smplh_body_model_path, fitted_shape_path=None):
    """Load SMPL-H data and create body model.

    SMPL-H has 52 joints (22 body + 30 hands, no face) and 10 shape parameters.
    Compatible with GMR since only body joints (0-21) are used for retargeting.

    Handles two formats:
    1. AMASS format: 'poses' array (N, 156) = root(3) + body(63) + hands(90)
    2. Standard format: separate 'root_orient', 'pose_body', hand pose arrays

    Args:
        smplh_file: Path to SMPL-H motion data file (.npz)
        smplh_body_model_path: Path to SMPL-H body models directory
        fitted_shape_path: Optional path to fitted shape parameters (.pkl)
                          If provided, uses optimized shape instead of data betas
    """
    smplh_data = np.load(smplh_file, allow_pickle=True)

    # Convert AMASS format to standard format if needed
    # Always prioritize 'poses' array when it exists with valid shape (N, 156).
    # Some datasets (e.g., MOYO) have both 'poses' and separate 'root_orient'/'pose_body'
    if "poses" in smplh_data and smplh_data["poses"].ndim == 2 and smplh_data["poses"].shape[1] == 156:
        # AMASS SMPL+H format: poses = [root(3), body(63), left_hand(45), right_hand(45)]
        poses = smplh_data["poses"]
        smplh_data = dict(smplh_data)  # Convert to mutable dict
        smplh_data["root_orient"] = poses[:, :3]
        smplh_data["pose_body"] = poses[:, 3:66]
        smplh_data["left_hand_pose"] = poses[:, 66:111]
        smplh_data["right_hand_pose"] = poses[:, 111:156]
        # AMASS uses 'mocap_framerate' key
        if "mocap_framerate" in smplh_data:
            smplh_data["mocap_frame_rate"] = torch.tensor(smplh_data["mocap_framerate"])
    else:
        # For non-AMASS files, convert to mutable dict for metadata storage
        smplh_data = dict(smplh_data)

    # Handle gender field (can be string or numpy array)
    gender = smplh_data["gender"]
    if hasattr(gender, 'item'):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    gender = str(gender)

    # Use SMPLH_Parser which handles AMASS data properly
    body_model = SMPLH_Parser(
        model_path=smplh_body_model_path,
        gender="neutral",  # AMASS uses neutral gender
        use_pca=False,
    )

    # Reconstruct full poses array for get_joints_verts (N, 156)
    # Build pose vector - hand poses are optional in AMASS
    pose_parts = [
        torch.tensor(smplh_data["root_orient"]).float(),
        torch.tensor(smplh_data["pose_body"]).float(),
    ]

    # Add hand poses if available
    if "left_hand_pose" in smplh_data and "right_hand_pose" in smplh_data:
        pose_parts.extend([
            torch.tensor(smplh_data["left_hand_pose"]).float(),
            torch.tensor(smplh_data["right_hand_pose"]).float(),
        ])
    else:
        # Use zeros for hand poses if not in data (45 dims = 15 joints × 3)
        n_frames = smplh_data["root_orient"].shape[0]
        pose_parts.extend([
            torch.zeros(n_frames, 45).float(),  # left hand: 15 joints × 3
            torch.zeros(n_frames, 45).float(),  # right hand: 15 joints × 3
        ])

    poses = torch.cat(pose_parts, dim=1)

    # Get betas - use fitted shape if provided, otherwise use data betas
    if fitted_shape_path is not None:
        fitted_shape, fitted_scale, fitted_metrics = load_fitted_shape(fitted_shape_path)
        betas = torch.from_numpy(fitted_shape).float()  # (1, 16)
        print(f"[load_smplh_file] Using fitted shape from {fitted_shape_path}")
        print(f"  Fitted scale: {fitted_scale[0]:.4f}, Beta[0]: {betas[0, 0]:.4f}")
    else:
        betas = torch.tensor(smplh_data["betas"]).float()
        if betas.ndim == 1:
            betas = betas.view(1, -1)

    # Get translation
    trans = torch.tensor(smplh_data["trans"]).float()

    # Apply offset_z and height_scale (MuscleMimic-style)
    if fitted_shape_path is not None:
        # Apply z-offset first
        trans[:, 2] += offset_z

        # Apply height scaling (preserve initial height)
        trans[:, :2] *= height_scale  # Scale x, y
        trans[:, 2] = (trans[:, 2] - trans[0, 2]) * height_scale + trans[0, 2]  # Scale z preserving initial

        print(f"  Applied offset_z and height_scale to translations")

    # IMPORTANT: Repeat betas for each frame to match batch size
    num_frames = poses.shape[0]
    betas = betas.repeat(num_frames, 1)

    # Use the custom get_joints_verts method that handles 16 betas properly
    vertices, joints = body_model.get_joints_verts(
        pose=poses,
        th_betas=betas,
        th_trans=trans,
    )

    # Apply fitted scale directly to joints
    # This scales the joints to match robot proportions
    if fitted_shape_path is not None:
        fitted_scale_val = float(fitted_scale.flat[0]) if isinstance(fitted_scale, np.ndarray) else float(fitted_scale)

        # Apply scale: (joints - pelvis) * scale + pelvis
        # This preserves root position while scaling all joints relative to pelvis
        pelvis_pos = joints[:, 0:1, :]  # (N, 1, 3)
        joints = (joints - pelvis_pos) * fitted_scale_val + pelvis_pos
        print(f"  Applied fitted scale {fitted_scale_val:.4f} directly to SMPL joints")

        # Store fitted_scale in smplh_data so GMR knows to skip additional scaling
        smplh_data["fitted_scale"] = fitted_scale_val

    # Create output dict similar to smplx output with all necessary attributes
    class SMPLHOutput:
        def __init__(self, vertices, joints, full_pose, global_orient, body_pose,
                     left_hand_pose, right_hand_pose, betas):
            self.vertices = vertices
            self.joints = joints
            self.full_pose = full_pose
            self.global_orient = global_orient
            self.body_pose = body_pose
            self.left_hand_pose = left_hand_pose
            self.right_hand_pose = right_hand_pose
            self.betas = betas

    # Build hand poses - use from data if available, otherwise use zeros
    if "left_hand_pose" in smplh_data and "right_hand_pose" in smplh_data:
        left_hand_pose = torch.tensor(smplh_data["left_hand_pose"]).float()
        right_hand_pose = torch.tensor(smplh_data["right_hand_pose"]).float()
    else:
        n_frames = smplh_data["root_orient"].shape[0]
        left_hand_pose = torch.zeros(n_frames, 45).float()
        right_hand_pose = torch.zeros(n_frames, 45).float()

    smplh_output = SMPLHOutput(
        vertices=vertices,
        joints=joints,
        full_pose=poses,
        global_orient=torch.tensor(smplh_data["root_orient"]).float(),
        body_pose=torch.tensor(smplh_data["pose_body"]).float(),
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        betas=betas,
    )

    # Height calculation
    if fitted_shape_path is not None:
        # With fitted shape, joints are already correctly scaled
        # Compute actual height from scaled joints for reference
        head_idx = 15
        left_foot_idx = 10
        right_foot_idx = 11
        human_height = float(joints[0, head_idx, 2] - min(joints[0, left_foot_idx, 2], joints[0, right_foot_idx, 2]))
        print(f"  Actual SMPL height after scaling: {human_height:.3f} m")
    else:
        # Use beta[0] heuristic for height-based scaling
        if len(smplh_data["betas"].shape) == 1:
            human_height = 1.66 + 0.1 * smplh_data["betas"][0]
        else:
            human_height = 1.66 + 0.1 * smplh_data["betas"][0, 0]

    return smplh_data, body_model, smplh_output, human_height


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data['betas'].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplh_data(smplh_data, body_model, smplh_output, curr_frame):
    """Extract SMPL-H joint data for a single frame.

    Since SMPL-H body joints (0-21) are identical to SMPL-X, we use the same logic.
    """
    return get_smplx_data(smplh_data, body_model, smplh_output, curr_frame)


def get_smplh_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps=30):
    """Extract SMPL-H joint data for all frames with FPS conversion.

    Since SMPL-H body joints (0-21) are identical to SMPL-X, we use the same logic.
    """
    return get_smplx_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps)


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
    return smplx_data_frames, aligned_fps



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        new_num_frames = num_frames // frame_skip
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = utils.quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
