import argparse
import pathlib
import os, re
import time
import matplotlib.pyplot as plt
import math
import mujoco
from utils import make_converted_amass_path
from utils import fix_amass_file

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplh_file, get_smplh_data_offline_fast

from rich import print

left_shoulder_tendons = [
    "DELT1_tendon_left",
    "DELT2_tendon_left",
    "DELT3_tendon_left",
    "SUPSP_tendon_left",
    "INFSP_tendon_left",
    "SUBSC_tendon_left",
    "TMIN_tendon_left",
    "TMAJ_tendon_left",
    "PECM1_tendon_left",
    "PECM2_tendon_left",
    "PECM3_tendon_left",
    "LAT1_tendon_left",
    "LAT2_tendon_left",
    "LAT3_tendon_left",
    "CORB_tendon_left",
    "TRIlong_tendon_left",
    "TRIlat_tendon_left",
    "TRImed_tendon_left",
    "ANC_tendon_left",
    "SUP_tendon_left",
    "BIClong_tendon_left",
    "BICshort_tendon_left",
    "BRA_tendon_left",
    "BRD_tendon_left",
]

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplh_file",
        help="SMPL-H motion file to load (AMASS format).",
        type=str,
        # required=True,
        default="/media/data/share/AMASS/KIT/3/tennis_forehand_right04_poses.npz",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1",
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "myofullbody"],
        default="myofullbody",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    args = parser.parse_args()


    SMPLH_FOLDER = HERE / ".." / "assets" / "body_models" / "smplh"


    # Load SMPL-H trajectory
    smplh_data, body_model, smplh_output, actual_human_height = load_smplh_file(
        args.smplh_file, SMPLH_FOLDER
    )


    # align fps
    tgt_fps = 30
    smplh_data_frames, aligned_fps = get_smplh_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps=tgt_fps)


    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplh",
        tgt_robot=args.robot,
    )


    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplh_file.split('/')[-1].split('.')[0]}.mp4",)


    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
        qvel_list = []
        xpos_list = []
        xquat_list = []
        xmat_list = []
        cvel_list = []
        subtree_com_list = []
        site_xpos_list = []
        site_xmat_list = []
        model = mujoco.MjModel.from_xml_path(str(HERE / ".." / "assets" / "skeleton" / "full_body_model" / "body" / "myofullbody.xml"))
        data = mujoco.MjData(model)

    # Start the viewer
    i = 0
    time_total =[]

    while True:
        if args.loop:
            i = (i + 1) % len(smplh_data_frames)
        else:
            i += 1
            if i >= len(smplh_data_frames):
                break

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Update task targets.
        smplh_data = smplh_data_frames[i]
        #print(smplh_data)

        # retarget
        time1 = time.time()
        qpos = retarget.retarget(smplh_data, offset_to_ground=True)
        time2 = time.time()
        time_total.append(time2 - time1)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplh_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )

        if args.save_path is not None:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            qpos_list.append(qpos)
            qvel_list.append(np.array(data.qvel.copy()))
            xpos_list.append(np.array(data.xpos.copy()))
            xmat_list.append(np.array(data.xmat.copy()))
            xquat_list.append(np.array(data.xquat.copy()))      # wxyz
            cvel_list.append(np.array(data.cvel.copy()))
            subtree_com_list.append(np.array(data.subtree_com.copy()))
            #site_xpos_list.append(np.array(data.site_xpos.copy()))
            #site_xmat_list.append(np.array(data.site_xmat.copy()))

    print(f"\n[INFO] Average retargeting time per frame: {np.mean(time_total)*1000:.2f} ms")



    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

        save_path = make_converted_amass_path(args.smplh_file, args.robot)
        npz_path = save_path.replace(".pkl", ".npz").replace(".pickle", ".npz")

        qpos_arr         = np.array(qpos_list)
        qvel_arr         = np.array(qvel_list)
        xpos_arr         = np.array(xpos_list)
        xquat_arr        = np.array(xquat_list)
        cvel_arr         = np.array(cvel_list)
        subtree_com_arr  = np.array(subtree_com_list)
        mj_body_name = ["pelvis","lumbar1", "head", "humerus_l", "ulna_l", "lunate_l", "humerus_r", "ulna_r" ,"lunate_r",
                     "femur_l", "tibia_l", "calcn_l", "toes_l", "femur_r", "tibia_r", "calcn_r", "toes_r"]
        body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in mj_body_name]
        for xpos, xmat in zip(xpos_list, xmat_list):
            site_xpos_list.append(xpos[body_ids])   # (17, 3)
            site_xmat_list.append(xmat[body_ids])  # (17, 4)

        #print(site_xpos_list)

        site_xpos_arr    = np.array(site_xpos_list)
        site_xmat_arr    = np.array(site_xmat_list)

        body_names = np.array(
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        )

        joint_names = np.array([
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
        ], dtype=object)

        njnt = model.njnt
        jnt_type = model.jnt_type.copy()          # (njnt,)
        jnt_qposadr = model.jnt_qposadr.copy()    # (njnt,)
        jnt_dofadr = model.jnt_dofadr.copy()      # (njnt,)

        # Body count and site count
        nbody = model.nbody
        nsite = model.nsite

        site_names = mj_body_name[:]

        #changing the site name so that it is compatible with musclemimic training framework.
        body2sitemimic = {
            "pelvis": "pelvis_mimic",
            "lumbar1": "upper_body_mimic",
            "head": "head_mimic",
            # Left arm (note: left uses capital L suffix)
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            # Right arm (note: right has no suffix)
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "talus_r": "right_ankle_mimic",
            "toes_r": "right_toes_mimic",
        }

        for i, name in enumerate(site_names):
            if name in body2sitemimic:
                site_names[i] = body2sitemimic[name]

        T = qpos_arr.shape[0]
        split_points = np.array([0, T - 1])

        np.savez(
            npz_path,
            frequency = aligned_fps,
            qpos=qpos_arr,
            qvel=qvel_arr,
            xpos=xpos_arr,
            xquat=xquat_arr,
            cvel=cvel_arr,
            subtree_com=subtree_com_arr,
            site_xpos=site_xpos_arr,
            site_xmat=site_xmat_arr,
            body_names=body_names,
            site_names=site_names,
            joint_names=joint_names,
            njnt=njnt,
            jnt_type=jnt_type,
            #jnt_qposadr=jnt_qposadr,
            #jnt_dofadr=jnt_dofadr,
            nbody=nbody,
            nsite=nsite,
            split_points=split_points
        )

        print(f"Full kinematic NPZ saved to {npz_path}")

        print("[Info] Converting GMR generated trajectory to Musclemimic training compatible trajectory ... ")
        fix_amass_file(npz_path)

    print("\n[INFO] Saving full joint trajectory...")

    robot_motion_viewer.close()
