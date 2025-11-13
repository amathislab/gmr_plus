import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplh_file, get_smplh_data_offline_fast

from rich import print

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplh_file",
        help="SMPL-H motion file to load.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                 "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
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

    parser.add_argument(
        "--fitted_shape",
        type=str,
        default=None,
        help="Path to fitted shape parameters (.pkl). Auto-detects if available for the robot.",
    )

    args = parser.parse_args()

    # Use local SMPL-H body models in assets folder
    SMPLH_FOLDER = HERE.parent / "assets" / "body_models" / "smplh"

    # Auto-detect fitted shape if not provided
    fitted_shape_path = args.fitted_shape
    # Handle empty string to explicitly disable fitted shape
    if fitted_shape_path == '':
        fitted_shape_path = None
        print("[Info] Fitted shape disabled, using height-based scaling")
    elif fitted_shape_path is None:
        # Try to find fitted shape for this robot
        auto_fitted_path = HERE.parent / "assets" / "fitted_shapes" / f"{args.robot}_shape.pkl"
        if auto_fitted_path.exists():
            print(f"[Info] Found fitted shape at {auto_fitted_path}, using data-driven shape fitting")
            fitted_shape_path = str(auto_fitted_path)

    # Load SMPL-H trajectory
    # When fitted_shape_path is provided, load_smplh_file returns effective height
    # computed from fitted scale, so GMR uses the fitted scale directly
    smplh_data, body_model, smplh_output, actual_human_height = load_smplh_file(
        args.smplh_file, SMPLH_FOLDER, fitted_shape_path=fitted_shape_path
    )

    # align fps
    tgt_fps = 30
    smplh_data_frames, aligned_fps = get_smplh_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps=tgt_fps)


    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplh",
        tgt_robot=args.robot,
        use_fitted_shape=(fitted_shape_path is not None),
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

    # Start the viewer
    i = 0

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

        # retarget
        qpos = retarget.retarget(smplh_data)

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
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
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



    robot_motion_viewer.close()
