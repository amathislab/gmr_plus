
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from .utils.shape_fitting import load_fitted_shape
from rich import print
from mink.tasks.equality_constraint_task import EqualityConstraintTask


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        use_fitted_shape: bool=False,  # Whether using fitted shape
        fitted_shape_path: str=None,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        # Auto-enable fitted shape if path is provided
        if fitted_shape_path is not None and not use_fitted_shape:
            use_fitted_shape = True
            if verbose:
                print(f"[GMR] Auto-enabled use_fitted_shape=True (fitted_shape_path provided)")

        self.use_fitted_shape = use_fitted_shape
        self.fitted_shape_path = fitted_shape_path
        self.learned_offsets = None

        # adjust the human scale table
        if use_fitted_shape:
            # When using fitted shape, the fitted_scale is already applied directly to SMPL joints
            # in load_smplh_file(). We do not apply any additional scaling here.
            for key in ik_config["human_scale_table"].keys():
                ik_config["human_scale_table"][key] = 1.0
            if verbose:
                print(f"[GMR] Using fitted shape: fitted_scale already applied to SMPL joints, skipping manual scaling (scale_table=1.0)")
            if fitted_shape_path is not None:
                if verbose:
                    print(f"[GMR] Attempting to load fitted shape from {fitted_shape_path}")
                try:
                    shape, scale, offset_z, height_scale, offsets = load_fitted_shape(fitted_shape_path)
                    self.fitted_shape = shape
                    self.fitted_scale = scale
                    self.fitted_offset_z = offset_z
                    self.fitted_height_scale = height_scale
                    self.learned_offsets = offsets
                    if verbose:
                        print(f"[GMR] ✓ Loaded fitted shape successfully")
                        print(f"  offset_z: {offset_z:.4f} m, height_scale: {height_scale:.4f}")
                        print(f"  Offsets keys: {offsets.keys() if offsets else 'None'}")
                        if offsets and "pos_offsets" in offsets:
                            print(f"  ✓ Loaded learned offsets: {len(offsets['pos_offsets'])} joints")
                except Exception as e:
                    import traceback
                    if verbose:
                        print(f"[GMR] ✗ Failed to load fitted shape from {fitted_shape_path}")
                        print(f"  Error: {e}")
                        traceback.print_exc()
        else:
            # Without fitted shape, use height-based scaling
            if actual_human_height is not None:
                ratio = actual_human_height / ik_config["human_height_assumption"]
            else:
                ratio = 1.0

            for key in ik_config["human_scale_table"].keys():
                ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        # Note: self.learned_offsets is already initialized above (line 72) and set during fitted shape loading

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0
    
    def _add_equality_tasks(self, cur_task):
        model = self.configuration.model

        for eq_id in range(model.neq):
            if model.eq_type[eq_id] == mj.mjtEq.mjEQ_JOINT:
                task = EqualityConstraintTask(model, eq_id)
                task.weight = 5.0
                cur_task.append(task)

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []

        if self.use_ik_match_table1:
            self._add_equality_tasks(self.tasks1)

        if self.use_ik_match_table2:
            self._add_equality_tasks(self.tasks2)
        
        
        # learned offsets already loaded in __init__, no need to reload

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                pos, rot = self._resolve_offsets(body_name, pos_offset, rot_offset)
                self.pos_offsets1[body_name] = pos
                self.rot_offsets1[body_name] = rot
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                pos, rot = self._resolve_offsets(body_name, pos_offset, rot_offset)
                self.pos_offsets2[body_name] = pos
                self.rot_offsets2[body_name] = rot
                self.tasks2.append(task)
                self.task_errors2[task] = []

    def _resolve_offsets(self, body_name, default_pos, default_rot):
        if self.learned_offsets is not None:
            pos_offsets = self.learned_offsets.get("pos_offsets", {})
            rot_offsets = self.learned_offsets.get("rot_offsets", {})
            if body_name in pos_offsets and body_name in rot_offsets:
                return (
                    np.array(pos_offsets[body_name], dtype=float),
                    R.from_quat(rot_offsets[body_name], scalar_first=True),
                )

        pos = np.array(default_pos, dtype=float) - self.ground
        rot = R.from_quat(default_rot, scalar_first=True)
        return pos, rot

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        # Apply offsets: either learned offsets from shape fitting, or hard-coded JSON offsets
        # Learned offsets are computed by compute_alignment_offsets() in robot's T-pose local frame (centered at roots)
        # They rotate with robot's target rotation at runtime (see offset_human_data implementation)
        use_learned_offsets = (
            self.learned_offsets is not None and
            "pos_offsets" in self.learned_offsets and
            "rot_offsets" in self.learned_offsets
        )

        if use_learned_offsets:
            # Convert learned offsets dict to format expected by offset_human_data()
            learned_pos_offsets = {}
            learned_rot_offsets = {}
            for body_name in self.learned_offsets["pos_offsets"].keys():
                learned_pos_offsets[body_name] = np.array(self.learned_offsets["pos_offsets"][body_name])
                learned_rot_offsets[body_name] = R.from_quat(self.learned_offsets["rot_offsets"][body_name], scalar_first=True)

            # Apply learned offsets
            human_data = self.offset_human_data(human_data, learned_pos_offsets, learned_rot_offsets)
        else:
            # Use hard-coded JSON offsets
            human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)

        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                # Offsets (either learned or JSON) have already been applied globally to human_data
                # So we just set the targets directly
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                # Offsets (either learned or JSON) have already been applied globally to human_data
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
        
        final_curr_pos = []
        final_tgt_pos  = []
        self.frame_tasks_2 = [t for t in self.tasks2 if isinstance(t, mink.FrameTask)]

        for task in self.frame_tasks_2:

            # === Get current pose after IK converged ===
            
            T_curr = self.configuration.get_transform_frame_to_world(
                task.frame_name,
                task.frame_type,     # always "body" in your setup
            )

            # === Get TARGET pose stored inside the task ===
            T_tgt = task.transform_target_to_world

            # === Extract xyz translations ===
            # mink.SE3 uses `.translation()` just like jaxlie
            p_curr = np.array(T_curr.translation())
            p_tgt  = np.array(T_tgt.translation())

            final_curr_pos.append(p_curr)
            final_tgt_pos.append(p_tgt)
        
        final_curr_pos = np.stack(final_curr_pos)   # (N, 3)
        final_tgt_pos  = np.stack(final_tgt_pos)    # (N, 3)

        body_diff = final_curr_pos - final_tgt_pos
        body_diff_norm = np.linalg.norm(body_diff, axis=1)

        return self.configuration.data.qpos.copy(), body_diff_norm


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0
        lowest_pos = np.inf

        # instead of using the food, we use the lowest point as the offset point
        for body_name, (pos, quat) in human_data.items():
            # pos is expected to be (x, y, z)
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name

        '''
        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        '''
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos.copy(), quat.copy()]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            new_pos = pos - np.array([0, 0, self.ground_offset])
            human_data[body_name] = (new_pos, quat)  # Create new tuple instead of modifying existing one
        return human_data
