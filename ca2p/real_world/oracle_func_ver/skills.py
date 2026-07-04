import rospy
import numpy as np
from pick_and_place_module.eef_control import MoveGroupControl
from pick_and_place_module.grasping import GripperInterface
from copy import deepcopy
from math import pi
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PrimitiveSkill:
    def __init__(self, gripper_offset=0.05, intermediate_z_stop=0.3, intermediate_distance=0.09, speed=0.13,
                 push_length=0.02, pull_length=0.02, sweep_count=3, sweep_width=0.03):
        self.gripper_offset = gripper_offset
        self.intermediate_z_stop = intermediate_z_stop
        self.intermediate_distance = intermediate_distance
        # self.push_length = push_length
        # self.pull_length = pull_length
        self.sweep_count = sweep_count
        self.sweep_width = sweep_width
        self.pose0 = None
        self.pose1 = None
        self.target_pose = None
        self.waypoint_density = 5
        self.moveit_control = MoveGroupControl(speed)
        self.gripper = GripperInterface()

    # === Utility Functions ===
    # These functions support primitive skills or handle auxiliary tasks

    def interpolate_pose(self, start, end, steps):
        """Linear interpolation between start and end points."""
        start_array = np.array(start)
        end_array = np.array(end)
        array_list = [start_array + (end_array - start_array) * i / (steps - 1) for i in range(steps)]
        return [array.tolist() for array in array_list]

    def setPose0(self, x, y, z, roll, pitch, yaw):
        self.pose0 = [x, y, z, roll + pi/4, pitch, yaw]

    def setPose1(self, x, y, z, roll, pitch, yaw):
        self.pose1 = [x, y, z, roll + pi/4, pitch, yaw]

    def setTargetPose(self, x, y, z, roll, pitch, yaw):
        self.target_pose = [x, y, z, roll + pi/4, pitch, yaw]

    def go_to_ready_pose(self):
        self.moveit_control.go_to_joint_state(0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
        rospy.sleep(2)
        print("go to ready pose")

    def current_pose(self):
        move_group = self.moveit_control
        quaternion_pose = move_group.get_current_pose()
        current_euler_pose = euler_from_quaternion((
            quaternion_pose.orientation.x,
            quaternion_pose.orientation.y,
            quaternion_pose.orientation.z,
            quaternion_pose.orientation.w
        ))
        return current_euler_pose
    
    def getPose(self,):
        pose = self.moveit_control.get_current_pose().position
        orientation = self.moveit_control.get_current_pose().orientation

        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]

        # Convert quaternion to roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Extract position components
        pose0_x, pose0_y, pose0_z = pose.x, pose.y, pose.z

        # Assign orientation components
        pose0_roll, pose0_pitch, pose0_yaw = roll, pitch, yaw

        # adjust
        pose0_yaw = pose0_yaw + np.pi/4
        pose0_pitch = pose0_pitch + np.pi
        pose0_roll = pose0_roll + np.pi
        print(f"({pose0_x:.7f}, {pose0_y:.7f}, {pose0_z:.7f}, {pose0_yaw:.7f}, {pose0_pitch:.7f}, {pose0_roll:.7f})")
        
        return [pose0_x, pose0_y, pose0_z, pose0_yaw, pose0_pitch, pose0_roll]

    # === Primitive Skills ===
    # These functions represent core robotic actions

    def execute_pick_and_place(self, gripper_force=5, axis=0):
        move_group = self.moveit_control
        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)

        # Pick waypoints
        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.pose0)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        intermediate_pose = deepcopy(self.pose0)
        intermediate_pose[2] = self.intermediate_z_stop
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
        
        destination_pose = deepcopy(self.pose0)
        destination_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(intermediate_pose, destination_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing pick waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])
        
        self.gripper.grasp(0.01, gripper_force)
        rospy.sleep(1)

        # Place waypoints
        current_pose_list = deepcopy(self.pose1)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        intermediate_pose = deepcopy(self.pose1)
        intermediate_pose[2] = self.intermediate_z_stop
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
        
        waypoints += self.interpolate_pose(intermediate_pose, self.pose1, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing place waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)
        self.go_to_ready_pose()

    def execute_pick(self, gripper_force=5, axis=0):
        move_group = self.moveit_control
        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)

        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
       
        intermediate_pose = deepcopy(self.target_pose)

        # intermediate_pose[2] = -self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance
        elif axis == 2:
            intermediate_pose[axis] += self.intermediate_distance
       
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
       
        destination_pose = deepcopy(self.target_pose)
        destination_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(intermediate_pose, destination_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing pick waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        self.gripper.grasp(0.005, gripper_force)
        rospy.sleep(1)
        # self.go_to_ready_pose()

    def execute_place(self, axis=0):
        move_group = self.moveit_control
        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
       
        intermediate_pose = deepcopy(self.target_pose)
        # intermediate_pose[2] += self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance
        elif axis == 2:
            intermediate_pose[axis] += self.intermediate_distance
       
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
        waypoints += self.interpolate_pose(intermediate_pose, self.target_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing place waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)
        self.go_to_ready_pose()

    def execute_push(self, gripper_force=5, axis=0, distance=0.1):
        move_group = self.moveit_control
        if axis < 2:
            self.target_pose[axis] -= distance
        elif axis == 2:
            self.target_pose[axis] += distance
        
        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance

        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
        
        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(intermediate_pose, approach_pose, self.waypoint_density)
        
        push_pose = deepcopy(self.target_pose)
        if axis < 2:
            push_pose[axis] += distance
        elif axis == 2:
            push_pose[axis] -= distance
            
        push_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(approach_pose, push_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing push waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)
        self.go_to_ready_pose()

    def execute_pull(self, gripper_force=5, axis=0, distance=0.02):
        move_group = self.moveit_control

        rospy.sleep(1)

        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints = self.interpolate_pose(current_pose_list, approach_pose, self.waypoint_density)
        
        pull_pose = deepcopy(self.target_pose)
        pull_pose[axis] -= distance
        pull_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(approach_pose, pull_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing pull waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        rospy.sleep(1)
        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)
        self.go_to_ready_pose()

    def execute_sweep(self, axis=0, distance=3):
        move_group = self.moveit_control
        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance
        
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)

        for _ in range(self.sweep_count):
            sweep_positive = deepcopy(self.target_pose)
            sweep_positive[axis] += distance
            sweep_positive[2] += self.gripper_offset + 0.025
            waypoints.append(sweep_positive)

            sweep_negative = deepcopy(self.target_pose)
            sweep_negative[axis] -= distance
            sweep_negative[2] += self.gripper_offset + 0.025
            waypoints.append(sweep_negative)

        for waypoint in waypoints:
            rospy.loginfo("Executing sweep waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

    def execute_rotate(self, gripper_force=5, axis=0):
        move_group = self.moveit_control
        self.gripper.grasp(0.1, 0)
        rospy.sleep(1)

        current_pose = move_group.get_current_pose().position
        current_pose_list = deepcopy(self.target_pose)
        current_pose_list[0], current_pose_list[1], current_pose_list[2] = current_pose.x, current_pose.y, current_pose.z
        
        intermediate_pose = deepcopy(self.target_pose)
        intermediate_pose[2] = self.intermediate_z_stop
        if axis < 2:
            intermediate_pose[axis] -= self.intermediate_distance
        
        waypoints = self.interpolate_pose(current_pose_list, intermediate_pose, self.waypoint_density)
        
        approach_pose = deepcopy(self.target_pose)
        approach_pose[2] += self.gripper_offset
        waypoints += self.interpolate_pose(intermediate_pose, approach_pose, self.waypoint_density)

        for waypoint in waypoints:
            rospy.loginfo("Executing rotate waypoint: %s", waypoint)
            move_group.go_to_pose_goal(waypoint[0], waypoint[1], waypoint[2], waypoint[3], waypoint[4], waypoint[5])

        self.gripper.grasp(0.01, gripper_force)
        rospy.sleep(1)

        current_joint_values = move_group.get_current_joint_states()
        current_joint_values[-1] -= pi/2
        move_group.go_to_joint_state(*current_joint_values)

        target_pose = deepcopy(self.target_pose)
        target_pose[2] += 0.2
        move_group.go_to_pose_goal(target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])

    def execute_go(self):
        move_group = self.moveit_control
        move_group.go_to_pose_goal(self.target_pose[0], self.target_pose[1], self.target_pose[2],
                                  self.target_pose[3], self.target_pose[4], self.target_pose[5])
        rospy.loginfo("Executing go waypoint: %s", self.target_pose)

    def execute_gripper(self, width1, force1):
        self.gripper.grasp(width1, force1)
