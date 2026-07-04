import numpy as np
import tf.transformations as tf_trans
from pick_and_place_module.skills import PrimitiveSkill
from object_dictionary import objects

class RobotController:
    def __init__(self, velocity=0.002, acceleration=0.4):
        self.ps = PrimitiveSkill(velocity, acceleration)

    def robot_move(self):
        """Parameter usage examples for every robot action function."""

        gripper_force = 50
        
        # ========================================

        # ========================================
        self.ps.go_to_ready_pose()
        
        # ========================================

        # setTargetPose(x, y, z, roll, pitch, yaw)

        # ========================================
        self.ps.setTargetPose(0.5, 0.1, 0.20, 0, np.pi, 0)
        
        # ========================================

        # ========================================
        self.ps.execute_go()
        
        # ========================================

        # execute_pick(gripper_force=5, axis=0)


        # ========================================
        self.ps.setTargetPose(objects["can"], 0, np.pi, 0)
        self.ps.execute_pick(gripper_force, axis=2)
        
        # ========================================

        # execute_place(axis=0)

        # ========================================
        self.ps.setTargetPose(0.3, 0.1, 0.20, 0, np.pi, 0)
        self.ps.execute_place(axis=2)
        
        # ========================================

        # execute_push(gripper_force=5, axis=0, distance=0.1)



        # ========================================
        self.ps.setTargetPose(0.4, 0.0, 0.20, 0, np.pi, 0)
        self.ps.execute_push(gripper_force=5, axis=0, distance=0.08)
        
        # ========================================

        # execute_pull(gripper_force=5, axis=0, distance=0.02)



        # ========================================
        self.ps.setTargetPose(0.4, 0.0, 0.20, 0, np.pi, 0)
        self.ps.execute_pull(gripper_force=7, axis=1, distance=0.03)
        
        # ========================================

        # execute_sweep(axis=0, distance=3)


        # ========================================
        self.ps.setTargetPose(0.4, 0.0, 0.20, 0, np.pi, 0)
        self.ps.execute_sweep(axis=0, distance=0.04)
        
        # ========================================

        # execute_rotate(gripper_force=5, axis=0)


        # ========================================
        self.ps.setTargetPose(0.4, 0.0, 0.20, 0, np.pi, 0)
        self.ps.execute_rotate(gripper_force=6, axis=2)
        
        # ========================================

        # execute_gripper(width, force)


        # ========================================
        self.ps.execute_gripper(0.1, 0)
        self.ps.execute_gripper(0.005, 5)
        self.ps.execute_gripper(0.02, 10)
        
        # ========================================



        # execute_pick_and_place(gripper_force=5, axis=0)
        # ========================================
        self.ps.setPose0(0.5, 0.1, 0.20, 0, np.pi/2, 0)
        self.ps.setPose1(0.3, -0.1, 0.20, 0, np.pi/2, np.pi/4)
        self.ps.execute_pick_and_place(gripper_force=6, axis=0)
        
        # ========================================

        # getPose() -> [x, y, z, roll, pitch, yaw]
        # ========================================
        current_pose = self.ps.getPose()
        print(f"Current robot pose: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, z={current_pose[2]:.3f}")
        print(f"Current robot orientation: roll={current_pose[3]:.3f}, pitch={current_pose[4]:.3f}, yaw={current_pose[5]:.3f}")

    def simple_pick_and_place_example(self):
        """Simple pick-and-place example."""

        self.ps.go_to_ready_pose()
        

        self.ps.setTargetPose(0.5, 0.15, 0.20, 0, np.pi, 0)
        self.ps.execute_pick(gripper_force=7, axis=2)
        

        self.ps.setTargetPose(0.3, -0.15, 0.20, 0, np.pi, 0)
        self.ps.execute_place(axis=2)
        
        print("Pick-and-place task completed!")

    def push_pull_example(self):
        """Push/pull example."""

        self.ps.go_to_ready_pose()
        

        self.ps.setTargetPose(0.4, 0.0, 0.2, 0, np.pi, 0)
        

        self.ps.execute_push(gripper_force=5, axis=0, distance=0.05)
        

        self.ps.setTargetPose(0.45, 0.0, 0.2, 0, np.pi, 0)
        

        self.ps.execute_pull(gripper_force=8, axis=0, distance=0.03)
        
        print("Push/pull task completed!")

def main():
    """Main function to initialize and run the robot controller."""
    robot = RobotController()
    

    

    robot.robot_move()
    

    # robot.simple_pick_and_place_example()
    
    

    # robot.push_pull_example()

if __name__ == "__main__":
    main()
