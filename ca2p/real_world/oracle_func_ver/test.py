import numpy as np
import tf.transformations as tf_trans
from pick_and_place_module.skills import PrimitiveSkill
from object_dictionary import objects

class RobotController:
    
    def __init__(self, velocity=0.002, acceleration=0.4):
        self.ps = PrimitiveSkill(velocity, acceleration)
        self.gripper_force = 50

    def robot_move(self):
        self.ps.go_to_ready_pose()
        
        self.ps.setTargetPose(objects['dice'], 0, np.pi, 0)
        self.ps.execute_pick(self.gripper_force, axis=2) 
        
        self.ps.setTargetPose(objects['target'], 0, np.pi, 0)
        self.ps.execute_go()
        
        self.ps.execute_gripper(0.1, 0)

        self.ps.go_to_ready_pose()

def main():
    """Main function to initialize and run the robot controller."""
    robot = RobotController()
    robot.robot_move()

if __name__ == "__main__":
    main()