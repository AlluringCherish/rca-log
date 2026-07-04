import numpy as np
import tf.transformations as tf_trans
import os
import sys
sys.path.append('/home/franka-emika/robot_interface')
from pick_and_place_module.skills import PrimitiveSkill
from Object_Detection.object_detection_utils import process_object_detection_from_file, load_coordinates_from_file, get_object_coordinate, postprocess_coordinates
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_tasks import close_box_task

class RobotController:
    def __init__(self, velocity=0.002, acceleration=0.4):
        self.ps = PrimitiveSkill(velocity, acceleration)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        object_list_path = os.path.join(current_dir, 'object_list.txt')
        self.coordinates_file_path = os.path.join(current_dir, 'object_list_position.py')

        self.object_coordinates = load_coordinates_from_file(self.coordinates_file_path)
        
    
    def get_obj(self, object_name):
        return get_object_coordinate(object_name, self.coordinates_file_path)
    
    def refresh_coordinates(self):
        self.object_coordinates = load_coordinates_from_file(self.coordinates_file_path)
        return self.object_coordinates

    def robot_move(self):
        close_box_task(self.ps, self.get_obj)

def main():
    """Main function to initialize and run the robot controller."""
    robot = RobotController()
    robot.robot_move()

if __name__ == "__main__":
    main()