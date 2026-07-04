    self.ps.setTargetPose(*get_obj_func(handle_obj))
    self.ps.execute_pick(gripper_force, axis=0)
    self.ps.setTargetPose(*get_obj_func(handle_obj))
    self.ps.execute_push(distance  = 0.10, axis = 0)