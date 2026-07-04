        self.primitive_skill.setTargetPose(*get_obj_func('pick_and_lift_target'))
        self.primitive_skill.execute_pick(gripper_force, axis=2)

        self.primitive_skill.setTargetPose(*get_obj_func('pick_and_lift_success'))
        self.primitive_skill.execute_place(axis=2)
