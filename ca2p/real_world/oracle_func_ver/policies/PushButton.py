        self.primitive_skill.setTargetPose(*get_obj_func('button'))
        self.primitive_skill.execute_go()

        self.primitive_skill.execute_push(gripper_force=gripper_force, axis=2, distance=0.02)
