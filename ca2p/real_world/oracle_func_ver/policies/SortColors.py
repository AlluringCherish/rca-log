        if objects is None:
            objects = {'red': ['red_obj1', 'red_obj2'], 'blue': ['blue_obj1', 'blue_obj2'], 'green': ['green_obj1', 'green_obj2']}
        if color_zones is None:
            color_zones = {'red': (0.3, 0.0), 'blue': (0.5, 0.0), 'green': (0.7, 0.0)}

        self.ps.go_to_ready_pose()

        for color, obj_list in objects.items():
            zone_x, zone_y = color_zones[color]
            for i, obj_name in enumerate(obj_list):
                self.ps.setTargetPose(*get_obj_func(obj_name))
                self.ps.execute_pick(gripper_force, axis=2)

                self.ps.setTargetPose(zone_x, zone_y + i * 0.05, 0.1, 0, np.pi, 0)
                self.ps.execute_place(axis=2)

        self.ps.go_to_ready_pose()