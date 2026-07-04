import numpy as np

def alternating_pattern_task(ps, get_obj_func, cubes=None, cylinders=None, start_x=0.4, y_pos=0.0, spacing=0.08, gripper_force=50):
    if cubes is None:
        cubes = ['cube_1', 'cube_2', 'cube_3']
    if cylinders is None:
        cylinders = ['cylinder_1', 'cylinder_2', 'cylinder_3']

    ps.go_to_ready_pose()

    for i in range(6):
        if i % 2 == 0:
            obj_name = cubes[i // 2]
        else:
            obj_name = cylinders[i // 2]

        ps.setTargetPose(*get_obj_func(obj_name))
        ps.execute_pick(gripper_force, axis=2)

        target_x = start_x + i * spacing
        ps.setTargetPose(target_x, y_pos, 0.1, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def arrange_in_grid_task(ps, get_obj_func, object_list=None, start_x=0.5, start_y=-0.2, spacing=0.1, gripper_force=50, grid_cols=3):
    if object_list is None:
        object_list = [f'object_{i}' for i in range(1, 10)]

    ps.go_to_ready_pose()

    for i, obj_name in enumerate(object_list):
        row = i // grid_cols
        col = i % grid_cols
        target_x = start_x + col * spacing
        target_y = start_y + row * spacing

        ps.setTargetPose(*get_obj_func(obj_name))
        ps.execute_pick(gripper_force, axis=2)

        ps.setTargetPose(target_x, target_y, 0.1, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def circular_arrangement_task(ps, get_obj_func, object_list=None, num_objects=8, center_x=0.5, center_y=0.0, radius=0.15, gripper_force=50):
    if object_list is None:
        object_list = [f'object_{i+1}' for i in range(num_objects)]
    else:
        num_objects = len(object_list)

    ps.go_to_ready_pose()

    for i, obj_name in enumerate(object_list):
        angle = 2 * np.pi * i / num_objects

        ps.setTargetPose(*get_obj_func(obj_name))
        ps.execute_pick(gripper_force, axis=2)

        target_x = center_x + radius * np.cos(angle)
        target_y = center_y + radius * np.sin(angle)

        ps.setTargetPose(target_x, target_y, 0.1, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def clear_workspace_task(ps, get_obj_func, object_list=None, num_objects=10, storage_x=0.7, storage_y=0.3, storage_spacing=0.06, objects_per_row=5, gripper_force=50):
    if object_list is None:
        object_list = [f'clutter_{i+1}' for i in range(num_objects)]
    else:
        num_objects = len(object_list)

    ps.go_to_ready_pose()

    for i, obj_name in enumerate(object_list):
        row = i // objects_per_row
        col = i % objects_per_row

        ps.setTargetPose(*get_obj_func(obj_name))
        ps.execute_pick(gripper_force, axis=2)

        target_x = storage_x + col * storage_spacing
        target_y = storage_y + row * storage_spacing

        ps.setTargetPose(target_x, target_y, 0.1, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def close_box_task(ps, get_obj_func, lid_obj='box_lid', base_obj='box_base', gripper_force=50, gripper_close=0.02, gripper_open=0.1):
    ps.go_to_ready_pose()

    ps.setTargetPose(*get_obj_func(lid_obj))
    ps.execute_go()

    ps.execute_gripper(gripper_close, gripper_force)

    ps.setTargetPose(*get_obj_func(base_obj))
    ps.execute_rotate(gripper_force, axis=1)

    ps.execute_gripper(gripper_open, 0)

    ps.go_to_ready_pose()

def insert_peg_task(ps, get_obj_func, peg_obj='peg', hole_obj='hole', gripper_force=50):
    ps.go_to_ready_pose()

    ps.setTargetPose(*get_obj_func(peg_obj))
    ps.execute_pick(gripper_force, axis=2)

    hole_pos = get_obj_func(hole_obj)
    ps.setTargetPose(hole_pos[0], hole_pos[1], hole_pos[2] + 0.05, hole_pos[3], hole_pos[4], hole_pos[5])
    ps.execute_go()

    ps.setTargetPose(*hole_pos)
    ps.execute_insert(gripper_force, axis=2)

    ps.execute_gripper(0.1, 0)

    ps.go_to_ready_pose()

def open_drawer_task(ps, get_obj_func, handle_obj='drawer_handle', gripper_force=50, pull_distance=0.15):
    ps.go_to_ready_pose()
    ps.setTargetPose(*get_obj_func(handle_obj))
    ps.execute_pick(gripper_force, axis=0)
    ps.setTargetPose(*get_obj_func(handle_obj))
    ps.execute_pull(distance  = 0.10, axis = 0)

    ps.go_to_ready_pose()

def close_drawer_task(ps, get_obj_func, handle_obj='drawer_handle', gripper_force=50, pull_distance=0.15):
    ps.go_to_ready_pose()
    ps.setTargetPose(*get_obj_func(handle_obj))
    ps.execute_pick(gripper_force, axis=0)
    ps.setTargetPose(*get_obj_func(handle_obj))
    ps.execute_push(distance  = 0.10, axis = 0)

    ps.go_to_ready_pose()


def pick_and_lift_task(ps, get_obj_func, target_obj='object', lift_height=0.2, gripper_force=50):
    ps.go_to_ready_pose()

    ps.setTargetPose(*get_obj_func(target_obj))
    ps.execute_pick(gripper_force, axis=2)

    current_pos = get_obj_func(target_obj)
    ps.setTargetPose(current_pos[0], current_pos[1], current_pos[2] + lift_height, 0, np.pi, 0)
    ps.execute_lift(axis=2)

    ps.go_to_ready_pose()

def pick_up_cup_task(ps, get_obj_func, cup_obj='cup1', gripper_force=50):
    ps.go_to_ready_pose()

    ps.setTargetPose(*get_obj_func(cup_obj))
    ps.execute_pick(gripper_force, axis=2)

    ps.setTargetPose(*get_obj_func(cup_obj))
    ps.execute_go()

    ps.go_to_ready_pose()

def pour_water_task(ps, get_obj_func, source_cup='cup1', target_cup='cup2', pour_height=0.15, gripper_force=50):
    ps.go_to_ready_pose()

    ps.setTargetPose(*get_obj_func(source_cup))
    ps.execute_pick(gripper_force, axis=2)

    target_pos = get_obj_func(target_cup)
    ps.setTargetPose(target_pos[0], target_pos[1], target_pos[2] + pour_height, 0, np.pi, 0)
    ps.execute_go()

    ps.execute_pour(angle=90, axis=1)

    ps.execute_pour(angle=-90, axis=1)

    ps.go_to_ready_pose()

def push_button_task(ps, get_obj_func, button_obj='button', push_depth=0.01, gripper_force=50):
    ps.go_to_ready_pose()

    button_pos = get_obj_func(button_obj)
    ps.setTargetPose(button_pos[0], button_pos[1], button_pos[2] + 0.05, button_pos[3], button_pos[4], button_pos[5])
    ps.execute_go()

    ps.setTargetPose(button_pos[0], button_pos[1], button_pos[2] - push_depth, button_pos[3], button_pos[4], button_pos[5])
    ps.execute_push(gripper_force, axis=2)

    ps.setTargetPose(button_pos[0], button_pos[1], button_pos[2] + 0.05, button_pos[3], button_pos[4], button_pos[5])
    ps.execute_go()

    ps.go_to_ready_pose()

def put_rubbish_in_bin_task(ps, get_obj_func, rubbish_list=None, bin_obj='bin', gripper_force=50):
    if rubbish_list is None:
        rubbish_list = ['rubbish_1', 'rubbish_2', 'rubbish_3']

    ps.go_to_ready_pose()

    bin_pos = get_obj_func(bin_obj)

    for rubbish in rubbish_list:
        ps.setTargetPose(*get_obj_func(rubbish))
        ps.execute_pick(gripper_force, axis=2)

        ps.setTargetPose(bin_pos[0], bin_pos[1], bin_pos[2] + 0.2, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def pyramid_stacking_task(ps, get_obj_func, blocks=None, base_x=0.5, base_y=0.0, layer_height=0.05, spacing=0.06, gripper_force=50):
    if blocks is None:
        blocks = [f'block_{i+1}' for i in range(6)]

    ps.go_to_ready_pose()

    layer_configs = [3, 2, 1]
    block_idx = 0

    for layer, num_blocks in enumerate(layer_configs):
        z_offset = layer * layer_height
        x_offset = (2 - num_blocks) * spacing / 2

        for i in range(num_blocks):
            if block_idx >= len(blocks):
                break

            ps.setTargetPose(*get_obj_func(blocks[block_idx]))
            ps.execute_pick(gripper_force, axis=2)

            target_x = base_x + x_offset + i * spacing
            target_y = base_y
            target_z = 0.1 + z_offset

            ps.setTargetPose(target_x, target_y, target_z, 0, np.pi, 0)
            ps.execute_place(axis=2)

            block_idx += 1

    ps.go_to_ready_pose()

def sort_by_size_task(ps, get_obj_func, objects=None, size_order=None, start_x=0.4, y_pos=0.0, spacing=0.08, gripper_force=50):
    if objects is None:
        objects = ['small_obj', 'medium_obj', 'large_obj']
    if size_order is None:
        size_order = [0, 1, 2]

    ps.go_to_ready_pose()

    sorted_objects = [objects[i] for i in size_order]

    for i, obj_name in enumerate(sorted_objects):
        ps.setTargetPose(*get_obj_func(obj_name))
        ps.execute_pick(gripper_force, axis=2)

        target_x = start_x + i * spacing
        ps.setTargetPose(target_x, y_pos, 0.1, 0, np.pi, 0)
        ps.execute_place(axis=2)

    ps.go_to_ready_pose()

def sort_colors_task(ps, get_obj_func, objects=None, color_zones=None, gripper_force=50):
    if objects is None:
        objects = {'red': ['red_obj1', 'red_obj2'], 'blue': ['blue_obj1', 'blue_obj2'], 'green': ['green_obj1', 'green_obj2']}
    if color_zones is None:
        color_zones = {'red': (0.3, 0.0), 'blue': (0.5, 0.0), 'green': (0.7, 0.0)}

    ps.go_to_ready_pose()

    for color, obj_list in objects.items():
        zone_x, zone_y = color_zones[color]
        for i, obj_name in enumerate(obj_list):
            ps.setTargetPose(*get_obj_func(obj_name))
            ps.execute_pick(gripper_force, axis=2)

            ps.setTargetPose(zone_x, zone_y + i * 0.05, 0.1, 0, np.pi, 0)
            ps.execute_place(axis=2)

    ps.go_to_ready_pose()