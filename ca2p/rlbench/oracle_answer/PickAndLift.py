def run_task(env, task, object_class_mapping, descriptions = None, obs = None, variation_index: int = 1):
    """
    Main function to run the PickAndLift task with the given environment and task.
    Task Description: Pick up the red block and lift it up to the target. Grasp the red block to the target. Lift the red block up to the target.
    Objects in the given environment: ['pick_and_lift_target', 'stack_blocks_distractor0', 'stack_blocks_distractor1', 'pick_and_lift_success]
    Example usage of run_action: run_action(skill_function, 'object_name', offset=[0,0,0.05], approach_distance=0.2, timeout=4.0)
    """
    # The following lines perform the stacking using available skills:
    obs, reward, done = run_action(pick, 'pick_and_lift_target', offset=[0.0, 0.0, 0.01])
    if done:    return
    obs, reward, done = run_action(place, 'pick_and_lift_success')
    if done:    return
    # code_end