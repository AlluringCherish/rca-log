def run_task(env, task, object_class_mapping, descriptions = None, obs = None, variation_index: int = 0):
    """
    Main function to run the PickAndLift task with the given environment and task.
    Task Description: Pick up the red cup. Grasp the red cup and lift it. Lift the red cup.
    Objects in the given environment: ['cup1', 'cup2', 'success']
    Example usage of run_action: run_action(skill_function, 'object_name', offset=[0,0,0.05], approach_distance=0.2, timeout=4.0)
    """
    # The following lines perform the stacking using available skills:
    target_cup = f"cup{variation_index + 1}"
    obs, reward, done = run_action(pick, target_cup, offset=[0.0, 0.03, 0.04])
    if done:    return
    obs, reward, done = run_action(place, 'success', offset=[0.0, 0.0, 0.1])
    if done:    return
    # code_end