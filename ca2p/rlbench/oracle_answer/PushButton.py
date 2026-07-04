def run_task(env, task, object_class_mapping, descriptions = None, obs = None, variation_index: int = 0):
    """
    Main function to run the PickAndLift task with the given environment and task.
    Task Description: Push the maroon button. Push down the maroon button. Press the button with the maroon base. Press the maroon button.
    Objects in the given environment: ['push_button_target', 'target_button_topPlate', 'target_button_joint']
    Example usage of run_action: run_action(skill_function, 'object_name', offset=[0,0,0.05], approach_distance=0.2, timeout=4.0)
    """
    # The following lines perform the stacking using available skills:
    obs, reward, done = run_action(push, 'push_button_target', offset=[0.0, 0.0, -0.03])
    if done:    return
    # code_end