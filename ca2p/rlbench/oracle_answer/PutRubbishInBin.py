def run_task(env, task, object_class_mapping, descriptions = None, obs = None, variation_index: int = 0):
    """
    Main function to run the PickAndLift task with the given environment and task.
    Task Description: Put rubbish in bin. Drop the rubbish into the bin. Pick up the rubbish and leave it in the trash can. Throw away the trash, leaving any other objects alone. Chuck way any rubbish on the table rubbish.
    Objects in the given environment: ['rubbish', 'tomato1', 'tomato2', 'success']
    Example usage of run_action: run_action(skill_function, 'object_name', offset=[0,0,0.05], approach_distance=0.2, timeout=4.0)
    """
    # The following lines perform the stacking using available skills:
    obs, reward, done = run_action(pick, 'rubbish')
    if done:    return
    obs, reward, done = run_action(place, 'success', offset=[0.0, 0.0, 0.3])
    if done:    return
    # code_end