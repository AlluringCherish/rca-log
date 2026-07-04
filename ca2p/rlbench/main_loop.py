import argparse
import importlib
import numpy as np

from env import setup_environment, shutdown_environment

from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.helper import object_names
from utils.feedback import Feedback, FeedbackWithError

from rlbench.gym import RLBenchEnv
from utils.code_postprocess import postprocess

def patched_close(self) -> None:
    pass

RLBenchEnv.close = patched_close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an arbitrary RLBench task by class name"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="RLBench task class name, e.g. PushButton, OpenDrawer, PutRubbishInBin, MyCustomTask"
    )
    args = parser.parse_args()

    tasks_module = importlib.import_module("rlbench.tasks")
    try:
        task_cls = getattr(tasks_module, args.task)
    except AttributeError:
        raise SystemExit(f"Error: '{args.task}'.")
    
    # env setup
    env, task = setup_environment(task_cls)

    descriptions, obs = task.reset()
    
    objects, objects_dict = object_names(task)
    print(f"[Main] [Object_Start]\nObjects in the task: {objects}\n[Object_End]")
    
    timestep = 0
    while True:

        if env.is_shutdown():
            print("[Main] Environment is shutdown. Please restart the environment.")
            break
        
        print(f"[Main] Objects in the task: {object_names(task)}")
        
        cmd = input(">>> ")

        if cmd.strip().lower() in ("exit", "quit"):
            print("\n[Main] Exiting RLBench interactive console.")
            shutdown_environment(env)
            break

        if cmd.strip().lower() == "restart":
            print("\n[Main] Restarting the task...")
            descriptions, obs = task.reset()
            print("[Main] Task restarted successfully.")
            continue
         
        if cmd.strip().lower() == "run":
            timestep += 1
            # policy validation
            policy_file_path = f"./tasks/{args.task}.py"
            policy_code = None
            with open(policy_file_path, "r", encoding="utf-8") as f:
                policy_code = f.read()
            assert policy_code is not None
            type_map = dict()
            for obj in objects_dict:
                type_map[obj['name']] = obj['type']
            # cleaned_policy_code = postprocess(policy_code, type_map)
            with open(policy_file_path, "w", encoding="utf-8") as f:
                f.write(policy_code)

            module = importlib.import_module(f"tasks.{args.task}")
            importlib.reload(module)
            run_task = getattr(module, "run_task")
            goal_condition, all_mets = task.get_goal_condition()
            
            try:
                run_task(env, task, type_map, args.task, descriptions, obs)
            except SkillFailure as e: 
                feedback = FeedbackWithError(
                    env=env,
                    task=task,
                    error_message=getattr(e, 'message', str(e)),
                    skill_type=getattr(e, 'skill_type', 'unknown'),
                    attempted_action=getattr(e, 'attempted_action', None),
                    waypoint_index=getattr(e, 'waypoint_index', -1),
                    step_index=getattr(e, 'step_index', -1),
                    original_robot_pos=getattr(e, 'original_robot_pos', None),
                    done=True
                )
                print(f"[Main] {feedback}")
                continue

            except PathOutOfWorkspace as e:
                feedback = FeedbackWithError(
                    env=env,
                    task=task,
                    error_message=getattr(e, 'message', str(e)),
                    skill_type=getattr(e, 'skill_type', 'unknown'),
                    attempted_action=getattr(e, 'attempted_action', None),
                    waypoint_index=getattr(e, 'waypoint_index', -1),
                    step_index=getattr(e, 'step_index', -1),
                    original_robot_pos=getattr(e, 'original_robot_pos', None),
                    done=True
                )
                print(f"[Main] {feedback}")
                shutdown_environment(env)
                break

            except BaseException as e:

                print(f"[Main] [Feedback] An unexpected error occurred: {type(e).__name__} - {e}")
                continue
        
        success, terminate = task._task.success()
        if success:
            print("[Main] [Feedback] Task completed successfully! [Main]")
            shutdown_environment(env)
            break
        else:
            try:
                feedback = Feedback(env, task)
                print(f"[Main] {feedback}")
            except Exception as e:
                print(f"[Main] UNCOMPLETED TASK FEEDBACK GEN ERROR: {e}")
            continue
