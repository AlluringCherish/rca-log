import traceback
from typing import Any, List, Tuple, Optional
import time
from modules.catp_agent import CATPAgent


class TaskExecutor:
    """
    Task execution loop for CATP
    Implements Algorithm 1 from the paper
    """
    
    def __init__(self, agent: CATPAgent, environment):
        """
        Initialize task executor
        
        Args:
            agent: CATP agent instance
            environment: Environment for task execution
        """
        self.agent = agent
        self.env = environment
        
    def execute_scenario(self) -> None:
        """
        Execute complete scenario with multiple tasks
        Implements Algorithm 1(A) - Task Execution Loop
        """
        scenario_done = False
        max_attempts = 10  # Prevent infinite loop
        attempts = 0
        
        while not scenario_done and attempts < max_attempts:
            # Execute single task
            success = self.execute_task()
            
            # Check if scenario is complete
            scenario_info = self.env.get_scenario_info() if hasattr(self.env, 'get_scenario_info') else {}
            scenario_done = scenario_info.get('scenario_done', False)
            
            if success:
                print(f"Task completed successfully")
            else:
                print(f"Task failed")
                
            attempts += 1
            
        if attempts >= max_attempts:
            print(f"Reached maximum attempts ({max_attempts})")
                
    def execute_task(self) -> bool:
        """
        Execute a single task with error handling and code repair
        
        Returns:
            True if task succeeded, False otherwise
        """
        # Initialize task
        t = 0
        observation, instruction = self.env.reset()
        
        # Generate initial policy code
        print(f"Generating policy for: {instruction}")
        executable_code = self.agent.generate(observation, instruction)
        
        # Execute the generated code to get action sequence
        try:
            actions = self.execute_code(executable_code)
        except Exception as e:
            print(f"Initial code generation failed: {e}")
            return False
            
        # Execute actions in environment
        task_done = False
        task_success = False
        
        while not task_done and t < len(actions):
            try:
                # Execute current action
                action = actions[t]
                observation, info = self.env.step(action)
                
            except Exception as e:
                # Handle execution error with code repair
                print(f"Execution error at step {t}: {e}")
                
                # Edit the faulty code
                executable_code = self.agent.edit(e, executable_code)
                
                # Re-execute to get new action sequence
                try:
                    actions = self.execute_code(executable_code)
                    # Continue from current timestep
                    continue
                except Exception as repair_error:
                    print(f"Code repair failed: {repair_error}")
                    return False
                    
            # Update timestep
            t += 1
            
            # Check task completion
            task_done = info.get('is_done', False)
            task_success = info.get('is_success', False)
            
        # Update cache if task succeeded
        if task_success:
            print("Task succeeded - updating cache")
            self.agent.update(executable_code)
            
        return task_success
        
    def execute_code(self, code: str) -> List[Any]:
        """
        Execute policy code to generate action sequence
        
        Args:
            code: Executable policy code
            
        Returns:
            List of actions to execute
        """
        # Create execution namespace with environment
        namespace = {
            'env': self.env,
            'observation': self.env.get_observation() if hasattr(self.env, 'get_observation') else None
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Extract actions from namespace
        if 'actions' in namespace:
            return namespace['actions']
        elif 'execute_task' in namespace:
            # If code defines execute_task function, call it
            return namespace['execute_task'](self.env)
        else:
            # Try to find any list of actions
            for key, value in namespace.items():
                if isinstance(value, list) and key not in ['env', 'observation']:
                    return value
                    
        return []


class SimulatedEnvironment:
    """
    Simulated environment for testing CATP
    """
    
    def __init__(self, num_tasks: int = 5):
        """
        Initialize simulated environment
        
        Args:
            num_tasks: Number of tasks in scenario
        """
        self.num_tasks = num_tasks
        self.current_task = 0
        self.current_step = 0
        self.max_steps = 10
        self.observation = None
        
    def reset(self) -> Tuple[Any, str]:
        """
        Reset environment for new task
        
        Returns:
            Initial observation and task instruction
        """
        self.current_step = 0
        self.observation = {
            'position': [0, 0, 0],
            'objects': ['cube', 'sphere', 'cylinder'],
            'task_id': self.current_task
        }
        
        instructions = [
            "Pick up the red cube and place it on the table",
            "Stack all blocks in ascending order of size",
            "Move the robot arm to position [1, 2, 3]",
            "Grasp the sphere and rotate it 90 degrees",
            "Sort objects by color from left to right"
        ]
        
        instruction = instructions[self.current_task % len(instructions)]
        
        return self.observation, instruction
        
    def step(self, action: Any) -> Tuple[Any, dict]:
        """
        Execute action in environment
        
        Args:
            action: Action to execute
            
        Returns:
            New observation and info dictionary
        """
        self.current_step += 1
        
        # Update observation based on action
        if isinstance(action, dict):
            if 'position' in action:
                self.observation['position'] = action['position']
            if 'grasp' in action:
                self.observation['grasped'] = action['grasp']
                
        # Check if task is done
        is_done = self.current_step >= self.max_steps
        
        # Simulate success (simplified)
        is_success = is_done and (self.current_step % 2 == 0)
        
        if is_done and is_success:
            self.current_task += 1
            
        info = {
            'is_done': is_done,
            'is_success': is_success,
            'scenario_done': self.current_task >= self.num_tasks
        }
        
        return self.observation, info
        
    def get_observation(self) -> Any:
        """Get current observation"""
        return self.observation
        
    def get_scenario_info(self) -> dict:
        """Get scenario information"""
        return {
            'current_task': self.current_task,
            'total_tasks': self.num_tasks,
            'scenario_done': self.current_task >= self.num_tasks
        }