"""Oracle policy collection for real-world robotic tasks"""

from pathlib import Path
import importlib.util

# List of available oracle policies
ORACLE_POLICIES = [
    "PickUpCup",
    "PickAndLift",
    "PushButton",
    "OpenDrawer",
    "PutRubbishInBin",
    "StackBlocks",
    "SortColors",
    "InsertPeg",
    "CloseBox",
    "PourWater",
]

def load_policy(policy_name: str):
    """Dynamically load a policy module by name"""
    if policy_name not in ORACLE_POLICIES:
        raise ValueError(f"Policy {policy_name} not found. Available policies: {ORACLE_POLICIES}")

    policy_path = Path(__file__).parent / f"{policy_name}.py"
    spec = importlib.util.spec_from_file_location(policy_name, policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_all_policies():
    """Return a dictionary of all available policies"""
    policies = {}
    for policy_name in ORACLE_POLICIES:
        try:
            policies[policy_name] = load_policy(policy_name)
        except Exception as e:
            print(f"Warning: Could not load policy {policy_name}: {e}")
    return policies