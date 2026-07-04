"""
Feedback system for real-world experiments
Compares generated code with oracle and provides feedback
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def extract_skills(code: str) -> List[str]:
    """Extract skill calls from code"""
    pattern = r'self\.ps\.(\w+)\([^)]*\)'
    matches = re.findall(pattern, code)
    return matches

def extract_objects(code: str) -> List[str]:
    """Extract object references from code"""
    pattern = r"objects\['([^']+)'\]"
    matches = re.findall(pattern, code)
    return matches

def extract_loops(code: str) -> bool:
    """Check if code has loops"""
    return 'for ' in code or 'while ' in code

def compare_with_oracle(generated_code: str, oracle_code: str, task_name: str) -> Tuple[bool, str]:
    """
    Compare generated code with oracle code and generate feedback
    Returns: (success: bool, feedback: str)
    """
    generated_skills = extract_skills(generated_code)
    oracle_skills = extract_skills(oracle_code)

    generated_objects = extract_objects(generated_code)
    oracle_objects = extract_objects(oracle_code)

    generated_has_loop = extract_loops(generated_code)
    oracle_has_loop = extract_loops(oracle_code)

    feedback_messages = []

    # Check skill sequence
    if generated_skills != oracle_skills:
        # Find differences
        missing_skills = set(oracle_skills) - set(generated_skills)
        extra_skills = set(generated_skills) - set(oracle_skills)

        if missing_skills:
            feedback_messages.append(f"Missing skills: {', '.join(missing_skills)}")
        if extra_skills:
            feedback_messages.append(f"Unnecessary skills: {', '.join(extra_skills)}")

        # Check order if same skills but different order
        if set(generated_skills) == set(oracle_skills):
            feedback_messages.append("Skill order is incorrect")

    # Check objects
    if set(generated_objects) != set(oracle_objects):
        missing_objects = set(oracle_objects) - set(generated_objects)
        wrong_objects = set(generated_objects) - set(oracle_objects)

        if missing_objects:
            feedback_messages.append(f"Missing objects: {', '.join(missing_objects)}")
        if wrong_objects:
            feedback_messages.append(f"Wrong objects used: {', '.join(wrong_objects)}")

    # Check loop structure
    if generated_has_loop != oracle_has_loop:
        if oracle_has_loop and not generated_has_loop:
            feedback_messages.append("Missing loop structure for multiple objects")
        elif not oracle_has_loop and generated_has_loop:
            feedback_messages.append("Unnecessary loop structure")

    # Check specific patterns for common tasks
    if "trash" in task_name.lower():
        # Check if handling multiple trash
        if oracle_has_loop and not generated_has_loop:
            feedback_messages.append("Need to handle multiple trash items with a loop")

        # Check if trash_bin is referenced
        if 'trash_bin' in oracle_objects and 'trash_bin' not in generated_objects:
            feedback_messages.append("Missing trash_bin target location")

    # Generate success status
    success = len(feedback_messages) == 0

    # Create feedback string
    if success:
        feedback = "Code matches oracle successfully!"
    else:
        feedback = "Code issues found:\n" + "\n".join(f"- {msg}" for msg in feedback_messages)

    return success, feedback

def identify_error_location(code: str, feedback: str) -> Tuple[int, int, str]:
    """
    Identify where in the code the error is and what needs to be fixed
    Returns: (start_line, end_line, error_type)
    """
    lines = code.split('\n')

    # Find error location based on feedback
    if "Missing loop structure" in feedback:
        # Need to wrap existing pick/place in a loop
        for i, line in enumerate(lines):
            if 'execute_pick' in line:
                # Find the block that needs to be wrapped in loop
                start = i
                end = i
                for j in range(i+1, len(lines)):
                    if 'execute_place' in lines[j]:
                        end = j
                        break
                return start, end, "add_loop"

    elif "Wrong objects used" in feedback:
        # Find wrong object references
        for match in re.findall(r"Wrong objects used: ([^\n]+)", feedback):
            wrong_objs = match.split(', ')
            for wrong_obj in wrong_objs:
                for i, line in enumerate(lines):
                    if wrong_obj in line:
                        return i, i, "replace_object"

    elif "Missing objects" in feedback:
        # Need to add more operations for missing objects
        for i, line in enumerate(lines):
            if 'go_to_ready_pose' in line and i > 0:  # Find last go_to_ready_pose
                return i-1, i-1, "add_operations"

    # Default: regenerate middle part
    return 1, len(lines)-2, "regenerate"

def generate_infilling_prompt(original_code: str, feedback: str, instruction: str) -> Tuple[str, str, Tuple[int, int, str]]:
    """
    Generate infilling prompt for CATP - only for the error part
    Returns: (prompt, infill_type, error_location)
    """
    lines = original_code.split('\n')
    error_location = identify_error_location(original_code, feedback)
    start_line, end_line, error_type = error_location

    error_part = '\n'.join(lines[start_line:end_line+1])

    if error_type == "add_loop":
        # Generate loop wrapper
        prompt = f"""Task: {instruction}
Need loop for multiple items. Wrap this:
{error_part}
Generate loop (NO COMMENTS):
        for i in range("""
        return prompt, "add_loop", error_location

    elif error_type == "replace_object":
        # Generate corrected object reference
        prompt = f"""Fix wrong object in:
{error_part}
Generate corrected line (NO COMMENTS):
        self.ps."""
        return prompt, "replace_line", error_location

    elif error_type == "add_operations":
        # Generate missing operations
        prompt = f"""Task: {instruction}
{feedback}
Add missing operations (NO COMMENTS):
        """
        return prompt, "add_lines", error_location

    else:
        # Generate replacement for error part
        prompt = f"""Fix: {feedback}
Replace:
{error_part}
With corrected code (NO COMMENTS):
        """
        return prompt, "replace_block", error_location

def apply_infilling(original_code: str, infilled_code: str, infill_type: str, error_location: Tuple[int, int, str]) -> str:
    """
    Apply the infilled code to the original code
    """
    lines = original_code.split('\n')
    start_line, end_line, _ = error_location

    # Clean up infilled code
    infilled_lines = [line for line in infilled_code.split('\n') if line.strip()]

    if infill_type == "add_loop":
        # Wrap existing code in loop
        loop_lines = []
        # Add the loop start from infilled
        loop_lines.extend(infilled_lines)
        # Indent the original code inside loop
        for i in range(start_line, end_line+1):
            if lines[i].strip():
                loop_lines.append("    " + lines[i])
        # Replace with loop version
        new_lines = lines[:start_line] + loop_lines + lines[end_line+1:]

    elif infill_type == "replace_line":
        # Replace single line
        new_lines = lines[:start_line] + infilled_lines + lines[end_line+1:]

    elif infill_type == "add_lines":
        # Insert new lines before last go_to_ready_pose
        new_lines = lines[:start_line+1] + infilled_lines + lines[start_line+1:]

    elif infill_type == "replace_block":
        # Replace entire block
        new_lines = lines[:start_line] + infilled_lines + lines[end_line+1:]

    else:
        new_lines = lines

    return '\n'.join(new_lines)

def generate_regeneration_prompt(instruction: str, feedback: str) -> str:
    """
    Generate regeneration prompt for other methods based on feedback
    """
    prompt = f"""Task: {instruction}

Previous attempt had these issues:
{feedback}

Generate corrected robot_move() method body (NO COMMENTS):
        self.ps.go_to_ready_pose()
        """
    return prompt

def save_error_code(task_name: str, error_code: str, feedback: str, method: str, inst_num: int = None, seed: int = None, output_dir: str = "./error_codes"):
    """
    Save error code and feedback for analysis
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Build filename: method_taskname_inst{num}_seed{num}_error.py
    filename_parts = [method, task_name]

    if inst_num is not None:
        filename_parts.append(f"inst{inst_num}")

    if seed is not None:
        filename_parts.append(f"seed{seed+1}")

    filename_parts.append("error")
    filename = "_".join(filename_parts) + ".py"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"""# Task: {task_name}
# Method: {method}
# Instruction: #{inst_num if inst_num else 'N/A'}
# Seed: {seed if seed is not None else 'N/A'}
# Feedback: {feedback}

# ===== ERROR CODE =====
{error_code}
""")

    return filepath