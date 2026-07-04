#!/usr/bin/env python3
"""
Feedback-based Code Regeneration System
- Compares generated code with answer policies
- Generates feedback using LLM
- Regenerates code based on feedback
- CATP uses infilling for fast correction
- Other methods regenerate entire code
"""

import os
import sys
import json
import torch
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN", "")

@dataclass
class CodeComparison:
    """Result of comparing generated code with answer"""
    task_name: str
    instruction: str
    generated_code: str
    answer_code: str
    feedback: str
    error_type: str  # 'missing_step', 'wrong_order', 'wrong_function', 'wrong_params'
    error_location: int  # line number where error occurs

class FeedbackGenerator:
    """Generate feedback by comparing generated code with answer policy"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def compare_codes(self, generated_path: str, answer_path: str, task_name: str, instruction: str) -> CodeComparison:
        """Compare generated code with answer and generate feedback"""
        
        # Read generated code
        if not os.path.exists(generated_path):
            logger.error(f"Generated code not found: {generated_path}")
            return None
            
        with open(generated_path, 'r') as f:
            generated_code = f.read()
            
        # Read answer code
        if not os.path.exists(answer_path):
            logger.error(f"Answer code not found: {answer_path}")
            return None
            
        with open(answer_path, 'r') as f:
            answer_code = f.read()
        
        # Extract robot_move method from both codes
        generated_method = self._extract_robot_move(generated_code)
        answer_method = self._extract_robot_move(answer_code)
        
        # Generate feedback using LLM
        feedback, error_type, error_location = self._generate_feedback(
            instruction, generated_method, answer_method
        )
        
        return CodeComparison(
            task_name=task_name,
            instruction=instruction,
            generated_code=generated_method,
            answer_code=answer_method,
            feedback=feedback,
            error_type=error_type,
            error_location=error_location
        )
    
    def _extract_robot_move(self, code: str) -> str:
        """Extract robot_move method from full code"""
        lines = code.split('\n')
        in_method = False
        method_lines = []
        indent_level = None

        for line in lines:
            if 'def robot_move(self):' in line:
                in_method = True
                method_lines.append(line)
                continue

            if in_method:
                # Check if we've exited the method
                if line and not line.startswith(' ') and not line.startswith('\t'):
                    if 'def ' in line:
                        break
                method_lines.append(line)

        return '\n'.join(method_lines)

    def _clean_code_for_comparison(self, code: str) -> str:
        """Clean code for comparison - remove comments, extra spaces"""
        lines = []
        for line in code.split('\n'):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            # Skip empty lines
            if not line.strip():
                continue
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            lines.append(line.strip())
        return '\n'.join(lines)

    def _codes_are_similar(self, generated: str, answer: str) -> bool:
        """Check if generated code is similar enough to answer"""

        # Extract key functions and their order
        gen_functions = []
        ans_functions = []

        for line in generated.split('\n'):
            if '_task(' in line:
                # Extract function name
                func_match = re.search(r'(\w+_task)\s*\(', line)
                if func_match:
                    gen_functions.append(func_match.group(1))
            elif 'setTargetPose' in line:
                # Extract object being targeted
                obj_match = re.search(r"get_obj\(['\"](\w+)['\"]\)", line)
                if obj_match:
                    gen_functions.append(f"target_{obj_match.group(1)}")

        for line in answer.split('\n'):
            if '_task(' in line:
                func_match = re.search(r'(\w+_task)\s*\(', line)
                if func_match:
                    ans_functions.append(func_match.group(1))
            elif 'setTargetPose' in line:
                obj_match = re.search(r"get_obj\(['\"](\w+)['\"]\)", line)
                if obj_match:
                    ans_functions.append(f"target_{obj_match.group(1)}")

        # Check if main functions match
        if gen_functions == ans_functions:
            return True

        # Check if key task functions are present
        gen_tasks = [f for f in gen_functions if '_task' in f]
        ans_tasks = [f for f in ans_functions if '_task' in f]

        # If task functions match and order is same, it's probably good enough
        if gen_tasks == ans_tasks and len(gen_tasks) > 0:
            logger.info("Main task functions match. Minor differences acceptable.")
            return True

        # Check sequence similarity (allow some differences)
        matches = 0
        for i, func in enumerate(gen_functions):
            if i < len(ans_functions) and func == ans_functions[i]:
                matches += 1

        # If 80% or more matches in sequence, consider it similar
        if len(ans_functions) > 0 and matches / len(ans_functions) >= 0.8:
            logger.info(f"Code similarity: {matches}/{len(ans_functions)} functions match")
            return True

        return False
    
    def _generate_feedback(self, instruction: str, generated: str, answer: str) -> Tuple[str, str, int]:
        """Use LLM to generate feedback"""

        # First check if codes are similar enough
        generated_clean = self._clean_code_for_comparison(generated)
        answer_clean = self._clean_code_for_comparison(answer)

        # Simple similarity check - if key functions are present and in correct order
        if self._codes_are_similar(generated_clean, answer_clean):
            logger.info("Generated code is similar enough to answer. No regeneration needed.")
            return "CORRECT", "correct", -1

        prompt = f"""Compare the generated robot code with the correct answer and provide specific feedback.

Task: {instruction}

Generated Code:
{generated}

Correct Answer:
{answer}

Analyze what's wrong with the generated code. Focus on:
1. Missing steps or function calls
2. Wrong order of operations
3. Wrong function used
4. Wrong parameters passed
5. Missing object manipulations

Provide feedback in this format:
ERROR_TYPE: [missing_step/wrong_order/wrong_function/wrong_params]
ERROR_LINE: [line number where the error is, starting from 1]
FEEDBACK: [specific description of what's wrong and how to fix it]

Be very specific. For example:
- "Missing open_drawer_task before placing die"
- "Wrong parameter: should be 'trash_bin' not 'bin'"
- "Missing die_red pickup before placement"
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Parse the response
        error_type = "missing_step"  # default
        error_line = 1  # default
        feedback = response
        
        if "ERROR_TYPE:" in response:
            match = re.search(r"ERROR_TYPE:\s*\[?(\w+)\]?", response)
            if match:
                error_type = match.group(1)
        
        if "ERROR_LINE:" in response:
            match = re.search(r"ERROR_LINE:\s*\[?(\d+)\]?", response)
            if match:
                error_line = int(match.group(1))
        
        if "FEEDBACK:" in response:
            match = re.search(r"FEEDBACK:\s*(.+)", response, re.DOTALL)
            if match:
                feedback = match.group(1).strip()
        
        logger.info(f"Generated feedback: {feedback[:100]}...")
        return feedback, error_type, error_line

class CodeRegenerator:
    """Regenerate code based on feedback"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def regenerate_catp(self, comparison: CodeComparison) -> str:
        """Use infilling for CATP - only generate the fix"""
        
        lines = comparison.generated_code.split('\n')
        error_line = comparison.error_location - 1  # Convert to 0-based
        
        # Create infilling prompt based on error type
        if comparison.error_type == "missing_step":
            # Insert missing step
            before = '\n'.join(lines[:error_line])
            after = '\n'.join(lines[error_line:])
            
            prompt = f"""Fix the code by adding the missing step.
Task: {comparison.instruction}
Feedback: {comparison.feedback}

Code before fix:
{before}
<INSERT_HERE>
{after}

Generate only the missing line(s) to insert at <INSERT_HERE>:"""
            
        elif comparison.error_type == "wrong_params":
            # Fix parameter in specific line
            if error_line < len(lines):
                wrong_line = lines[error_line]
                prompt = f"""Fix the parameter in this line.
Task: {comparison.instruction}
Feedback: {comparison.feedback}

Wrong line: {wrong_line}
Fixed line:"""
            else:
                prompt = f"""Fix based on feedback: {comparison.feedback}
Generate the corrected line:"""
        
        else:  # wrong_order, wrong_function
            # For more complex fixes, regenerate a small section
            start = max(0, error_line - 2)
            end = min(len(lines), error_line + 3)
            context = '\n'.join(lines[start:end])
            
            prompt = f"""Fix this code section.
Task: {comparison.instruction}  
Feedback: {comparison.feedback}

Wrong code section:
{context}

Fixed code section:"""
        
        # Generate the fix (only 1-2 lines)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=50,  # Small fix only
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        fix = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Apply the fix
        if comparison.error_type == "missing_step":
            lines.insert(error_line, fix.strip())
        elif comparison.error_type == "wrong_params" and error_line < len(lines):
            lines[error_line] = fix.strip()
        else:
            # Replace the section
            start = max(0, error_line - 2)
            end = min(len(lines), error_line + 3)
            fix_lines = fix.strip().split('\n')
            lines[start:end] = fix_lines
        
        return '\n'.join(lines)
    
    def regenerate_other(self, comparison: CodeComparison, method: str) -> str:
        """Regenerate entire code for other methods (CAP, LRLL, RAGCache)"""
        
        # Create regeneration prompt with feedback
        prompt = f"""Task: {comparison.instruction}

Previous attempt had this error: {comparison.feedback}

Correct implementation should be like:
{comparison.answer_code}

Generate the corrected robot_move method:
def robot_move(self):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        regenerated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Ensure it starts with proper method definition
        if not regenerated.strip().startswith('def robot_move'):
            regenerated = f"def robot_move(self):\n{regenerated}"
        
        return regenerated

def save_regenerated_code(task_name: str, code: str, method: str, output_dir: str = "./generated_code"):
    """Save regenerated code to file"""
    
    filepath = os.path.join(output_dir, f"{method}_{task_name}_regenerated.py")
    
    # Create full Python file
    full_code = f"""import numpy as np
import tf.transformations as tf_trans
import os
import sys
sys.path.append('/home/franka-emika/robot_interface')
from pick_and_place_module.skills import PrimitiveSkill
from Object_Detection.object_detection_utils import process_object_detection_from_file, load_coordinates_from_file, get_object_coordinate, postprocess_coordinates
from robot_tasks import *

class RobotController:
    def __init__(self, velocity=0.002, acceleration=0.4):
        self.ps = PrimitiveSkill(velocity, acceleration)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        object_list_path = os.path.join(current_dir, 'object_list.txt')
        self.coordinates_file_path = os.path.join(current_dir, 'object_list_position.py')
        
        self.object_coordinates = load_coordinates_from_file(self.coordinates_file_path)
    
    def get_obj(self, object_name):
        return get_object_coordinate(object_name, self.coordinates_file_path)
    
    def refresh_coordinates(self):
        self.object_coordinates = load_coordinates_from_file(self.coordinates_file_path)
        return self.object_coordinates

{code}

def main():
    robot = RobotController()
    robot.robot_move()

if __name__ == "__main__":
    main()
"""
    
    with open(filepath, 'w') as f:
        f.write(full_code)
    
    logger.info(f"Saved regenerated code to {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Feedback-based Code Regeneration")
    parser.add_argument('--method', required=True, choices=['catp', 'lrll', 'cap', 'ragcache'])
    parser.add_argument('--task', required=True, help='Task name (e.g., ComplexTask)')
    parser.add_argument('--inst_num', type=int, default=0, help='Instruction number')
    parser.add_argument('--generated_dir', default='./generated_code', help='Directory with generated code')
    parser.add_argument('--answer_dir', default='./answer_policy', help='Directory with answer policies')
    parser.add_argument('--model', default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help='Model for feedback generation')
    
    args = parser.parse_args()
    
    # Load instructions
    with open('instructions.jsonl', 'r') as f:
        instructions = [json.loads(line) for line in f]
    
    # Find matching instruction
    instruction = None
    for inst in instructions:
        if inst['task'] == args.task:
            instruction = inst['instruction']
            break
    
    if not instruction:
        logger.error(f"Task {args.task} not found in instructions")
        return
    
    logger.info(f"Processing task: {args.task}, instruction: {instruction}")
    
    # Load model for feedback and regeneration
    logger.info(f"Loading model: {args.model}")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Paths
    generated_path = os.path.join(args.generated_dir, f"{args.method}_{args.task}_inst{args.inst_num}_seed0.py")
    answer_path = os.path.join(args.answer_dir, f"{args.task}.py")
    
    # Generate feedback
    logger.info("Generating feedback...")
    feedback_gen = FeedbackGenerator(model, tokenizer)
    comparison = feedback_gen.compare_codes(generated_path, answer_path, args.task, instruction)
    
    if not comparison:
        logger.error("Failed to generate feedback")
        return
    
    logger.info(f"Feedback: {comparison.feedback}")
    logger.info(f"Error type: {comparison.error_type}")
    logger.info(f"Error location: line {comparison.error_location}")

    # Check if regeneration is needed
    if comparison.error_type == "correct":
        logger.info("✓ Generated code is correct or similar enough. No regeneration needed.")
        print("\n" + "="*50)
        print("CODE VALIDATION RESULTS")
        print("="*50)
        print(f"Method: {args.method}")
        print(f"Task: {args.task}")
        print(f"Result: ✓ CORRECT - No regeneration needed")
        print("Generated code is similar enough to the answer policy.")
        return

    # Regenerate code
    logger.info(f"Regenerating code using {args.method}...")
    regenerator = CodeRegenerator(model, tokenizer)

    if args.method == 'catp':
        # Use infilling for fast fix
        regenerated_code = regenerator.regenerate_catp(comparison)
        logger.info("CATP: Used infilling for fast correction")
    else:
        # Full regeneration for other methods
        regenerated_code = regenerator.regenerate_other(comparison, args.method)
        logger.info(f"{args.method.upper()}: Regenerated entire code")

    # Save regenerated code
    output_path = save_regenerated_code(args.task, regenerated_code, args.method, args.generated_dir)
    logger.info(f"Regeneration complete: {output_path}")
    
    # Show comparison
    print("\n" + "="*50)
    print("REGENERATION RESULTS")
    print("="*50)
    print(f"Method: {args.method}")
    print(f"Task: {args.task}")
    print(f"Feedback: {comparison.feedback}")
    print(f"Error Type: {comparison.error_type}")
    print(f"Error Location: Line {comparison.error_location}")
    print("\nRegenerated Code Preview:")
    print(regenerated_code[:500] + "..." if len(regenerated_code) > 500 else regenerated_code)

if __name__ == "__main__":
    main()
