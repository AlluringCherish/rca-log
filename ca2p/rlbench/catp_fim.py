import torch
import re
from typing import Tuple, Optional, List
from transformers import StoppingCriteria, StoppingCriteriaList


class DefStoppingCriteria(StoppingCriteria):
    """Stop generation when 'def ' is encountered in generated output"""
    def __init__(self, tokenizer, initial_length):
        self.tokenizer = tokenizer
        self.initial_length = initial_length
        
    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) <= self.initial_length:
            return False
        generated_ids = input_ids[0][self.initial_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "\ndef " in generated_text or generated_text.strip().startswith("def "):
            return True
        return False


class FIMCodeModifier:
    """Line-level code modification using FIM (Fill-in-the-Middle)."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Qwen models FIM tokens
        model_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else ""


        if "Qwen" in model_name or "qwen" in model_name.lower():
            # Qwen models
            self.fim_prefix = "<|fim_prefix|>"
            self.fim_middle = "<|fim_middle|>"
            self.fim_suffix = "<|fim_suffix|>"
        elif "CodeLlama" in model_name or "codellama" in model_name.lower():
            # CodeLlama models
            self.fim_prefix = "<PRE>"
            self.fim_middle = "<MID>"
            self.fim_suffix = "<SUF>"
        elif "starcoder" in model_name.lower():
            # StarCoder models
            self.fim_prefix = "<fim_prefix>"
            self.fim_middle = "<fim_middle>"
            self.fim_suffix = "<fim_suffix>"
        elif "deepseek" in model_name.lower():
            # DeepSeek Coder
            self.fim_prefix = "<|fim▁begin|>"
            self.fim_middle = "<|fim▁hole|>"
            self.fim_suffix = "<|fim▁end|>"
        else:
            # Default (Qwen style)
            self.fim_prefix = "<|fim_prefix|>"
            self.fim_middle = "<|fim_middle|>"
            self.fim_suffix = "<|fim_suffix|>"
            print(f"[FIM] Warning: Unknown model {model_name}, using default Qwen FIM tokens")

        print(f"[FIM] Model: {model_name}")
        print(f"[FIM] Using tokens: prefix={self.fim_prefix}, middle={self.fim_middle}, suffix={self.fim_suffix}")
        
    def parse_error_feedback(self, feedback: str) -> dict:
        """Extract detailed information from error feedback."""
        print(f"\n[FIM] ========== Parsing Feedback ==========\n[FIM] Feedback: {feedback[:200]}..." if len(feedback) > 200 else f"\n[FIM] ========== Parsing Feedback ==========\n[FIM] Feedback: {feedback}")
        
        error_info = {
            'type': None,
            'line_no': None,
            'function_name': None,
            'variable_name': None,
            'message': feedback
        }
        

        # Format 1: "NameError: name 'x' is not defined"
        # Format 2: "NameError - name 'x' is not defined"
        name_error_match = re.search(r"NameError[:\s\-]+name '(\w+)' is not defined", feedback)
        if name_error_match:
            error_info['type'] = 'NameError'
            error_info['variable_name'] = name_error_match.group(1)
            print(f"[FIM] ✓ Identified NameError for variable: '{error_info['variable_name']}'")
            

        line_match = re.search(r"line (\d+)", feedback)
        if line_match:
            error_info['line_no'] = int(line_match.group(1))
            print(f"[FIM] ✓ Found error at line: {error_info['line_no']}")
            

        attr_error_match = re.search(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", feedback)
        if attr_error_match:
            error_info['type'] = 'AttributeError'
            error_info['variable_name'] = attr_error_match.group(2)
            print(f"[FIM] ✓ Identified AttributeError: '{attr_error_match.group(1)}' has no attribute '{attr_error_match.group(2)}'")
            

        if "TypeError" in feedback:
            error_info['type'] = 'TypeError'
            print(f"[FIM] ✓ Identified TypeError")
            

        if "IndexError" in feedback:
            error_info['type'] = 'IndexError'
            print(f"[FIM] ✓ Identified IndexError")
        
        if not error_info['type']:
            print(f"[FIM] ⚠ Could not identify specific error type")
        
        print(f"[FIM] ========================================\n")
        return error_info
    
    def find_error_line(self, code: str, error_info: dict) -> Tuple[int, str]:
        """Locate the line where the error occurred."""
        lines = code.split('\n')
        

        if error_info['line_no'] and error_info['line_no'] <= len(lines):
            return error_info['line_no'] - 1, lines[error_info['line_no'] - 1]
        

        if error_info['variable_name']:
            for i, line in enumerate(lines):
                if error_info['variable_name'] in line:

                    if not line.strip().startswith('#'):
                        return i, line
        
        return -1, ""
    
    def create_fim_prompt_for_line(self, code: str, line_idx: int, error_info: dict, 
                                   instruction: str = None) -> str:
        """Create an FIM prompt for the specified line."""
        lines = code.split('\n')
        
        if line_idx < 0 or line_idx >= len(lines):
            return None
            
        error_line = lines[line_idx]
        

        context_before = max(0, line_idx - 3)
        context_after = min(len(lines), line_idx + 4)
        

        prefix_lines = lines[:line_idx]
        if context_before > 0:
            prefix_lines = lines[:context_before] + ['    # ...'] + lines[context_before:line_idx]
        prefix = '\n'.join(prefix_lines)
        

        suffix_lines = lines[line_idx + 1:]
        if context_after < len(lines):
            suffix_lines = lines[line_idx + 1:context_after] + ['    # ...'] + lines[context_after:]
        suffix = '\n'.join(suffix_lines) if suffix_lines else ""
        

        error_type = error_info['type']
        

        hint = ""
        if error_type == 'NameError' and error_info['variable_name']:
            hint = f"\n# Fix: Replace undefined '{error_info['variable_name']}' with correct function/variable"
        elif error_type == 'AttributeError':
            hint = f"\n# Fix: Use correct attribute or method"
        elif error_type == 'TypeError':
            hint = f"\n# Fix: Correct the type mismatch"
        elif error_type == 'IndexError':
            hint = f"\n# Fix: Use valid index or check bounds"
        

        inst = f"\n# Task: {instruction}" if instruction else ""
        
        prompt = f"""{hint}{inst}
{self.fim_prefix}{prefix}
{self.fim_suffix}{suffix}
{self.fim_middle}"""
        
        return prompt
    
    def generate_fixed_line(self, prompt: str, max_tokens: int = 128, past_key_values=None) -> str:
        """Generate a corrected line using FIM.

        Args:
            prompt: FIM-style prompt
            max_tokens: maximum number of tokens to generate
            past_key_values: optional precomputed KV cache
        """
        print(f"[FIM] Generating fixed line with FIM...")
        print(f"[FIM] Max tokens: {max_tokens}")
        if past_key_values:
            print(f"[FIM] Using pre-computed KV cache")
        
        embed_device = self.model.model.embed_tokens.weight.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(embed_device)
        
        initial_length = len(inputs.input_ids[0])
        stopping_criteria = StoppingCriteriaList([
            DefStoppingCriteria(self.tokenizer, initial_length)
        ])
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                past_key_values=past_key_values,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        

        if self.fim_middle in generated:
            generated_part = generated.split(self.fim_middle)[-1]
            if self.fim_suffix in generated_part:
                generated_part = generated_part.split(self.fim_suffix)[0]
            

            lines = generated_part.strip().split('\n')
            if lines:
                result = lines[0]
                print(f"[FIM] ✓ Generated line: {result}")
                return result
        
        print(f"[FIM] ⚠ Failed to generate valid line")
        return ""
    
    def fix_code_with_fim(self, code: str, feedback: str, instruction: str = None,
                          gripper_position: str = None, available_objects: list = None,
                          policy_manager=None, task_name: str = None) -> str:
        """Intelligently modify code using FIM.

        Args:
            code: code segment to update
            feedback: error feedback text
            instruction: task description
            gripper_position: current gripper position
            available_objects: list of available objects
            policy_manager: optional cache manager
            task_name: task name used for cache selection
        """
        print(f"\n[FIM] ========== Starting Code Fix ==========" )
        print(f"[FIM] Instruction: {instruction}" if instruction else "[FIM] No instruction provided")
        print(f"[FIM] Gripper position: {gripper_position}" if gripper_position else "[FIM] No gripper position")
        print(f"[FIM] Available objects: {available_objects}" if available_objects else "[FIM] No object list provided")
        

        merged_kv_cache = None
        if policy_manager and task_name:
            print(f"[FIM] Using policy manager to get relevant code caches for task: {task_name}")
            

            selected_blocks = policy_manager.select_relevant_blocks(
                instruction if instruction else "", 
                task_name, 
                perplexity_threshold=20.0
            )
            
            if selected_blocks:
                print(f"[FIM] Found {len(selected_blocks)} relevant blocks with low perplexity")
                

                kv_caches_to_merge = []
                

                if policy_manager.helper_code_cache is not None:
                    kv_caches_to_merge.append(policy_manager.helper_code_cache)
                    print("[FIM] Added helper code cache")
                

                if policy_manager.skill_interface_cache is not None:
                    kv_caches_to_merge.append(policy_manager.skill_interface_cache)
                    print("[FIM] Added skill interface cache")
                

                for block_id, block_info in selected_blocks.items():
                    if block_info.get('code_kv'):
                        kv_caches_to_merge.append(block_info['code_kv'])
                        print(f"[FIM] Added code cache from block: {block_id}")
                

                if kv_caches_to_merge:
                    try:
                        merged_kv_cache = policy_manager.merge_kv_caches(kv_caches_to_merge)
                        print(f"[FIM] Successfully merged {len(kv_caches_to_merge)} KV caches for FIM")
                    except Exception as e:
                        print(f"[FIM] Failed to merge KV caches: {e}")
                        merged_kv_cache = None
        

        if self.is_semantic_feedback(feedback):
            print(f"[FIM] Detected semantic feedback - using semantic fix")
            return self.fix_semantic_error(code, feedback, instruction, gripper_position, available_objects,
                                          past_key_values=merged_kv_cache)
        

        error_info = self.parse_error_feedback(feedback)
        
        if not error_info['type']:

            print(f"[FIM] No clear error type - trying semantic fix")
            return self.fix_semantic_error(code, feedback, instruction, gripper_position, available_objects,
                                          past_key_values=merged_kv_cache)
        

        line_idx, error_line = self.find_error_line(code, error_info)
        
        if line_idx < 0:
            print(f"[FIM] ⚠ Could not locate error line in code")
            print(f"[FIM] Falling back to semantic fix")
            return self.fix_semantic_error(code, feedback, instruction, gripper_position, available_objects,
                                          past_key_values=merged_kv_cache)
        
        print(f"[FIM] ✓ Found error at line {line_idx + 1}: {error_line.strip()}")
        print(f"[FIM] Error type: {error_info['type']}")
        

        print(f"[FIM] Creating FIM prompt for line {line_idx + 1}...")
        fim_prompt = self.create_fim_prompt_for_line(code, line_idx, error_info, instruction)
        
        if not fim_prompt:
            print(f"[FIM] ⚠ Failed to create FIM prompt")
            return code
        
        print(f"[FIM] ✓ FIM prompt created successfully")
        

        print(f"[FIM] Generating fixed line...")
        fixed_line = self.generate_fixed_line(fim_prompt, past_key_values=merged_kv_cache)
        
        if not fixed_line:
            print(f"[FIM] ⚠ Failed to generate fixed line")
            print(f"[FIM] Returning original code")
            return code
        

        lines = code.split('\n')
        

        original_indent = len(error_line) - len(error_line.lstrip())
        fixed_line = ' ' * original_indent + fixed_line.strip()
        
        print(f"\n[FIM] ========== Code Modification ==========" )
        print(f"[FIM] Original line {line_idx + 1}: {lines[line_idx]}")
        print(f"[FIM] Fixed line {line_idx + 1}: {fixed_line}")
        
        lines[line_idx] = fixed_line
        fixed_code = '\n'.join(lines)
        
        print(f"[FIM] ✓ Successfully replaced line {line_idx + 1}")
        

        print(f"\n[FIM] ========== Complete Fixed Code ==========")
        print(fixed_code)
        print(f"[FIM] ========================================\n")
        
        return fixed_code
    
    def fix_task_not_done(self, code: str, instruction: str = None, gripper_position: str = None, 
                          available_objects: list = None) -> str:
        """Modify the code when the task fails by trying a completely new strategy."""
        print(f"[FIM] Task not done - trying alternative approaches...")
        

        if available_objects:
            print(f"[FIM] Using provided objects: {available_objects}")
        
        import re
        import random
        
        lines = code.split('\n')
        fixed_lines = []
        

        actions = []
        for i, line in enumerate(lines):
            if 'run_action(' in line and not line.strip().startswith('#'):
                match = re.search(r'run_action\((\w+),\s*[\'"]([^\'\"]+)[\'"]', line)
                if match:
                    actions.append({
                        'line_idx': i,
                        'line': line,
                        'skill': match.group(1),
                        'object': match.group(2)
                    })
        
        print(f"[FIM] Found {len(actions)} actions in code")
        

        all_objects = set()
        

        if available_objects:
            if isinstance(available_objects, list):
                all_objects.update(available_objects)
            elif isinstance(available_objects, str):

                obj_matches = re.findall(r'[\'"]([^\'\"]+)[\'"]', available_objects)
                all_objects.update(obj_matches)
        

        if instruction:

            obj_list_match = re.search(r'Objects.*?:\s*\[(.*?)\]', instruction)
            if obj_list_match:
                obj_str = obj_list_match.group(1)
                objs = re.findall(r'[\'"]([^\'\"]+)[\'"]', obj_str)
                all_objects.update(objs)
        

        if not all_objects:
            for line in lines:

                obj_matches = re.findall(r'[\'"]([^\'\"]+)[\'"]', line)
                for obj in obj_matches:

                    if any(keyword in obj.lower() for keyword in ['target', 'block', 'success', 'cube', 'object', 'grasp', 'marker', 'cup', 'plate', 'button']):
                        all_objects.add(obj)
        

        used_objects = {action['object'] for action in actions}
        unused_objects = all_objects - used_objects
        
        print(f"[FIM] Available objects: {all_objects}")
        print(f"[FIM] Unused objects: {unused_objects}")
        

        if len(actions) == 0:
            print(f"[FIM] No actions found - generating complete new solution")
            return self.generate_code_from_scratch(lines, instruction, all_objects)
        

        if len(actions) >= 2:
            for i in range(len(actions)-1):
                if (actions[i]['skill'] == actions[i+1]['skill'] and 
                    actions[i]['object'] == actions[i+1]['object']):
                    print(f"[FIM] ✓ Duplicate action found - generating new sequence")
                    

                    if instruction:
                        new_code = self.generate_new_sequence(lines, actions, instruction, all_objects)
                        if new_code != code:
                            return new_code
        

        if unused_objects and len(actions) > 0:
            print(f"[FIM] Trying different objects from unused set")
            

            alternative_obj = None
            first_action = actions[0]
            

            if first_action['skill'] in ['push', 'slide'] and 'block' in unused_objects:
                alternative_obj = 'block'

            elif first_action['skill'] == 'pick':
                for obj in unused_objects:
                    if 'grasp' in obj or 'cube' in obj or 'block' in obj:
                        alternative_obj = obj
                        break
            

            if not alternative_obj:
                alternative_obj = list(unused_objects)[0]
            

            line_idx = first_action['line_idx']
            indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            lines[line_idx] = f"{' ' * indent}obs, reward, done = run_action({first_action['skill']}, '{alternative_obj}')"
            print(f"[FIM] ✓ Changed object from '{first_action['object']}' to '{alternative_obj}'")
            
            fixed_code = '\n'.join(lines)
            print(f"\n[FIM] ========== Complete Fixed Code ==========\n{fixed_code}\n[FIM] ========================================\n")
            return fixed_code
        

        print(f"[FIM] Generating completely new skill sequence")
        new_code = self.generate_alternative_solution(lines, instruction, all_objects)
        if new_code != code:
            return new_code
        
        print(f"[FIM] ⚠ All alternative approaches exhausted")
        print(f"\n[FIM] ========== Returning Original Code ==========\n{code}\n[FIM] ========================================\n")
        return code
    
    def generate_code_from_scratch(self, lines, instruction, all_objects):
        """Create a full action sequence when no actions exist in the code."""
        print(f"[FIM] Generating complete solution from instruction")
        
        inst_lower = instruction.lower() if instruction else ""
        

        best_object = self.find_best_object_from_instruction(inst_lower, all_objects)
        

        if 'slide' in inst_lower or 'push' in inst_lower:

            obj = best_object if best_object else ('block' if 'block' in str(all_objects) else list(all_objects)[0])
            return self.create_simple_action(lines, 'push', obj)
        elif 'pick' in inst_lower and 'lift' in inst_lower:

            obj = list(all_objects)[0] if all_objects else 'target'
            return self.create_double_action(lines, 'pick', obj, 'place', obj)
        elif 'stack' in inst_lower:

            objs = list(all_objects)
            if len(objs) >= 2:
                return self.create_double_action(lines, 'pick', objs[0], 'place', objs[1])
            else:
                return self.create_simple_action(lines, 'pick', objs[0] if objs else 'target')
        else:

            obj = list(all_objects)[0] if all_objects else 'target'
            return self.create_double_action(lines, 'pick', obj, 'place', obj)
    
    def find_best_object_from_instruction(self, inst_lower, all_objects):
        """Locate objects mentioned in the instruction and return them."""
        if not all_objects:
            return None
            

        for obj in all_objects:
            obj_lower = obj.lower()

            obj_parts = obj_lower.replace('_', ' ').split()
            

            if obj_lower in inst_lower:
                print(f"[FIM] Found exact match: '{obj}' in instruction")
                return obj
            

            for part in obj_parts:
                if len(part) > 2 and part in inst_lower:
                    print(f"[FIM] Found partial match: '{obj}' (matched '{part}')")
                    return obj
        
        return None
    
    def create_simple_action(self, lines, skill, obj):
        """Generate a single action."""
        new_lines = []
        for line in lines:
            new_lines.append(line)

            if 'def run_task' in line or '"""' in line and len(new_lines) > 10:
                new_lines.append("    obs, reward, done = run_action({}, '{}')".format(skill, obj))
                new_lines.append("    if done:")
                new_lines.append("        return")
                break
        
        result = '\n'.join(new_lines)
        print(f"\n[FIM] ========== Generated Code (Simple Action) ==========\n{result}\n[FIM] ========================================\n")
        return result
    
    def create_double_action(self, lines, skill1, obj1, skill2, obj2):
        """Generate two actions."""
        new_lines = []
        for line in lines:
            new_lines.append(line)

            if 'def run_task' in line or '"""' in line and len(new_lines) > 10:
                new_lines.append("    obs, reward, done = run_action({}, '{}')".format(skill1, obj1))
                new_lines.append("    if done:")
                new_lines.append("        return")
                new_lines.append("    obs, reward, done = run_action({}, '{}')".format(skill2, obj2))
                new_lines.append("    if done:")
                new_lines.append("        return")
                break
        
        result = '\n'.join(new_lines)
        print(f"\n[FIM] ========== Generated Code (Double Action) ==========\n{result}\n[FIM] ========================================\n")
        return result
    
    def generate_new_sequence(self, lines, actions, instruction, all_objects):
        """Create a brand-new action sequence from the instruction."""
        print(f"[FIM] Generating new sequence based on instruction: {instruction}")
        
        inst_lower = instruction.lower() if instruction else ""
        new_lines = []
        action_start_idx = actions[0]['line_idx'] if actions else -1
        

        skills_to_use = []
        objects_to_use = list(all_objects) if all_objects else ['target']
        

        if 'lift' in inst_lower or 'raise' in inst_lower:
            skills_to_use = ['pick', 'place']  # pick and lift
        elif 'push' in inst_lower or 'slide' in inst_lower:

            skills_to_use = ['push']  # Just push the block

            if 'block' in str(all_objects) and 'target' in str(all_objects):
                objects_to_use = ['block']
        elif 'stack' in inst_lower:
            skills_to_use = ['pick', 'place', 'pick', 'place']  # stack multiple objects
        elif 'open' in inst_lower:
            skills_to_use = ['move', 'open_gripper']
        elif 'close' in inst_lower:
            skills_to_use = ['move', 'close_gripper']
        elif 'place' in inst_lower and 'pick' in inst_lower:
            skills_to_use = ['pick', 'place']
        else:

            skills_to_use = ['move', 'pick', 'place']
        
        print(f"[FIM] New skill sequence: {skills_to_use}")
        

        for i, line in enumerate(lines):
            if i == action_start_idx and skills_to_use:

                indent = len(line) - len(line.lstrip())
                
                for j, skill in enumerate(skills_to_use):

                    if skill in ['open_gripper', 'close_gripper']:
                        new_lines.append(f"{' ' * indent}obs, reward, done = run_action({skill})")
                    else:
                        obj_idx = j % len(objects_to_use)
                        obj = objects_to_use[obj_idx]
                        new_lines.append(f"{' ' * indent}obs, reward, done = run_action({skill}, '{obj}')")
                    new_lines.append(f"{' ' * indent}if done:")
                    new_lines.append(f"{' ' * (indent+4)}return")
                

                continue
            elif any(i == a['line_idx'] for a in actions):

                continue
            else:
                new_lines.append(line)
        
        result = '\n'.join(new_lines)
        print(f"\n[FIM] ========== Generated New Sequence ==========\n{result}\n[FIM] ========================================\n")
        return result
    
    def generate_alternative_solution(self, lines, instruction, all_objects):
        """Attempt to solve the task with a completely different approach."""
        print(f"[FIM] Attempting completely different approach")
        
        import random
        

        all_skills = ['pick', 'place', 'move', 'push', 'open_gripper', 'close_gripper']
        

        inst_lower = instruction.lower() if instruction else ""
        

        priority_skills = []
        if 'pick' in inst_lower:
            priority_skills.append('pick')
        if 'place' in inst_lower or 'put' in inst_lower:
            priority_skills.append('place')
        if 'push' in inst_lower or 'slide' in inst_lower:
            priority_skills.append('push')
        if 'move' in inst_lower:
            priority_skills.append('move')
        

        if not priority_skills:
            priority_skills = ['pick', 'place']
        

        if len(priority_skills) < 3:
            remaining_skills = [s for s in all_skills if s not in priority_skills and not s.endswith('_gripper')]
            if remaining_skills:
                priority_skills.append(random.choice(remaining_skills))
        
        print(f"[FIM] Alternative skill combination: {priority_skills}")
        

        new_lines = []
        action_inserted = False
        
        for line in lines:
            if 'run_action(' in line and not action_inserted:

                indent = len(line) - len(line.lstrip())
                

                objects = list(all_objects) if all_objects else ['target']
                
                for skill in priority_skills:
                    if skill in ['open_gripper', 'close_gripper']:
                        new_lines.append(f"{' ' * indent}obs, reward, done = run_action({skill})")
                    else:

                        obj = random.choice(objects) if objects else 'target'
                        new_lines.append(f"{' ' * indent}obs, reward, done = run_action({skill}, '{obj}')")
                    new_lines.append(f"{' ' * indent}if done:")
                    new_lines.append(f"{' ' * (indent+4)}return")
                
                action_inserted = True
                continue
            elif 'run_action(' in line:

                continue
            else:
                new_lines.append(line)
        
        result = '\n'.join(new_lines)
        print(f"\n[FIM] ========== Generated Alternative Solution ==========\n{result}\n[FIM] ========================================\n")
        return result
    
    def is_semantic_feedback(self, feedback: str) -> bool:
        """Check whether the feedback indicates a semantic error."""
        semantic_keywords = [
            "wrong object", "incorrect object", "should use",
            "target first", "pick before", "place after",
            "wrong skill", "incorrect action", "need to",
            "instead of", "should be", "must be",
            "task has not been done", "task not completed",
            "task failed", "check if", "action sequence"
        ]
        feedback_lower = feedback.lower()
        return any(keyword in feedback_lower for keyword in semantic_keywords)
    
    def fix_semantic_error(self, code: str, feedback: str, instruction: str = None,
                           gripper_position: str = None, available_objects: list = None,
                           past_key_values=None) -> str:
        """Adjust code based on semantic feedback.

        Args:
            past_key_values: optional precomputed KV cache
        """
        print(f"\n[FIM] ========== Semantic Fix ==========" )
        print(f"[FIM] Feedback: {feedback[:150]}..." if len(feedback) > 150 else f"[FIM] Feedback: {feedback}")
        if available_objects:
            print(f"[FIM] Available objects for semantic fix: {available_objects}")
        
        lines = code.split('\n')
        fixed_lines = []
        

        feedback_lower = feedback.lower()
        

        if "task has not been done" in feedback_lower or "task not completed" in feedback_lower:
            print(f"[FIM] Detected 'task not done' feedback - analyzing action sequence")
            return self.fix_task_not_done(code, instruction, gripper_position, available_objects)
        

        if ("check if" in feedback_lower or "task execution failed" in feedback_lower or 
            "task failed" in feedback_lower or ("using" in feedback_lower and "correctly" in feedback_lower)):
            print(f"[FIM] Detected generic/check feedback - trying alternative approach")
            return self.fix_task_not_done(code, instruction, gripper_position, available_objects)
        
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
            

            if 'run_action(' in line:

                import re
                match = re.search(r'run_action\((\w+),\s*[\'"]([^\'\"]+)[\'"]', line)
                if match:
                    current_skill = match.group(1)
                    current_object = match.group(2)
                    

                    new_skill = current_skill
                    new_object = current_object
                    

                    if "wrong object" in feedback_lower or "should use" in feedback_lower:

                        obj_match = re.search(r"should use ['\"]?(\w+)['\"]?", feedback_lower)
                        if obj_match:
                            new_object = obj_match.group(1)
                        elif "target" in feedback_lower and current_object != "target":
                            new_object = "target"
                        elif "block" in feedback_lower and "target" in current_object:
                            new_object = "block"
                    

                    if "wrong skill" in feedback_lower or "should be" in feedback_lower:
                        if "push" in feedback_lower and current_skill != "push":
                            new_skill = "push"
                        elif "pick" in feedback_lower and current_skill != "pick":
                            new_skill = "pick"
                        elif "place" in feedback_lower and current_skill != "place":
                            new_skill = "place"
                        elif "move" in feedback_lower and current_skill != "move":
                            new_skill = "move"
                    

                    if new_skill != current_skill or new_object != current_object:
                        indent = len(line) - len(line.lstrip())
                        fixed_line = f"{' ' * indent}obs, reward, done = run_action({new_skill}, '{new_object}')"
                        fixed_lines.append(fixed_line)
                        print(f"[FIM] ✓ Modified action:")
                        print(f"[FIM]   Original: {line.strip()}")
                        print(f"[FIM]   Fixed:    {fixed_line.strip()}")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        fixed_code = '\n'.join(fixed_lines)
        
        if fixed_code != code:
            print(f"[FIM] ✓ Semantic fix applied successfully")
            print(f"\n[FIM] ========== Complete Fixed Code ==========\n{fixed_code}\n[FIM] ========================================\n")
        else:
            print(f"[FIM] ⚠ No semantic changes made")
            print(f"[FIM] ========================================\n")
        
        return fixed_code
    
    def fix_multiple_errors(self, code: str, feedback: str, instruction: str = None,
                           max_iterations: int = 3) -> str:
        """Fix multiple errors sequentially."""
        print(f"\n[FIM] ========== Multiple Error Fix ==========" )
        print(f"[FIM] Max iterations: {max_iterations}")
        
        fixed_code = code
        
        for i in range(max_iterations):
            print(f"\n[FIM] Iteration {i+1}/{max_iterations}")

            new_code = self.fix_code_with_fim(fixed_code, feedback, instruction)
            
            if new_code == fixed_code:

                print(f"[FIM] No more changes possible, stopping at iteration {i+1}")
                break
                
            fixed_code = new_code
            print(f"[FIM] ✓ Code modified in iteration {i+1}")
            



        
        print(f"\n[FIM] ========== Final Code After {i+1} Iterations ==========\n{fixed_code}")
        print(f"[FIM] ========================================\n")
        return fixed_code



def apply_fim_fix(model, tokenizer, code: str, feedback: str, 
                  instruction: str = None, gripper_position: str = None, 
                  available_objects: list = None) -> str:
    """Apply FIM-based code edits."""
    print(f"\n{'='*50}")
    print(f"[FIM] STARTING FIM CODE MODIFICATION")
    print(f"{'='*50}\n")
    
    modifier = FIMCodeModifier(model, tokenizer)
    result = modifier.fix_code_with_fim(code, feedback, instruction, gripper_position, available_objects)
    
    print(f"\n{'='*50}")
    print(f"[FIM] FIM CODE MODIFICATION COMPLETE")
    print(f"{'='*50}\n")
    
    return result
