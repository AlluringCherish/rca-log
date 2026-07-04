"""
Function Cache Manager for Robot Tasks
Manages caching of individual functions from robot_tasks.py
"""

import os
import torch
import inspect
import hashlib
import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import ast
from transformers.cache_utils import DynamicCache
import logging

# Add safe globals for torch serialization
try:
    from transformers.cache_utils import DynamicCache
    torch.serialization.add_safe_globals([DynamicCache])
    # Also try to add DynamicLayer if it exists
    try:
        from transformers.cache_utils import DynamicLayer
        torch.serialization.add_safe_globals([DynamicLayer])
    except ImportError:
        pass
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class FunctionCache:
    """Cache for individual function"""
    function_name: str
    summary: str  # Function interface and docstring
    code: str
    parameters: Dict[str, Any]  # Parameter names and default values
    code_kv_cache: Optional[DynamicCache] = None
    summary_kv_cache: Optional[DynamicCache] = None

    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'function_name': self.function_name,
            'summary': self.summary,
            'code': self.code,
            'parameters': self.parameters
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            function_name=data['function_name'],
            summary=data['summary'],
            code=data['code'],
            parameters=data['parameters']
        )

class FunctionCacheManager:
    """Manages caching of robot task functions"""

    def __init__(self, cache_dir: str = "./function_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.functions_cache: Dict[str, FunctionCache] = {}
        self.load_all_caches()

    def extract_function_info(self, func) -> Tuple[str, Dict[str, Any], str]:
        """Extract function signature, parameters, and docstring"""
        sig = inspect.signature(func)
        params = {}

        # Extract parameters and their default values
        for name, param in sig.parameters.items():
            if name in ['ps', 'get_obj_func']:  # Skip these system parameters
                continue
            if param.default != inspect.Parameter.empty:
                params[name] = param.default
            else:
                params[name] = None

        # Get function source code
        source = inspect.getsource(func)

        # Create summary with interface and docstring
        docstring = inspect.getdoc(func) or "No description available"
        param_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        summary = f"""Function: {func.__name__}({param_str})
Description: {docstring}
Parameters:
"""
        for name, default in params.items():
            summary += f"  - {name}: {type(default).__name__ if default is not None else 'Any'} = {repr(default)}\n"

        return summary, params, source

    def cache_function(self, func_name: str, func) -> FunctionCache:
        """Cache a single function"""
        summary, params, code = self.extract_function_info(func)

        func_cache = FunctionCache(
            function_name=func_name,
            summary=summary,
            code=code,
            parameters=params
        )

        # Save to disk
        self._save_function_cache(func_cache)

        # Store in memory
        self.functions_cache[func_name] = func_cache

        return func_cache

    def cache_all_functions(self):
        """Cache all functions from robot_tasks.py"""
        import sys
        import importlib

        # Import robot_tasks module
        sys.path.append('./oracle_func_ver')
        import robot_tasks

        # Get all task functions
        task_functions = [
            name for name in dir(robot_tasks)
            if name.endswith('_task') and callable(getattr(robot_tasks, name))
        ]

        logger.info(f"Found {len(task_functions)} task functions to cache")

        for func_name in task_functions:
            func = getattr(robot_tasks, func_name)
            cache = self.cache_function(func_name, func)
            logger.info(f"Cached function: {func_name}")

        return self.functions_cache

    def _save_function_cache(self, func_cache: FunctionCache):
        """Save function cache to disk"""
        # Save summary
        summary_path = os.path.join(self.cache_dir, f"{func_cache.function_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(func_cache.summary)

        # Save code
        code_path = os.path.join(self.cache_dir, f"{func_cache.function_name}_code.py")
        with open(code_path, 'w') as f:
            f.write(func_cache.code)

        # Save metadata
        meta_path = os.path.join(self.cache_dir, f"{func_cache.function_name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'function_name': func_cache.function_name,
                'parameters': func_cache.parameters
            }, f, indent=2)

        # Save KV caches if they exist
        if func_cache.code_kv_cache is not None:
            kv_path = os.path.join(self.cache_dir, f"{func_cache.function_name}_code_kv.pt")
            torch.save(func_cache.code_kv_cache, kv_path)

        if func_cache.summary_kv_cache is not None:
            kv_path = os.path.join(self.cache_dir, f"{func_cache.function_name}_summary_kv.pt")
            torch.save(func_cache.summary_kv_cache, kv_path)

    def load_function_cache(self, func_name: str) -> Optional[FunctionCache]:
        """Load a specific function cache from disk"""
        summary_path = os.path.join(self.cache_dir, f"{func_name}_summary.txt")
        code_path = os.path.join(self.cache_dir, f"{func_name}_code.py")
        meta_path = os.path.join(self.cache_dir, f"{func_name}_meta.json")

        if not all(os.path.exists(p) for p in [summary_path, code_path, meta_path]):
            return None

        # Load summary
        with open(summary_path, 'r') as f:
            summary = f.read()

        # Load code
        with open(code_path, 'r') as f:
            code = f.read()

        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        func_cache = FunctionCache(
            function_name=meta['function_name'],
            summary=summary,
            code=code,
            parameters=meta['parameters']
        )

        # Load KV caches if they exist
        code_kv_path = os.path.join(self.cache_dir, f"{func_name}_code_kv.pt")
        if os.path.exists(code_kv_path):
            try:
                func_cache.code_kv_cache = torch.load(code_kv_path, weights_only=False)
            except Exception as e:
                logger.warning(f"Could not load code KV cache for {func_name}: {e}")
                func_cache.code_kv_cache = None

        summary_kv_path = os.path.join(self.cache_dir, f"{func_name}_summary_kv.pt")
        if os.path.exists(summary_kv_path):
            try:
                func_cache.summary_kv_cache = torch.load(summary_kv_path, weights_only=False)
            except Exception as e:
                logger.warning(f"Could not load summary KV cache for {func_name}: {e}")
                func_cache.summary_kv_cache = None

        return func_cache

    def load_all_caches(self):
        """Load all function caches from disk"""
        if not os.path.exists(self.cache_dir):
            return

        # Find all cached functions
        meta_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_meta.json')]

        for meta_file in meta_files:
            func_name = meta_file.replace('_meta.json', '')
            func_cache = self.load_function_cache(func_name)
            if func_cache:
                self.functions_cache[func_name] = func_cache

        logger.info(f"Loaded {len(self.functions_cache)} function caches")

    def get_function_cache(self, func_name: str) -> Optional[FunctionCache]:
        """Get a specific function cache"""
        return self.functions_cache.get(func_name)

    def get_all_summaries(self) -> Dict[str, str]:
        """Get all function summaries for retrieval"""
        return {
            name: cache.summary
            for name, cache in self.functions_cache.items()
        }

    def get_relevant_functions_by_perplexity(self, task_description: str, model, tokenizer, top_k: int = 3) -> list:
        """Get most relevant functions using perplexity scoring"""
        import torch

        results = []
        test_prompt = f"Robot task: {task_description}\nCode:\n"

        for func_name, cache in self.functions_cache.items():
            # Generate function call for this task
            func_call = self.generate_function_call(func_name, task_description)
            func_call_indented = ' ' * 8 + func_call

            # Compute perplexity
            full_text = test_prompt + func_call_indented
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            prompt_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

            prompt_len = prompt_inputs['input_ids'].shape[-1]

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Calculate perplexity only for the code part
                shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., prompt_len:].contiguous()

                # Compute cross entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                perplexity = torch.exp(loss).item()

            results.append((func_name, cache, perplexity))

        # Sort by perplexity (lower is better)
        results.sort(key=lambda x: x[2])

        # Return top_k with lowest perplexity
        return [(name, cache) for name, cache, _ in results[:top_k]]

    def get_relevant_functions(self, task_description: str, top_k: int = 3) -> list:
        """Get most relevant functions for a task with improved matching"""
        relevant = []
        task_lower = task_description.lower()

        # Extract key action words and objects
        action_words = ['put', 'place', 'pick', 'lift', 'pour', 'push', 'pull', 'open', 'close',
                       'insert', 'stack', 'arrange', 'sort', 'clear', 'collect', 'throw']

        task_action = None
        for action in action_words:
            if action in task_lower:
                task_action = action
                break

        for name, cache in self.functions_cache.items():
            score = 0
            func_name_lower = name.lower()

            # Exact action match gets highest score
            if task_action and task_action in func_name_lower:
                score += 10

            # Pattern matching for similar tasks
            # Special handling for "throw" -> "put" mapping
            if 'throw' in task_lower and 'put' in func_name_lower and 'bin' in func_name_lower:
                score += 10
            # "place" is similar to "put"
            if 'place' in task_lower and 'put' in func_name_lower:
                score += 9
            if 'put' in task_lower and 'put' in func_name_lower:
                score += 10
            if 'pick' in task_lower and ('pick' in func_name_lower or 'lift' in func_name_lower):
                score += 8
            if 'sort' in task_lower and 'sort' in func_name_lower:
                score += 8
            if 'arrange' in task_lower and ('arrange' in func_name_lower or 'grid' in func_name_lower or 'circular' in func_name_lower):
                score += 7
            if 'stack' in task_lower and ('stack' in func_name_lower or 'pyramid' in func_name_lower):
                score += 7

            # Check for object similarity - improved matching
            # Match "trash", "rubbish", "garbage", "waste" interchangeably
            if any(word in task_lower for word in ['trash', 'rubbish', 'garbage', 'waste']):
                if 'rubbish' in func_name_lower or 'trash' in func_name_lower:
                    score += 8

            # Check for bin/basket/container
            if any(word in task_lower for word in ['bin', 'basket', 'container']):
                if 'bin' in func_name_lower:
                    score += 5

            # Cup matching
            if 'cup' in task_lower and 'cup' in func_name_lower:
                score += 5
            # Box matching
            if 'box' in task_lower and 'box' in func_name_lower:
                score += 5
            # Drawer matching
            if 'drawer' in task_lower and 'drawer' in func_name_lower:
                score += 5

            # General keyword matching
            task_words = task_lower.split()
            for word in task_words:
                if len(word) > 3 and word in func_name_lower:
                    score += 2
                if len(word) > 3 and word in cache.summary.lower():
                    score += 1

            if score > 0:
                relevant.append((name, cache, score))

        # Sort by score and return top_k
        relevant.sort(key=lambda x: x[2], reverse=True)
        return [(name, cache) for name, cache, _ in relevant[:top_k]]

    def generate_function_call(self, func_name: str, task_description: str = None, **kwargs) -> str:
        """Generate a function call with parameters extracted from task description"""
        cache = self.functions_cache.get(func_name)
        if not cache:
            return f"# Function {func_name} not found in cache"

        # Build parameter string
        param_str = "self.ps, self.get_obj"

        # Try to extract parameters from task description if provided
        if task_description and not kwargs:
            task_lower = task_description.lower()

            # Improved object extraction
            import re

            # Clean up the instruction
            clean_inst = task_description.replace(',', ' ').replace(' and ', ' ')

            # Extract all potential objects
            all_objects = re.findall(r'\b([a-zA-Z]+_[a-zA-Z0-9_]+|[a-zA-Z]+\d+)\b', clean_inst)

            # Separate items from targets based on 'in' or 'into'
            items = []
            targets = []

            if ' in ' in task_description or ' into ' in task_description:
                parts = re.split(r'\s+in\s+|\s+into\s+', task_description)
                if len(parts) >= 2:
                    # Items are before 'in/into'
                    item_part = parts[0]
                    target_part = parts[1]

                    items = re.findall(r'\b([a-zA-Z]+_[a-zA-Z0-9_]+|[a-zA-Z]+\d+)\b', item_part)
                    targets = re.findall(r'\b([a-zA-Z]+_[a-zA-Z0-9_]+|[a-zA-Z]+\d+|bin|sink|basket|container)\b', target_part)

            if not items and all_objects:
                # Fallback: classify based on keywords
                for obj in all_objects:
                    if any(keyword in obj.lower() for keyword in ['bin', 'basket', 'sink', 'container']):
                        targets.append(obj)
                    else:
                        items.append(obj)

            # Map common task patterns to function parameters
            if 'put' in func_name.lower() and 'bin' in func_name.lower():
                # For put_rubbish_in_bin_task
                if items:
                    rubbish_list = [f"'{item}'" for item in items]
                    kwargs['rubbish_list'] = f"[{', '.join(rubbish_list)}]"

                if targets:
                    # Use first target as bin
                    kwargs['bin_obj'] = f"'{targets[0]}'"
                else:
                    # Default to 'bin' if no target specified
                    kwargs['bin_obj'] = "'bin'"

        # Add kwargs to parameter string
        for param_name, default_value in cache.parameters.items():
            if param_name in kwargs:
                value = kwargs[param_name]
                param_str += f", {param_name}={value}"

        return f"{func_name}({param_str})"

    def save_kv_cache(self, func_name: str, code_kv: Optional[DynamicCache] = None,
                     summary_kv: Optional[DynamicCache] = None):
        """Save KV cache for a function"""
        if func_name not in self.functions_cache:
            logger.warning(f"Function {func_name} not in cache")
            return

        cache = self.functions_cache[func_name]

        if code_kv is not None:
            cache.code_kv_cache = code_kv
            kv_path = os.path.join(self.cache_dir, f"{func_name}_code_kv.pt")
            torch.save(code_kv, kv_path)

        if summary_kv is not None:
            cache.summary_kv_cache = summary_kv
            kv_path = os.path.join(self.cache_dir, f"{func_name}_summary_kv.pt")
            torch.save(summary_kv, kv_path)