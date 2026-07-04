 #!/usr/bin/env python3
"""
CATP Prompt-Only Version: No post-processing, pure prompt-based generation
"""

import torch
import argparse
import os
import json
import re
import numpy as np
from time import time, sleep
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Import the FIM modifier for feedback fixes only
from catp_fim import FIMCodeModifier

import cag.dataset as cagds
import cag.similarity as cagsim
from api_requests import init_task, run_task, shutdown_task, SSHClient
import config_remote
from transfer_clean_code import remove_comments_and_docstrings

# Configure logging
import sys
log_format = "%(asctime)s - %(levelname)s - %(message)s"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/catp_prompt_only.log', mode='a')
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


@dataclass
class PolicyBlock:
    """Policy block structure"""
    block_id: str
    task_name: str
    code: str
    summary: str
    code_kv_cache: Optional[DynamicCache] = None
    summary_kv_cache: Optional[DynamicCache] = None
    success_rate: float = 0.0
    usage_count: int = 0
    last_access_time: float = 0.0  # For temporal locality
    is_offloaded: bool = False  # Track if cache is offloaded to CPU
    
    # Locality components
    temporal_score: float = 1.0  # Recency of use [0, 1]
    spatial_score: float = 1.0   # Proximity in call trace [0, 1]
    semantic_score: float = 1.0  # Functional diversity via perplexity [0, 1]
    composite_locality: float = 1.0  # Weighted composite score
    
    # Metadata for scoring
    last_perplexity: float = 0.0  # Last measured perplexity
    call_distance: int = 0  # Distance from current execution


class SimplePolicyManager:
    """Simple policy KV Cache manager without post-processing"""
    
    def __init__(self, model, tokenizer, cache_dir="./catp_cache", clear_cache=True):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.policy_blocks: Dict[str, PolicyBlock] = {}
        self.task_hierarchy = defaultdict(list)
        self.skill_interface_cache = None
        self.skill_interface_text = ""
        self.helper_code_cache = None  # Cache for run_action helper
        self.helper_code_text = ""
        
        # Memory management parameters
        self.gpu_memory_threshold = 0.85  # Start offloading when GPU memory exceeds 85%
        self.reserved_gpu_memory_gb = 2.0  # Reserve 2GB as buffer
        self.offloaded_caches = {}  # Track CPU-offloaded caches
        
        # Locality scoring weights (from paper)
        self.w_temporal = 0.4
        self.w_spatial = 0.3
        self.w_semantic = 0.3
        
        # Execution trace for spatial locality
        self.execution_trace = []  # List of recently executed block_ids (ordered by recency)
        self.max_trace_length = 20
        
        # Cache eviction history for better decision making
        self.eviction_history = []  # Track which blocks were evicted
        
        # Cache hit tracking
        self.cache_hits = 0  # Number of times cached blocks were used
        self.cache_requests = 0  # Total number of cache requests
        self.task_cache_hits = defaultdict(int)  # Track hits per task
        self.task_cache_requests = defaultdict(int)  # Track requests per task
        
        # Effective hit tracking (blocks that pass perplexity threshold)
        self.effective_hits = 0  # Blocks actually used after perplexity filtering
        self.total_blocks_evaluated = 0  # Total blocks evaluated with perplexity
        self.blocks_passed_threshold = 0  # Blocks that passed perplexity threshold
        self.direct_hits = 0  # Exact task name matches
        self.partial_hits = 0  # Similar task matches
        self.cross_task_transfers = 0  # Blocks from completely different tasks
        
        # Clear existing cache if requested
        if clear_cache and os.path.exists(cache_dir):
            import shutil
            logger.info(f"Clearing existing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Don't load existing caches if we cleared them
        if not clear_cache:
            self.load_existing_caches()
        
        self.create_helper_code_cache()  # Create helper code cache on init
        
    def check_gpu_memory(self) -> Tuple[float, float]:
        """Check current GPU memory usage
        Returns: (used_gb, total_gb)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return allocated, total
        return 0.0, 0.0
    
    # DEPRECATED - Replaced by offload_by_locality
    # def offload_least_recent_caches(self, num_blocks: int = 1):
    #     """This method is deprecated. Use offload_by_locality() instead."""
    #     logger.warning("[DEPRECATED] offload_least_recent_caches is deprecated. Use offload_by_locality()")
    #     self.offload_by_locality(num_blocks)
    
    def _move_cache_to_cpu(self, cache: Optional[DynamicCache]) -> Optional[DynamicCache]:
        """Move a DynamicCache to CPU memory"""
        if cache is None:
            return None
        
        cpu_cache = DynamicCache()
        for i in range(len(cache.key_cache)):
            cpu_cache.key_cache.append(cache.key_cache[i].cpu())
            cpu_cache.value_cache.append(cache.value_cache[i].cpu())
        
        if hasattr(cache, '_seen_tokens'):
            cpu_cache._seen_tokens = cache._seen_tokens
        if hasattr(cache, 'cache_position'):
            cpu_cache.cache_position = cache.cache_position.cpu() if cache.cache_position is not None else None
        
        return cpu_cache
    
    def restore_cache_from_cpu(self, block_id: str):
        """Restore a cache from CPU to GPU when needed
        
        Before restoring, checks if GPU memory is sufficient.
        If not, offloads other low-locality blocks first.
        """
        block = self.policy_blocks.get(block_id)
        if not block or not block.is_offloaded:
            return
        
        if block_id in self.offloaded_caches:
            # Check GPU memory before restoration
            used_gb, total_gb = self.check_gpu_memory()
            
            # Estimate size of cache to restore (rough estimate: 1GB per block)
            estimated_size_gb = 1.0
            
            # If not enough space, offload other blocks first
            if (used_gb + estimated_size_gb) > (total_gb * self.gpu_memory_threshold):
                logger.info(f"[MEMORY] Need to offload blocks before restoring {block.task_name}")
                # Offload 1-2 blocks with lowest locality (excluding the one we want to restore)
                eligible = [
                    (bid, b.composite_locality) 
                    for bid, b in self.policy_blocks.items()
                    if bid != block_id and not b.is_offloaded and b.code_kv_cache is not None
                ]
                eligible.sort(key=lambda x: x[1])
                
                if eligible:
                    to_offload = [eligible[0][0]]
                    if len(eligible) > 1 and eligible[1][1] < 0.3:  # Also offload second if very low locality
                        to_offload.append(eligible[1][0])
                    
                    for bid in to_offload:
                        self.offload_by_locality(1)
                        break  # offload_by_locality will handle the selection
            
            cpu_caches = self.offloaded_caches[block_id]
            embed_device = self.model.model.embed_tokens.weight.device
            
            # Restore to GPU
            if cpu_caches['code']:
                block.code_kv_cache = self._move_cache_to_gpu(cpu_caches['code'], embed_device)
            if cpu_caches['summary']:
                block.summary_kv_cache = self._move_cache_to_gpu(cpu_caches['summary'], embed_device)
            
            block.is_offloaded = False
            
            # Track restoration time
            restore_time = time()
            offload_duration = restore_time - cpu_caches.get('offload_time', 0)
            
            del self.offloaded_caches[block_id]
            
            logger.info(f"[MEMORY] Restored {block.task_name} to GPU (was offloaded for {offload_duration:.1f}s)")
    
    def _move_cache_to_gpu(self, cache: Optional[DynamicCache], device) -> Optional[DynamicCache]:
        """Move a DynamicCache from CPU to GPU"""
        if cache is None:
            return None
        
        gpu_cache = DynamicCache()
        for i in range(len(cache.key_cache)):
            gpu_cache.key_cache.append(cache.key_cache[i].to(device))
            gpu_cache.value_cache.append(cache.value_cache[i].to(device))
        
        if hasattr(cache, '_seen_tokens'):
            gpu_cache._seen_tokens = cache._seen_tokens
        if hasattr(cache, 'cache_position'):
            gpu_cache.cache_position = cache.cache_position.to(device) if cache.cache_position is not None else None
        
        return gpu_cache
    
    def compute_temporal_locality(self, block: PolicyBlock) -> float:
        """Compute temporal locality score based on recency"""
        if block.last_access_time == 0:
            return 0.0
        
        current_time = time()
        time_diff = current_time - block.last_access_time
        
        # Exponential decay: more recent = higher score
        # Score approaches 0 as time_diff increases
        decay_rate = 0.01  # Adjust based on typical usage patterns
        score = np.exp(-decay_rate * time_diff)
        
        return min(1.0, max(0.0, score))
    
    def compute_spatial_locality(self, block_id: str) -> float:
        """Compute spatial locality based on call trace proximity"""
        if block_id not in self.execution_trace:
            return 0.0
        
        # Find most recent occurrence in trace
        try:
            distance = self.execution_trace[::-1].index(block_id)
            # Normalize: closer in trace = higher score
            score = 1.0 - (distance / self.max_trace_length)
            return min(1.0, max(0.0, score))
        except ValueError:
            return 0.0
    
    def compute_semantic_locality(self, block: PolicyBlock) -> float:
        """Compute semantic locality based on perplexity (functional diversity)"""
        if block.last_perplexity == 0:
            return 0.5  # Default middle score if no perplexity
        
        # Lower perplexity = better match = higher locality
        # Normalize perplexity to [0, 1] range
        # Assuming perplexity typically ranges from 1 to 100
        normalized = 1.0 - min(1.0, block.last_perplexity / 100.0)
        
        return min(1.0, max(0.0, normalized))
    
    def update_composite_locality(self, block_id: str):
        """Update composite locality score for a block"""
        block = self.policy_blocks.get(block_id)
        if not block:
            return
        
        # Compute individual locality components
        block.temporal_score = self.compute_temporal_locality(block)
        block.spatial_score = self.compute_spatial_locality(block_id)
        block.semantic_score = self.compute_semantic_locality(block)
        
        # Compute weighted composite score
        block.composite_locality = (
            self.w_temporal * block.temporal_score +
            self.w_spatial * block.spatial_score +
            self.w_semantic * block.semantic_score
        )
        
        logger.debug(f"[LOCALITY] {block.task_name}: temporal={block.temporal_score:.2f}, "
                    f"spatial={block.spatial_score:.2f}, semantic={block.semantic_score:.2f}, "
                    f"composite={block.composite_locality:.2f}")
    
    def update_all_locality_scores(self):
        """Update locality scores for all blocks"""
        for block_id in self.policy_blocks:
            self.update_composite_locality(block_id)
    
    def select_blocks_for_offloading(self, num_blocks: int) -> List[str]:
        """Select blocks with lowest composite locality for offloading"""
        # Update all scores first
        self.update_all_locality_scores()
        
        # Sort by composite locality (lowest first)
        eligible_blocks = [
            (block_id, block.composite_locality)
            for block_id, block in self.policy_blocks.items()
            if not block.is_offloaded and block.code_kv_cache is not None
        ]
        
        eligible_blocks.sort(key=lambda x: x[1])
        
        # Return block_ids with lowest scores
        return [block_id for block_id, _ in eligible_blocks[:num_blocks]]
    
    def offload_to_cpu(self, block_id: str):
        """Offload a specific block to CPU memory"""
        block = self.policy_blocks.get(block_id)
        if not block or block.is_offloaded:
            return
        
        # Move cache to CPU
        cpu_code_cache = self._move_cache_to_cpu(block.code_kv_cache)
        cpu_summary_cache = self._move_cache_to_cpu(block.summary_kv_cache) if block.summary_kv_cache else None
        
        # Store CPU caches
        self.offloaded_caches[block_id] = {
            'code': cpu_code_cache,
            'summary': cpu_summary_cache,
            'offload_time': time(),
            'locality_at_offload': block.composite_locality
        }
        
        # Clear GPU caches
        block.code_kv_cache = None
        block.summary_kv_cache = None
        block.is_offloaded = True
        
        logger.debug(f"[MEMORY] Offloaded {block.task_name} to CPU")
    
    def offload_by_locality(self, num_blocks: int = 1, max_offload: int = None):
        """Offload blocks based on composite locality score
        
        This implements the cache replacement policy from the paper:
        - Blocks with lowest ℓ(f_k) scores are evicted first
        - Maintains functional diversity while managing memory
        
        Args:
            num_blocks: Target number of blocks to offload
            max_offload: Maximum number of blocks to offload (limits num_blocks)
        """
        if max_offload is not None:
            num_blocks = min(num_blocks, max_offload)
        
        blocks_to_offload = self.select_blocks_for_offloading(num_blocks)
        
        offloaded_info = []
        for block_id in blocks_to_offload:
            block = self.policy_blocks[block_id]
            
            # Record eviction for analysis
            eviction_record = {
                'block_id': block_id,
                'task_name': block.task_name,
                'locality_score': block.composite_locality,
                'temporal': block.temporal_score,
                'spatial': block.spatial_score, 
                'semantic': block.semantic_score,
                'time': time()
            }
            self.eviction_history.append(eviction_record)
            
            # Move cache to CPU
            cpu_code_cache = self._move_cache_to_cpu(block.code_kv_cache)
            cpu_summary_cache = self._move_cache_to_cpu(block.summary_kv_cache) if block.summary_kv_cache else None
            
            # Store CPU caches
            self.offloaded_caches[block_id] = {
                'code': cpu_code_cache,
                'summary': cpu_summary_cache,
                'offload_time': time(),
                'locality_at_offload': block.composite_locality
            }
            
            # Clear GPU caches
            block.code_kv_cache = None
            block.summary_kv_cache = None
            block.is_offloaded = True
            
            offloaded_info.append(f"{block.task_name}(ℓ={block.composite_locality:.3f})")
            
        if blocks_to_offload:
            torch.cuda.empty_cache()
            used_gb, total_gb = self.check_gpu_memory()
            logger.info(f"[MEMORY] Offloaded {len(blocks_to_offload)} blocks: {', '.join(offloaded_info)}")
            logger.info(f"[MEMORY] GPU memory after: {used_gb:.2f}/{total_gb:.2f} GB ({used_gb/total_gb:.1%})")
    
    def manage_memory_before_generation(self, min_free_gb: float = 1.5):
        """Check and manage GPU memory before generation
        
        Args:
            min_free_gb: Minimum free GPU memory to maintain (default 1.5GB)
        """
        # Clear cache first for accurate reading
        torch.cuda.empty_cache()
        
        used_gb, total_gb = self.check_gpu_memory()
        free_gb = total_gb - used_gb
        usage_ratio = used_gb / total_gb if total_gb > 0 else 0
        
        logger.info(f"[MEMORY] GPU before generation: {used_gb:.2f}/{total_gb:.2f} GB used, {free_gb:.2f} GB free ({usage_ratio:.1%})")
        
        # More aggressive offloading when memory is low
        if free_gb < min_free_gb:
            logger.warning(f"[MEMORY] Low GPU memory! Need {min_free_gb}GB free, have {free_gb:.2f}GB")
            
            # Get all GPU-resident blocks
            gpu_blocks = [b for b in self.policy_blocks.values() 
                         if not b.is_offloaded and (b.code_kv_cache is not None or b.summary_kv_cache is not None)]
            
            if gpu_blocks:
                # Estimate memory per block
                avg_block_gb = used_gb / max(1, len(gpu_blocks))
                blocks_to_free = max(1, int((min_free_gb - free_gb + 0.5) / avg_block_gb))  # +0.5GB buffer
                blocks_to_free = min(blocks_to_free, len(gpu_blocks) - 1)  # Keep at least 1
                
                logger.info(f"[MEMORY] Offloading {blocks_to_free}/{len(gpu_blocks)} blocks (est. {blocks_to_free * avg_block_gb:.2f}GB)")
                self.offload_by_locality(blocks_to_free)
                
                # Clear cache and check again
                torch.cuda.empty_cache()
                used_after, _ = self.check_gpu_memory()
                free_after = total_gb - used_after
                logger.info(f"[MEMORY] After offload: {used_after:.2f}GB used, {free_after:.2f}GB free")
        elif usage_ratio > self.gpu_memory_threshold:
            logger.warning(f"[MEMORY] High usage ratio {usage_ratio:.1%}, offloading some blocks")
            num_to_offload = max(1, len(self.policy_blocks) // 5)
            self.offload_by_locality(num_to_offload)
    
    def compute_kv_cache(self, text: str, cache_key: Optional[str] = None) -> Optional[DynamicCache]:
        """Compute (or load) KV cache for provided text"""
        cache_path = None
        if cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                try:
                    logger.info(f"Loading existing KV cache from {cache_path}")
                    return torch.load(cache_path, map_location='cpu', weights_only=False)
                except Exception as exc:
                    logger.warning(f"Failed to load KV cache {cache_path}: {exc}")

        embed_device = self.model.model.embed_tokens.weight.device
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(embed_device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )

        cache = outputs.past_key_values
        if cache and getattr(cache, 'key_cache', None):
            seq_len = cache.key_cache[0].shape[2]
            device = cache.key_cache[0].device
            if not hasattr(cache, 'cache_position') or cache.cache_position is None:
                cache.cache_position = torch.arange(0, seq_len, dtype=torch.long, device=device)
            cache._seen_tokens = seq_len

        if cache_key and cache:
            try:
                torch.save(cache, cache_path)
                logger.info(f"Saved KV cache to {cache_path}")
            except Exception as exc:
                logger.error(f"Failed to save KV cache {cache_path}: {exc}")

        return cache
    
    def create_policy_summary(self, code: str, task_name: str) -> str:
        """Create detailed summary of policy code"""
        # Extract all run_action calls with their parameters
        action_pattern = re.findall(r'run_action\((\w+),\s*[\'"]([^\'\"]+)[\'"]', code)
        
        # Build sequence of actions
        action_sequence = []
        objects_used = set()
        skills_used = set()
        
        for skill, obj in action_pattern:
            action_sequence.append(f"{skill}('{obj}')")
            objects_used.add(obj)
            skills_used.add(skill)
        
        # Analyze task pattern
        task_type = "manipulation"
        if "push" in skills_used or "slide" in task_name.lower():
            task_type = "pushing/sliding"
        elif "pick" in skills_used and "place" in skills_used:
            task_type = "pick-and-place"
        elif "reach" in task_name.lower():
            task_type = "reaching"
        elif "lamp" in task_name.lower() or "button" in task_name.lower():
            task_type = "button/switch activation"
        
        # Build comprehensive summary
        summary = f"Task: {task_name}\n"
        summary += f"Type: {task_type}\n"
        summary += f"Objects: {', '.join(sorted(objects_used))}\n"
        summary += f"Skills used: {', '.join(sorted(skills_used))}\n"
        summary += f"Action sequence: {' -> '.join(action_sequence[:5])}"
        if len(action_sequence) > 5:
            summary += f" ... ({len(action_sequence)} total actions)"
        summary += "\n"
        
        # Add structural information
        if 'while' in code or 'for' in code:
            summary += "Structure: Contains loops\n"
        if 'if done:' in code:
            summary += "Has early termination checks\n"
        
        return summary
    
    def store_policy_block(self, task_name: str, code: str, skip_if_exists: bool = True, is_example: bool = False):
        """Store policy block and create KV cache with locality management"""
        if skip_if_exists and task_name in self.task_hierarchy:
            existing_blocks = self.task_hierarchy[task_name]
            if existing_blocks:
                logger.info(f"Task {task_name} already has cached blocks, skipping")
                return existing_blocks[0]

        # Check memory before adding new cache - need extra space
        self.manage_memory_before_generation(min_free_gb=2.0)

        block_id = f"{task_name}_{int(time())}"
        summary = self.create_policy_summary(code, task_name)
        original_code = code
        if not is_example:
            code = remove_comments_and_docstrings(code)

        block = PolicyBlock(
            block_id=block_id,
            task_name=task_name,
            code=code,
            summary=summary,
            last_access_time=time(),  # Set initial access time
            temporal_score=1.0,  # New block has high temporal locality
            spatial_score=1.0,   # New block is in current trace
            semantic_score=1.0,  # New block is highly relevant
            composite_locality=1.0  # Start with maximum locality
        )

        if is_example:
            full_code_context = f"""### Robot Control Task: {task_name}
### Cached policy implementation:
{original_code}
### End of example"""
            block.code_kv_cache = self.compute_kv_cache(full_code_context, cache_key=f"{block_id}_code")
        else:
            block.code_kv_cache = self.compute_kv_cache(code, cache_key=f"{block_id}_code")

        block.summary_kv_cache = self.compute_kv_cache(summary, cache_key=f"{block_id}_summary")

        self.policy_blocks[block_id] = block
        self.task_hierarchy[task_name].append(block_id)

        # Update execution trace for new block
        self.execution_trace.append(block_id)
        if len(self.execution_trace) > self.max_trace_length:
            self.execution_trace = self.execution_trace[-self.max_trace_length:]

        logger.info(f"Stored block {block_id} for task {task_name} with max locality")

        return block_id
    
    def create_helper_code_cache(self):
        """Create KV cache for helper code. For prompt-only we skip explicit helper caching."""
        self.helper_code_text = ""
        self.helper_code_cache = None
        logger.info("Skipping helper code cache for faster generation")

    def calculate_perplexity(self, query: str, context: str) -> float:
        """Calculate perplexity of a query given context"""
        try:
            full_text = f"""{context}

Task: {query}

Solution:"""
            device = self.model.model.embed_tokens.weight.device
            inputs = self.tokenizer(
                full_text,
                return_tensors='pt',
                max_length=1024,
                truncation=True,
                padding=True
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = torch.clamp(outputs.loss, max=10.0)
                perplexity = torch.exp(loss).item()

            return perplexity
        except Exception as exc:
            logger.error(f"Error calculating perplexity: {exc}")
            return float('inf')
    
    def load_skill_interfaces(self, skill_file_path: str):
        """Load skill interfaces and create (or load) their KV cache."""
        logger.info(f"Loading skill interfaces from {skill_file_path}")
        skill_interfaces = []

        if os.path.exists(skill_file_path):
            with open(skill_file_path, 'r') as f:
                for line in f:
                    line_dict = json.loads(line)
                    interface = line_dict['code_interface']
                    description = line_dict['description']
                    skill_interfaces.append(f"{interface}# {description}")

            self.skill_interface_text = "### AVAILABLE SKILLS" + "".join(skill_interfaces)
            self.skill_interface_cache = self.compute_kv_cache(
                self.skill_interface_text,
                cache_key="skill_interfaces"
            )
            logger.info("Skill interface cache ready")
        else:
            logger.warning("Skill interface file not found; skipping cache creation")
    
    def compute_instruction_perplexity(self, instruction: str, kv_cache: DynamicCache) -> float:
        """Compute perplexity of instruction given cached context"""
        embed_device = self.model.model.embed_tokens.weight.device
        
        if kv_cache and kv_cache.key_cache:
            if kv_cache.key_cache[0].device != embed_device:
                for i in range(len(kv_cache.key_cache)):
                    kv_cache.key_cache[i] = kv_cache.key_cache[i].to(embed_device)
                    kv_cache.value_cache[i] = kv_cache.value_cache[i].to(embed_device)
                if hasattr(kv_cache, 'cache_position') and kv_cache.cache_position is not None:
                    kv_cache.cache_position = kv_cache.cache_position.to(embed_device)
        
        input_ids = self.tokenizer.encode(instruction, return_tensors="pt").to(embed_device)
        
        try:
            with torch.no_grad():
                cached_tokens = kv_cache._seen_tokens if hasattr(kv_cache, '_seen_tokens') else kv_cache.key_cache[0].shape[2]
                labels = input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=kv_cache,
                    labels=labels,
                    return_dict=True,
                    use_cache=True
                )
                perplexity = torch.exp(outputs.loss).item()
            
            logger.info(f"Context: {cached_tokens} tokens, Perplexity: {perplexity:.2f}")
            
        except Exception as e:
            logger.error(f"Error computing perplexity: {e}")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=input_ids,
                    return_dict=True
                )
                perplexity = torch.exp(outputs.loss).item()
        
        return perplexity
    
    def select_relevant_blocks(self, instruction: str, task_name: str,
                              perplexity_threshold: float = 100.0) -> Dict[str, Dict]:
        """Select all GPU-resident cached blocks without perplexity filtering."""
        selected_blocks: Dict[str, Dict] = {}

        # Track cache request statistics
        self.cache_requests += 1
        self.task_cache_requests[task_name] += 1

        def _store_block(block_id: str, block: PolicyBlock, block_type: str, origin: str) -> None:
            if block_id in selected_blocks:
                return

            if block_type == 'full':
                code_kv = block.code_kv_cache
                summary_kv = block.summary_kv_cache
            else:
                code_kv = None
                summary_kv = block.summary_kv_cache

            if code_kv is None and summary_kv is None:
                return

            block.last_access_time = time()
            block.usage_count += 1
            block.last_perplexity = 0.0

            if block_id in self.execution_trace:
                self.execution_trace.remove(block_id)
            self.execution_trace.append(block_id)
            if len(self.execution_trace) > self.max_trace_length:
                self.execution_trace = self.execution_trace[-self.max_trace_length:]

            self.update_composite_locality(block_id)

            self.cache_hits += 1
            self.task_cache_hits[task_name] += 1
            if origin == 'direct':
                self.direct_hits += 1
            else:
                self.partial_hits += 1
                self.cross_task_transfers += 1

            selected_blocks[block_id] = {
                'type': block_type,
                'code_kv': code_kv,
                'summary_kv': summary_kv,
                'code': block.code
            }

        # Step 1: include direct matches already on GPU
        direct_gpu_blocks = []
        for block_id in self.task_hierarchy.get(task_name, []):
            block = self.policy_blocks[block_id]
            if block.is_offloaded:
                continue
            if block.code_kv_cache is None and block.summary_kv_cache is None:
                continue
            direct_gpu_blocks.append(block_id)

        if direct_gpu_blocks:
            logger.info(f"Using {len(direct_gpu_blocks)} direct GPU-resident caches for task {task_name}")
            for block_id in direct_gpu_blocks:
                block = self.policy_blocks[block_id]
                block_type = 'full' if block.code_kv_cache is not None else 'summary_only'
                _store_block(block_id, block, block_type, origin='direct')

        # Step 2: include every other GPU resident block
        additional_gpu_blocks = []
        for block_id, block in self.policy_blocks.items():
            if block_id in selected_blocks:
                continue
            if block.is_offloaded:
                continue
            if block.code_kv_cache is None and block.summary_kv_cache is None:
                continue
            additional_gpu_blocks.append(block_id)

        if additional_gpu_blocks:
            logger.info(f"Adding {len(additional_gpu_blocks)} additional GPU-resident caches")
            for block_id in additional_gpu_blocks:
                block = self.policy_blocks[block_id]
                block_type = 'full' if block.code_kv_cache is not None else 'summary_only'
                _store_block(block_id, block, block_type, origin='other')

        if not selected_blocks:
            logger.info("No GPU-resident caches available; generation will proceed without KV cache assistance")

        logger.info(f"Selected {len(selected_blocks)} GPU-resident blocks for generation")
        return selected_blocks
    
    def find_similar_tasks(self, task_name: str) -> List[str]:
        """Find similar tasks"""
        similar = []
        task_keywords = set(re.findall(r'[A-Z][a-z]+', task_name))
        
        for other_task in self.task_hierarchy.keys():
            if other_task == task_name:
                continue
            other_keywords = set(re.findall(r'[A-Z][a-z]+', other_task))
            if len(task_keywords & other_keywords) > 0:
                similar.append(other_task)
        
        return similar[:3]
    
    def merge_kv_caches(self, kv_caches: List[DynamicCache]) -> Optional[DynamicCache]:
        """Merge multiple KV caches hierarchically."""
        if not kv_caches:
            return None

        valid_caches = [cache for cache in kv_caches if cache is not None]
        if not valid_caches:
            return None

        if len(valid_caches) == 1:
            return valid_caches[0]

        logger.info(f"Merging {len(valid_caches)} KV caches hierarchically")
        merged = DynamicCache()
        base_cache = valid_caches[0]

        try:
            if hasattr(base_cache, 'key_cache') and hasattr(base_cache, 'value_cache'):
                num_layers = len(base_cache.key_cache)

                for layer_idx in range(num_layers):
                    keys_to_concat = []
                    values_to_concat = []

                    for cache in valid_caches:
                        if cache and hasattr(cache, 'key_cache') and len(cache.key_cache) > layer_idx:
                            keys_to_concat.append(cache.key_cache[layer_idx])
                            values_to_concat.append(cache.value_cache[layer_idx])

                    if keys_to_concat:
                        merged_keys = torch.cat(keys_to_concat, dim=2)
                        merged_values = torch.cat(values_to_concat, dim=2)
                        merged.update(merged_keys, merged_values, layer_idx)

                logger.info(f"Successfully merged {len(valid_caches)} caches")
            else:
                logger.warning("Cache structure not recognized, using first cache only")
                return valid_caches[0]
        except Exception as exc:
            logger.error(f"Error merging caches: {exc}")
            return valid_caches[0]

        return merged
    
    def load_existing_caches(self):
        """Load existing KV caches from cache directory"""
        if not os.path.exists(self.cache_dir):
            return
        
        logger.info(f"Loading existing caches from {self.cache_dir}")
        
        skill_cache_path = os.path.join(self.cache_dir, "skill_interfaces.pt")
        if os.path.exists(skill_cache_path):
            try:
                self.skill_interface_cache = torch.load(skill_cache_path, map_location='cpu', weights_only=False)
                logger.info(f"Loaded skill interface cache")
            except Exception as e:
                logger.error(f"Failed to load skill interface cache: {e}")
        
        cache_files = {}
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_code.pt') or filename.endswith('_summary.pt'):
                if '_code.pt' in filename:
                    block_id = filename.replace('_code.pt', '')
                    cache_type = 'code'
                elif '_summary.pt' in filename:
                    block_id = filename.replace('_summary.pt', '')
                    cache_type = 'summary'
                else:
                    continue
                
                if block_id not in cache_files:
                    cache_files[block_id] = {}
                cache_files[block_id][cache_type] = filename
        
        loaded_count = 0
        for block_id, files in cache_files.items():
            if 'code' in files and 'summary' in files:
                try:
                    task_name = '_'.join(block_id.split('_')[:-1])
                    
                    code_cache_path = os.path.join(self.cache_dir, files['code'])
                    summary_cache_path = os.path.join(self.cache_dir, files['summary'])
                    
                    code_kv_cache = torch.load(code_cache_path, map_location='cpu', weights_only=False)
                    summary_kv_cache = torch.load(summary_cache_path, map_location='cpu', weights_only=False)
                    
                    block = PolicyBlock(
                        block_id=block_id,
                        task_name=task_name,
                        code="",
                        summary="",
                        code_kv_cache=code_kv_cache,
                        summary_kv_cache=summary_kv_cache
                    )
                    
                    self.policy_blocks[block_id] = block
                    self.task_hierarchy[task_name].append(block_id)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load block {block_id}: {e}")
        
        logger.info(f"Loaded {loaded_count} cached blocks")

def _get_dataset_with_adaptation(dataset_name: str, max_questions: Optional[int], is_adaptation: bool):
        """
        Wrap cagds.get to pass is_adaptation when available, but keep backward compatibility.
        """
        try:
            # Newer signature supports is_adaptation
            _, dataset = cagds.get(dataset_name, max_questions=max_questions, is_adaptation=is_adaptation)
        except TypeError:
            # Fallback for older signature
            _, dataset = cagds.get(dataset_name, max_questions=max_questions)
        return list(dataset)

def generate_with_prompt_only(
    model,
    tokenizer,
    policy_manager: SimplePolicyManager,
    instruction: str,
    task_name: str,
    main_objects: List[str],
    perplexity_threshold: float = 20.0,
    max_new_tokens: int = 128,
    use_summary_cache: bool = True
) -> Tuple[str, float, int, float]:
    """Generate code using prompt only - no post-processing
    
    Args:
        use_summary_cache: If True, use summary KV cache. If False, only use code cache.
    """
    
    # Memory management before generation
    policy_manager.manage_memory_before_generation()
    
    # Select relevant blocks
    logger.info(f"Selecting relevant blocks for task: {task_name}")
    selected_blocks = policy_manager.select_relevant_blocks(
        instruction, task_name, perplexity_threshold
    )
    
    # Collect and merge KV caches
    kv_caches_to_merge = []
    
    # Always include helper code cache first for consistency
    if policy_manager.helper_code_cache is not None:
        kv_caches_to_merge.append(policy_manager.helper_code_cache)
        logger.info("Added helper code cache to KV caches")
    
    if policy_manager.skill_interface_cache is not None:
        kv_caches_to_merge.append(policy_manager.skill_interface_cache)
    
    retrieved_code_text = []

    for block_id, block_info in selected_blocks.items():
        block_code = block_info.get('code')
        if block_code:
            retrieved_code_text.append(block_code)

        block_type = block_info.get('type')
        if block_type == 'full':
            code_cache = block_info.get('code_kv')
            if code_cache is not None:
                kv_caches_to_merge.append(code_cache)
            if use_summary_cache:
                summary_cache = block_info.get('summary_kv')
                if summary_cache is not None:
                    kv_caches_to_merge.append(summary_cache)
            else:
                logger.info("[ABLATION] Skipping summary cache")
        elif block_type == 'summary_only':
            if use_summary_cache:
                summary_cache = block_info.get('summary_kv')
                if summary_cache is not None:
                    kv_caches_to_merge.append(summary_cache)
                    logger.info("Added summary-only KV cache")
            else:
                logger.info("[ABLATION] Summary-only cache skipped due to configuration")

    if retrieved_code_text:
        logger.info(f"Collected {len(retrieved_code_text)} cached snippets for reuse")

    merged_kv_cache = None
    if kv_caches_to_merge:
        try:
            merged_kv_cache = policy_manager.merge_kv_caches(kv_caches_to_merge)
            logger.info(f"Successfully merged {len(kv_caches_to_merge)} KV caches")
        except Exception as e:
            logger.error(f"Failed to merge KV caches: {e}")
    
    # Build strong prompt that enforces correct format
    
    
    # Build more specific prompt based on task
    available_objs = [obj for obj in main_objects if obj != 'success']
    obj_list_str = ', '.join([f"'{obj}'" for obj in available_objs])
    
    # Give even more explicit starting point
    # Task-specific prompting
    task_specific_hint = ""

    
    # Very explicit prompt with clear start
    prompt = f"""
    # MANDATORY: Use these EXACT objects and skills. NO placeholders!
    # Available objects: {obj_list_str}
    # Available skills: move, pick, place, push, open_gripper, close_gripper
    {task_specific_hint}#
    # Then continue with the remaining implementation.
    #
    # Starting code (continue from here):
    obs, reward, done = run_action(skill, object, offset(if needed))
    {instruction}"""
    
    logger.info(f"Prompt:\n{prompt[:500]}...")
    
    # More aggressive memory management before generation
    policy_manager.manage_memory_before_generation(min_free_gb=3.0)  # Keep 3GB free for generation
    torch.cuda.empty_cache()  # Force clear cache
    
    embed_device = model.model.embed_tokens.weight.device
    inputs = tokenizer(prompt, return_tensors="pt").to(embed_device)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time()
    
    # Token-by-token generation
    input_ids = inputs.input_ids
    origin_len = input_ids.shape[-1]
    output_ids = input_ids.clone()
    next_token = input_ids
    past_key_values = merged_kv_cache
    
    stop_phrases = ["# code_end", "\ndef ", "\nif __name__", "\n\n\n", "```python", "# Add your", "# ...", "# Your code here"]
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # More frequent and aggressive memory check
            if step % 20 == 0:  # Check every 20 steps instead of 30
                used_gb, total_gb = policy_manager.check_gpu_memory()
                free_gb = total_gb - used_gb
                
                if free_gb < 1.0:  # Higher emergency threshold 1GB instead of 0.5GB
                    logger.warning(f"[MEMORY] Critical! Step {step}, free: {free_gb:.2f}GB")
                    # Emergency offload
                    gpu_blocks = [bid for bid, b in policy_manager.policy_blocks.items() 
                                 if not b.is_offloaded and b.code_kv_cache is not None]
                    if gpu_blocks:
                        # Offload half of GPU blocks
                        num_to_offload = max(1, len(gpu_blocks) // 2)
                        policy_manager.offload_by_locality(num_blocks=num_to_offload)
                    torch.cuda.empty_cache()
            
            try:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    do_sample = False,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True).to(embed_device)
                past_key_values = outputs.past_key_values
            except torch.cuda.OutOfMemoryError:
                logger.error(f"[MEMORY] OOM at step {step}! Emergency cleanup...")
                
                # Offload ALL blocks except essential
                gpu_blocks = [bid for bid, b in policy_manager.policy_blocks.items() 
                             if not b.is_offloaded]
                
                if gpu_blocks:
                    logger.info(f"[MEMORY] Emergency offloading {len(gpu_blocks)} blocks")
                    # Use the existing offload_by_locality method
                    policy_manager.offload_by_locality(num_blocks=len(gpu_blocks))
                
                torch.cuda.empty_cache()
                
                # Reset generation state and continue with no cache
                past_key_values = None
                
                # Try once more
                try:
                    outputs = model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        do_sample = False,
                        use_cache=True
                    )
                    
                    logits = outputs.logits[:, -1, :]
                    next_token = logits.argmax(dim=-1, keepdim=True).to(embed_device)
                    past_key_values = outputs.past_key_values
                except:
                    logger.error("[MEMORY] Still OOM! Aborting.")
                    generated_text = "# OOM Error - could not complete generation"
                    break
            
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            gen_ids = output_ids[:, origin_len:]
            generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            should_stop = False
            for phrase in stop_phrases:
                if phrase in generated_text:
                    idx = generated_text.find(phrase)
                    generated_text = generated_text[:idx].rstrip()
                    should_stop = True
                    break
            
            if should_stop:
                break
            
            eos = tokenizer.eos_token_id
            if next_token.item() == eos:
                break
    
    end_time = time()
    generation_time = end_time - start_time
    
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    
    generated_tokens = len(gen_ids[0]) if 'gen_ids' in locals() else 0
    
    # Build complete function body
    # The prompt already starts with "obs, reward, done = run_action("
    # So we need to complete it properly
    lines = []
    
    # Add the first line (started in prompt)
    lines.append(generated_text)
    
    # Parse the rest of the generated text to build proper lines
    if '\n' in generated_text:
        remaining_lines = generated_text.split('\n')[1:]
        for line in remaining_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                if not line.startswith('    '):
                    line = '    ' + stripped
                lines.append(line)
    
    # # Ensure proper return statement
    # if not lines or not any('return' in line for line in lines[-3:]):
    #     lines.append("    return obs, reward, done")
    
    generated_code = '\n'.join(lines)
    
    logger.info(f"Generated code ({len(generated_code)} chars)")
    logger.info(generated_code)
    
    return generated_code, generation_time, generated_tokens, peak_alloc


def prompt_only_test(args: argparse.Namespace):
    """Test CATP with prompt-only approach
    
    Ablation flags:
    - feedback_mode: 'fim' (line-level fix) or 'regenerate' (full regeneration)
    - use_summary_cache: True/False to enable/disable summary cache usage
    """
    
    # Load model
    if args.quantized:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(args.modelname, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            args.modelname,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.modelname, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            args.modelname,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )
    
    # Initialize manager
    policy_manager = SimplePolicyManager(model, tokenizer)
    infiller = FIMCodeModifier(model, tokenizer)
    
    # Load skill interfaces
    skill_file_path = "./functions.jsonl"
    policy_manager.load_skill_interfaces(skill_file_path)
    
    # Load policies with memory management
    policy_folder = "./answer"
    if os.path.exists(policy_folder):
        policy_files = [f for f in os.listdir(policy_folder) if f.endswith(".py")]
        logger.info(f"Loading {len(policy_files)} policy files...")
        
        # Load in batches to avoid OOM
        batch_size = 5
        for i, filename in enumerate(policy_files):
            task_name = filename[:-3]
            if task_name not in policy_manager.task_hierarchy:
                with open(os.path.join(policy_folder, filename), 'r') as f:
                    code = f.read()
                    policy_manager.store_policy_block(task_name, code, is_example=True)
            
            # Check memory every few files
            if (i + 1) % batch_size == 0 and i + 1 < len(policy_files):
                logger.info(f"Loaded {i+1}/{len(policy_files)} policies, checking memory...")
                policy_manager.manage_memory_before_generation(min_free_gb=1.0)
    
    logger.info(f"Total blocks available: {len(policy_manager.policy_blocks)}")
    
    # Initialize cache hit tracking for this test run
    test_cache_hits = 0
    test_cache_requests = 0
    
    # Load dataset
    dataset = _get_dataset_with_adaptation(args.dataset, args.maxQuestion, args.adaptation)
    
    # SSH client
    ssh_client = SSHClient()
    if not ssh_client.create_ssh_client():
        print("Failed to create SSH connection.")
        return
    
    _config = config_remote
    
    # Results storage
    results = {
        "task_names": [],
        "success": [],
        "similarity": [],
        "generation_time": [],
        "generated_tokens": [],
        "peak_memory": []
    }
    
    REPLAN_LIMIT = 3
    
    for task_idx, (task, ground_truth) in enumerate(dataset):
        print(f"\n{'='*50}")
        print(f"Task {task_idx + 1}/{len(dataset)}: {task}")
        print('='*50)
        
        try:
            req_success, session_id, initial_objects, initial_instruction, descriptions = init_task(
                task, args.perturbation, args.adaptation
            )
        except TypeError:
            # Older init_task without adaptation arg
            req_success, session_id, initial_objects, initial_instruction, descriptions = init_task(
                task, args.perturbation
            )
        
        if not req_success:
            sleep(3)
            continue
        
        is_success = False
        best_similarity = 0
        total_time = 0
        total_tokens = 0
        peak_memory = 0
        
        for attempt in range(REPLAN_LIMIT):
            torch.cuda.empty_cache()
            
            logger.info(f"Attempt {attempt + 1}/{REPLAN_LIMIT}")
            
            if attempt == 0:
                # First attempt: Generate with prompt-only approach
                generated_code, gen_time, gen_tokens, peak_mem = generate_with_prompt_only(
                    model, tokenizer, policy_manager,
                    initial_instruction, task, initial_objects,
                    perplexity_threshold=args.perplexity_threshold,
                    use_summary_cache=args.use_summary_cache
                )
                new_instruction = initial_instruction
            else:
                # Retry based on feedback_mode
                if args.feedback_mode == 'regenerate':
                    # Full regeneration with feedback
                    logger.info(f"[ABLATION] Regenerating code based on feedback")
                    feedback_instruction = f"""{new_instruction}
                    
                    # IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT:
                    # {feedback}
                    # Gripper position: {gripper_position}
                    # 
                    # Fix the issues mentioned above and generate correct code.
                    # Focus on using the right objects and skills for the task."""
                    
                    generated_code, gen_time, gen_tokens, peak_mem = generate_with_prompt_only(
                        model, tokenizer, policy_manager,
                        feedback_instruction, task, initial_objects,
                        perplexity_threshold=args.perplexity_threshold,
                        use_summary_cache=args.use_summary_cache
                    )
                else:  # default 'fim'
                    # FIM line-level fix
                    logger.info(f"[ABLATION] Using FIM to fix code based on feedback")
                    
                    # If no specific feedback, try common fixes based on task patterns
                    if "No feedback available" in feedback or "Task execution failed" in feedback or "did not complete successfully" in feedback:
                        logger.info("Generic feedback detected, applying heuristic fixes...")
                        
                        # Common fix patterns based on task type
                        if 'pick' in task.lower() or 'lift' in task.lower():
                            # For pick tasks, ensure we pick object before success
                            feedback = "Wrong sequence. Should pick the object first before going to success location."
                        elif 'push' in task.lower() or 'slide' in task.lower():
                            # For push tasks, check if using right skill
                            feedback = "Check if using push skill correctly with the right object."
                        elif 'button' in task.lower() or 'switch' in task.lower():
                            # Button tasks need closed gripper
                            feedback = "Button tasks require close_gripper first to make a fist."
                        else:
                            # Generic object/skill mismatch
                            feedback = "Possible wrong object or skill used. Check the task requirements."
                    
                    fixed_code = infiller.fix_code_with_fim(
                        generated_code, feedback, new_instruction, gripper_position, initial_objects
                    )
                    
                    # If FIM didn't change anything, try more aggressive fix
                    if fixed_code == generated_code:
                        logger.warning("FIM did not modify code, attempting alternative fix...")
                        # Try swapping objects if multiple exist
                        lines = generated_code.split('\n')
                        modified = False
                        for i, line in enumerate(lines):
                            if 'run_action' in line and not modified:
                                # Simple heuristic: if task failed, try different object
                                if "'target'" in line and "'block'" in str(initial_objects):
                                    lines[i] = line.replace("'target'", "'block'")
                                    modified = True
                                    logger.info(f"Swapped target->block in line {i}")
                                elif "'block'" in line and "'target'" in str(initial_objects):
                                    lines[i] = line.replace("'block'", "'target'")
                                    modified = True
                                    logger.info(f"Swapped block->target in line {i}")
                        
                        if modified:
                            fixed_code = '\n'.join(lines)
                    
                    generated_code = fixed_code
                    gen_time = time() - start_time
                    gen_tokens = len(tokenizer.encode(generated_code))
                    peak_mem = torch.cuda.max_memory_allocated()
            
            total_time += gen_time
            total_tokens += gen_tokens
            peak_memory = max(peak_memory, peak_mem)
            
            # Send and run code
            local_path = f"{_config.LOCAL_PATH}{task}.py"
            with open(local_path, "w") as f:
                f.write(generated_code)
            
            remote_path = f"{_config.REMOTE_PATH}{task}.py"
            if not ssh_client.send_file(local_path, remote_path):
                break
            
            is_terminated, session_id, run_data = run_task(session_id)
            
            if is_terminated:
                if run_data and run_data.get("success"):
                    is_success = True
                    block_id = policy_manager.store_policy_block(task, generated_code)
                    print(f"✓ Task {task} succeeded!")
                break
            
            # Collect feedback
            if run_data is None:
                # Task failed but no specific feedback - generate generic feedback
                feedback = "Task execution failed. Check if the correct objects are being used and the action sequence is appropriate for the task."
                gripper_position = ""
                new_instruction = initial_instruction
                logger.warning("Task failed without specific feedback, using generic feedback")
            else:
                feedback = run_data.get("feedback", "")
                if not feedback:
                    # Empty feedback but run_data exists - still a failure
                    feedback = "Task did not complete successfully. Review the action sequence and object interactions."
                gripper_position = run_data.get("gripper_position", "")
                new_instruction = run_data.get("new_instruction", initial_instruction)
            
            logger.info(f"Feedback: {feedback[:200]}...")
            start_time = time()
        
        # Cleanup
        if not is_terminated:
            shutdown_task(session_id)
        
        # Calculate similarity
        similarity = cagsim.bert(generated_code, ground_truth)
        best_similarity = max(best_similarity, similarity)
        
        # Save results
        results["task_names"].append(task)
        results["success"].append(is_success)
        results["similarity"].append(best_similarity)
        results["generation_time"].append(total_time / (attempt + 1))
        results["generated_tokens"].append(total_tokens / (attempt + 1))
        results["peak_memory"].append(peak_memory / (1024**2))
        
        print(f"\nResults for {task}:")
        print(f"  Success: {is_success}")
        print(f"  Similarity: {best_similarity:.4f}")
        print(f"  Avg Gen Time: {results['generation_time'][-1]:.2f}s")
        print(f"  Avg Tokens: {results['generated_tokens'][-1]:.0f}")
        print(f"  Peak Memory: {results['peak_memory'][-1]:.2f} MB")
        
        # Print cache hit ratios so far
        if policy_manager.cache_requests > 0:
            current_hit_ratio = policy_manager.cache_hits / policy_manager.cache_requests
            print(f"  Cache Hit Ratio (cumulative): {current_hit_ratio:.2%} ({policy_manager.cache_hits}/{policy_manager.cache_requests})")
            
            if policy_manager.total_blocks_evaluated > 0:
                effective_ratio = policy_manager.effective_hits / policy_manager.total_blocks_evaluated
                print(f"  Effective Hit Ratio: {effective_ratio:.2%} ({policy_manager.effective_hits}/{policy_manager.total_blocks_evaluated})")
        
        with open(args.output, "a") as f:
            f.write(f"[{task_idx}] Task: {task}, Success: {is_success}, ")
            f.write(f"Similarity: {best_similarity:.4f}, ")
            f.write(f"Time: {results['generation_time'][-1]:.2f}s, ")
            f.write(f"Tokens: {results['generated_tokens'][-1]:.0f}, ")
            f.write(f"Memory: {results['peak_memory'][-1]:.2f} MB\n")
    
    ssh_client.close()
    
    # Final statistics
    if results["success"]:
        avg_success = sum(results["success"]) / len(results["success"])
        avg_similarity = sum(results["similarity"]) / len(results["similarity"])
        avg_time = sum(results["generation_time"]) / len(results["generation_time"])
        avg_tokens = sum(results["generated_tokens"]) / len(results["generated_tokens"])
        avg_memory = sum(results["peak_memory"]) / len(results["peak_memory"])
        
        print("\n" + "="*50)
        print("CATP Prompt-Only Results:")
        print(f"  Success Rate: {avg_success:.2%}")
        print(f"  Avg Similarity: {avg_similarity:.4f}")
        print(f"  Avg Generation Time: {avg_time:.2f}s")
        print(f"  Avg Generated Tokens: {avg_tokens:.0f}")
        print(f"  Avg Peak Memory: {avg_memory:.2f} MB")
        
        # Print comprehensive cache statistics
        print("\n" + "="*50)
        print("Cache Statistics:")
        
        # Basic hit ratio
        if policy_manager.cache_requests > 0:
            overall_hit_ratio = policy_manager.cache_hits / policy_manager.cache_requests
            print(f"  Overall Cache Hit Ratio: {overall_hit_ratio:.2%} ({policy_manager.cache_hits}/{policy_manager.cache_requests})")
        else:
            print(f"  Overall Cache Hit Ratio: N/A (no cache requests)")
        
        # Effective hit ratio (blocks that passed perplexity threshold)
        if policy_manager.total_blocks_evaluated > 0:
            effective_ratio = policy_manager.effective_hits / policy_manager.total_blocks_evaluated
            threshold_pass_ratio = policy_manager.blocks_passed_threshold / policy_manager.total_blocks_evaluated
            print(f"\n  Effective Hit Ratio: {effective_ratio:.2%} ({policy_manager.effective_hits}/{policy_manager.total_blocks_evaluated})")
            print(f"  Threshold Pass Ratio: {threshold_pass_ratio:.2%} ({policy_manager.blocks_passed_threshold}/{policy_manager.total_blocks_evaluated})")
        
        # Hit type breakdown
        print(f"\n  Hit Type Breakdown:")
        print(f"    Direct Hits (exact task): {policy_manager.direct_hits}")
        print(f"    Partial Hits (similar task): {policy_manager.partial_hits}")
        print(f"    Cross-task Transfers: {policy_manager.cross_task_transfers}")
        
        # Per-task cache statistics if available
        if policy_manager.task_cache_requests:
            print("\n  Per-Task Cache Hit Ratios:")
            for task_name in sorted(policy_manager.task_cache_requests.keys()):
                task_requests = policy_manager.task_cache_requests[task_name]
                task_hits = policy_manager.task_cache_hits[task_name]
                if task_requests > 0:
                    task_ratio = task_hits / task_requests
                    print(f"    {task_name}: {task_ratio:.2%} ({task_hits}/{task_requests})")
        
        with open(args.output, "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write("CATP Prompt-Only Final Results:\n")
            f.write(f"  Success Rate: {avg_success:.2%}\n")
            f.write(f"  Avg Similarity: {avg_similarity:.4f}\n")
            f.write(f"  Avg Generation Time: {avg_time:.2f}s\n")
            f.write(f"  Avg Generated Tokens: {avg_tokens:.0f}\n")
            f.write(f"  Avg Peak Memory: {avg_memory:.2f} MB\n")
            
            # Write comprehensive cache statistics to file
            f.write("\n" + "="*50 + "\n")
            f.write("Cache Statistics:\n")
            if policy_manager.cache_requests > 0:
                overall_hit_ratio = policy_manager.cache_hits / policy_manager.cache_requests
                f.write(f"  Overall Cache Hit Ratio: {overall_hit_ratio:.2%} ({policy_manager.cache_hits}/{policy_manager.cache_requests})\n")
            
            if policy_manager.total_blocks_evaluated > 0:
                effective_ratio = policy_manager.effective_hits / policy_manager.total_blocks_evaluated
                threshold_pass_ratio = policy_manager.blocks_passed_threshold / policy_manager.total_blocks_evaluated
                f.write(f"  Effective Hit Ratio: {effective_ratio:.2%} ({policy_manager.effective_hits}/{policy_manager.total_blocks_evaluated})\n")
                f.write(f"  Threshold Pass Ratio: {threshold_pass_ratio:.2%} ({policy_manager.blocks_passed_threshold}/{policy_manager.total_blocks_evaluated})\n")
            
            f.write(f"\nHit Type Breakdown:\n")
            f.write(f"  Direct Hits: {policy_manager.direct_hits}\n")
            f.write(f"  Partial Hits: {policy_manager.partial_hits}\n")
            f.write(f"  Cross-task Transfers: {policy_manager.cross_task_transfers}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CATP Prompt-Only Version")
    
    parser.add_argument('--modelname', default="Qwen/Qwen2.5-Coder-14B-Instruct", 
                       type=str, help='Model name')
    parser.add_argument('--quantized', default=True, type=bool, 
                       help='Use quantized model')
    parser.add_argument('--output', required=True, type=str, 
                       help='Output file for results')
    parser.add_argument('--dataset', default='rlbench', 
                       choices=['rlbench', 'kis', 'squad-dev'], 
                       help='Dataset to use')
    parser.add_argument('--perplexity_threshold', default=15.0, type=float,
                       help='Perplexity threshold for block selection')
    parser.add_argument('--maxQuestion', default=10, type=int, 
                       help='Maximum number of questions')
    parser.add_argument('--perturbation', default=False, action='store_true',
                       help='Use perturbation setting')
    parser.add_argument('--randomSeed', default=2, type=int, 
                       help='Random seed')
    parser.add_argument('--adaptation', default=False, action='store_true',
                       help='Use adaptation setting (dataset/init_task will adapt)')
    # Ablation test flags
    parser.add_argument('--feedback_mode', default='fim', 
                       choices=['fim', 'regenerate'],
                       help='Feedback handling: fim (line-level fix) or regenerate (full regeneration)')
    parser.add_argument('--use_summary_cache', default=True, type=bool,
                       help='Whether to use summary KV cache (ablation test)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("CATP Prompt-Only System")
    print("  - No post-processing")
    print("  - Pure prompt-based generation")
    print("  - Clear format enforcement in prompt")
    print(f"  - Perplexity Threshold: {args.perplexity_threshold}")
    print(f"  - Feedback Mode: {args.feedback_mode}")
    print(f"  - Use Summary Cache: {args.use_summary_cache}")
    print(f"  - Adaptation: {args.adaptation}")
    print("="*50)
    
    # Log ablation settings
    logger.info(f"[ABLATION] Feedback mode: {args.feedback_mode}")
    logger.info(f"[ABLATION] Use summary cache: {args.use_summary_cache}")
    
    prompt_only_test(args)