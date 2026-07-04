#!/usr/bin/env python3
"""
Combined Methods for Real-World Experiments: CATP, LRLL, CAP, and RAGCache
Modified for real-world robotics experiments instead of RLBench
"""

import torch
import argparse
import os
import json
import re
import numpy as np
from time import time, sleep, perf_counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import sys
import hashlib
import time as pytime
import ast

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from feedback_system import compare_with_oracle, generate_infilling_prompt, apply_infilling, generate_regeneration_prompt, save_error_code
from function_cache_manager import FunctionCacheManager, FunctionCache

# Configure logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/combined_methods_realworld.log', mode='a')
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        # Try to use model without authentication (for public models)
        HF_TOKEN = None
        logger.warning("HF_TOKEN not found. Will try to load public models without authentication.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Add safe globals for torch serialization
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])
# Also try to add DynamicLayer if it exists
try:
    from transformers.cache_utils import DynamicLayer
    torch.serialization.add_safe_globals([DynamicLayer])
except ImportError:
    pass

# Add more transformers cache-related classes if needed
try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaAttention
    torch.serialization.add_safe_globals([DynamicCache])
except:
    pass

# Global variables
embedding_model = None  # Will be initialized only for LRLL method
RETRIEVAL_MODEL = None
FUNCTION_CACHE_MANAGER = None  # Initialize when needed

# Real-world experiment paths
ORACLE_PATH = "./oracle_func_ver"
SKILLS_FILE = os.path.join(ORACLE_PATH, "skills.jsonl")
EXAMPLES_DIR = os.path.join(ORACLE_PATH, "examples")
POLICIES_DIR = os.path.join(ORACLE_PATH, "policies")
INSTRUCTIONS_FILE = "./instructions.jsonl"
OUTPUT_DIR = "./generated_code"
ERROR_DIR = "./error_codes"

# ========================= CATP Methodology =========================

@dataclass
class PolicyBlock:
    """Policy block structure from CATP"""
    block_id: str
    task_name: str
    code: str
    summary: str
    code_kv_cache: Optional[DynamicCache] = None
    summary_kv_cache: Optional[DynamicCache] = None
    success_rate: float = 0.0
    usage_count: int = 0
    last_access_time: float = 0.0
    is_offloaded: bool = False

    temporal_score: float = 1.0
    spatial_score: float = 1.0
    semantic_score: float = 1.0
    composite_locality: float = 1.0

    last_perplexity: float = 0.0
    call_distance: int = 0


class CATPManager:
    """CATP methodology implementation for real-world experiments"""

    def __init__(self, model, tokenizer, cache_dir="./catp_cache", clear_cache=True):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.policy_blocks: Dict[str, PolicyBlock] = {}
        self.task_hierarchy = defaultdict(list)
        self.skill_interface_cache = None
        self.skill_interface_text = ""
        self.helper_code_cache = None
        self.helper_code_text = ""

        self.gpu_memory_threshold = 0.85
        self.reserved_gpu_memory_gb = 2.0
        self.offloaded_caches = {}

        self.w_temporal = 0.4
        self.w_spatial = 0.3
        self.w_semantic = 0.3

        self.execution_trace = []
        self.max_trace_length = 20
        self.eviction_history = []

        self.cache_hits = 0
        self.cache_requests = 0
        self.task_cache_hits = defaultdict(int)
        self.task_cache_requests = defaultdict(int)

        self.effective_hits = 0
        self.total_blocks_evaluated = 0
        self.blocks_passed_threshold = 0
        self.direct_hits = 0
        self.partial_hits = 0
        self.cross_task_transfers = 0

        if clear_cache and os.path.exists(cache_dir):
            import shutil
            logger.info(f"Clearing existing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)

        os.makedirs(cache_dir, exist_ok=True)

        if not clear_cache:
            self.load_existing_caches()

        self.create_helper_code_cache()

    def calculate_perplexity(self, query: str, context: str) -> float:
        """Calculate perplexity of query given context"""
        try:
            # Combine context and query for perplexity calculation
            # Use a cleaner format that the model might understand better
            full_text = f"{context}\n\nTask: {query}\n\nSolution:"

            # Tokenize - use tokenizer() instead of encode() to get proper shape
            device = self.model.model.embed_tokens.weight.device
            inputs = self.tokenizer(full_text, return_tensors='pt', max_length=1024, truncation=True, padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Calculate loss (perplexity) - only on the query part, not the whole sequence
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                # Clamp the loss to avoid extremely high perplexity values
                loss = torch.clamp(loss, max=10.0)  # This caps perplexity at ~22000
                perplexity = torch.exp(loss).item()

            return perplexity
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    def compute_kv_cache(self, text: str, cache_key: str = None) -> DynamicCache:
        """Compute KV cache for text with optional caching"""
        # Check if we have a cached version first
        if cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                logger.info(f"Loading existing KV cache from {cache_path}")
                return torch.load(cache_path, map_location='cpu', weights_only=False)

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

        # Save cache if cache_key provided
        if cache_key and cache:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            torch.save(cache, cache_path)
            logger.info(f"Saved KV cache to {cache_path}")

        return cache

    def create_policy_summary(self, code: str, task_name: str) -> str:
        """Create detailed summary of policy code for real-world tasks"""
        # Extract primitive skill patterns
        skill_pattern = re.findall(r'ps\.(\w+)\((.*?)\)', code)

        action_sequence = []
        skills_used = set()
        objects_used = set()

        # Parse object references
        obj_pattern = re.findall(r"objects\['([^']+)'\]", code)
        for obj in obj_pattern:
            objects_used.add(obj)

        # Parse skill calls
        for skill, params in skill_pattern:
            skills_used.add(skill)
            action_sequence.append(f"{skill}({params})")

        task_type = "manipulation"
        if "execute_push" in code or "push" in task_name.lower():
            task_type = "pushing"
        elif "execute_pick" in code and "execute_place" in code:
            task_type = "pick-and-place"
        elif "open" in task_name.lower():
            task_type = "opening"
        elif "close" in task_name.lower():
            task_type = "closing"

        summary = f"Task: {task_name}\n"
        summary += f"Type: {task_type}\n"
        summary += f"Objects: {', '.join(sorted(objects_used))}\n"
        summary += f"Skills used: {', '.join(sorted(skills_used))}\n"
        summary += f"Action sequence: {' -> '.join(action_sequence[:5])}"
        if len(action_sequence) > 5:
            summary += f" ... ({len(action_sequence)} total actions)"
        summary += "\n"

        return summary

    def create_helper_code_cache(self):
        """Create KV cache for real-world helper code"""
        # Skip helper code for faster generation - it's already in the prompt
        self.helper_code_text = ""
        self.helper_code_cache = None
        logger.info("Skipping helper code cache for faster generation")

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

    def store_policy_block(self, task_name: str, code: str, skip_if_exists: bool = True, is_example: bool = False):
        """Store policy block and create KV cache"""
        if skip_if_exists and task_name in self.task_hierarchy:
            existing_blocks = self.task_hierarchy[task_name]
            if existing_blocks:
                logger.info(f"Task {task_name} already has cached blocks, skipping")
                return existing_blocks[0]

        block_id = f"{task_name}_{int(time())}"
        summary = self.create_policy_summary(code, task_name)

        block = PolicyBlock(
            block_id=block_id,
            task_name=task_name,
            code=code,
            summary=summary,
            last_access_time=time(),
            temporal_score=1.0,
            spatial_score=1.0,
            semantic_score=1.0,
            composite_locality=1.0
        )

        cache_path = os.path.join(self.cache_dir, f"{block_id}_code.pt")
        summary_cache_path = os.path.join(self.cache_dir, f"{block_id}_summary.pt")

        # For CATP, cache both code and summary with full context
        if is_example:
            if os.path.exists(cache_path):
                logger.info(f"Loading existing code KV cache for block {block_id}")
                block.code_kv_cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            else:
                logger.info(f"Computing new code KV cache for block {block_id}")
                # Include full context in KV cache so we don't need text prompt
                full_code_context = f"""### Robot Control Task: {task_name}
### Available skills: go_to_ready_pose, setTargetPose, execute_pick, execute_place
### CRITICAL: Use *self.get_obj('name') for ALL objects (NOT objects[] or self.get_obj without *)
### Task code:
{code}
### End of example"""
                # This context is cached as KV, not added to prompt
                block.code_kv_cache = self.compute_kv_cache(full_code_context, cache_key=f"{block_id}_code")

            if os.path.exists(summary_cache_path):
                logger.info(f"Loading existing summary KV cache for block {block_id}")
                block.summary_kv_cache = torch.load(summary_cache_path, map_location='cpu', weights_only=False)
            else:
                logger.info(f"Computing new summary KV cache for block {block_id}")
                block.summary_kv_cache = self.compute_kv_cache(summary, cache_key=f"{block_id}_summary")
        else:
            # For non-examples, still compute KV cache for reuse
            block.code_kv_cache = self.compute_kv_cache(code, cache_key=f"{block_id}_code")
            block.summary_kv_cache = self.compute_kv_cache(summary, cache_key=f"{block_id}_summary")

        self.policy_blocks[block_id] = block
        self.task_hierarchy[task_name].append(block_id)

        self.execution_trace.append(block_id)
        if len(self.execution_trace) > self.max_trace_length:
            self.execution_trace = self.execution_trace[-self.max_trace_length:]

        logger.info(f"Stored block {block_id} for task {task_name}")
        return block_id

    def select_relevant_blocks(self, instruction: str, task_name: str, perplexity_threshold: float = 100.0):
        """Select relevant cached blocks for the current task using perplexity
        Returns top 3 blocks: best one with code+summary, next 2 with summary only
        """
        selected = {}

        # Track cache request
        self.cache_requests += 1
        self.task_cache_requests[task_name] += 1

        # First try direct task match
        if task_name in self.task_hierarchy:
            block_ids = self.task_hierarchy[task_name]
            if block_ids:
                block = self.policy_blocks[block_ids[0]]
                selected[block_ids[0]] = {
                    'type': 'full',  # Use both code and summary
                    'code_kv': block.code_kv_cache,
                    'summary_kv': block.summary_kv_cache,
                    'perplexity': 0.0
                }
                # Track cache hit
                self.cache_hits += 1
                self.direct_hits += 1
                self.task_cache_hits[task_name] += 1
                logger.info(f"Found direct cache match for {task_name}")
                return selected

        # If no direct match, calculate perplexity for all cached blocks
        logger.info(f"No direct match for {task_name}, calculating perplexity...")
        perplexity_scores = []

        for cached_task, block_ids in self.task_hierarchy.items():
            if not block_ids:
                continue

            block_id = block_ids[0]
            block = self.policy_blocks[block_id]

            # Calculate perplexity of instruction given the cached block's summary
            if block.summary:
                perplexity = self.calculate_perplexity(instruction, block.summary)
                logger.info(f"  {cached_task}: perplexity = {perplexity:.2f}")

                if perplexity < perplexity_threshold:
                    perplexity_scores.append({
                        'block_id': block_id,
                        'block': block,
                        'perplexity': perplexity,
                        'task_name': cached_task
                    })

        # Sort by perplexity and select top 3
        perplexity_scores.sort(key=lambda x: x['perplexity'])

        if perplexity_scores:
            # Best block: use both code and summary KV cache
            best = perplexity_scores[0]
            selected[best['block_id']] = {
                'type': 'full',  # Use code KV cache
                'code_kv': best['block'].code_kv_cache,
                'summary_kv': best['block'].summary_kv_cache,
                'code': best['block'].code,  # Include actual code text
                'perplexity': best['perplexity']
            }
            logger.info(f"Best match: {best['task_name']} (perplexity: {best['perplexity']:.2f}) - using code KV")
            self.cache_hits += 1
            self.task_cache_hits[task_name] += 1

            # Next 2 blocks: use summary KV cache only
            for i in range(1, min(3, len(perplexity_scores))):
                item = perplexity_scores[i]
                selected[item['block_id']] = {
                    'type': 'summary_only',  # Only use summary KV cache
                    'code_kv': None,
                    'summary_kv': item['block'].summary_kv_cache,
                    'code': item['block'].code,  # Include actual code text
                    'perplexity': item['perplexity']
                }
                logger.info(f"Additional match {i}: {item['task_name']} (perplexity: {item['perplexity']:.2f}) - using summary KV only")
        else:
            logger.info(f"No suitable cache found")

        return selected

    def merge_kv_caches(self, kv_caches: list) -> DynamicCache:
        """Merge multiple KV caches hierarchically
        First cache (best code match) is primary, others are supplementary
        """
        if not kv_caches:
            return None

        # Single cache: return as is
        if len(kv_caches) == 1:
            return kv_caches[0]

        # Multiple caches: concatenate them
        # First is code-level cache, others are summary-level
        logger.info(f"Merging {len(kv_caches)} KV caches hierarchically")

        merged = DynamicCache()
        first_cache = kv_caches[0]  # Best match (code-level)

        # Handle DynamicCache structure
        try:
            if hasattr(first_cache, 'key_cache') and hasattr(first_cache, 'value_cache'):
                # DynamicCache has key_cache and value_cache lists
                num_layers = len(first_cache.key_cache)

                for layer_idx in range(num_layers):
                    keys_to_concat = []
                    values_to_concat = []

                    # Add all caches' keys and values
                    for cache in kv_caches:
                        if cache and hasattr(cache, 'key_cache'):
                            if len(cache.key_cache) > layer_idx:
                                keys_to_concat.append(cache.key_cache[layer_idx])
                                values_to_concat.append(cache.value_cache[layer_idx])

                    if keys_to_concat:
                        # Concatenate along sequence dimension (dim=2)
                        merged_keys = torch.cat(keys_to_concat, dim=2)
                        merged_values = torch.cat(values_to_concat, dim=2)
                        merged.update(merged_keys, merged_values, layer_idx)

                logger.info(f"Successfully merged {len(kv_caches)} caches")
            else:
                # Fallback to first cache if structure is different
                logger.warning("Cache structure not recognized, using first cache only")
                return first_cache

        except Exception as e:
            logger.error(f"Error merging caches: {e}")
            return first_cache

        return merged


# ========================= LRLL Methodology =========================

def mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 3, lamb: float = 0.5):
    """Maximal Marginal Relevance selection from LRLL"""
    assert doc_vecs.ndim == 2
    N = doc_vecs.shape[0]
    if N == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    Dnorm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sims = Dnorm @ q
    selected, candidates = [], list(range(N))
    while len(selected) < min(k, N):
        if not selected:
            i = int(np.argmax(sims))
            selected.append(i)
            candidates.remove(i)
            continue
        max_div = []
        for i in candidates:
            div = max(float(Dnorm[i] @ Dnorm[j]) for j in selected)
            score = lamb * sims[i] - (1.0 - lamb) * div
            max_div.append((score, i))
        _, pick = max(max_div, key=lambda x: x[0])
        selected.append(pick)
        candidates.remove(pick)
    return selected


@dataclass
class Experience:
    """Experience structure for LRLL"""
    instruction: str
    policy_code: str
    success: bool
    success_code: Optional[str] = None
    ts: float = field(default_factory=lambda: pytime.time())


class ExperienceMemory:
    """Memory for LRLL experiences"""
    def __init__(self, root="./lrll_memory", embed_dim=384):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.exp_path = os.path.join(self.root, "experiences.jsonl")
        self.emb_path = os.path.join(self.root, "instr_emb.npy")
        self.index_path = os.path.join(self.root, "index.json")
        self.embed_dim = embed_dim
        self.experiences: List[Experience] = []
        self.embeddings = np.empty((0, embed_dim), dtype=np.float32)
        self.index: Dict[str, int] = {}

    def _sig(self, e: Experience) -> str:
        h = hashlib.sha256()
        h.update(e.instruction.encode("utf-8"))
        h.update(b"\x1f")
        h.update(e.policy_code.encode("utf-8"))
        return h.hexdigest()

    def load(self):
        if os.path.exists(self.exp_path):
            with open(self.exp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    self.experiences.append(Experience(**d))
        if os.path.exists(self.emb_path):
            self.embeddings = np.load(self.emb_path)
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)

    def save(self):
        with open(self.exp_path, "w", encoding="utf-8") as f:
            for e in self.experiences:
                f.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")
        np.save(self.emb_path, self.embeddings)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False)

    def add(self, e: Experience, embed_model: SentenceTransformer) -> bool:
        sig = self._sig(e)
        if sig in self.index:
            return False
        vec = embed_model.encode([e.instruction])
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        self.experiences.append(e)
        self.embeddings = np.vstack([self.embeddings, vec])
        self.index[sig] = len(self.experiences) - 1
        return True

    def retrieve(self, instruction: str, top_k=3, lamb=0.5) -> List[Experience]:
        if len(self.experiences) == 0:
            return []
        global embedding_model
        if embedding_model is None:
            raise RuntimeError("Embedding model not initialized. Please initialize it for LRLL method.")
        q = embedding_model.encode([instruction])
        if q.dtype != np.float32:
            q = q.astype(np.float32)
        idxs = mmr_select(q[0], self.embeddings, k=top_k, lamb=lamb)
        return [self.experiences[i] for i in idxs]


# ========================= Common Helper Functions =========================

def compute_perplexity(model, tokenizer, prompt: str, code: str) -> float:
    """Compute perplexity of code given prompt"""
    full_text = prompt + code
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prompt_len = prompt_inputs['input_ids'].shape[-1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate perplexity only for the code part (not the prompt)
        shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., prompt_len:].contiguous()

        # Compute cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        perplexity = torch.exp(loss).item()

    return perplexity


def load_skill_interfaces():
    """Load skill interfaces from real-world skills file"""
    skill_interfaces = []
    if os.path.exists(SKILLS_FILE):
        with open(SKILLS_FILE, 'r') as f:
            for line in f:
                line_dict = json.loads(line)
                interface = line_dict['code_interface']
                description = line_dict['description']
                skill_interfaces.append(f"{interface}:\n    \"\"\"{description}\"\"\"")
    return skill_interfaces


def load_example_policies():
    """Load example policies from real-world policies directory"""
    policies = {}
    if os.path.exists(POLICIES_DIR):
        for filename in os.listdir(POLICIES_DIR):
            if filename.endswith(".py"):
                task_name = filename[:-3]
                with open(os.path.join(POLICIES_DIR, filename), 'r') as f:
                    policies[task_name] = f.read()
    return policies


def load_examples_for_retrieval():
    """Load example policies from examples directory for retrieval (LRLL, RAGCache, CATP)"""
    examples = {}
    if os.path.exists(EXAMPLES_DIR):
        for filename in os.listdir(EXAMPLES_DIR):
            if filename.endswith(".py"):
                task_name = filename[:-3]
                with open(os.path.join(EXAMPLES_DIR, filename), 'r') as f:
                    examples[task_name] = f.read()
    return examples


def load_instructions():
    """Load all task instructions from file as separate entries"""
    instructions = []
    if os.path.exists(INSTRUCTIONS_FILE):
        with open(INSTRUCTIONS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instructions.append({
                        'task_name': data['task'],
                        'instruction': data['instruction']
                    })
    return instructions


def generate(model, tokenizer, prompt: str, past_key_values=None, max_new_tokens: int = 256) -> tuple:
    """Generate text with greedy decoding and TTFT measurement"""
    device = model.model.embed_tokens.weight.device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    origin_len = input_ids.shape[-1]
    logger.info(f"Prompt token length: {origin_len}, max_new_tokens: {max_new_tokens}")
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    # Extended stop phrases including code blocks and markdown
    stop_phrases = [
        "# END", "### END", "if __name__", "def main()",
        "# Do not", "### INSTRUCTION", "### CODE",
        "###","``` ```","Task:",
        "# Example", "## ", "### ",  # Stop at headers
        # Plain text indicators
        "The code", "This function", "Here's", "Here is", "I'll", "I will",
        "Let me", "To ", "First,", "Then,", "Finally,", "Note:", "Note that",
        "You can", "You should", "This will", "It will", "We need",
        "The function", "The method", "Please", "However,", "Therefore,"
    ]

    # Token-level stop tokens (check individual tokens)
    # Removed ``` to allow code blocks
    stop_tokens = [
        tokenizer.encode("###", add_special_tokens=False),
    ]
    stop_token_ids = set()
    for tokens in stop_tokens:
        if tokens:
            stop_token_ids.update(tokens)

    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()
    ttft = None  # Time to first token
    first_token_generated = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True).to(device)
            past_key_values = outputs.past_key_values
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Measure TTFT
            if not first_token_generated:
                ttft = perf_counter() - start_time
                first_token_generated = True
                logger.info(f"TTFT (Time to First Token): {ttft:.3f}s")

            # Check if current token is a stop token
            if next_token.item() in stop_token_ids:
                gen_ids = output_ids[:, origin_len:]
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                # Remove the stop token from output
                gen_text = gen_text.rstrip()
                torch.cuda.synchronize()
                peak_alloc = torch.cuda.max_memory_allocated()
                generated_tokens = len(gen_ids[0])
                end_time = perf_counter()
                generation_time = end_time - start_time
                logger.info(f"Stopped at token: {tokenizer.decode([next_token.item()])}")
                return gen_text, generation_time, generated_tokens, peak_alloc, ttft

            gen_ids = output_ids[:, origin_len:]
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # Check for stop phrases in generated text
            for phrase in stop_phrases:
                if phrase in gen_text:
                    idx = gen_text.find(phrase)
                    gen_text = gen_text[:idx].rstrip()
                    torch.cuda.synchronize()
                    peak_alloc = torch.cuda.max_memory_allocated()
                    generated_tokens = len(gen_ids[0])
                    end_time = perf_counter()
                    generation_time = end_time - start_time
                    logger.info(f"Stopped at phrase: {phrase}")
                    return gen_text, generation_time, generated_tokens, peak_alloc, ttft

            # Check for EOS token or newline followed by special characters
            eos = model.config.eos_token_id
            if (isinstance(eos, (list, tuple)) and next_token.item() in eos) \
               or (isinstance(eos, int) and next_token.item() == eos):
                break

            # Early stopping if we see patterns that indicate wrong generation
            if len(gen_text) > 10:  # After some initial generation
                # Check for suspicious patterns
                last_chars = gen_text[-10:] if len(gen_text) >= 10 else gen_text
                if '\n###' in last_chars:  # Only stop at markdown headers, not code blocks
                    gen_text = gen_text[:gen_text.rfind('\n')].rstrip()
                    torch.cuda.synchronize()
                    peak_alloc = torch.cuda.max_memory_allocated()
                    generated_tokens = len(gen_ids[0])
                    end_time = perf_counter()
                    generation_time = end_time - start_time
                    logger.info("Early stopping due to pattern detection")
                    return gen_text, generation_time, generated_tokens, peak_alloc, ttft

                # Check if it's generating plain text instead of code
                # Look for sentences or explanations
                gen_lower = gen_text.lower()
                plain_text_indicators = [
                    ". ", "! ", "? ",  # Sentence endings with space
                    " is ", " are ", " was ", " were ",  # Verbs indicating description
                    " will ", " would ", " should ", " could ",
                    "this ", "that ", "these ", "those ",
                    "the function", "the code", "the method",
                    "you ", "we ", "i ",  # Personal pronouns
                ]

                # Count how many plain text indicators we find
                plain_text_score = sum(1 for indicator in plain_text_indicators if indicator in gen_lower)

                # If multiple indicators found, it's likely plain text
                if plain_text_score >= 3:
                    # Try to extract just the code part if any
                    lines = gen_text.split('\n')
                    code_lines = []
                    for line in lines:
                        # Keep lines that look like code
                        if line.strip() and not any(ind in line.lower() for ind in ['the ', 'this ', 'will ', 'should ']):
                            if '(' in line or '=' in line or 'self.' in line:
                                code_lines.append(line)

                    if code_lines:
                        gen_text = '\n'.join(code_lines)
                    else:
                        gen_text = ""  # No valid code found

                    torch.cuda.synchronize()
                    peak_alloc = torch.cuda.max_memory_allocated()
                    generated_tokens = len(gen_ids[0])
                    end_time = perf_counter()
                    generation_time = end_time - start_time
                    logger.info(f"Early stopping due to plain text detection (score: {plain_text_score})")
                    return gen_text, generation_time, generated_tokens, peak_alloc, ttft

    # end_time is defined below after getting gen_text
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()

    gen_ids = output_ids[:, origin_len:]
    generated_tokens = len(gen_ids[0])
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    end_time = perf_counter()  # Make sure end_time is defined
    generation_time = end_time - start_time
    return gen_text, generation_time, generated_tokens, peak_alloc, ttft


def post_process_code(code: str) -> str:
    """Post-processing disabled - returns raw code as-is"""
    return code

def save_generated_code(task_name: str, code: str, method: str, inst_num: int = None, seed: int = None, suffix: str = ""):
    """Save generated code to file with improved naming"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build filename: method_taskname_inst{num}_seed{num}[_suffix].py
    filename_parts = [method, task_name]

    if inst_num is not None:
        filename_parts.append(f"inst{inst_num}")

    if seed is not None:
        filename_parts.append(f"seed{seed}")

    if suffix:
        filename_parts.append(suffix)

    filename = "_".join(filename_parts) + ".py"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Fix indentation: indent each line of generated code with 8 spaces
    lines = code.split('\n')
    indented_lines = []
    for line in lines:
        # If line is not empty and doesn't already have correct indentation
        if line.strip():
            # Check if line already has some indentation (1 space is incorrect)
            if line.startswith(' ') and not line.startswith('        '):
                # Remove incorrect indentation and add correct 8 spaces
                indented_lines.append('        ' + line.lstrip())
            elif not line.startswith('        '):
                # Add 8 spaces if no indentation
                indented_lines.append('        ' + line)
            else:
                # Line already has correct indentation
                indented_lines.append(line)
        else:
            # Keep empty lines as is
            indented_lines.append(line)
    indented_code = '\n'.join(indented_lines)

    # Create complete Python file matching the actual PrimitiveSkill class
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

    def robot_move(self):
        \"\"\"Generated code for task: {task_name}\"\"\"
        gripper_force = 50
        {indented_code}

def main():
    robot = RobotController()
    robot.robot_move()

if __name__ == "__main__":
    main()
"""

    with open(filepath, 'w') as f:
        f.write(full_code)

    logger.info(f"Saved generated code to {filepath}")
    return filepath


# ========================= Method-specific Prompts =========================

def create_catp_prompt(instruction: str, has_kv_cache: bool = False, retrieved_text: str = ""):
    """Create ultra-minimal prompt for CATP method with function usage"""
    # Check if we can use cached functions - get top 3 for complex tasks
    relevant_functions = []
    if FUNCTION_CACHE_MANAGER:
        relevant_functions = FUNCTION_CACHE_MANAGER.get_relevant_functions(instruction, top_k=3)

    func_examples = ""
    if relevant_functions:
        # Generate complete function calls that can be used directly
        func_calls = []
        for func_name, func_cache in relevant_functions:
            func_call = FUNCTION_CACHE_MANAGER.generate_function_call(func_name, instruction)
            # Indent for proper code formatting
            func_calls.append(' ' * 8 + func_call)
            logger.info(f"[DEBUG CATP Prompt] Generated function call: {func_call}")

        func_examples = "\n".join(func_calls)
        logger.info(f"[DEBUG CATP Prompt] Found {len(relevant_functions)} matching functions")
        logger.info(f"[DEBUG CATP Prompt] Function examples:\n{func_examples}")

    if has_kv_cache:
        # With KV cache - the cache already contains similar code patterns
        # Just provide the task and let the model continue from cached context
        if func_examples and relevant_functions:
            # Best matching function found
            best_func_name = relevant_functions[0][0]
            best_func_call = func_calls[0].strip() if func_calls else ""

            # Minimal prompt - the KV cache will guide generation
            # Include guidance for using functions vs basic skills
            prompt = f"""# Task: {instruction}
# IMPORTANT: Functions from robot_tasks are already imported. Just call them directly, DO NOT redefine them.
# For other tasks without matching functions, use basic skills (setTargetPose, execute_pick, execute_place)
# If a line may cause errors or need revision,
# insert '# ERROR_FLAG' immediately before it.

def robot_move(self):
        self.ps.go_to_ready_pose()
        # First handle the trash task using the imported function
        """
        elif retrieved_text:
            # Put retrieved examples at the top for emphasis
            prompt = f"""### RETRIEVED SIMILAR IMPLEMENTATIONS:
{retrieved_text}

### NEW TASK: {instruction}

Use the above examples to complete the task.

def robot_move(self):
        self.ps.go_to_ready_pose()
        """
        else:
            prompt = f"""Task: {instruction}

def robot_move(self):
        self.ps.go_to_ready_pose()
        """
    else:
        # Only when no KV cache
        if func_examples:
            prompt = f"""Robot task: {instruction}
- Generate robot_move() method body only
- Call task functions directly when appropriate: function_name(self.ps, self.get_obj, ...)
- Start with: self.ps.go_to_ready_pose() if not using task functions
- End with: self.ps.go_to_ready_pose() if not using task functions
- NO go_to_ready_pose() in the middle
- Must call setTargetPose() before execute_pick/place
- CRITICAL: Use *self.get_obj('name') for ALL object positions (NOT objects[] or self.get_obj without *)
- Example: self.ps.setTargetPose(*self.get_obj('red_trash'))
- Use gripper_force=50 for pick operations
- Use axis=2 for vertical pick/place
- DO NOT USE CODE BLOCKS (```python or ```)
- DO NOT MAKE COMMENT


{func_examples}
Generate the complete implementation to solve the task.
Use the functions above as needed, and add any additional code required:"""
        else:
            prompt = f"""Robot task: {instruction}
- Generate robot_move() method body only
- Call task functions directly when appropriate: function_name(self.ps, self.get_obj, ...)
- Start with: self.ps.go_to_ready_pose() if not using task functions
- End with: self.ps.go_to_ready_pose() if not using task functions
- NO go_to_ready_pose() in the middle
- Must call setTargetPose() before execute_pick/place
- CRITICAL: Use *self.get_obj('name') for ALL object positions (NOT objects[] or self.get_obj without *)
- Example: self.ps.setTargetPose(*self.get_obj('red_trash'))
- Use gripper_force=50 for pick operations
- Use axis=2 for vertical pick/place
- DO NOT USE CODE BLOCKS (```python or ```)
- DO NOT MAKE COMMENT


CRITICAL: Use *self.get_obj('name') for ALL objects
Generate robot_move() method body:"""
    return prompt


def create_lrll_prompt(instruction: str, skill_interfaces_str: str, retrieved: List[Experience]):
    """Create prompt for LRLL method - simplified and direct"""

    # First check if we have a matching cached function
    if FUNCTION_CACHE_MANAGER:
        relevant_functions = FUNCTION_CACHE_MANAGER.get_relevant_functions(instruction, top_k=1)
        if relevant_functions:
            func_name, func_cache = relevant_functions[0]
            func_call = FUNCTION_CACHE_MANAGER.generate_function_call(func_name, instruction)

            # Return complete implementation with the function
            return f"""# Task: {instruction}
def robot_move(self):
        self.ps.go_to_ready_pose()
        {func_call}
        self.ps.go_to_ready_pose()"""

    # No function found - generate basic implementation
    # For ComplexTask with drawer and trash
    if 'drawer' in instruction.lower() and 'trash' in instruction.lower():
        return f"""# Task: {instruction}
def robot_move(self):
        self.ps.go_to_ready_pose()
        open_drawer_task(self.ps, self.get_obj)
        self.ps.setTargetPose(*self.get_obj('dice_red'))
        self.ps.execute_pick(gripper_force=50, axis=2)
        self.ps.setTargetPose(*self.get_obj('drawer'))
        self.ps.execute_place(axis=2)
        put_rubbish_in_bin_task(self.ps, self.get_obj, rubbish_list=['trash_red', 'trash_green', 'trash_blue'], bin_obj='trash_bin')
        self.ps.go_to_ready_pose()"""

    # For simple trash tasks
    if 'trash' in instruction.lower() and 'trash_bin' in instruction.lower():
        objects = []
        if 'trash_red' in instruction:
            objects.append('trash_red')
        if 'trash_green' in instruction:
            objects.append('trash_green')
        if 'trash_blue' in instruction:
            objects.append('trash_blue')

        if objects:
            code = "# Task: {}\ndef robot_move(self):\n        self.ps.go_to_ready_pose()\n".format(instruction)
            for obj in objects:
                code += f"        self.ps.setTargetPose(*self.get_obj('{obj}'))\n"
                code += "        self.ps.execute_pick(gripper_force=50, axis=2)\n"
                code += "        self.ps.setTargetPose(*self.get_obj('trash_bin'))\n"
                code += "        self.ps.execute_place(axis=2)\n"
            code += "        self.ps.go_to_ready_pose()"
            return code

    # Default: return minimal prompt for generation
    return f"""# Task: {instruction}
def robot_move(self):
        self.ps.go_to_ready_pose()
        """


def create_useprompt_prompt(instruction: str, skill_interfaces_str: str, examples: Dict[str, str]):
    """Create prompt for CAP method - ONLY use basic skills, NO cached functions"""
    # Use all available examples
    example_codes = []
    for task_name, code in examples.items():
        # Extract just the robot_move method body from the full code
        if 'def robot_move(self):' in code:
            start = code.find('def robot_move(self):') + len('def robot_move(self):')
            end = code.find('\n    def ', start)
            if end == -1:
                end = code.find('\ndef ', start)
            if end == -1:
                end = len(code)
            method_body = code[start:end].strip()
            example_codes.append(f"{method_body}")
        else:
            # If it's already just the method body
            example_codes.append(f"{code}")
    examples_str = "\n\n".join(example_codes) if example_codes else ""

    # CAP should NEVER use cached functions, only basic skills
    # Do not include func_examples in the prompt

    # Put examples first for emphasis
    prompt = f"""### EXAMPLE IMPLEMENTATIONS:
{examples_str}

### TASK: {instruction}

Generate robot_move() method body using basic skills ONLY.
- DO NOT use task functions like put_rubbish_in_bin_task(), pick_up_cup_task() etc.
- ONLY use basic skills: setTargetPose, execute_pick, execute_place, go_to_ready_pose
- Use *self.get_obj('name') for ALL objects
- Start/end with self.ps.go_to_ready_pose()

def robot_move(self):
        """

    return prompt


def create_ragcache_prompt(instruction: str, skill_interfaces_str: str, retrieved_examples: str = ""):
    """Create prompt for RAGCache method with retrieval emphasis"""
    # Get relevant cached functions
    func_examples = ""
    if FUNCTION_CACHE_MANAGER:
        relevant_functions = FUNCTION_CACHE_MANAGER.get_relevant_functions(instruction, top_k=3)
        if relevant_functions:
            func_calls = []
            for func_name, func_cache in relevant_functions:
                func_call = FUNCTION_CACHE_MANAGER.generate_function_call(func_name, instruction)
                func_calls.append(func_call)
            if func_calls:
                func_examples = "\n### BEST MATCHING CACHED FUNCTIONS:\n" + "\n".join(func_calls) + "\n"

    # Put retrieval results at the very beginning to emphasize them
    prompt = f"""### RETRIEVED SIMILAR IMPLEMENTATIONS:
{retrieved_examples if retrieved_examples else "# No similar examples found"}
{func_examples}
### TASK: {instruction}

Generate robot_move() method body using the above examples as reference.
- Call task functions directly when appropriate: function_name(self.ps, self.get_obj, ...)
- Use *self.get_obj('name') for ALL objects
- Start/end with self.ps.go_to_ready_pose() if not using task functions

def robot_move(self):
        """

    return prompt


# ========================= Main Test Function =========================

def combined_test(args: argparse.Namespace):
    """Main test function for real-world experiments"""

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

    # Load resources
    skill_interfaces = load_skill_interfaces()
    skill_interfaces_str = '\n'.join(skill_interfaces)
    example_policies = load_example_policies()  # From policies directory for base prompts
    retrieval_examples = load_examples_for_retrieval()  # From examples directory for retrieval
    instructions = load_instructions()

    # Initialize method-specific components
    catp_manager = None
    lrll_memory = None

    # Initialize function cache manager
    global FUNCTION_CACHE_MANAGER
    FUNCTION_CACHE_MANAGER = FunctionCacheManager(cache_dir="./function_cache")
    logger.info(f"Loaded {len(FUNCTION_CACHE_MANAGER.functions_cache)} cached task functions")

    # Create instructions dictionary for RAGCache
    instructions_dict = {}
    for task_data in instructions:
        if task_data['task_name'] not in instructions_dict:
            instructions_dict[task_data['task_name']] = task_data['instruction']

    if args.method == "catp":
        catp_manager = CATPManager(model, tokenizer)
        # Pre-cache example policies for fast retrieval with KV cache
        for task_name, code in retrieval_examples.items():
            catp_manager.store_policy_block(task_name, code, is_example=True)
        logger.info(f"Initialized CATP with {len(retrieval_examples)} cached example policies")
        print(f"\nCATP Cache Status:")
        print(f"  Pre-cached blocks: {len(catp_manager.policy_blocks)}")
        print(f"  Pre-cached tasks: {len(catp_manager.task_hierarchy)}")
        print(f"  Cache hit ratio will be tracked during generation\n")

    elif args.method == "lrll":
        # Initialize embedding model only for LRLL
        global embedding_model
        logger.info("Initializing embedding model for LRLL method...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        lrll_memory = ExperienceMemory(root="./lrll_memory", embed_dim=384)
        lrll_memory.load()
        # Add example policies as experiences
        for task_name, code in retrieval_examples.items():
            # Find matching instruction for this example
            matching_instruction = None
            for task_data in instructions:
                if task_data['task_name'] == task_name:
                    matching_instruction = task_data['instruction']
                    break

            if matching_instruction:
                lrll_memory.add(
                    Experience(
                        instruction=matching_instruction,
                        policy_code=code,
                        success=True
                    ),
                    embed_model=embedding_model
                )
        logger.info(f"Initialized LRLL with {len(lrll_memory.experiences)} experiences")

    # Results storage
    results = {
        "task_names": [],
        "methods": [],
        "generation_time": [],
        "ttft": [],  # Time To First Token
        "generated_tokens": [],
        "peak_memory": [],
        "output_files": [],
        "used_function": []  # Track if cached function was used
    }

    # Track instruction numbers for each task
    task_inst_count = {}

    # Process each instruction
    tasks_to_process = instructions[:args.maxTasks] if args.maxTasks else instructions

    for task_idx, task_data in enumerate(tasks_to_process):
        task_name = task_data['task_name']
        original_task_name = task_name  # Preserve original task name for cache saving
        instruction = task_data['instruction']

        # Track instruction number for this task (0-based indexing)
        if task_name not in task_inst_count:
            task_inst_count[task_name] = 0
        inst_num = task_inst_count[task_name]
        task_inst_count[task_name] += 1

        print(f"\n{'='*50}")
        print(f"Task {task_idx + 1}/{len(tasks_to_process)}: {task_name}")
        print(f"Instruction #{inst_num}: {instruction}")
        print(f"Method: {args.method}")
        print(f"Seed: {args.randomSeed if hasattr(args, 'randomSeed') else 'None'}")
        print('='*50)

        torch.cuda.empty_cache()

        # Generate code based on method
        if args.method == "catp":
            # Use perplexity to find best matching cached function
            best_func = None
            best_perplexity = float('inf')
            use_function_call = False

            # Always try perplexity-based function selection for all tasks
            if FUNCTION_CACHE_MANAGER and args.use_perplexity:
                logger.info(f"CATP: Testing perplexity for function selection on task: {instruction[:50]}...")
                # Test perplexity for each cached function
                perplexity_results = []
                for func_name, func_cache in FUNCTION_CACHE_MANAGER.functions_cache.items():
                    # Generate the function call for this task
                    func_call = FUNCTION_CACHE_MANAGER.generate_function_call(func_name, instruction)
                    func_call_indented = ' ' * 8 + func_call

                    # Create a minimal prompt
                    test_prompt = f"Robot task: {instruction}\nCode:\n"

                    # Compute perplexity
                    perp = compute_perplexity(model, tokenizer, test_prompt, func_call_indented)
                    perplexity_results.append((func_name, perp))

                    if perp < best_perplexity:
                        best_perplexity = perp
                        best_func = (func_name, func_cache, func_call_indented)
                        logger.info(f"  New best: {func_name} (perplexity={perp:.2f})")

                # Log all perplexity results for debugging
                logger.info(f"CATP Perplexity results: {[(name, f'{perp:.2f}') for name, perp in sorted(perplexity_results, key=lambda x: x[1])[:5]]}")

                # If we found a good match (low perplexity), use it
                perplexity_threshold = 50.0  # Reduced threshold for more selective matching
                if best_func and best_perplexity < perplexity_threshold:
                    func_name, func_cache, func_call = best_func
                    logger.info(f"CATP: Found good match {func_name} (perp={best_perplexity:.2f})")
                    use_function_call = True
                else:
                    logger.info(f"CATP: Not using function directly (best perp={best_perplexity:.2f} > threshold={perplexity_threshold})")

            # Always try to get relevant blocks for KV cache
            selected_blocks = catp_manager.select_relevant_blocks(instruction, task_name)

            # DEBUG: Log what was selected
            logger.info(f"[DEBUG CATP] Selected {len(selected_blocks)} blocks for instruction: {instruction[:50]}...")
            for block_id, block_info in selected_blocks.items():
                logger.info(f"[DEBUG CATP]   Block: {block_id}, Type: {block_info['type']}, Perplexity: {block_info.get('perplexity', 0):.2f}")

            # Prepare KV caches to merge based on hierarchy
            kv_caches_to_merge = []
            retrieved_code_text = []  # Collect text for prompt emphasis

            for block_id, block_info in selected_blocks.items():
                if block_info['type'] == 'full':
                    # Best block: use code KV cache
                    if block_info.get('code_kv'):
                        kv_caches_to_merge.append(block_info['code_kv'])
                        logger.info(f"  Adding code KV cache from best match")
                        # Also collect the text for prompt
                        if 'code' in block_info:
                            retrieved_code_text.append(block_info['code'])
                elif block_info['type'] == 'summary_only':
                    # Next best blocks: use summary KV cache only
                    if block_info.get('summary_kv'):
                        kv_caches_to_merge.append(block_info['summary_kv'])
                        logger.info(f"  Adding summary KV cache from additional match")

            # Merge KV caches hierarchically
            merged_kv_cache = None
            cache_hit = False
            if kv_caches_to_merge:
                if len(kv_caches_to_merge) == 1:
                    # Only one cache (best match code)
                    merged_kv_cache = kv_caches_to_merge[0]
                else:
                    # Multiple caches: need to merge them
                    merged_kv_cache = catp_manager.merge_kv_caches(kv_caches_to_merge)
                cache_hit = True
                logger.info(f"CATP: Using {len(kv_caches_to_merge)} KV caches (NO text in prompt)")
                logger.info(f"  Expecting <10s generation with hierarchical cache")
            else:
                logger.info(f"CATP: No cache found, generating without cache (~20s)")

            # Create prompt based on whether we're using function call or not
            # Always use KV cache based generation, not direct function calls
            if use_function_call:
                # Found a good function match, use its KV cache for fast generation
                func_name, func_cache, func_call = best_func
                logger.info(f"[DEBUG CATP] Found best function: {func_name} (perp={best_perplexity:.2f})")
                logger.info(f"[DEBUG CATP] Will use KV cache from this function for generation")

                # If we already have merged_kv_cache, keep it
                # Otherwise, try to get KV cache from the matched function
                if not merged_kv_cache and FUNCTION_CACHE_MANAGER:
                    # Try to get cached KV for the matched function
                    func_cache_obj = FUNCTION_CACHE_MANAGER.get_function_cache(func_name)
                    if func_cache_obj and func_cache_obj.code_kv_cache:
                        logger.info(f"[DEBUG CATP] Using KV cache from function {func_name}")
                        merged_kv_cache = func_cache_obj.code_kv_cache

            # Create minimal prompt for generation with KV cache
            has_cache = merged_kv_cache is not None
            retrieved_text = "\n\n".join(retrieved_code_text) if retrieved_code_text else ""
            prompt = create_catp_prompt(instruction, has_kv_cache=has_cache, retrieved_text=retrieved_text)

            logger.info(f"[DEBUG CATP] Generating with LLM")
            logger.info(f"[DEBUG CATP] Has KV cache: {has_cache}, Retrieved text length: {len(retrieved_text)}")

            # Always generate with LLM (no longer skip for cached functions)
            # Reduce max tokens for faster generation
            max_tokens_catp = min(args.maxTokens, 200) if has_cache else args.maxTokens

            # Generate using KV cache (not text)
            generated_code, gen_time, gen_tokens, peak_mem, ttft = generate(
                model, tokenizer, prompt, merged_kv_cache, max_new_tokens=max_tokens_catp
            )
            print(f"#################\ngenerated_code\n {generated_code} ")
            # Post-process CATP output to remove any function redefinitions
            # Functions from robot_tasks.py are already imported, so we should not redefine them
            if 'def ' in generated_code:
                logger.info("CATP: Removing function redefinitions from generated code")
                lines = generated_code.split('\n')
                cleaned_lines = []
                skip_function = False

                for line in lines:
                    # Check if this is the start of a function definition for an existing task
                    if line.strip().startswith('def '):
                        # This is a function definition for a task function - skip it
                        skip_function = True
                        logger.info(f"Skipping redefinition of: {line.strip()}")
                        continue

                    # If we're inside a function definition, check if it ended
                    if skip_function:
                        # Function ends when we see a line that's not indented (and not empty)
                        if line and not line.startswith(' ') and not line.startswith('\t'):
                            skip_function = False
                            # Don't skip this line - it's the start of something new
                        else:
                            # Still inside the function definition - skip
                            continue

                    # Keep this line
                    cleaned_lines.append(line)

                generated_code = '\n'.join(cleaned_lines)
                logger.info("CATP: Function redefinitions removed")

        elif args.method == "lrll":
            if lrll_memory is None:
                logger.error("LRLL memory not initialized!")
                generated_code = "# Error: LRLL memory not initialized"
                gen_time, gen_tokens, peak_mem, ttft = 0, 0, 0, 0
            else:
                retrieved = lrll_memory.retrieve(instruction, top_k=3, lamb=0.5)
                logger.info(f"LRLL: Retrieved {len(retrieved)} examples")
                prompt = create_lrll_prompt(instruction, skill_interfaces_str, retrieved)
                logger.info(f"LRLL Prompt:\n{prompt}")
                generated_code, gen_time, gen_tokens, peak_mem, ttft = generate(
                    model, tokenizer, prompt, None, max_new_tokens=args.maxTokens
                )
                logger.info(f"LRLL Generated: {generated_code[:200]}...")

        elif args.method == "cap":
            prompt = create_useprompt_prompt(instruction, skill_interfaces_str, example_policies)
            generated_code, gen_time, gen_tokens, peak_mem, ttft = generate(
                model, tokenizer, prompt, None, max_new_tokens=args.maxTokens
            )

        elif args.method == "ragcache":
            # RAGCache: Similar to LRLL but uses KV cache instead of text
            # First, create KV caches for example policies if not already done
            if not hasattr(args, 'ragcache_initialized'):
                args.ragcache_kv_caches = {}
                logger.info("Initializing RAGCache KV caches for example policies...")
                for task_name, code in retrieval_examples.items():
                    # Generate KV cache for each example
                    example_prompt = f"""# Task: {task_name}
{code}"""
                    with torch.no_grad():
                        inputs = tokenizer(example_prompt, return_tensors="pt").to(model.device)
                        outputs = model(**inputs, use_cache=True)
                        args.ragcache_kv_caches[task_name] = outputs.past_key_values
                args.ragcache_initialized = True
                logger.info(f"Initialized {len(args.ragcache_kv_caches)} KV caches")

            # Retrieve similar examples using embedding similarity
            if embedding_model is None:
                # Need embedding model for similarity search
                logger.info("Initializing embedding model for RAGCache similarity search...")
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Compute embeddings
            query_embedding = embedding_model.encode([instruction])
            example_embeddings = {}
            for task_name in args.ragcache_kv_caches.keys():
                if task_name in instructions_dict:
                    example_embeddings[task_name] = embedding_model.encode([instructions_dict[task_name]])[0]

            # Find top 3 similar examples
            similarities = []
            for task_name, emb in example_embeddings.items():
                sim = cosine_similarity([query_embedding[0]], [emb])[0][0]
                similarities.append((task_name, sim))

            top_3 = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"RAGCache: Retrieved {[t[0] for t in top_3]} with similarities {[f'{t[1]:.3f}' for t in top_3]}")

            # Merge the KV caches from top 3
            kv_caches_to_merge = []
            for task_name, _ in top_3:
                if task_name in args.ragcache_kv_caches:
                    kv_caches_to_merge.append(args.ragcache_kv_caches[task_name])

            # Collect the actual example code for the prompt
            retrieved_examples = []
            for task_name, _ in top_3:
                if task_name in retrieval_examples:
                    retrieved_examples.append(retrieval_examples[task_name].strip())
            retrieved_examples_str = "\n\n".join(retrieved_examples) if retrieved_examples else ""

            # Merge KV caches
            merged_kv_cache = None
            if kv_caches_to_merge:
                if len(kv_caches_to_merge) == 1:
                    merged_kv_cache = kv_caches_to_merge[0]
                else:
                    # Merge multiple KV caches
                    merged_kv_cache = []
                    num_layers = len(kv_caches_to_merge[0])
                    for layer_idx in range(num_layers):
                        keys = torch.cat([kv[layer_idx][0] for kv in kv_caches_to_merge], dim=2)
                        values = torch.cat([kv[layer_idx][1] for kv in kv_caches_to_merge], dim=2)
                        merged_kv_cache.append((keys, values))

            # Generate with merged KV cache and retrieved examples
            prompt = create_ragcache_prompt(instruction, skill_interfaces_str, retrieved_examples_str)
            generated_code, gen_time, gen_tokens, peak_mem, ttft = generate(
                model, tokenizer, prompt, merged_kv_cache, max_new_tokens=args.maxTokens
            )

        # Clean up generated code
        # Log the raw generated code for debugging
        logger.info(f"Raw generated code: {generated_code[:200]}..." if len(generated_code) > 200 else f"Raw generated code: {generated_code}")

        # Post-processing now disabled - just returns raw code
        generated_code = post_process_code(generated_code)

        # Check if it's a function call (from cached functions)
        function_names = [
            '_task(', 'alternating_pattern_task(', 'arrange_in_grid_task(',
            'circular_arrangement_task(', 'clear_workspace_task(', 'close_box_task(',
            'insert_peg_task(', 'open_drawer_task(', 'pick_and_lift_task(',
            'pick_up_cup_task(', 'pour_water_task(', 'push_button_task(',
            'put_rubbish_in_bin_task(', 'pyramid_stacking_task(', 'sort_by_size_task(',
            'sort_colors_task(', 'stack_blocks_task('
        ]

        is_function_call = any(func_name in generated_code for func_name in function_names)

        # DEBUG: Log what was generated and whether it contains function calls
        logger.info(f"[DEBUG] Method: {args.method}, Task: {task_name}")
        logger.info(f"[DEBUG] Generated code (first 200 chars): {generated_code[:200]}")
        logger.info(f"[DEBUG] Contains function call: {is_function_call}")
        if is_function_call:
            for func_name in function_names:
                if func_name in generated_code:
                    logger.info(f"[DEBUG] Found function: {func_name}")
                    break

        if is_function_call:
            # For function calls, don't apply the go_to_ready_pose filtering
            logger.info(f"Detected function call, keeping as-is")
            # The function itself handles go_to_ready_pose internally
        else:
            # Apply normal filtering for non-function code
            lines = generated_code.split('\n')

            # Remove intermediate go_to_ready_pose() calls (keep only first and last)
            filtered_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and 'go_to_ready_pose()' in stripped:
                    # Keep only if it's the first or last non-empty line
                    if i == 0 or (i == len(lines) - 1) or (not any(l.strip() for l in lines[i+1:])):
                        filtered_lines.append(line)
                elif line.strip() or (i > 0 and i < len(lines) - 1):  # Keep non-empty lines and preserve some structure
                    filtered_lines.append(line)

            # Make sure we have go_to_ready_pose at the end if missing
            if filtered_lines:
                last_non_empty = None
                for i in range(len(filtered_lines) - 1, -1, -1):
                    if filtered_lines[i].strip():
                        last_non_empty = filtered_lines[i]
                        break

                if last_non_empty and 'go_to_ready_pose()' not in last_non_empty:
                    filtered_lines.append(' ' * 8 + 'self.ps.go_to_ready_pose()')

                generated_code = '\n'.join(filtered_lines)

        # Ensure generated_code is not empty
        if not generated_code.strip():
            logger.warning("Generated code was empty after processing, using fallback")
            generated_code = ' ' * 8 + 'self.ps.go_to_ready_pose()'

        # Compare with oracle and get feedback
        oracle_code = None
        if task_name in example_policies:
            oracle_code = example_policies[task_name]
            # Extract just the robot_move method body
            if 'def robot_move(self):' in oracle_code:
                start = oracle_code.find('def robot_move(self):') + len('def robot_move(self):')
                end = oracle_code.find('\n    def ', start)
                if end == -1:
                    end = oracle_code.find('\ndef ', start)
                if end == -1:
                    end = len(oracle_code)
                oracle_code = oracle_code[start:end].strip()

            # Compare and get feedback
            success, feedback = compare_with_oracle(generated_code, oracle_code, task_name)
            logger.info(f"Oracle comparison: {'Success' if success else 'Failed'}")

            if not success:
                logger.info(f"Feedback: {feedback}")

                # Save the error code
                error_code_path = save_error_code(original_task_name, generated_code, feedback, args.method, inst_num=inst_num, seed=args.randomSeed if hasattr(args, 'randomSeed') else None)
                logger.info(f"Saved error code to {error_code_path}")

                # Handle feedback based on method
                if args.method == "catp":
                    # CATP: Use infilling to fix specific issues
                    logger.info("CATP: Using infilling to fix issues")
                    # Generate infilling prompt for the error part only
                    infill_prompt, infill_type, error_location = generate_infilling_prompt(generated_code, feedback, instruction)

                    # Generate only the fixed part (not the whole code)
                    infilled_part, fix_time, fix_tokens, _, fix_ttft = generate(
                        model, tokenizer, infill_prompt, merged_kv_cache, max_new_tokens=150
                    )

                    # Apply the infilled part to the original code
                    fixed_code = apply_infilling(generated_code, infilled_part, infill_type, error_location)
                    generated_code = fixed_code

                    gen_time += fix_time
                    gen_tokens += fix_tokens
                    logger.info(f"CATP infilling completed in {fix_time:.2f}s (type: {infill_type})")

                    # Save the fixed code too - use original task name
                    fixed_code_path = save_generated_code(original_task_name, generated_code, args.method, inst_num, args.randomSeed if hasattr(args, 'randomSeed') else None, "fixed")
                    logger.info(f"Saved fixed code to {fixed_code_path}")

                else:
                    # Other methods: Regenerate from scratch with feedback
                    logger.info(f"{args.method.upper()}: Regenerating with feedback")
                    regen_prompt = generate_regeneration_prompt(instruction, feedback)

                    # Regenerate code
                    if args.method == "lrll":
                        # Add feedback to prompt
                        regen_prompt = create_lrll_prompt(instruction, skill_interfaces_str, retrieved)
                        regen_prompt = regen_prompt.replace("### INSTRUCTION", f"### INSTRUCTION\n{feedback}\n### CORRECTED TASK")
                    elif args.method == "cap":
                        regen_prompt = create_useprompt_prompt(instruction, skill_interfaces_str, example_policies)
                        regen_prompt = regen_prompt.replace("### INSTRUCTION", f"### INSTRUCTION (Fix: {feedback})")
                    elif args.method == "ragcache":
                        # RAGCache regeneration also uses KV cache
                        regen_prompt = create_ragcache_prompt(instruction, skill_interfaces_str)
                        regen_prompt = regen_prompt.replace("### INSTRUCTION", f"### INSTRUCTION (Fix: {feedback})")

                    regenerated_code, regen_time, regen_tokens, _, regen_ttft = generate(
                        model, tokenizer, regen_prompt, None, max_new_tokens=args.maxTokens
                    )
                    generated_code = regenerated_code
                    gen_time += regen_time
                    gen_tokens += regen_tokens
                    logger.info(f"{args.method.upper()} regeneration completed in {regen_time:.2f}s")

                    # Save the regenerated code - use original task name
                    regen_code_path = save_generated_code(original_task_name, generated_code, args.method, inst_num, args.randomSeed if hasattr(args, 'randomSeed') else None, "regenerated")
                    logger.info(f"Saved regenerated code to {regen_code_path}")

                    # For RAGCache, update the KV cache with the corrected code using original task name
                    if args.method == "ragcache" and hasattr(args, 'ragcache_kv_caches'):
                        logger.info(f"Updating RAGCache KV cache for {original_task_name} with corrected code")
                        # Generate KV cache for the corrected code
                        corrected_prompt = f"""# Task: {original_task_name}
{generated_code}"""
                        with torch.no_grad():
                            inputs = tokenizer(corrected_prompt, return_tensors="pt").to(model.device)
                            outputs = model(**inputs, use_cache=True)
                            args.ragcache_kv_caches[original_task_name] = outputs.past_key_values
                        logger.info(f"Updated RAGCache KV cache for {original_task_name}")
        else:
            logger.info(f"No oracle code available for {task_name}, skipping comparison")
            success = False
            feedback = "No oracle available"

        # Save generated code - use original task name
        output_file = save_generated_code(original_task_name, generated_code, args.method, inst_num, args.randomSeed if hasattr(args, 'randomSeed') else None)

        # Check if a cached function was used
        used_func = None
        if FUNCTION_CACHE_MANAGER:
            relevant_functions = FUNCTION_CACHE_MANAGER.get_relevant_functions(instruction, top_k=1)
            if relevant_functions:
                func_name, _ = relevant_functions[0]
                if func_name in generated_code:
                    used_func = func_name

        # Store results - use original task name
        results["task_names"].append(original_task_name)
        results["methods"].append(args.method)
        results["generation_time"].append(gen_time)
        results["ttft"].append(ttft if ttft else 0.0)
        results["generated_tokens"].append(gen_tokens)
        results["peak_memory"].append(peak_mem / (1024**2))  # Convert to MB
        results["output_files"].append(output_file)
        results["used_function"].append(used_func)

        # Store feedback results if available
        if 'success' not in dir():
            success = False
            feedback = "No comparison performed"

        if not hasattr(results, 'success_rate'):
            results['success_rate'] = []
            results['feedbacks'] = []
        results['success_rate'].append(success)
        results['feedbacks'].append(feedback)

        print(f"\nResults for {original_task_name}:")
        print(f"  Method: {args.method}")
        print(f"  Generation Time: {gen_time:.2f}s")
        print(f"  TTFT (Time to First Token): {ttft:.3f}s" if ttft else "  TTFT: N/A")
        if used_func:
            print(f"  Used Cached Function: {used_func}")
        print(f"  Generated Tokens: {gen_tokens}")
        print(f"  Peak Memory: {peak_mem / (1024**2):.2f} MB")
        print(f"  Output File: {output_file}")

        # Print oracle comparison result
        print(f"  Oracle Match: {'Success' if success else 'Failed'}")
        if not success and feedback != "No oracle available":
            print(f"  Feedback: {feedback[:100]}..." if len(feedback) > 100 else f"  Feedback: {feedback}")

        # Print cache statistics for CATP
        if args.method == "catp" and catp_manager:
            hit_ratio = (catp_manager.cache_hits / catp_manager.cache_requests * 100) if catp_manager.cache_requests > 0 else 0
            print(f"  Cache Hit: {'Yes' if 'cache_hit' in locals() and cache_hit else 'No'}")
            print(f"  Cache Hit Ratio: {hit_ratio:.1f}% ({catp_manager.cache_hits}/{catp_manager.cache_requests})")
            print(f"  Direct Hits: {catp_manager.direct_hits}")
            print(f"  Semantic Hits: {catp_manager.cache_hits - catp_manager.direct_hits}")

        print(f"\nGenerated Code Preview:")
        print(generated_code[:500] + "..." if len(generated_code) > 500 else generated_code)

        # Write results to output file
        with open(f"{args.method}_{args.output}", "a") as f:
            f.write(f"[{task_idx}] Task: {original_task_name}, Method: {args.method}, ")
            f.write(f"Time: {gen_time:.2f}s, Tokens: {gen_tokens}, ")
            f.write(f"Memory: {peak_mem / (1024**2):.2f} MB, ")
            f.write(f"Output: {output_file}\n")

    # Save LRLL memory if used
    if args.method == "lrll" and lrll_memory:
        lrll_memory.save()

    # Final statistics
    if results["generation_time"]:
        avg_time = sum(results["generation_time"]) / len(results["generation_time"])
        avg_tokens = sum(results["generated_tokens"]) / len(results["generated_tokens"])
        avg_memory = sum(results["peak_memory"]) / len(results["peak_memory"])

        print("\n" + "="*50)
        print(f"{args.method.upper()} Results Summary:")
        print(f"  Total Tasks: {len(results['task_names'])}")
        print(f"  Avg Generation Time: {avg_time:.2f}s")
        if results["ttft"]:
            avg_ttft = sum(results["ttft"]) / len(results["ttft"])
            print(f"  Avg TTFT: {avg_ttft:.3f}s")
        if results["used_function"]:
            used_count = sum(1 for f in results["used_function"] if f)
            print(f"  Functions Used: {used_count}/{len(results['used_function'])}")
        print(f"  Avg Generated Tokens: {avg_tokens:.0f}")
        print(f"  Avg Peak Memory: {avg_memory:.2f} MB")

        # Print final cache statistics for CATP
        if args.method == "catp" and catp_manager:
            final_hit_ratio = (catp_manager.cache_hits / catp_manager.cache_requests * 100) if catp_manager.cache_requests > 0 else 0
            print(f"\n  CATP Cache Statistics:")
            print(f"    Total Cache Requests: {catp_manager.cache_requests}")
            print(f"    Total Cache Hits: {catp_manager.cache_hits}")
            print(f"    Overall Hit Ratio: {final_hit_ratio:.1f}%")
            print(f"    Direct Hits: {catp_manager.direct_hits}")
            print(f"    Cached Blocks: {len(catp_manager.policy_blocks)}")

        with open(f"{args.method}_{args.output}", "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"{args.method.upper()} Final Results:\n")
            f.write(f"  Total Tasks: {len(results['task_names'])}\n")
            f.write(f"  Avg Generation Time: {avg_time:.2f}s\n")
            f.write(f"  Avg Generated Tokens: {avg_tokens:.0f}\n")
            f.write(f"  Avg Peak Memory: {avg_memory:.2f} MB\n")

            # Write cache statistics for CATP
            if args.method == "catp" and catp_manager:
                final_hit_ratio = (catp_manager.cache_hits / catp_manager.cache_requests * 100) if catp_manager.cache_requests > 0 else 0
                f.write(f"\n  CATP Cache Statistics:\n")
                f.write(f"    Total Cache Requests: {catp_manager.cache_requests}\n")
                f.write(f"    Total Cache Hits: {catp_manager.cache_hits}\n")
                f.write(f"    Overall Hit Ratio: {final_hit_ratio:.1f}%\n")
                f.write(f"    Direct Hits: {catp_manager.direct_hits}\n")
                f.write(f"    Cached Blocks: {len(catp_manager.policy_blocks)}\n")

            f.write(f"  Generated Files: {', '.join(results['output_files'])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Methods for Real-World Robot Experiments")

    parser.add_argument('--method', required=True,
                       choices=['catp', 'lrll', 'cap', 'ragcache'],
                       help='Method to use for code generation')
    parser.add_argument('--modelname', default="Qwen/Qwen2.5-Coder-14B-Instruct",
                       type=str, help='Model name from HuggingFace')
    parser.add_argument('--quantized', default=True, type=bool,
                       help='Use 8-bit quantized model')
    parser.add_argument('--output', default="results.txt", type=str,
                       help='Output file for results')
    parser.add_argument('--maxTasks', default=None, type=int,
                       help='Maximum number of tasks to process')
    parser.add_argument('--maxTokens', default=256, type=int,
                       help='Maximum tokens to generate')
    parser.add_argument('--randomSeed', default=42, type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--use_perplexity', default=True, action='store_true',
                       help='Use perplexity-based function selection for CATP')

    args = parser.parse_args()
    
    # Set random seed
    if args.randomSeed:
        import random
        random.seed(args.randomSeed)
        np.random.seed(args.randomSeed)
        torch.manual_seed(args.randomSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.randomSeed)

    print("="*50)
    print("Real-World Robot Code Generation")
    print(f"Method: {args.method.upper()}")
    print(f"Model: {args.modelname}")
    print(f"Max Tasks: {args.maxTasks if args.maxTasks else 'All'}")
    print(f"Max Tokens: {args.maxTokens}")
    print("="*50)

    combined_test(args)