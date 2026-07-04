import os
import json
import time
import hashlib
import torch
import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import wandb
# Import CATP modules
from modules.catp_agent import CATPAgent
from modules.code_llm_wrapper import CodeLLMWrapper
from modules.hierarchical_cache import HierarchicalCodeCache, FunctionInterface, FunctionCode
from modules.utils import info_print, error_print, debug_print

# Add safe globals for torch.load
torch.serialization.add_safe_globals([FunctionInterface, FunctionCode, DynamicCache])

# --- Logging helpers (same as alf_teach_epic) ---

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using DEVICE hint: {DEVICE}")

# --- Global Variables ---
FILE_ORDER = None
SKILL_CODE_PREFIX = ""
SKILL_CODE_SUFFIX = "\n\n# Now generate the plan:\n"

# CATP Agent instead of KV dict
CATP_AGENT = None
CATP_INFO = {}

# Retrieval model for semantic search (compatible with alf_teach_epic)
RETRIEVAL_MODEL = None

RUNTIME_GUIDLINES = """Adhere to these stringent guidelines:
1. Use only the classes and functions defined previously. Do not create functions that are not provided above.
2. Make sure that you output a consistent plan. For example, opening of the same object should not occur in successive steps.
3. Make sure the output is consistent with the proper affordances of objects. For example, a couch cannot be opened, so your output should never include the open() function for this object, but a fridge can be opened.
4. The input is dialogue between <Driver> and <Commander>. Interpret the dialogue into robot actions. Do not output any dialogue.
5. Object categories should only be chosen from the following classes: ShowerDoor, Cabinet, CounterTop, Sink, Towel, HandTowel, TowelHolder, SoapBar, ToiletPaper, ToiletPaperHanger, HandTowelHolder, SoapBottle, GarbageCan, Candle, ScrubBrush, Plunger, SinkBasin, Cloth, SprayBottle, Toilet, Faucet, ShowerHead, Box, Bed, Book, DeskLamp, BasketBall, Pen, Pillow, Pencil, CellPhone, KeyChain, Painting, CreditCard, AlarmClock, CD, Laptop, Drawer, SideTable, Chair, Blinds, Desk, Curtains, Dresser, Watch, Television, WateringCan, Newspaper, FloorLamp, RemoteControl, HousePlant, Statue, Ottoman, ArmChair, Sofa, DogBed, BaseballBat, TennisRacket, VacuumCleaner, Mug, ShelvingUnit, Shelf, StoveBurner, Apple, Lettuce, Bottle, Egg, Microwave, CoffeeMachine, Fork, Fridge, WineBottle, Spatula, Bread, Tomato, Pan, Cup, Pot, SaltShaker, Potato, PepperShaker, ButterKnife, StoveKnob, Toaster, DishSponge, Spoon, Plate, Knife, DiningTable, Bowl, LaundryHamper, Vase, Stool, CoffeeTable, Poster, Bathtub, TissueBox, Footstool, BathtubBasin, ShowerCurtain, TVStand, Boots, RoomDecor, PaperTowelRoll, Ladle, Kettle, Safe, GarbageBag, TeddyBear, TableTopDecor, Dumbbell, Desktop, AluminumFoil, Window, LightSwitch, AppleSliced, BreadSliced, LettuceSliced, PotatoSliced, TomatoSliced
6. You can only pick up one object at a time. If the agent is holding an object, the agent should place or put down the object before attempting to pick up a second object.
7. Each object instance should instantiate a different InteractionObject class even if two object instances are the same object category.
Follow the output format provided earlier. Think step by step to carry out the instruction.

Write a Python script that could be executed by a household robot for the following:
"""

# --- Helper functions from alf_teach_epic ---
def to_camel_case(s):
    """Convert snake_case to CamelCase"""
    return ''.join(word.capitalize() for word in s.split('_'))

def to_words(s):
    """Convert CamelCase to words"""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s).lower()

# --- Resource Loading ---
def load_resources():
    """
    Load resources for CATP:
      - FILE_ORDER list (paths of all example files)
      - SKILL_CODE_PREFIX script
      - RETRIEVAL_MODEL for semantic search
    """
    global FILE_ORDER, SKILL_CODE_PREFIX, RETRIEVAL_MODEL

    info_print("Loading resources (FILE_ORDER, SKILL_CODE_PREFIX, RETRIEVAL_MODEL)...")

    file_order_path = './helper_alf/example/file_order.txt'
    skill_code_path = './prompt/api_corrective.py'

    # FILE_ORDER - try to load, create dummy if not exists
    try:
        with open(file_order_path, 'r', encoding='utf-8') as f:
            FILE_ORDER = [line.strip() for line in f if line.strip()]
        info_print(f"Loaded FILE_ORDER: {len(FILE_ORDER)} files")
    except:
        FILE_ORDER = []
        info_print("FILE_ORDER not found, using empty list")

    # Skill code - try to load, use default if not exists
    try:
        with open(skill_code_path, 'r', encoding='utf-8') as f:
            SKILL_CODE_PREFIX = f.read()
        info_print("Loaded SKILL_CODE_PREFIX")
    except:
        SKILL_CODE_PREFIX = "# Default skill code\n"
        info_print("Using default SKILL_CODE_PREFIX")
    
    # CATP uses perplexity-based retrieval, not sentence transformers
    info_print("CATP uses perplexity-based retrieval (not SentenceTransformer)")
    RETRIEVAL_MODEL = None  # Not needed for CATP

# --- Model Loading ---
info_print("Loading LLM (using CATP wrapper)...")
try:
    # Try to load Qwen model like alf_teach_epic
    llm_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B', use_fast=True)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-Coder-7B',
        load_in_8bit=True,
        device_map='auto',
        torch_dtype="auto",
    )
    llm_model.eval()
    llm_model.generation_config.pad_token_id = llm_tokenizer.pad_token_id
    llm_model.generation_config.eos_token_id = llm_tokenizer.eos_token_id

    info_print("LLM loaded (8-bit, Qwen).")
    
    # Create CodeLLMWrapper and directly set the loaded model
    code_llm = CodeLLMWrapper(model_name='placeholder', device=str(DEVICE), offload_callback=None)  # Don't load model again
    code_llm.model = llm_model
    code_llm.tokenizer = llm_tokenizer
    
except Exception as e:
    error_print(f"Failed to load Qwen model: {e}")
    info_print("Using placeholder CodeLLM")
    code_llm = CodeLLMWrapper(model_name='placeholder', device=str(DEVICE), offload_callback=None)
    llm_model = None
    llm_tokenizer = None

# --- Flask App ---
app = Flask(__name__)

# --- CATP Integration Functions ---
def save_cache_to_disk():
    """Save current cache to disk"""
    if CATP_AGENT is None:
        return
    
    try:
        cache_path = "./cache/catp_kv_cache.pt"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'interface_layer': CATP_AGENT.cache.interface_layer,
            'code_layer': CATP_AGENT.cache.code_layer,
            'info': CATP_INFO
        }, cache_path)
        print(f"[Cache] Saved {len(CATP_AGENT.cache.interface_layer)} caches to {cache_path}")
    except Exception as e:
        print(f"[Cache] Failed to save: {e}")

def initialize_catp():
    """Initialize CATP agent"""
    global CATP_AGENT, CATP_INFO
    
    try:
        CATP_AGENT = CATPAgent(
            code_llm=code_llm,
            max_cache_size=100,  # Total cache capacity (CPU + GPU)
            perplexity_threshold=10.0,
            device=str(DEVICE)
        )
        
        # Set GPU cache limit based on available memory
        # Reduce GPU cache limit to prevent OOM issues
        CATP_AGENT.cache.max_gpu_cache_size = 10  # Keep fewer caches on GPU to avoid OOM
        
        # Set up offload callback for OOM handling
        if code_llm is not None:
            code_llm.offload_callback = lambda: CATP_AGENT.cache.offload_lowest_locality_cache()
        
        CATP_INFO = {
            'initialized': True,
            'cache_size': 0,
            'build_time': time.time()
        }
        
        info_print("CATP agent initialized")
        return True
    except Exception as e:
        error_print(f"Failed to initialize CATP: {e}")
        return False

def build_catp_cache_from_examples(selected_indices=None):
    """
    Build CATP cache from example plan files (like alf_teach_epic)
    Forward entire file content directly to create KV cache
    """
    global CATP_INFO
    
    if FILE_ORDER is None:
        error_print("FILE_ORDER is None. Call load_resources() first.")
        return False
        
    if CATP_AGENT is None:
        error_print("CATP agent not initialized")
        return False

    try:
        t0 = time.time()
        total_tokens = 0
        
        # 1) Add skill code prefix + suffix to cache
        info_print("Building skill code KV cache...")
        skill_prompt = (SKILL_CODE_PREFIX or "") + (SKILL_CODE_SUFFIX or "")
        if skill_prompt and code_llm and code_llm.model:
            # Forward the skill code to create KV cache
            skill_kv = preprocess_knowledge(code_llm.model, code_llm.tokenizer, skill_prompt)
            
            # Store in cache using the hierarchical structure
            CATP_AGENT.cache.add_function(
                function_id="skill_code",
                interface_text=skill_prompt[:200],  # First 200 chars as interface
                code_text=skill_prompt,
                interface_kv=skill_kv,  # This will be stored as kv_cache
                code_kv=skill_kv,
                perplexity=0.0
            )
            if hasattr(skill_kv, 'key_cache') and len(skill_kv.key_cache) > 0:
                total_tokens += skill_kv.key_cache[0].shape[-2]
            info_print(f"Skill cache built: {total_tokens} tokens")
        
        # 2) Build KV cache for each example file
        info_print("Building example KV caches...")
        indices = list(range(len(FILE_ORDER))) if selected_indices is None else list(selected_indices)
        files_used = 0
        
        for idx in indices:
            file_path = FILE_ORDER[idx]
            full_path = f'./prompt/examples/examples_plan/{file_path}'
            if not os.path.exists(full_path):
                full_path = file_path
            if not os.path.exists(full_path):
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    txt = f.read()

                # Extract example name from file path
                example_name = os.path.basename(file_path).replace('.txt', '').replace('.py', '')
                
                if code_llm and code_llm.model:
                    # Forward the entire file content to create KV cache (like alf_teach_epic)
                    example_kv = preprocess_knowledge(code_llm.model, code_llm.tokenizer, txt)
                    
                    # Store in CATP cache
                    CATP_AGENT.cache.add_function(
                        function_id=example_name,
                        interface_text=txt[:200],  # First 200 chars as interface for retrieval
                        code_text=txt,
                        interface_kv=example_kv,
                        code_kv=example_kv,
                        perplexity=0.0
                    )
                    
                    if hasattr(example_kv, 'key_cache') and len(example_kv.key_cache) > 0:
                        example_tokens = example_kv.key_cache[0].shape[-2]
                        total_tokens += example_tokens
                    
                files_used += 1

                if files_used % 10 == 0:
                    info_print(f"Processed {files_used}/{len(indices)} example files")
                    
            except Exception as e:
                error_print(f"Failed to process {file_path}: {e}")
        
        # 3) Build runtime guidelines cache
        info_print("Building runtime guidelines cache...")
        if code_llm and code_llm.model:
            runtime_kv = preprocess_knowledge(code_llm.model, code_llm.tokenizer, RUNTIME_GUIDLINES)
            CATP_AGENT.cache.add_function(
                function_id="guidelines",
                interface_text=RUNTIME_GUIDLINES[:200],
                code_text=RUNTIME_GUIDLINES,
                interface_kv=runtime_kv,
                code_kv=runtime_kv,
                perplexity=0.0
            )
            if hasattr(runtime_kv, 'key_cache') and len(runtime_kv.key_cache) > 0:
                total_tokens += runtime_kv.key_cache[0].shape[-2]
        
        CATP_INFO = {
            'files_used': files_used,
            'total_tokens': total_tokens,
            'num_caches': len(CATP_AGENT.cache.interface_layer),
            'build_seconds': time.time() - t0
        }
        
        info_print(f"[CATP Cache] Built: files={files_used}, "
                  f"total_tokens={total_tokens}, time={CATP_INFO['build_seconds']:.2f}s")
        
        # Save cache to file
        cache_path = "./cache/catp_kv_cache.pt"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'interface_layer': CATP_AGENT.cache.interface_layer,
            'code_layer': CATP_AGENT.cache.code_layer,
            'info': CATP_INFO
        }, cache_path)
        info_print(f"Saved CATP KV cache to {cache_path}")
        
        return True

    except Exception as e:
        error_print(f"Build failed: {e}")
        import traceback; traceback.print_exc()
        return False


def preprocess_knowledge(model, tokenizer, prompt: str) -> DynamicCache:
    """
    Prepare knowledge kv cache (same as alf_teach_epic)
    Forward the prompt through the model to create KV cache
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess
        
    Returns:
        DynamicCache: KV Cache
    """
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    
    return outputs.past_key_values

def generate_with_catp(prompt: str, max_new_tokens: int = 1024, temperature: float = 0.0):
    """
    Generate using CATP with KV cache (like alf_teach_epic)
    Now with batch perplexity computation for faster retrieval
    """
    if CATP_AGENT is None:
        raise ValueError("CATP agent not initialized")
    
    if code_llm is None or code_llm.model is None:
        raise ValueError("Model not loaded")
    
    retrieval_start = time.time()
    
    # Parse instruction from prompt
    instruction = prompt.strip()
    
    # Use batch retrieval for faster perplexity computation
    top_k = 3  # Select top 3 most relevant
    retrieved_interfaces = CATP_AGENT.cache.retrieve_interfaces_batch(
        instruction=instruction,
        model=code_llm,
        top_k=top_k
    )
    
    # Extract PPL scores for logging
    ppl_scores = {}
    selected_func_ids = []
    cache_hits = 0  # Count of caches already in GPU memory
    total_caches_needed = 0
    
    # Collect KV caches in order
    kv_list = []
    kv_length = 0
    
    # 1) Add skill code KV if available (always include first)
    if "skill_code" in CATP_AGENT.cache.interface_layer:
        skill_interface = CATP_AGENT.cache.interface_layer["skill_code"]
        if skill_interface.kv_cache is not None:
            kv_list.append(skill_interface.kv_cache)
            if hasattr(skill_interface.kv_cache, 'key_cache'):
                kv_length += skill_interface.kv_cache.key_cache[0].shape[-2]
                # Check if cache was already on GPU (cache hit)
                if CATP_AGENT.cache.cache_on_gpu.get("skill_code", False):
                    cache_hits += 1
                total_caches_needed += 1
    
    # Store initial GPU status before retrieval (for cache hit calculation)
    initial_gpu_status = {}
    for func_id, _, _ in retrieved_interfaces:
        initial_gpu_status[func_id] = CATP_AGENT.cache.cache_on_gpu.get(func_id, False)
    
    # 2) Add top-k retrieved examples from batch retrieval
    for func_id, kv_cache, ppl in retrieved_interfaces:
        ppl_scores[func_id] = ppl  # Store PPL score
        selected_func_ids.append(func_id)
        
        if func_id != "skill_code" and func_id != "guidelines":
            if kv_cache is not None:
                kv_list.append(kv_cache)
                if hasattr(kv_cache, 'key_cache'):
                    kv_length += kv_cache.key_cache[0].shape[-2]
                    # Check if cache was already on GPU before retrieval (cache hit)
                    if initial_gpu_status.get(func_id, False):
                        cache_hits += 1
                    total_caches_needed += 1
    
    # 3) Add runtime guidelines
    if "guidelines" in CATP_AGENT.cache.interface_layer:
        guidelines_interface = CATP_AGENT.cache.interface_layer["guidelines"]
        if guidelines_interface.kv_cache is not None:
            kv_list.append(guidelines_interface.kv_cache)
            if hasattr(guidelines_interface.kv_cache, 'key_cache'):
                kv_length += guidelines_interface.kv_cache.key_cache[0].shape[-2]
                # Check if cache was already on GPU (cache hit)
                if CATP_AGENT.cache.cache_on_gpu.get("guidelines", False):
                    cache_hits += 1
                total_caches_needed += 1
    
    retrieval_time = time.time() - retrieval_start
    
    # Calculate cache hit ratio
    cache_hit_ratio = cache_hits / max(total_caches_needed, 1)
    
    # Get memory footprint before generation
    memory_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    # Concatenate KV caches
    if kv_list:
        full_kv = DynamicCache()
        num_layers = len(kv_list[0].key_cache)
        
        # Get target device (should be GPU)
        device = code_llm.model.model.embed_tokens.weight.device
        
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            
            for kv in kv_list:
                if layer_idx < len(kv.key_cache):
                    # Ensure all tensors are on the same device before concatenation
                    layer_keys.append(kv.key_cache[layer_idx].to(device))
                    layer_values.append(kv.value_cache[layer_idx].to(device))
            
            if layer_keys:
                concatenated_keys = torch.cat(layer_keys, dim=2)
                concatenated_values = torch.cat(layer_values, dim=2)
                full_kv.key_cache.append(concatenated_keys)
                full_kv.value_cache.append(concatenated_values)
    else:
        full_kv = None
    
    # Clean up KV cache to original length
    if full_kv is not None:
        clean_up(full_kv, kv_length)
    
    # Generate using custom generation loop (like alf_teach_epic)
    generation_start = time.time()
    
    # Encode prompt
    device = code_llm.model.model.embed_tokens.weight.device
    input_ids = code_llm.tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate with KV cache
    output_ids = generate_with_kv_cache(
        code_llm.model, code_llm.tokenizer, input_ids, full_kv,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    
    generation_time = time.time() - generation_start
    
    # Get memory footprint after generation
    memory_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    memory_footprint = memory_after - memory_before
    
    # Decode
    generated_text = code_llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Store generated code temporarily (will be saved to cache only on success confirmation)
    gen_id = None
    if CATP_AGENT and generated_text and len(generated_text.strip()) > 50:  # Only save meaningful outputs
        try:
            # Generate unique ID based on instruction hash
            import hashlib
            gen_id = f"gen_{hashlib.md5(instruction.encode()).hexdigest()[:8]}"
            
            # Store in pending cache (not yet confirmed)
            if not hasattr(CATP_AGENT, 'pending_cache'):
                CATP_AGENT.pending_cache = {}
            
            # Prepare cache data but don't add yet
            CATP_AGENT.pending_cache[gen_id] = {
                'instruction': instruction[:200],
                'generated_text': generated_text,
                'timestamp': time.time()
            }
            
            print(f"[Cache] Prepared generated code for potential caching: {gen_id}")
        except Exception as e:
            print(f"[Cache] Failed to prepare generated code: {e}")
    
    # Get cache statistics
    cache_stats = CATP_AGENT.cache.get_cache_stats()
    
    # Print PPL scores and locality info
    if ppl_scores:
        print(f"[PPL Scores] Top-{len(ppl_scores)} retrieved:")
        for idx, (func_id, ppl) in enumerate(ppl_scores.items(), 1):
            # Get locality score for this cache
            locality_score = CATP_AGENT.cache.locality_scores.get(func_id, 0.0)
            usage_count = CATP_AGENT.cache.usage_count.get(func_id, 0)
            on_gpu = CATP_AGENT.cache.cache_on_gpu.get(func_id, False)
            location = "GPU" if on_gpu else "CPU"
            print(f"  {idx}. {func_id}: PPL={ppl:.2f}, Locality={locality_score:.3f}, Usage={usage_count}, Loc={location}")
        print(f"[Cache Stats] GPU: {cache_stats['gpu_caches']}/{cache_stats['max_gpu_capacity']}, "
              f"CPU: {cache_stats['cpu_caches']}, Hit Rate: {cache_hit_ratio:.2%}")
    
    # Get locality metrics
    locality_scores_list = [CATP_AGENT.cache.locality_scores.get(fid, 0.0) 
                            for fid in selected_func_ids]
    avg_locality = sum(locality_scores_list) / len(locality_scores_list) if locality_scores_list else 0
    
    # Log metrics to wandb
    if wandb and wandb.run is not None:
        wandb.log({
            "cache_hit_ratio": cache_hit_ratio,
            "memory_footprint_gb": memory_footprint,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_inference_time": retrieval_time + generation_time,
            "kv_cache_tokens": kv_length,
            "num_caches_used": total_caches_needed,
            "tokens_generated": len(output_ids[0]),
            "avg_ppl_score": sum(ppl_scores.values()) / len(ppl_scores) if ppl_scores else 0,
            "min_ppl_score": min(ppl_scores.values()) if ppl_scores else 0,
            "max_ppl_score": max(ppl_scores.values()) if ppl_scores else 0,
            "avg_locality_score": avg_locality,
            "min_locality_score": min(locality_scores_list) if locality_scores_list else 0,
            "max_locality_score": max(locality_scores_list) if locality_scores_list else 0,
        })
    
    # Collect locality info for each cache
    locality_info = {}
    for func_id in selected_func_ids:
        locality_info[func_id] = {
            'locality_score': CATP_AGENT.cache.locality_scores.get(func_id, 0.0),
            'usage_count': CATP_AGENT.cache.usage_count.get(func_id, 0),
            'last_access': CATP_AGENT.cache.last_access_time.get(func_id, 0)
        }
    
    return {
        'text': generated_text,
        'tokens': len(output_ids[0]),
        'retrieval_time': retrieval_time,
        'generation_time': generation_time,
        'kv_cache_tokens': kv_length,
        'recomputed_tokens': 0,
        'ppl_scores': ppl_scores,
        'cache_hit_ratio': cache_hit_ratio,
        'memory_footprint_gb': memory_footprint,
        'selected_caches': selected_func_ids,
        'locality_info': locality_info,
        'avg_locality_score': avg_locality,
        'cache_id': gen_id  # Include cache ID for success confirmation
    }


def generate_with_kv_cache(model, tokenizer, input_ids: torch.Tensor, past_key_values, 
                           max_new_tokens: int = 1024, temperature: float = 0.0) -> torch.Tensor:
    """
    Generate text with KV cache using model.generate()
    """
    device = model.model.embed_tokens.weight.device
    
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Calculate the length of past_key_values if present
    past_length = 0
    if past_key_values is not None and hasattr(past_key_values, 'key_cache') and len(past_key_values.key_cache) > 0:
        past_length = past_key_values.key_cache[0].shape[-2]
        # Extend attention mask for past_key_values
        attention_mask = torch.cat([
            torch.ones((input_ids.shape[0], past_length), dtype=torch.long, device=device),
            attention_mask
        ], dim=1)

    with torch.no_grad():
        # Use simpler generation parameters to avoid cache position issues
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        
        if temperature > 0:
            generation_config["temperature"] = temperature
        
        # Only pass past_key_values if it's not empty
        if past_key_values is not None and len(past_key_values.key_cache) > 0:
            generation_config["past_key_values"] = past_key_values
            
        output_ids = model.generate(
            input_ids=input_ids,
            **generation_config
        )

    gen_ids = output_ids[:, origin_len:]
    return gen_ids


def clean_up(kv: DynamicCache, origin_len: int):
    """Truncate the KV Cache to the original length (same as alf_teach_epic)"""
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

def parse_llm_output(text):
    """Parse LLM output (same as alf_teach_epic)"""
    if not text or not text.strip():
        return text
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('. ', 1)
        if len(parts) > 1 and parts[0].isdigit():
            lines.append(parts[1])
        else:
            lines.append(line)
    return '\n'.join(lines) if lines else text

# --- Endpoints (Same as alf_teach_epic) ---
@app.route("/generate_catp", methods=["POST"])
def generate_cache_endpoint():
    """Generate using CATP (compatible with alf_teach_epic)"""
    if not request.is_json:
        return jsonify({"error": "Request format is not JSON."}), 400
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'."}), 400

    try:
        if CATP_AGENT is None:
            return jsonify({"error": "CATP not initialized."}), 500

        start_time = time.time()
        generation_prompt = f"\n{prompt}\n\n# Generated plan:\n"
        
        # Use CATP generation
        result = generate_with_catp(
            generation_prompt,
            max_new_tokens=256,
            temperature=0.0
        )
        
        result_text = parse_llm_output(result['text'])
        total_time = time.time() - start_time

        response = jsonify({
            "program": result_text,
            "tokens_generated": result['tokens'],
            "inference_time": total_time,
            "retrieval_time": result['retrieval_time'],
            "generation_time": result['generation_time'],
            "kv_cache_tokens": result['kv_cache_tokens'],
            "cache_status": "catp",
            "cache_info": CATP_INFO,
            "ppl_scores": result.get('ppl_scores', {}),
            "cache_hit_ratio": result.get('cache_hit_ratio', 0),
            "memory_footprint_gb": result.get('memory_footprint_gb', 0),
            "selected_caches": result.get('selected_caches', []),
            "locality_info": result.get('locality_info', {}),
            "avg_locality_score": result.get('avg_locality_score', 0),
            "cache_id": result.get('cache_id',0)  # Include cache_id for success confirmation
        })
        
        # Clean up memory after request
        torch.cuda.empty_cache()
        
        return response

    except Exception as e:
        error_print(f"Error in generation: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate_sub_rag", methods=["POST"])
def generate_sub_task_fast():
    """Generate subtask directly without caching"""
    if not request.is_json:
        return jsonify({"error": "Request format is not JSON."}), 400
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Missing 'prompt'."}), 400
    if code_llm is None or code_llm.model is None:
        return jsonify({"error": "Model not loaded."}), 500

    start_time = time.time()
    sub_prompt = f"\n{prompt}\n"
    
    # Directly use model.generate() without caching
    device = code_llm.model.model.embed_tokens.weight.device
    input_ids = code_llm.tokenizer.encode(sub_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = code_llm.model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=None,
            do_sample=False,
            pad_token_id=code_llm.tokenizer.pad_token_id if code_llm.tokenizer.pad_token_id is not None else code_llm.tokenizer.eos_token_id
        )
    
    generated_text = code_llm.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    result_text = parse_llm_output(generated_text)
    total_time = time.time() - start_time
    
    # response = jsonify({
    #     "program": result_text,
    #     "tokens_generated": len(output_ids[0]) - input_ids.shape[-1],
    #     "inference_time": total_time,
    #     "generation_time": total_time,
    # })
    response = jsonify({
        "program": result_text,
        "tokens_generated": len(output_ids[0]) - input_ids.shape[-1],
        "inference_time": total_time,
        "retrieval_time": 0,
        "generation_time": total_time,
        "recomputed_tokens": 0,
        "ppl_scores": 0,
        "cache_hit_ratio":0,
        "memory_footprint_gb": 0,
        "selected_caches": 0,
        })
    # Clean up memory after request
    torch.cuda.empty_cache()
    
    return response

@app.route("/get_prompt_plan_catp", methods=["POST"])
def get_prompt_plan_endpoint():
    """Get prompt for plan generation (compatible endpoint)"""
    data = request.get_json()
    command = data.get("command", "")
    cache_key = "PLAN:" + hashlib.md5(command.encode()).hexdigest()[:16]
    return jsonify({"final_prompt": f"Generate a plan for:\n{command}\n", "cache_key": cache_key})

@app.route("/get_prompt_replan_catp", methods=["POST"])
def get_prompt_replan_endpoint():
    """Get prompt for replanning (compatible endpoint)"""
    data = request.get_json()
    command = data.get("command", "")
    cache_key = "REPLAN:" + hashlib.md5(command.encode()).hexdigest()[:16]
    return jsonify({"final_prompt": f"Fix the plan for:\n{command}\n", "cache_key": cache_key})

@app.route("/cache_stats", methods=["GET"])
def get_cache_stats():
    """Get CATP cache statistics"""
    stats = {
        "catp_cache": {
            "enabled": CATP_AGENT is not None,
            "interfaces_cached": len(CATP_AGENT.cache.interface_layer) if CATP_AGENT else 0,
            "codes_cached": len(CATP_AGENT.cache.code_layer) if CATP_AGENT else 0,
            "total_functions": CATP_INFO.get("total_functions", 0),
            "built_in_seconds": CATP_INFO.get("build_seconds", None),
        }
    }
    
    # Add GPU/CPU cache distribution
    if CATP_AGENT:
        cache_dist = CATP_AGENT.cache.get_cache_stats()
        stats["catp_cache"].update({
            "gpu_caches": cache_dist['gpu_caches'],
            "cpu_caches": cache_dist['cpu_caches'],
            "gpu_utilization": cache_dist['gpu_utilization'],
            "max_gpu_capacity": cache_dist['max_gpu_capacity'],
            "total_capacity": cache_dist['max_total_capacity']
        })
    
    try:
        stats["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
    except Exception:
        pass
    return jsonify(stats)

@app.route("/reload_cache", methods=["POST"])
def reload_cache():
    """Reload CATP cache from file"""
    try:
        cache_path = "./cache/catp_kv_cache.pt"
        if os.path.exists(cache_path):
            cache_data = torch.load(cache_path, map_location=DEVICE, weights_only=False)
            
            if CATP_AGENT is None:
                initialize_catp()
                
            CATP_AGENT.cache.interface_layer = cache_data.get('interface_layer', {})
            CATP_AGENT.cache.code_layer = cache_data.get('code_layer', {})
            
            global CATP_INFO
            CATP_INFO = cache_data.get('info', {})
            
            return jsonify({
                "status": "Cache reloaded",
                "interfaces": len(CATP_AGENT.cache.interface_layer),
                "codes": len(CATP_AGENT.cache.code_layer)
            })
        else:
            return jsonify({"error": "Cache file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/confirm_success", methods=["POST"])
def confirm_success():
    """Confirm successful execution of generated code and save to cache"""
    if not request.is_json:
        return jsonify({"error": "Request format is not JSON."}), 400
    
    data = request.get_json()
    cache_id = data.get("cache_id")
    success = data.get("success", True)
    
    if not cache_id:
        return jsonify({"error": "Missing 'cache_id'."}), 400
    
    if CATP_AGENT is None:
        return jsonify({"error": "CATP not initialized."}), 500
    
    # Check if we have pending cache for this ID
    if not hasattr(CATP_AGENT, 'pending_cache'):
        CATP_AGENT.pending_cache = {}
    
    if cache_id not in CATP_AGENT.pending_cache:
        return jsonify({"error": f"No pending cache found for ID: {cache_id}"}), 404
    
    if success:
        try:
            # Get pending cache data
            cache_data = CATP_AGENT.pending_cache[cache_id]
            
            # Compute KV cache for the generated code
            gen_kv = preprocess_knowledge(code_llm.model, code_llm.tokenizer, cache_data['generated_text'])
            
            # Add to permanent cache
            CATP_AGENT.cache.add_function(
                function_id=cache_id,
                interface_text=cache_data['instruction'],
                code_text=cache_data['generated_text'],
                interface_kv=gen_kv,
                code_kv=gen_kv,
                perplexity=3.0  # Lower perplexity for successful code
            )
            
            # Remove from pending
            del CATP_AGENT.pending_cache[cache_id]
            
            print(f"[Cache] Successfully added {cache_id} to permanent cache")
            print(f"[Cache] Total cache size: {len(CATP_AGENT.cache.interface_layer)}")
            
            # Save cache to disk periodically
            if len(CATP_AGENT.cache.interface_layer) % 5 == 0:
                save_cache_to_disk()
            
            return jsonify({
                "status": "success",
                "message": f"Code {cache_id} added to cache",
                "cache_size": len(CATP_AGENT.cache.interface_layer)
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to add to cache: {str(e)}"}), 500
    else:
        # Remove from pending if execution failed
        if cache_id in CATP_AGENT.pending_cache:
            del CATP_AGENT.pending_cache[cache_id]
            print(f"[Cache] Removed failed code {cache_id} from pending")
        
        return jsonify({
            "status": "removed",
            "message": f"Code {cache_id} removed from pending"
        })

@app.route("/cleanup_pending", methods=["POST"])
def cleanup_pending():
    """Clean up old pending caches that were never confirmed"""
    if CATP_AGENT is None:
        return jsonify({"error": "CATP not initialized."}), 500
    
    if not hasattr(CATP_AGENT, 'pending_cache'):
        return jsonify({"message": "No pending caches to clean"}), 200
    
    current_time = time.time()
    timeout = 3600  # 1 hour timeout for pending caches
    
    cleaned = []
    for cache_id, cache_data in list(CATP_AGENT.pending_cache.items()):
        if current_time - cache_data['timestamp'] > timeout:
            del CATP_AGENT.pending_cache[cache_id]
            cleaned.append(cache_id)
    
    print(f"[Cache] Cleaned {len(cleaned)} old pending caches")
    
    return jsonify({
        "cleaned": len(cleaned),
        "remaining": len(CATP_AGENT.pending_cache),
        "cache_ids": cleaned
    })

@app.route("/search_examples", methods=["POST"])
def search_examples():
    """Search for relevant examples using CATP's retrieval"""
    if not request.is_json:
        return jsonify({"error": "Request format is not JSON."}), 400
    
    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 3)
    
    if not query:
        return jsonify({"error": "Missing 'query'."}), 400
    
    if CATP_AGENT is None:
        return jsonify({"error": "CATP not initialized."}), 500
    
    try:
        # Use CATP's retrieval mechanism
        interfaces = CATP_AGENT.cache.retrieve_interfaces(
            instruction=query,
            perplexity_threshold=20.0,  # Higher threshold for search
            model=code_llm
        )
        
        # Format matches
        matches = []
        for func_id, _ in interfaces[:top_k]:
            if func_id in CATP_AGENT.cache.interface_layer:
                interface = CATP_AGENT.cache.interface_layer[func_id]
                matches.append({
                    "name": func_id,
                    "similarity": 1.0 / (1.0 + interface.perplexity)  # Convert perplexity to similarity
                })
        
        return jsonify({"matches": matches})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Main ---
if __name__ == "__main__":
    try:
        wandb.init(project="server_final_alfred", job_type="inference_server_catp")
        info_print("wandb initialized (with CATP)")
    except Exception as e:
        error_print(f"wandb init failed: {e}")

    # 1) Load resources
    load_resources()

    # 2) Initialize CATP
    if not initialize_catp():
        error_print("Failed to initialize CATP. Exiting.")
        raise SystemExit(1)

    # 3) Try to load cache from file first, if not available, build it
    cache_path = "./cache/catp_kv_cache.pt"
    
    if os.path.exists(cache_path):
        try:
            cache_data = torch.load(cache_path, map_location=DEVICE, weights_only=False)
            CATP_AGENT.cache.interface_layer = cache_data.get('interface_layer', {})
            CATP_AGENT.cache.code_layer = cache_data.get('code_layer', {})
            CATP_INFO = cache_data.get('info', {})
            
            # Initialize all loaded caches as being on GPU
            for func_id in CATP_AGENT.cache.interface_layer:
                # Mark as on GPU since we just loaded them
                CATP_AGENT.cache.cache_on_gpu[func_id] = True
                # Initialize usage stats if not present
                if func_id not in CATP_AGENT.cache.usage_count:
                    CATP_AGENT.cache.usage_count[func_id] = 0
                if func_id not in CATP_AGENT.cache.last_access_time:
                    CATP_AGENT.cache.last_access_time[func_id] = 0
                if func_id not in CATP_AGENT.cache.locality_scores:
                    CATP_AGENT.cache._update_locality_score(func_id)
            
            # Now manage GPU memory to offload excess caches if needed
            CATP_AGENT.cache._manage_gpu_memory()
            
            info_print(f"Loaded CATP KV cache from file: {cache_path}")
        except Exception as e:
            error_print(f"Failed to load cache: {e}")
            built = build_catp_cache_from_examples()
            if not built:
                error_print("Failed to build CATP cache.")
    else:
        # Build cache from examples
        built = build_catp_cache_from_examples()
        if not built:
            error_print("Failed to build CATP cache.")

    # 4) Print stats
    info_print(f"CATP ready - Interfaces: {len(CATP_AGENT.cache.interface_layer)}, "
              f"Codes: {len(CATP_AGENT.cache.code_layer)}")
    try:
        info_print(f"GPU memory used: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    except Exception:
        pass

    app.run(host="0.0.0.0", port=5000)
