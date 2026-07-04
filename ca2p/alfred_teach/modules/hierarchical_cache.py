import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict
import ast


@dataclass
class FunctionInterface:
    """Function interface with signature and metadata"""
    function_id: str
    interface_text: str
    kv_cache: Optional[torch.Tensor] = None
    perplexity: float = 0.0
    
    
@dataclass
class FunctionCode:
    """Function implementation with code and KV cache"""
    function_id: str
    code_text: str
    kv_cache: Optional[torch.Tensor] = None


class HierarchicalCodeCache:
    """
    Hierarchical code cache with two layers:
    1. Function-Interface layer (I): stores signatures and semantic metadata
    2. Function-Code layer (C): stores validated implementations
    """
    
    def __init__(self, max_cache_size: int = 100, device: str = 'cuda', max_gpu_cache_size: int = 15):
        """
        Initialize the hierarchical code cache
        
        Args:
            max_cache_size: Maximum number of functions to cache (CPU + GPU)
            device: Device for KV cache tensors
            max_gpu_cache_size: Maximum number of caches to keep on GPU
        """
        self.max_cache_size = max_cache_size
        self.max_gpu_cache_size = max_gpu_cache_size
        
        # Ensure device is valid
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')  # Use first available GPU
        elif device.startswith('cuda:'):
            try:
                self.device = torch.device(device)
            except:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        # Function-Interface layer
        self.interface_layer: OrderedDict[str, FunctionInterface] = OrderedDict()
        
        # Function-Code layer
        self.code_layer: OrderedDict[str, FunctionCode] = OrderedDict()
        
        # Track cache device locations (True = GPU, False = CPU)
        self.cache_on_gpu: Dict[str, bool] = {}
        
        # Locality scores for cache management
        self.locality_scores: Dict[str, float] = {}
        
        # Weights for locality score computation
        self.w_temporal = 0.4
        self.w_spatial = 0.3
        self.w_semantic = 0.3
        
        # Usage statistics for temporal locality
        self.usage_count: Dict[str, int] = {}
        self.last_access_time: Dict[str, int] = {}
        self.current_time = 0
        
    def add_function(self, 
                     function_id: str,
                     interface_text: str,
                     code_text: str,
                     interface_kv: Optional[torch.Tensor] = None,
                     code_kv: Optional[torch.Tensor] = None,
                     perplexity: float = 0.0) -> None:
        """
        Add a function to both cache layers
        
        Args:
            function_id: Unique identifier for the function
            interface_text: Function signature and docstring
            code_text: Full function implementation
            interface_kv: Precomputed KV cache for interface
            code_kv: Precomputed KV cache for code
            perplexity: Perplexity score for semantic relevance
        """
        # Check cache capacity and evict if necessary
        if len(self.interface_layer) >= self.max_cache_size:
            self._evict_lowest_locality()
            
        # Add to interface layer
        self.interface_layer[function_id] = FunctionInterface(
            function_id=function_id,
            interface_text=interface_text,
            kv_cache=interface_kv,
            perplexity=perplexity
        )
        
        # Add to code layer
        self.code_layer[function_id] = FunctionCode(
            function_id=function_id,
            code_text=code_text,
            kv_cache=code_kv
        )
        
        # Mark as on GPU initially
        self.cache_on_gpu[function_id] = True if interface_kv is not None else False
        
        # Check GPU cache capacity and offload if needed
        self._manage_gpu_memory()
        
        # Initialize usage statistics
        self.usage_count[function_id] = 1
        self.last_access_time[function_id] = self.current_time
        self.current_time += 1
        
        # Update locality score
        self._update_locality_score(function_id)
        
    def retrieve_interfaces(self, 
                           instruction: str,
                           perplexity_threshold: float = 10.0,
                           model=None) -> List[Tuple[str, torch.Tensor]]:
        """
        Retrieve relevant function interfaces based on perplexity
        
        Args:
            instruction: Task instruction
            perplexity_threshold: Maximum perplexity for selection
            model: Language model for computing perplexity
            
        Returns:
            List of (function_id, kv_cache) tuples
        """
        selected_interfaces = []
        
        for func_id, interface in self.interface_layer.items():
            # Compute perplexity if model is provided
            if model is not None:
                perplexity = self._compute_perplexity(
                    interface.interface_text, instruction, model
                )
                interface.perplexity = perplexity
            else:
                perplexity = interface.perplexity
                
            # Select interfaces below threshold
            if perplexity <= perplexity_threshold:
                selected_interfaces.append((func_id, interface.kv_cache))
                
                # Update access time
                self.last_access_time[func_id] = self.current_time
                self.usage_count[func_id] += 1
                self.current_time += 1
                
        return selected_interfaces
    
    def retrieve_interfaces_batch(self, 
                                  instruction: str,
                                  perplexity_threshold: float = 10.0,
                                  model=None,
                                  top_k: int = 3) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Retrieve relevant function interfaces using batch perplexity computation
        
        Args:
            instruction: Task instruction
            perplexity_threshold: Maximum perplexity for selection (optional if using top_k)
            model: Language model for computing perplexity
            top_k: Number of top interfaces to retrieve
            
        Returns:
            List of (function_id, kv_cache, perplexity) tuples sorted by perplexity
        """
        if not self.interface_layer:
            return []
        
        # Collect all interfaces and their IDs
        func_ids = []
        interface_texts = []
        
        for func_id, interface in self.interface_layer.items(): 
            func_ids.append(func_id)
            # Truncate to reasonable length for perplexity computation
            text = interface.interface_text[:500] if len(interface.interface_text) > 500 else interface.interface_text
            interface_texts.append(text)
        
        # Compute perplexity in batch
        if model is not None and hasattr(model, 'compute_perplexity_batch'):
            # Use batch computation if available
            # More aggressive batch size reduction based on memory usage
            allocated_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            

            batch_size = 2

            print(f"[Memory] Using batch_size={batch_size} (Allocated: {allocated_gb:.1f}GB)")
            perplexities = model.compute_perplexity_batch(interface_texts, instruction, batch_size=batch_size)
        elif model is not None:
            # Fall back to sequential computation
            perplexities = []
            for text in interface_texts:
                ppl = self._compute_perplexity(text, instruction, model)
                perplexities.append(ppl)
        else:
            # Use cached perplexities
            perplexities = [self.interface_layer[fid].perplexity for fid in func_ids]
        
        # Combine results and sort by perplexity
        results = []
        for func_id, ppl in zip(func_ids, perplexities):
            interface = self.interface_layer[func_id]
            interface.perplexity = ppl  # Update cached perplexity
            results.append((func_id, interface.kv_cache, ppl))
        
        # Sort by perplexity (lower is better)
        results.sort(key=lambda x: x[2])
        
        # Select top-k or those below threshold
        if top_k > 0:
            selected = results[:top_k]
        else:
            selected = [(fid, kv, ppl) for fid, kv, ppl in results if ppl <= perplexity_threshold]
        
        # Update access statistics and ensure selected caches are on GPU
        for func_id, _, _ in selected:
            self.last_access_time[func_id] = self.current_time
            self.usage_count[func_id] = self.usage_count.get(func_id, 0) + 1
            self.current_time += 1
            self._update_locality_score(func_id)
            
            # Move to GPU if currently on CPU
            if func_id in self.cache_on_gpu and not self.cache_on_gpu[func_id]:
                self._move_to_gpu(func_id)
                # After moving to GPU, manage GPU memory
                self._manage_gpu_memory()
        
        return selected
        
    def retrieve_code(self, function_id: str) -> Optional[str]:
        """
        Retrieve function code by ID
        
        Args:
            function_id: Function identifier
            
        Returns:
            Function code text or None if not found
        """
        if function_id in self.code_layer:
            # Update access statistics
            self.last_access_time[function_id] = self.current_time
            self.usage_count[function_id] += 1
            self.current_time += 1
            
            return self.code_layer[function_id].code_text
        return None
        
    def retrieve_code_kv(self, function_id: str) -> Optional[torch.Tensor]:
        """
        Retrieve function code KV cache for editing
        
        Args:
            function_id: Function identifier
            
        Returns:
            Function code KV cache or None if not found
        """
        if function_id in self.code_layer:
            return self.code_layer[function_id].kv_cache
        return None
        
    def extract_functions_from_code(self, code: str) -> List[Tuple[str, str, str]]:
        """
        Extract function definitions from code
        
        Args:
            code: Python code containing function definitions
            
        Returns:
            List of (function_id, interface_text, code_text) tuples
        """
        functions = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function ID
                    func_id = node.name
                    
                    # Extract interface (signature + docstring)
                    interface_lines = []
                    
                    # Function signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    
                    signature = f"def {func_id}({', '.join(args)})"
                    if node.returns:
                        signature += f" -> {ast.unparse(node.returns)}"
                    signature += ":"
                    interface_lines.append(signature)
                    
                    # Docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        interface_lines.append(f'    """{docstring}"""')
                        
                    interface_text = '\n'.join(interface_lines)
                    
                    # Extract full function code
                    code_text = ast.unparse(node)
                    
                    functions.append((func_id, interface_text, code_text))
                    
        except SyntaxError:
            pass
            
        return functions
        
    def update_from_executed_code(self, executed_code: str, model=None) -> None:
        """
        Update cache with successfully executed functions
        
        Args:
            executed_code: Successfully executed policy code
            model: Language model for KV cache computation
        """
        functions = self.extract_functions_from_code(executed_code)
        
        for func_id, interface_text, code_text in functions:
            # Compute KV caches if model is provided
            interface_kv = None
            code_kv = None
            
            if model is not None:
                interface_kv = self._compute_kv_cache(interface_text, model)
                code_kv = self._compute_kv_cache(code_text, model)
                
            self.add_function(
                function_id=func_id,
                interface_text=interface_text,
                code_text=code_text,
                interface_kv=interface_kv,
                code_kv=code_kv
            )
            
    def _compute_perplexity(self, text: str, context: str, model) -> float:
        """
        Compute perplexity of text given context using the model
        
        Args:
            text: Text to compute perplexity for (interface)
            context: Context (instruction)
            model: Language model wrapper
            
        Returns:
            Perplexity score
        """
        if model is None or not hasattr(model, 'compute_perplexity'):
            # Fallback to random for testing
            return np.random.uniform(1, 20)
            
        # Use the model's compute_perplexity method
        # PPL(interface | instruction) - lower is better
        return model.compute_perplexity(text, context)
        
    def _compute_kv_cache(self, text: str, model) -> torch.Tensor:
        """
        Compute KV cache for text using the model
        
        Args:
            text: Text to compute KV cache for
            model: Language model
            
        Returns:
            KV cache tensor
        """
        # This is a placeholder - actual implementation depends on the model
        # In practice, this would extract KV states from the model's attention layers
        dummy_kv = torch.randn(1, 512, 768).to(self.device)  # Placeholder dimensions
        return dummy_kv
        
    def _update_locality_score(self, function_id: str) -> None:
        """
        Update composite locality score for a function
        
        Args:
            function_id: Function identifier
        """
        # Temporal locality (recency of use)
        time_diff = self.current_time - self.last_access_time.get(function_id, 0)
        l_temporal = np.exp(-0.1 * time_diff)  # Exponential decay
        
        # Spatial locality (frequency of use)
        usage = self.usage_count.get(function_id, 1)
        l_spatial = min(1.0, usage / 10.0)  # Normalize to [0, 1]
        
        # Semantic locality (based on perplexity)
        if function_id in self.interface_layer:
            perplexity = self.interface_layer[function_id].perplexity
            l_semantic = 1.0 / (1.0 + perplexity / 10.0)  # Lower perplexity = higher score
        else:
            l_semantic = 0.5
            
        # Composite score
        score = (self.w_temporal * l_temporal +
                self.w_spatial * l_spatial +
                self.w_semantic * l_semantic)
        
        self.locality_scores[function_id] = score
        
    def _move_to_cpu(self, function_id: str) -> None:
        """
        Move a cache from GPU to CPU memory
        """
        if function_id not in self.cache_on_gpu or not self.cache_on_gpu[function_id]:
            return  # Already on CPU
        
        # Move interface KV cache to CPU
        if function_id in self.interface_layer:
            interface = self.interface_layer[function_id]
            if interface.kv_cache is not None:
                if hasattr(interface.kv_cache, 'key_cache'):
                    # Move DynamicCache to CPU
                    for i in range(len(interface.kv_cache.key_cache)):
                        interface.kv_cache.key_cache[i] = interface.kv_cache.key_cache[i].cpu()
                        interface.kv_cache.value_cache[i] = interface.kv_cache.value_cache[i].cpu()
                else:
                    # Move regular tensor to CPU
                    interface.kv_cache = interface.kv_cache.cpu()
        
        # Move code KV cache to CPU
        if function_id in self.code_layer:
            code = self.code_layer[function_id]
            if code.kv_cache is not None:
                if hasattr(code.kv_cache, 'key_cache'):
                    # Move DynamicCache to CPU
                    for i in range(len(code.kv_cache.key_cache)):
                        code.kv_cache.key_cache[i] = code.kv_cache.key_cache[i].cpu()
                        code.kv_cache.value_cache[i] = code.kv_cache.value_cache[i].cpu()
                else:
                    # Move regular tensor to CPU
                    code.kv_cache = code.kv_cache.cpu()
        
        self.cache_on_gpu[function_id] = False
        print(f"[Cache] Offloaded {function_id} to CPU memory")
        
        # Clear CUDA cache to free up memory immediately
        torch.cuda.empty_cache()
    
    def _move_to_gpu(self, function_id: str) -> None:
        """
        Move a cache from CPU to GPU memory
        """
        if function_id not in self.cache_on_gpu or self.cache_on_gpu[function_id]:
            return  # Already on GPU
        
        # Move interface KV cache to GPU
        if function_id in self.interface_layer:
            interface = self.interface_layer[function_id]
            if interface.kv_cache is not None:
                if hasattr(interface.kv_cache, 'key_cache'):
                    # Move DynamicCache to GPU
                    for i in range(len(interface.kv_cache.key_cache)):
                        interface.kv_cache.key_cache[i] = interface.kv_cache.key_cache[i].to(self.device)
                        interface.kv_cache.value_cache[i] = interface.kv_cache.value_cache[i].to(self.device)
                else:
                    # Move regular tensor to GPU
                    interface.kv_cache = interface.kv_cache.to(self.device)
        
        # Move code KV cache to GPU
        if function_id in self.code_layer:
            code = self.code_layer[function_id]
            if code.kv_cache is not None:
                if hasattr(code.kv_cache, 'key_cache'):
                    # Move DynamicCache to GPU
                    for i in range(len(code.kv_cache.key_cache)):
                        code.kv_cache.key_cache[i] = code.kv_cache.key_cache[i].to(self.device)
                        code.kv_cache.value_cache[i] = code.kv_cache.value_cache[i].to(self.device)
                else:
                    # Move regular tensor to GPU
                    code.kv_cache = code.kv_cache.to(self.device)
        
        self.cache_on_gpu[function_id] = True
        print(f"[Cache] Loaded {function_id} to GPU memory")
    
    def _manage_gpu_memory(self) -> None:
        """
        Manage GPU memory by offloading caches with lowest locality scores
        """
        # Count caches on GPU
        gpu_caches = [fid for fid, on_gpu in self.cache_on_gpu.items() if on_gpu]
        
        if len(gpu_caches) <= self.max_gpu_cache_size:
            return  # Within GPU capacity
        
        # Sort by locality score and offload lowest scoring ones
        sorted_caches = sorted(gpu_caches, key=lambda x: self.locality_scores.get(x, 0))
        
        # Offload excess caches to CPU
        num_to_offload = len(gpu_caches) - self.max_gpu_cache_size
        for func_id in sorted_caches[:num_to_offload]:
            self._move_to_cpu(func_id)
            
    def offload_lowest_locality_cache(self) -> bool:
        """
        Offload the cache with the lowest locality score from GPU to CPU
        Called when OOM occurs during perplexity computation
        
        Returns:
            True if a cache was offloaded, False otherwise
        """
        # Find caches currently on GPU
        gpu_caches = [fid for fid, on_gpu in self.cache_on_gpu.items() if on_gpu]
        
        if not gpu_caches:
            print("[OOM Handler] No caches on GPU to offload")
            # Try to clear cache anyway
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            return False
            
        # Sort by locality score (ascending) - lowest locality first  
        sorted_caches = sorted(gpu_caches, key=lambda x: self.locality_scores.get(x, 0.5))
        
        # Check memory pressure and offload accordingly
        allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        if allocated > 22:  # Critical - offload more aggressively
            num_to_offload = min(3, len(sorted_caches))
        elif allocated > 20:  # High pressure
            num_to_offload = min(2, len(sorted_caches))
        else:
            num_to_offload = 1
            
        offloaded = []
        
        for i in range(num_to_offload):
            func_id_to_offload = sorted_caches[i]
            locality = self.locality_scores.get(func_id_to_offload, 0.5)
            print(f"[OOM Handler] Offloading {func_id_to_offload} (locality={locality:.3f}) to CPU due to OOM")
            self._move_to_cpu(func_id_to_offload)
            offloaded.append(func_id_to_offload)
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Log memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[OOM Handler] After offloading {len(offloaded)} caches: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        return True
    
    def _evict_lowest_locality(self) -> None:
        """
        Evict the function with the lowest locality score (complete removal, not offloading)
        """
        if not self.locality_scores:
            return
            
        # Find function with lowest score
        min_func_id = min(self.locality_scores, key=self.locality_scores.get)
        
        # Remove from all layers and tracking
        if min_func_id in self.interface_layer:
            del self.interface_layer[min_func_id]
        if min_func_id in self.code_layer:
            del self.code_layer[min_func_id]
        if min_func_id in self.locality_scores:
            del self.locality_scores[min_func_id]
        if min_func_id in self.usage_count:
            del self.usage_count[min_func_id]
        if min_func_id in self.last_access_time:
            del self.last_access_time[min_func_id]
        if min_func_id in self.cache_on_gpu:
            del self.cache_on_gpu[min_func_id]
            
    def clear(self) -> None:
        """Clear all cache entries"""
        self.interface_layer.clear()
        self.code_layer.clear()
        self.locality_scores.clear()
        self.usage_count.clear()
        self.last_access_time.clear()
        self.cache_on_gpu.clear()
        self.current_time = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including GPU/CPU distribution
        
        Returns:
            Dictionary with cache statistics
        """
        gpu_caches = sum(1 for on_gpu in self.cache_on_gpu.values() if on_gpu)
        cpu_caches = sum(1 for on_gpu in self.cache_on_gpu.values() if not on_gpu)
        
        return {
            'total_caches': len(self.interface_layer),
            'gpu_caches': gpu_caches,
            'cpu_caches': cpu_caches,
            'max_gpu_capacity': self.max_gpu_cache_size,
            'max_total_capacity': self.max_cache_size,
            'gpu_utilization': gpu_caches / self.max_gpu_cache_size if self.max_gpu_cache_size > 0 else 0,
            'cache_locations': dict(self.cache_on_gpu)
        }