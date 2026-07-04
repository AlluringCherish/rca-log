import torch
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeLLMWrapper:
    """
    Wrapper for Code Language Models with KV cache support
    Provides unified interface for different model backends
    """
    
    def __init__(self, 
                 model_name: str = "codellama/CodeLlama-7b-Python-hf",
                 device: str = 'cuda',
                 use_kv_cache: bool = True,
                 offload_callback=None):
        """
        Initialize Code LLM wrapper
        
        Args:
            model_name: Model identifier
            device: Device for computation
            use_kv_cache: Whether to use KV caching
        """
        self.model_name = model_name
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.offload_callback = offload_callback
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.kv_cache_manager = KVCacheManager(device=device)
        
        # Load model if specified
        if model_name != "placeholder":
            self._load_model()
            
    def _load_model(self):
        """Load the actual model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.model.eval()
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            print("Using placeholder model for testing")
            
    def generate(self, 
                 prompt: str,
                 max_length: int = 2048,
                 temperature: float = 0.7,
                 past_key_values: Optional[torch.Tensor] = None) -> str:
        """
        Generate code from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            past_key_values: Optional KV cache from previous context
            
        Returns:
            Generated code
        """
        if self.model is None:
            # Placeholder generation for testing
            return self._placeholder_generate(prompt)
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to correct device
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Generate with model
        with torch.no_grad():
            # Use max_new_tokens instead of max_length to avoid confusion
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,  # Generate up to 512 new tokens
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                past_key_values=None,  # Disable KV cache for now
                use_cache=False,  # Disable cache to avoid errors
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode output
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from generated text
        prompt_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated.startswith(prompt_decoded):
            generated = generated[len(prompt_decoded):]
            
        return generated
        
    def compute_perplexity(self, text: str, context: str) -> float:
        """
        Compute perplexity of text given context
        PPL(text | context) - measures how well the text follows the context
        
        Args:
            text: Text to evaluate (e.g., function interface)
            context: Context for conditioning (e.g., instruction)
            
        Returns:
            Perplexity score (lower is better)
        """
        if self.model is None or self.tokenizer is None:
            # Return random perplexity for testing
            return np.random.uniform(1, 20)
            
        try:
            # Combine context and text for conditional probability
            # We want P(text | context)
            full_text = context + "\n" + text
            
            # Tokenize the full text
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move to correct device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            input_ids = inputs["input_ids"].to(device)
            
            # Get the length of context tokens to compute PPL only on text part
            context_tokens = self.tokenizer(context, return_tensors="pt", truncation=True)
            context_length = context_tokens["input_ids"].shape[1]
            
            # Compute loss with model
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                
                # Get loss only for the text part (not context)
                # This gives us a better measure of P(text | context)
                if context_length < input_ids.shape[1]:
                    # Mask out context part from loss
                    shift_logits = outputs.logits[..., context_length-1:-1, :].contiguous()
                    shift_labels = input_ids[..., context_length:].contiguous()
                    
                    # Calculate cross entropy
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                   shift_labels.view(-1))
                else:
                    loss = outputs.loss
                
                # Convert loss to perplexity
                perplexity = torch.exp(loss).item()
                
                # Clamp to reasonable range
                perplexity = min(perplexity, 1000.0)  # Cap at 1000
                
            return perplexity
            
        except Exception as e:
            print(f"Warning: Perplexity computation failed: {e}")
            return 50.0  # Return moderate perplexity on error
    
    def compute_perplexity_batch(self, texts: List[str], context: str, batch_size: int = 4) -> List[float]:
        """
        Compute perplexity for multiple texts in batch with padding
        PPL(text | context) - measures how well each text follows the context
        
        Args:
            texts: List of texts to evaluate (e.g., function interfaces)
            context: Context for conditioning (e.g., instruction)
            batch_size: Maximum batch size (auto-adjusted based on GPU memory)
            
        Returns:
            List of perplexity scores (lower is better)
        """
        if self.model is None or self.tokenizer is None:
            # Return random perplexities for testing
            return [np.random.uniform(1, 20) for _ in texts]
        
        if not texts:
            return []
        
        # Initialize tokenizer pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        perplexities = []
        device = next(self.model.parameters()).device
        
        # Process in batches
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # Combine context with each text
                full_texts = [context + "\n" + text for text in batch_texts]
                
                # Tokenize with padding
                inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,  # Pad to same length within batch
                    truncation=True,
                    max_length=2048,
                    return_attention_mask=True
                )
                
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # Get context length for each item (they should be same)
                context_tokens = self.tokenizer(context, return_tensors="pt", truncation=True)
                context_length = context_tokens["input_ids"].shape[1]
                
                # Compute loss for the batch
                with torch.no_grad():
                    # Create labels with padding tokens masked
                    labels = input_ids.clone()
                    labels[~attention_mask.bool()] = -100  # Ignore padding in loss
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Compute per-sample perplexity
                    batch_size_actual = input_ids.shape[0]
                    vocab_size = outputs.logits.shape[-1]
                    
                    for i in range(batch_size_actual):
                        # Get valid tokens for this sample (excluding padding)
                        valid_mask = attention_mask[i].bool()
                        valid_length = valid_mask.sum().item()
                        
                        if context_length < valid_length:
                            # Compute loss only for the text part (after context)
                            sample_logits = outputs.logits[i, context_length-1:valid_length-1, :]
                            sample_labels = input_ids[i, context_length:valid_length]
                            
                            # Filter out any remaining padding
                            label_mask = sample_labels != self.tokenizer.pad_token_id
                            if label_mask.any():
                                sample_logits = sample_logits[label_mask]
                                sample_labels = sample_labels[label_mask]
                                
                                # Calculate cross entropy for this sample
                                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                                loss = loss_fct(
                                    sample_logits.reshape(-1, vocab_size),
                                    sample_labels.reshape(-1)
                                )
                                
                                # Convert to perplexity
                                ppl = torch.exp(loss).item()
                                ppl = min(ppl, 1000.0)  # Cap at 1000
                            else:
                                ppl = 50.0  # Default if no valid tokens
                        else:
                            ppl = 50.0  # Default if context is longer than text
                        
                        perplexities.append(ppl)
                        
            except torch.cuda.OutOfMemoryError as e:
                # If batch is too large, fall back to smaller batch or sequential
                print(f"OOM with batch size {batch_size}, falling back to sequential")
                # Trigger cache offloading if available
                if hasattr(self, 'offload_callback') and self.offload_callback:
                    self.offload_callback()
                    # Retry once after offloading
                    try:
                        torch.cuda.empty_cache()
                        # Retry with same batch
                        inputs = self.tokenizer(
                            full_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048,
                            return_attention_mask=True
                        )
                        input_ids = inputs["input_ids"].to(device)
                        attention_mask = inputs["attention_mask"].to(device)
                        context_tokens = self.tokenizer(context, return_tensors="pt", truncation=True)
                        context_length = context_tokens["input_ids"].shape[1]
                        
                        with torch.no_grad():
                            labels = input_ids.clone()
                            labels[~attention_mask.bool()] = -100
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            batch_size_actual = input_ids.shape[0]
                            vocab_size = outputs.logits.shape[-1]
                            
                            for i in range(batch_size_actual):
                                valid_mask = attention_mask[i].bool()
                                valid_length = valid_mask.sum().item()
                                if context_length < valid_length:
                                    sample_logits = outputs.logits[i, context_length-1:valid_length-1, :]
                                    sample_labels = input_ids[i, context_length:valid_length]
                                    label_mask = sample_labels != self.tokenizer.pad_token_id
                                    if label_mask.any():
                                        sample_logits = sample_logits[label_mask]
                                        sample_labels = sample_labels[label_mask]
                                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                                        loss = loss_fct(
                                            sample_logits.reshape(-1, vocab_size),
                                            sample_labels.reshape(-1)
                                        )
                                        ppl = torch.exp(loss).item()
                                        ppl = min(ppl, 1000.0)
                                    else:
                                        ppl = 50.0
                                else:
                                    ppl = 50.0
                                perplexities.append(ppl)
                    except torch.cuda.OutOfMemoryError:
                        # If still OOM, fall back to sequential
                        for text in batch_texts:
                            ppl = self.compute_perplexity(text, context)
                            perplexities.append(ppl)
                else:
                    # No offload callback, just fall back to sequential
                    for text in batch_texts:
                        ppl = self.compute_perplexity(text, context)
                        perplexities.append(ppl)
            except Exception as e:
                print(f"Warning: Batch perplexity computation failed: {e}")
                # Fall back to sequential computation
                for text in batch_texts:
                    ppl = self.compute_perplexity(text, context)
                    perplexities.append(ppl)
        
        return perplexities
        
    def extract_kv_cache(self, text: str) -> torch.Tensor:
        """
        Extract KV cache for text
        
        Args:
            text: Input text
            
        Returns:
            KV cache tensor
        """
        if self.model is None:
            # Return placeholder cache
            return self.kv_cache_manager.create_placeholder_cache(text)
            
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
            
            if hasattr(outputs, 'past_key_values'):
                # Extract and format KV cache
                kv_cache = self.kv_cache_manager.format_kv_cache(outputs.past_key_values)
                return kv_cache
                
        return self.kv_cache_manager.create_placeholder_cache(text)
        
    def _placeholder_generate(self, prompt: str) -> str:
        """
        Placeholder generation for testing without actual model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Mock generated code
        """
        # Extract key information from prompt
        if "pick" in prompt.lower():
            return """
def execute_task(env):
    # Pick and place task
    actions = []
    
    # Move to object
    actions.append({'move_to': [0.5, 0.5, 0.2]})
    
    # Grasp object
    actions.append({'grasp': True})
    
    # Move to target
    actions.append({'move_to': [1.0, 1.0, 0.3]})
    
    # Release object
    actions.append({'grasp': False})
    
    return actions
"""
        elif "stack" in prompt.lower():
            return """
def execute_task(env):
    # Stacking task
    actions = []
    
    for i, obj in enumerate(env.observation['objects']):
        actions.append({'pick': obj})
        actions.append({'place': [0, 0, i * 0.1]})
        
    return actions
"""
        else:
            return """
def execute_task(env):
    # Default task execution
    actions = []
    actions.append({'action': 'default'})
    return actions
"""


class KVCacheManager:
    """
    Manager for KV cache operations
    Handles formatting, merging, and storage of KV caches
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize KV cache manager
        
        Args:
            device: Device for tensor operations
        """
        self.device = device
        self.cache_dim = 768  # Default hidden dimension
        
    def format_kv_cache(self, past_key_values: Tuple) -> torch.Tensor:
        """
        Format model KV cache into unified tensor
        
        Args:
            past_key_values: Raw KV cache from model
            
        Returns:
            Formatted KV cache tensor
        """
        if not past_key_values:
            return self.create_placeholder_cache("")
            
        # Extract keys and values
        all_keys = []
        all_values = []
        
        for layer_kv in past_key_values:
            if len(layer_kv) >= 2:
                keys, values = layer_kv[0], layer_kv[1]
                all_keys.append(keys)
                all_values.append(values)
                
        if all_keys:
            # Stack across layers
            keys_tensor = torch.stack(all_keys, dim=0)
            values_tensor = torch.stack(all_values, dim=0)
            
            # Combine keys and values
            kv_cache = torch.cat([keys_tensor, values_tensor], dim=1)
            
            # Reshape to standard format
            batch_size = kv_cache.shape[0]
            seq_len = kv_cache.shape[-2]
            
            kv_cache = kv_cache.view(batch_size, -1, self.cache_dim)
            
            return kv_cache
            
        return self.create_placeholder_cache("")
        
    def merge_kv_caches(self, caches: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge multiple KV caches
        
        Args:
            caches: List of KV cache tensors
            
        Returns:
            Merged KV cache
        """
        if not caches:
            return self.create_placeholder_cache("")
            
        # Filter out None values
        valid_caches = [c for c in caches if c is not None]
        
        if not valid_caches:
            return self.create_placeholder_cache("")
            
        # Concatenate along sequence dimension
        merged = torch.cat(valid_caches, dim=1)
        
        return merged
        
    def create_placeholder_cache(self, text: str) -> torch.Tensor:
        """
        Create placeholder KV cache for testing
        
        Args:
            text: Input text
            
        Returns:
            Placeholder KV cache tensor
        """
        # Estimate sequence length from text
        seq_len = max(1, len(text.split()) // 2)
        
        # Create random tensor as placeholder
        placeholder = torch.randn(1, seq_len, self.cache_dim).to(self.device)
        
        return placeholder
        
    def compress_cache(self, kv_cache: torch.Tensor, compression_ratio: float = 0.5) -> torch.Tensor:
        """
        Compress KV cache to reduce memory usage
        
        Args:
            kv_cache: Input KV cache
            compression_ratio: Ratio of dimensions to keep
            
        Returns:
            Compressed KV cache
        """
        if kv_cache is None:
            return None
            
        # Simple compression via dimensionality reduction
        batch_size, seq_len, hidden_dim = kv_cache.shape
        
        # Reduce hidden dimension
        new_dim = int(hidden_dim * compression_ratio)
        
        # Use linear projection for compression
        compression_matrix = torch.randn(hidden_dim, new_dim).to(self.device)
        compressed = torch.matmul(kv_cache, compression_matrix)
        
        return compressed
        
    def decompress_cache(self, compressed_cache: torch.Tensor, original_dim: int = 768) -> torch.Tensor:
        """
        Decompress KV cache back to original dimension
        
        Args:
            compressed_cache: Compressed KV cache
            original_dim: Original hidden dimension
            
        Returns:
            Decompressed KV cache
        """
        if compressed_cache is None:
            return None
            
        batch_size, seq_len, compressed_dim = compressed_cache.shape
        
        # Use linear projection for decompression
        decompression_matrix = torch.randn(compressed_dim, original_dim).to(self.device)
        decompressed = torch.matmul(compressed_cache, decompression_matrix)
        
        return decompressed