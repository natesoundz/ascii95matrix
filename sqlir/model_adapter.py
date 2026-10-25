"""
ModelAdapter

Abstract interface for different language model backends.
Supports Llama, Qwen, Granite, DeepSeek, GPT-style models, and future architectures.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class LayerInfo:
    """Information about a model layer."""
    layer_id: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    name: str = ""


@dataclass
class TokenInfo:
    """Information about a token in the context."""
    token_id: int
    token_str: str
    embedding: Optional[np.ndarray] = None
    position: int = 0


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Each concrete model (Llama, Qwen, etc.) implements this interface.
    Provides unified API for:
    - Tokenization
    - Inference
    - Layer access
    - KV cache access
    - Residual stream access
    - Logit access
    """
    
    def __init__(self, model_name: str, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Decoded text
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        tokens: List[int],
        cache: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run forward pass through model.
        
        Args:
            tokens: Input token IDs
            cache: Optional KV cache from previous forward pass
        
        Returns:
            Tuple of (logits, kv_cache)
            logits: [vocab_size] array of next token logits
            kv_cache: KV cache for next forward pass (or for inspection)
        """
        pass
    
    @abstractmethod
    def get_layer_info(self, layer_id: int) -> LayerInfo:
        """Get information about a specific layer."""
        pass
    
    @abstractmethod
    def read_layer_state(self, layer_id: int, layer_type: str = 'residual') -> np.ndarray:
        """
        Read state from a specific layer.
        
        Args:
            layer_id: Which layer (0-indexed)
            layer_type: 'residual', 'attention', 'mlp', 'kv'
        
        Returns:
            State array from layer
        """
        pass
    
    @abstractmethod
    def write_layer_state(self, layer_id: int, state: np.ndarray, layer_type: str = 'residual') -> None:
        """
        Write state to a specific layer.
        
        Args:
            layer_id: Which layer (0-indexed)
            state: New state to write
            layer_type: 'residual', 'attention', 'mlp', 'kv'
        """
        pass
    
    @abstractmethod
    def read_kv_cache(self, layer_id: Optional[int] = None) -> Dict:
        """
        Read KV cache values.
        
        Args:
            layer_id: Which layer, or None for all layers
        
        Returns:
            KV cache dict: {layer_id: {'key': k, 'value': v}, ...}
        """
        pass
    
    @abstractmethod
    def write_kv_cache(self, kv_data: Dict) -> None:
        """
        Write KV cache values.
        
        Args:
            kv_data: KV cache dict to write
        """
        pass
    
    @abstractmethod
    def read_residual_stream(self, layer_id: int) -> np.ndarray:
        """
        Read residual stream activation at a layer.
        
        Args:
            layer_id: Which layer
        
        Returns:
            Residual stream activation
        """
        pass
    
    @abstractmethod
    def write_residual_stream(self, layer_id: int, activation: np.ndarray) -> None:
        """
        Modify residual stream activation.
        
        Args:
            layer_id: Which layer
            activation: New activation to add to residual
        """
        pass
    
    @abstractmethod
    def read_attention_weights(self, layer_id: int, head_id: Optional[int] = None) -> np.ndarray:
        """
        Read attention weights from a layer.
        
        Args:
            layer_id: Which layer
            head_id: Which attention head, or None for all
        
        Returns:
            Attention weight matrix [seq_len, seq_len] or [num_heads, seq_len, seq_len]
        """
        pass
    
    @abstractmethod
    def read_logits(self) -> np.ndarray:
        """
        Read current logits from output.
        
        Returns:
            Logits array [vocab_size]
        """
        pass
    
    @abstractmethod
    def modify_logits(self, modification: np.ndarray) -> None:
        """
        Add modification to logits before sampling.
        
        Args:
            modification: Values to add to logits
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> int:
        """
        Sample next token from logits.
        
        Args:
            logits: Logit array [vocab_size]
            temperature: Sampling temperature
            top_k: Keep only top-k tokens
            top_p: Keep tokens with cumulative probability > top_p
        
        Returns:
            Next token ID
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        hooks: Optional[Dict] = None
    ) -> str:
        """
        Generate text with optional hooks for intervention.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            hooks: Dict of {layer_id: callback_fn}
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_num_layers(self) -> int:
        """Get number of layers in model."""
        pass
    
    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Get hidden dimension size."""
        pass
    
    def __enter__(self):
        """Context manager support."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.unload()


class OllamaAdapter(ModelAdapter):
    """
    Adapter for Ollama models running locally.
    
    Uses llama-cpp-python for direct layer access.
    """
    
    def __init__(self, model_name: str = "llama2", verbose: bool = False):
        super().__init__(model_name, verbose)
        self.model = None
        self.is_loaded = False
    
    def load(self) -> None:
        """Load model via ollama."""
        if self.is_loaded:
            return
        
        try:
            # Note: This would require llama-cpp-python installed
            # For now, we'll create a mock implementation
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=f"models/{self.model_name}.gguf",
                n_gpu_layers=-1,  # Use GPU
                verbose=self.verbose
            )
            self.is_loaded = True
            
            if self.verbose:
                print(f"[OllamaAdapter] Loaded {self.model_name}")
        
        except ImportError:
            print("[OllamaAdapter] llama-cpp-python not installed")
            print("Install with: pip install llama-cpp-python")
            self.is_loaded = False
    
    def unload(self) -> None:
        """Unload model."""
        if self.model:
            del self.model
        self.is_loaded = False
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return self.model.tokenize(text.encode())
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return self.model.detokenize(tokens).decode()
    
    def forward(self, tokens: List[int], cache: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Forward pass."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # This would require embedding model's internal state
        # For now, return placeholder
        logits = np.random.randn(self.get_vocab_size())
        return logits, cache or {}
    
    def get_layer_info(self, layer_id: int) -> LayerInfo:
        """Get layer information."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return LayerInfo(
            layer_id=layer_id,
            num_heads=32,  # Model-dependent
            hidden_dim=self.get_hidden_dim(),
            mlp_dim=self.get_hidden_dim() * 4
        )
    
    def read_layer_state(self, layer_id: int, layer_type: str = 'residual') -> np.ndarray:
        """Read layer state."""
        raise NotImplementedError("Requires layer-level model access")
    
    def write_layer_state(self, layer_id: int, state: np.ndarray, layer_type: str = 'residual') -> None:
        """Write layer state."""
        raise NotImplementedError("Requires layer-level model access")
    
    def read_kv_cache(self, layer_id: Optional[int] = None) -> Dict:
        """Read KV cache."""
        raise NotImplementedError("Requires KV cache access")
    
    def write_kv_cache(self, kv_data: Dict) -> None:
        """Write KV cache."""
        raise NotImplementedError("Requires KV cache access")
    
    def read_residual_stream(self, layer_id: int) -> np.ndarray:
        """Read residual stream."""
        raise NotImplementedError("Requires layer-level model access")
    
    def write_residual_stream(self, layer_id: int, activation: np.ndarray) -> None:
        """Write residual stream."""
        raise NotImplementedError("Requires layer-level model access")
    
    def read_attention_weights(self, layer_id: int, head_id: Optional[int] = None) -> np.ndarray:
        """Read attention weights."""
        raise NotImplementedError("Requires layer-level model access")
    
    def read_logits(self) -> np.ndarray:
        """Read logits."""
        raise NotImplementedError("Requires logit access")
    
    def modify_logits(self, modification: np.ndarray) -> None:
        """Modify logits."""
        raise NotImplementedError("Requires logit access")
    
    def sample(self, logits: np.ndarray, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9) -> int:
        """Sample token."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Simple sampling
        if temperature > 0:
            logits = logits / temperature
        
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        return np.random.choice(len(logits), p=probs)
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, hooks: Optional[Dict] = None) -> str:
        """Generate text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        output = self.model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return output['choices'][0]['text']
    
    def get_vocab_size(self) -> int:
        """Get vocab size."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self.model.vocab_size()
    
    def get_num_layers(self) -> int:
        """Get number of layers."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        # Model-dependent
        return 32
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        # Model-dependent
        return 4096
