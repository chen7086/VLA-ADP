"""
SparseOpenVLA - VLA model with integrated sparse attention.

Uses inheritance to directly replace attention layers, following standard model extension patterns.
"""

import sys
import os
from typing import Dict, Optional, Any

# Add sparsevla path for importing sparse attention modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../sparsevla'))

from .openvla import OpenVLA
from vla_sparse_attention import VLASparseAttention, VLASparseAttentionConfig


class SparseOpenVLA(OpenVLA):
    """
    Sparse attention version of OpenVLA model.
    
    Replaces all transformer layer attention modules during initialization
    to enable efficient vision token sparsification.
    """
    
    def __init__(self, *args, sparse_config: Optional[VLASparseAttentionConfig] = None, **kwargs):
        # Initialize original model first
        super().__init__(*args, **kwargs)
        
        # Apply sparse configuration if provided
        if sparse_config is not None:
            self.sparse_config = sparse_config
            self._replace_attention_layers()
        else:
            self.sparse_config = None
    
    def _replace_attention_layers(self):
        """Replace all transformer layer self_attn modules with sparse versions."""
        layers = None
        
        # Try different paths to find transformer layers
        possible_paths = [
            'llm_backbone.llm.model.layers',  # OpenVLA via PrismaticVLM
            'language_model.model.layers',    # HF standard path
            'llm.model.layers',               # Alternative path
        ]
        
        for path in possible_paths:
            try:
                obj = self
                for attr in path.split('.'):
                    obj = getattr(obj, attr)
                layers = obj
                break
            except AttributeError:
                continue
        
        if layers is None:
            print("WARNING: Could not find transformer layers")
            return
        
        replaced_count = 0
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                original_attn = layer.self_attn
                layer.self_attn = VLASparseAttention(original_attn, self.sparse_config)
                replaced_count += 1
    

    
    @classmethod 
    def from_existing_vla(cls, existing_vla, sparse_config: VLASparseAttentionConfig):
        """
        Create sparse version from existing VLA model.
        
        Safe method: adds sparse functionality to existing model without changing class type.
        """
        # Add sparse configuration
        existing_vla.sparse_config = sparse_config
        

        
        # Replace attention layers using static method
        cls._replace_attention_layers_on_model(existing_vla, sparse_config)
        
        return existing_vla
    
    @staticmethod
    def _replace_attention_layers_on_model(model, sparse_config):
        """Static method to replace attention layers on specified model."""
        layers = None
        
        # Try different paths to find transformer layers
        possible_paths = [
            'llm_backbone.llm.model.layers',  # OpenVLA via PrismaticVLM
            'language_model.model.layers',    # HF standard path
            'llm.model.layers',               # Alternative path
        ]
        
        for path in possible_paths:
            try:
                obj = model
                for attr in path.split('.'):
                    obj = getattr(obj, attr)
                layers = obj
                break
            except AttributeError:
                continue
        
        if layers is None:
            print("WARNING: Could not find transformer layers")
            return
        
        replaced_count = 0
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                original_attn = layer.self_attn
                layer.self_attn = VLASparseAttention(original_attn, sparse_config)
                replaced_count += 1
    
    @classmethod
    def from_pretrained_with_sparse(cls, pretrained_model_name_or_path: str, 
                                   sparse_config: VLASparseAttentionConfig, **kwargs):
        """
        Convenience method to load pretrained model and apply sparse attention.
        """
        # Load original model first
        model = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Apply sparse configuration
        model.sparse_config = sparse_config
        model._replace_attention_layers()
        
        return model 