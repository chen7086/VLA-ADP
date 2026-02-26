"""
PruneOpenVLA - VLA model with integrated KV-pruning attention (PruneVLA).

This is a thin wrapper that applies pruning attention to an existing VLA model
by calling the external replacement function to swap self_attn modules in-place.
"""

from typing import Optional

try:
    from prunevla.prune_kv_attention import PruneVLAConfig, replace_attention_with_prune
    _PRUNE_AVAILABLE = True
except Exception:
    PruneVLAConfig = None  # type: ignore
    replace_attention_with_prune = None  # type: ignore
    _PRUNE_AVAILABLE = False


def apply_prune_on_existing_vla(model, prune_config: Optional[PruneVLAConfig]):
    """
    In-place apply KV-pruning attention on an existing VLA model.
    Returns the same model instance after replacement.
    """
    if prune_config is None or not _PRUNE_AVAILABLE:
        return model
    replace_attention_with_prune(model, prune_config)
    return model


