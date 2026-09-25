"""Kimi-Killer L3 walking skeleton — network from scratch in pure JAX.

Phase 1 (this package): the core network — KDA, Gated MLA, SiTU-GLU MLP, NoPE,
MTP, ViT-minimum, AttnRes — plus the optimizer, QAT fake-quant, tokenizer,
data pipeline and Orbax checkpoints. No MaxText/Flax dependency for the core
layers; MaxText integration is documented in ``net/maxtext/``.
"""

from .config import ModelConfig, load_config, save_config

__all__ = ["ModelConfig", "load_config", "save_config"]
