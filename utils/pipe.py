"""ComfyUI model-management integration.

The MOSS model and its audio codec are wrapped in one ``MossModelPatcher`` so
ComfyUI's model manager owns their VRAM: it moves them to the GPU when a
generation node asks, and evicts them to system RAM when another workflow
needs the space, exactly like a checkpoint loaded through the core nodes.

The Hugging Face remote-code modules cannot be partially offloaded or
weight-cast the way ComfyUI's own ``comfy.ops`` layers can, so the patcher
only ever moves the whole bundle at once.
"""

import torch

import comfy.model_management as mm
import comfy.model_patcher


class MossModelBundle(torch.nn.Module):
    """Container so the TTS model and the codec move as one unit."""

    def __init__(self, model, audio_tokenizer):
        super().__init__()
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        # ModelPatcher reads/writes this; the HF model's own `device` is a
        # read-only property, which is why the bundle exists.
        self.device = torch.device("cpu")


class MossModelPatcher(comfy.model_patcher.ModelPatcher):
    """All-or-nothing variant of ModelPatcher for remote-code models."""

    def _move(self, device):
        if self.model.device != device:
            self.model.to(device)
            self.model.device = device
        self.model.model_loaded_weight_memory = 0 if mm.is_device_cpu(device) else self.model_size()
        self.model.model_lowvram = False

    # Entry points used by comfy.model_management.LoadedModel.
    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        before = self.loaded_size()
        self._move(device_to)
        return self.loaded_size() - before

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Cannot free part of the bundle; LoadedModel falls through to detach().
        return 0

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        self._move(device_to if device_to is not None else self.load_device)

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        if load_weights:
            self._move(device_to if device_to is not None else self.load_device)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            self._move(device_to if device_to is not None else self.offload_device)
        return self.model

    def detach(self, unpatch_all=True):
        self._move(self.offload_device)
        return self.model


class MossPipe:
    """What the loader hands to the generation nodes (type MOSS_TTS_PIPE)."""

    def __init__(self, patcher, processor, sample_rate, model_id):
        self.patcher = patcher
        self.processor = processor
        self.sample_rate = sample_rate
        self.model_id = model_id

    @property
    def model(self):
        return self.patcher.model.model

    @property
    def device(self):
        return self.patcher.load_device


def _kv_bytes_per_token(model):
    """Rough KV-cache footprint per generated token for the text backbone."""
    config = model.config
    text = None
    for name in ("language_config", "qwen3_config", "text_config"):
        text = getattr(config, name, None)
        if text is not None:
            break
    if text is None:
        return 0
    layers = getattr(text, "num_hidden_layers", 0)
    kv_heads = getattr(text, "num_key_value_heads", None) or getattr(text, "num_attention_heads", 0)
    head_dim = getattr(text, "head_dim", None) or (
        getattr(text, "hidden_size", 0) // max(1, getattr(text, "num_attention_heads", 1))
    )
    dtype_size = 2 if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else 4
    return layers * kv_heads * head_dim * 2 * dtype_size


def estimate_generation_memory(model, total_tokens):
    """Activation + KV-cache budget to reserve on top of the weights."""
    return _kv_bytes_per_token(model) * total_tokens + 1024 * 1024 * 1024


def load(pipe, total_tokens):
    """Bring the model and codec onto the GPU through ComfyUI's manager.

    Returns ``(model, processor, device)``.  Call this before touching the
    codec (reference-audio encoding) or the model.
    """
    mm.load_models_gpu(
        [pipe.patcher],
        memory_required=estimate_generation_memory(pipe.model, total_tokens),
    )
    return pipe.model, pipe.processor, pipe.device
