import hashlib
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel

import comfy.model_management as mm
import folder_paths

from ..utils.constants import MODEL_VARIANTS, SAMPLE_RATE, default_codec_for
from ..utils import backend, compat

MODEL_ID_TTSD = "OpenMOSS-Team/MOSS-TTSD-v1.0"
MODEL_ID_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"

# Track loaded model so we can free VRAM before loading a new one
_current_model = None
_current_processor = None

# Store models in ComfyUI's models directory under moss-tts/
MOSS_MODELS_DIR = os.path.join(folder_paths.models_dir, "moss-tts")
os.makedirs(MOSS_MODELS_DIR, exist_ok=True)


def _resolve_local_dir(repo_id_or_path, check_for_updates=True):
    """If repo_id_or_path is a HF repo ID, download it into ComfyUI's
    models/moss-tts/ directory and return the local path.
    If it's already a local path, return it as-is.
    Works around MOSS processor bug: Path(repo_id) mangles / to \\ on Windows.

    With check_for_updates=False an existing download is used as-is, without
    contacting the Hub at all."""
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path
    # e.g. "OpenMOSS-Team/MOSS-TTS" → "OpenMOSS-Team--MOSS-TTS"
    safe_name = repo_id_or_path.replace("/", "--")
    local_dir = os.path.join(MOSS_MODELS_DIR, safe_name)
    if not check_for_updates and os.path.isfile(os.path.join(local_dir, "config.json")):
        return local_dir
    return snapshot_download(repo_id_or_path, local_dir=local_dir)


class MossTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (list(MODEL_VARIANTS.keys()),),
                "local_model_path": ("STRING", {"default": ""}),
                "codec_local_path": ("STRING", {"default": ""}),
                "check_for_updates": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "audio/MOSS-TTS"

    def load_model(self, model_variant, local_model_path, codec_local_path, check_for_updates=True):
        global _current_model, _current_processor

        device = mm.get_torch_device()
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        attn_implementation = backend.resolve_attn_implementation(device, dtype)

        mm.unload_all_models()

        # Free the previous MOSS model from VRAM
        if _current_processor is not None:
            if hasattr(_current_processor, "audio_tokenizer"):
                _current_processor.audio_tokenizer.cpu()
            _current_processor = None
        if _current_model is not None:
            _current_model.cpu()
            del _current_model
            _current_model = None
        torch.cuda.empty_cache()

        model_id = MODEL_VARIANTS[model_variant]
        model_path = local_model_path.strip() if local_model_path.strip() else model_id

        # Pre-resolve to local directory to avoid MOSS processor's
        # Path(repo_id) bug on Windows (converts / to \)
        local_dir = _resolve_local_dir(model_path, check_for_updates)

        # Always resolve the codec ourselves so it lands in models/moss-tts/
        # and honours check_for_updates, instead of letting the processor
        # pull it into the HF cache on its own.
        codec_path = codec_local_path.strip() or default_codec_for(model_id)
        processor_kwargs = {"codec_path": _resolve_local_dir(codec_path, check_for_updates)}
        if model_id == MODEL_ID_VOICE_GENERATOR:
            processor_kwargs["normalize_inputs"] = True

        # Bypasses AutoProcessor, which on transformers >= 5.20 injects kwargs
        # the MOSS remote processor cannot accept. See utils/compat.py.
        processor = compat.load_processor(local_dir, **processor_kwargs)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        if attn_implementation != "flash_attention_2" and hasattr(processor.audio_tokenizer, "set_attention_implementation"):
            # Audio-Tokenizer-v2 defaults to flash_attention_2 and only knows
            # sdpa as the alternative.
            processor.audio_tokenizer.set_attention_implementation("sdpa")

        model_kwargs = {}
        config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        if hasattr(config, "attn_implementation"):
            # Local-Transformer v1.5 reads its own config fields (defaulting
            # to flash_attention_2) rather than the from_pretrained kwarg.
            config.attn_implementation = attn_implementation
            config.local_transformer_attn_implementation = attn_implementation
            model_kwargs["config"] = config

        model = AutoModel.from_pretrained(
            local_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **model_kwargs,
        ).to(device)
        model.eval()

        # v1.5 Local runs at 48 kHz; everything else at 24 kHz.
        sample_rate = getattr(processor.model_config, "sampling_rate", None) or SAMPLE_RATE

        _current_model = model
        _current_processor = processor

        return ((model, processor, sample_rate, device, model_id),)

    @classmethod
    def IS_CHANGED(cls, model_variant, local_model_path, codec_local_path, check_for_updates=True):
        h = hashlib.md5()
        h.update(model_variant.encode())
        h.update(local_model_path.encode())
        h.update(codec_local_path.encode())
        h.update(str(check_for_updates).encode())
        return h.hexdigest()
