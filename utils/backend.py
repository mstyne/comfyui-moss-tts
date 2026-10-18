import importlib.metadata
import importlib.util
import torch

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


from .constants import LANGUAGE_NAMES, V15_MODEL_IDS

# The 1.0 Local-Transformer drives generation through HF GenerationMixin and
# needs the transformers 5.x shims in utils/compat.py.  Every other model,
# including Local-Transformer v1.5, ships its own self-contained generate().
LEGACY_LOCAL_MODEL_IDS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}


def is_legacy_local_model(model_id):
    return model_id in LEGACY_LOCAL_MODEL_IDS


def language_tag(model_id, language):
    """Translate the node's language choice into what the model was trained on.

    Returns None for "auto" so the prompt renders the field as unset.
    """
    if language == "auto":
        return None
    if model_id in V15_MODEL_IDS:
        return LANGUAGE_NAMES.get(language, language)
    return language


def run_generation(model, input_ids, attention_mask, model_id,
                   temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    """Unified generate call for the Delay and Local model APIs.

    All remote implementations expose the same ``generate`` keyword interface
    (``audio_*`` for the audio channels, ``text_*`` for the text channel).
    The 1.0 Local model additionally needs a couple of transformers 5.x
    workarounds; see utils/compat.py.
    """
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        audio_temperature=temperature,
        audio_top_p=top_p,
        audio_top_k=top_k,
        audio_repetition_penalty=repetition_penalty,
    )

    if is_legacy_local_model(model_id):
        from transformers import DynamicCache
        from . import compat

        compat.patch_local_model(model)

        # Text channel keeps its own sampling settings, independent of the
        # audio sliders exposed in the node.
        kwargs.update(
            text_temperature=1.5,
            text_top_p=1.0,
            text_top_k=top_k,
            text_repetition_penalty=1.0,
        )
        # transformers 5.x builds DynamicCache(config=model.config) and reads
        # config.num_hidden_layers, which MossTTSDelayConfig does not expose.
        # Supplying the cache ourselves skips that path.
        kwargs["past_key_values"] = DynamicCache()

    return model.generate(**kwargs)


def _flash_attn_available() -> bool:
    """Check that flash_attn is both importable and has valid package metadata."""
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        importlib.metadata.version("flash_attn")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def resolve_attn_implementation(device, dtype) -> str:
    if (
        str(device).startswith("cuda")
        and _flash_attn_available()
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"
