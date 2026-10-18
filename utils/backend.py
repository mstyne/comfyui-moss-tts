import importlib.metadata
import importlib.util
import torch

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


LOCAL_MODEL_IDS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}


def is_local_model(model_id):
    """Local models use HF GenerationMixin with a custom generation_config.
    Delay models use a custom generate() with audio_* kwargs."""
    return model_id in LOCAL_MODEL_IDS


def run_generation(model, input_ids, attention_mask, model_id,
                   temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    """Unified generate call for the Delay and Local model APIs.

    Both remote implementations now expose the same ``generate`` keyword
    interface (``audio_*`` for the audio channels, ``text_*`` for the text
    channel).  The Local model additionally needs a couple of transformers 5.x
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

    if is_local_model(model_id):
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
