import torch


def apply_handles(tensor, sample_rate, head_seconds, tail_seconds):
    """Pad silence before/after a [S] mono or [C, S] multi-channel tensor."""
    head_samples = int(head_seconds * sample_rate)
    tail_samples = int(tail_seconds * sample_rate)
    lead = tensor.shape[:-1]
    parts = []
    if head_samples > 0:
        parts.append(torch.zeros(*lead, head_samples, dtype=tensor.dtype, device=tensor.device))
    parts.append(tensor)
    if tail_samples > 0:
        parts.append(torch.zeros(*lead, tail_samples, dtype=tensor.dtype, device=tensor.device))
    return torch.cat(parts, dim=-1) if len(parts) > 1 else tensor


def comfyui_audio_to_moss_tensor(audio_dict):
    waveform = audio_dict["waveform"]   # [B, C, S]
    sr = audio_dict["sample_rate"]
    wav = waveform[0].mean(dim=0)       # [S]
    return wav, sr


def moss_tensor_to_comfyui_audio(tensor, sample_rate=24000):
    """Wrap a [S] mono or [C, S] multi-channel tensor as ComfyUI AUDIO [1, C, S]."""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return {
        "waveform": tensor.unsqueeze(0),
        "sample_rate": sample_rate,
    }


def resample_if_needed(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    import torchaudio
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)
