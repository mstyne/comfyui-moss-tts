MODEL_VARIANTS = {
    "MOSS-TTS (Delay 8B)": "OpenMOSS-Team/MOSS-TTS",
    "MOSS-TTS (Local 1.7B)": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "MOSS-TTS v1.5 (Delay 8B)": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "MOSS-TTS v1.5 (Local 4B, 48kHz stereo)": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "MOSS-TTSD v1.0": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "MOSS-SoundEffect": "OpenMOSS-Team/MOSS-SoundEffect",
}

DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# Models that use a codec other than DEFAULT_CODEC_PATH.
CODEC_PATHS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
}


def default_codec_for(model_id):
    return CODEC_PATHS.get(model_id, DEFAULT_CODEC_PATH)


# v1.5 models were trained with full language names in the prompt
# ("French"), while the 1.0 models use ISO codes ("fr").
V15_MODEL_IDS = {
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
}

# Every language listed on the MOSS-TTS v1.5 model card (31 languages).
# The 1.0 models document the first 20; the remainder were added in v1.5.
# Keys are the ISO codes passed to the 1.0 models; values are the full names
# passed to the v1.5 models. The dropdown keeps the original four first.
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "tr": "Turkish",
    # Added in v1.5
    "yue": "Cantonese",
    "fi": "Finnish",
    "he": "Hebrew",
    "hi": "Hindi",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "ro": "Romanian",
    "sw": "Swahili",
    "tl": "Tagalog",
    "th": "Thai",
    "vi": "Vietnamese",
}

LANGUAGE_CHOICES = ["auto"] + list(LANGUAGE_NAMES)

TOKENS_PER_SECOND = 12.5

DEFAULT_PARAMS = {
    "OpenMOSS-Team/MOSS-TTS": {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.0,
    },
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-TTS-v1.5": {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.0,
    },
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5": {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.0,
    },
    "OpenMOSS-Team/MOSS-TTSD-v1.0": {
        "temperature": 1.1,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-VoiceGenerator": {
        "temperature": 1.5,
        "top_p": 0.6,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-SoundEffect": {
        "temperature": 1.5,
        "top_p": 0.6,
        "top_k": 50,
        "repetition_penalty": 1.2,
    },
}

MAX_NEW_TOKENS_DEFAULT = 4096

SAMPLE_RATE = 24000
