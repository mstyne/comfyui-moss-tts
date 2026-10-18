"""Compatibility helpers for running the MOSS remote code on transformers 5.x.

The MOSS model code is fetched from Hugging Face via ``trust_remote_code`` and
was written against transformers 4.57.  transformers 5.20 changed a handful of
internals that the remote code relies on.  Everything here is idempotent and
only patches what is actually missing, so it stays a no-op on versions that do
not need it.
"""

import functools
import inspect
import json
import os
import sys
import types

from transformers.dynamic_module_utils import get_class_from_dynamic_module


def _processor_class_ref(local_dir):
    """Return the ``module.Class`` reference for the processor in ``local_dir``."""
    for name in ("processor_config.json", "config.json"):
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            auto_map = json.load(f).get("auto_map", {})
        ref = auto_map.get("AutoProcessor")
        if ref:
            return ref
    raise ValueError(f"No AutoProcessor entry found in auto_map under {local_dir}")


def load_processor(local_dir, **kwargs):
    """Load the MOSS processor without going through ``AutoProcessor``.

    transformers >= 5.20 ``AutoProcessor.from_pretrained`` injects ``revision``
    into the kwargs it forwards.  The MOSS remote ``from_pretrained`` passes all
    leftover kwargs to ``ProcessorMixin.__init__``, which now raises
    ``TypeError: Unexpected keyword argument revision``.  Resolving the class
    from ``auto_map`` ourselves sidesteps the injection.
    """
    cls = get_class_from_dynamic_module(_processor_class_ref(local_dir), local_dir)
    return cls.from_pretrained(local_dir, trust_remote_code=True, **kwargs)


def patch_local_model(model):
    """Make the MOSS Local-Transformer ``generate`` loop work on transformers 5.x.

    The Local model overrides ``GenerationMixin._sample`` and calls private
    helpers that 5.x removed or changed:

    1. ``_get_initial_cache_position`` no longer exists.  5.x creates
       ``cache_position`` inside ``prepare_inputs_for_generation`` for remote
       code models, so a pass-through shim is enough.
    2. ``create_causal_mask`` renamed ``input_embeds`` to ``inputs_embeds`` and
       dropped ``cache_position``.
    3. ``prepare_inputs_for_generation`` only slices ``input_ids`` down to the
       new tokens when ``next_sequence_length`` is given.  The remote sampling
       loop never passes it, so every step would re-feed the whole prompt on
       top of the KV cache.  Derive it from the cache length instead.

    Cached generation patched this way matches a no-cache forward pass to
    ~1e-7, same as on transformers 4.57.1.
    """
    if not hasattr(model, "_get_initial_cache_position"):
        model._get_initial_cache_position = types.MethodType(
            lambda self, cur_len, device, model_kwargs: model_kwargs, model
        )

    module = sys.modules.get(type(model).__module__)
    create_causal_mask = getattr(module, "create_causal_mask", None)
    if create_causal_mask is not None and not getattr(create_causal_mask, "_moss_compat", False):
        params = inspect.signature(create_causal_mask).parameters
        if "input_embeds" not in params:
            accepts_cache_position = "cache_position" in params

            @functools.wraps(create_causal_mask)
            def _create_causal_mask(*args, **kwargs):
                if "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                if not accepts_cache_position:
                    kwargs.pop("cache_position", None)
                return create_causal_mask(*args, **kwargs)

            _create_causal_mask._moss_compat = True
            module.create_causal_mask = _create_causal_mask

    prepare = model.prepare_inputs_for_generation
    if (
        "next_sequence_length" in inspect.signature(prepare).parameters
        and not getattr(prepare, "_moss_compat", False)
    ):
        # functools.wraps keeps the original signature visible to
        # GenerationMixin._validate_model_kwargs.
        @functools.wraps(prepare)
        def _prepare_inputs_for_generation(input_ids, **kwargs):
            past_key_values = kwargs.get("past_key_values")
            if kwargs.get("next_sequence_length") is None and past_key_values is not None:
                past_length = past_key_values.get_seq_length()
                if past_length > 0:
                    kwargs["next_sequence_length"] = input_ids.shape[1] - past_length
            return prepare(input_ids, **kwargs)

        _prepare_inputs_for_generation._moss_compat = True
        model.prepare_inputs_for_generation = _prepare_inputs_for_generation

    return model
