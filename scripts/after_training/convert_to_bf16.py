"""Convert a checkpoint (full model or PEFT adapter) to a BF16 checkpoint, normalization layers in fp32.

Handles causal-LM, sequence-classification and bare-base checkpoints, optionally merging a LoRA
adapter into its base model first. ``--verify`` reads the saved safetensors headers and asserts the
stored parameters are primarily bfloat16, raising on failure; ``--check_inference`` runs a short
generation / classification pass and reports its output. ``--output_dir`` is written fresh: every
``model*.safetensors`` or index the completed save did not produce is removed afterwards.

Usage:
    python scripts/after_training/convert_to_bf16.py --input_dir <src> --output_dir <dst> \
        [--model_type causal_lm|classifier|base] [--peft [--merge_adapter]] [--verify] [--check_inference]
"""

import argparse
import logging
import math
import os
from collections import Counter

import torch
from accelerate import PartialState
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.adapters import assert_no_expert_lora_adapter, merge_adapter_into_base
from src.checkpoint.format import ADAPTER_SAFETENSORS_FILE, DEFAULT_MAX_SHARD_SIZE
from src.checkpoint.tool_io import (
    apply_training_sidecars,
    checkpoint_shard_files,
    copy_training_sidecars,
    iter_checkpoint_shard_entries,
    preflight_model_load_resources,
    reject_in_place_conversion,
    reject_sharded_checkpoint,
    save_full_checkpoint,
)
from src.log import configure_cli_logging
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.model_preparation import auto_load_model, sanitize_generation_config
from src.models.loading.tokenizer_setup import load_processing_class, resolve_peft_processing_class
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.structure import is_normalization_module

configure_cli_logging()
logger = logging.getLogger(__name__)

# --model_type → the class the checkpoint loads with; None resolves it from the config
# (``auto_load_model``; AutoModelForCausalLM would drop a VLM's vision tower). Kept as one table so
# every load goes through the coverage gate rather than reaching ``from_pretrained`` directly.
_MODEL_CLASSES = {
    "causal_lm": None,
    "classifier": AutoModelForSequenceClassification,
    "base": AutoModel,
}

# Types a PEFT adapter can be merged into. A bare backbone has no adapter task type to load.
_PEFT_MODEL_TYPES = ("causal_lm", "classifier")

# --verify bar: share of saved float parameters that must be bf16, weighted by numel. Numerous but
# tiny fp32 norm leaves are expected, so the bar sits high enough that an fp32 embedding or head —
# few tensors, a large share of the parameters — still fails it.
_BF16_VERIFY_MIN_FRACTION = 0.99

# --check_inference smoke test: a handful of sampled tokens shows the converted weights still produce
# coherent text, and is short enough to cost seconds on any host.
_SMOKE_PROMPT = "Hello, world! I am a"
_SMOKE_MAX_NEW_TOKENS = 20
_SMOKE_TEMPERATURE = 0.7
_SMOKE_CLASSIFICATION_TEXT = "This is a test sentence for classification."


def load_model(model_path, model_type, is_peft=False, **load_kwargs):
    """Load ``model_path`` as ``model_type``, through the checkpoint-coverage gate.

    Raises on an unknown type, or a PEFT-unsupported one, rather than falling back to a different
    architecture head.

    ``is_peft`` loads the adapter's base and applies the adapter on top instead of handing the
    directory to PEFT's own auto-class: that class reaches ``from_pretrained`` for the base directly,
    where a truncated or key-mismatched checkpoint random-initializes the absent tensors with only a
    log line, and this tool would write them out as a finished bf16 model. A missing seq-cls head is
    legitimate only when the adapter declares one in ``modules_to_save`` (a classification adapter
    trained on a plain causal-LM base), so the exemption is read off that declaration.
    """
    if model_type not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {sorted(_MODEL_CLASSES)}")
    if not is_peft:
        return _load_verified(model_path, model_type, excuse_task_head=False, **load_kwargs)
    _reject_unsupported_peft_type(model_type)
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = _load_verified(
        peft_config.base_model_name_or_path,
        model_type,
        excuse_task_head=bool(getattr(peft_config, "modules_to_save", None)),
        **load_kwargs,
    )
    return PeftModel.from_pretrained(base_model, model_path)


def _reject_unsupported_peft_type(model_type: str) -> None:
    """Refuse a ``--model_type`` an adapter cannot be loaded onto: a bare backbone has no task type."""
    if model_type not in _PEFT_MODEL_TYPES:
        raise ValueError(f"PEFT is only supported for model types {sorted(_PEFT_MODEL_TYPES)}, not {model_type!r}")


def _load_verified(model_path, model_type, *, excuse_task_head: bool, **load_kwargs):
    """``model_path`` loaded as ``model_type``, missing-key report gated and buffers finalized.

    ``--check_inference`` generates from what this returns, and transformers 5 hands back every
    non-persistent buffer as uninitialized memory, so the repair belongs on the load rather than on
    whichever caller happens to run a forward.
    """
    model_class = _MODEL_CLASSES[model_type]
    if model_class is None:
        model = auto_load_model(model_path, excuse_task_head=excuse_task_head, **load_kwargs)
    else:
        model = from_pretrained_verified(model_class, model_path, excuse_task_head=excuse_task_head, **load_kwargs)
    finalize_loaded_model(model)
    return model


def verify_adapter_save(adapter_path):
    """Report on an unmerged PEFT save, which the bf16 assertion does not apply to.

    PEFT restores LoRA ``A``/``B`` to fp32 as it writes them (``autocast_adapter_dtype``), so a
    correct output on this path is entirely fp32 and the bf16 assertion does not apply. What is
    checkable is that the adapter carries tensors at all: an empty one loads as a no-op.
    """
    with safe_open(adapter_path, framework="pt") as reader:
        tensor_count = len(reader.keys())
    if not tensor_count:
        raise RuntimeError(f"{adapter_path} carries no tensors — the save produced an adapter with nothing in it.")
    logger.info(
        f"✓ Adapter-only save with {tensor_count} tensors. PEFT restores LoRA A/B to fp32 as it saves, "
        f"so the BF16 check applies only to a merged model — re-run with --merge_adapter to verify weights."
    )
    return True


def verify_model_conversion(model_path):
    """Whether the checkpoint's stored parameters are primarily bfloat16, read from the safetensors headers.

    Not a ``from_pretrained`` reload: that path casts every tensor to ``config.dtype`` on the way in,
    so a checkpoint still stored in fp32 comes back 100% bfloat16 and the check could never fail. The
    header carries each tensor's dtype and shape, so the parameter-weighted share costs no tensor
    reads.

    An unmerged PEFT save writes no ``model*.safetensors`` at all and is handed to
    :func:`verify_adapter_save` instead.
    """
    logger.info(f"Verifying BF16 conversion for model at {model_path}...")

    try:
        # Resolved eagerly, ahead of the lazy walk below, so the adapter fallback sees the
        # FileNotFoundError rather than having it surface from inside the loop.
        checkpoint_shard_files(model_path)
    except FileNotFoundError:
        adapter = os.path.join(model_path, ADAPTER_SAFETENSORS_FILE)
        if not os.path.isfile(adapter):
            raise
        return verify_adapter_save(adapter)

    dtype_params: Counter[str] = Counter()
    for _shard, reader, key in iter_checkpoint_shard_entries(model_path):
        header = reader.get_slice(key)
        dtype_params[header.get_dtype()] += math.prod(header.get_shape())

    # Float tensors only: an integer buffer can never be bf16, so counting it would dilute the share.
    # safetensors spells float dtypes F16/F32/F64/F8_*/BF16 and everything else I*/U*/BOOL.
    float_params = sum(count for dtype, count in dtype_params.items() if dtype.startswith(("F", "BF")))
    if not float_params:
        raise RuntimeError(f"{model_path} stores no floating-point tensors, so there is nothing to verify.")

    bf16_params = dtype_params["BF16"]
    logger.info(f"Stored parameters by dtype: {dict(sorted(dtype_params.items()))}")
    logger.info(f"- BF16: {bf16_params}/{float_params} float parameters ({bf16_params / float_params * 100:.2f}%)")

    if bf16_params / float_params > _BF16_VERIFY_MIN_FRACTION:
        logger.info("✓ Verification successful: Model correctly converted to BF16")
        return True
    logger.error("✗ Verification failed: Model is not stored primarily in BF16")
    return False


def run_test_inference(model_path, model_type, is_peft=False, trust_remote_code=True) -> None:
    """Run a simple inference test on the converted model and print what it produced.

    Diagnostic only: a broken conversion shows up in the printed text/logits, not as a return value,
    so the caller has nothing to branch on. ``--verify`` is the gate that raises. ``is_peft`` picks
    the adapter auto-class, the same split :func:`verify_model_conversion` makes.
    """
    logger.info(f"Running test inference on model at {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if model_type == "base":
        logger.warning("Test inference not implemented for base model type — skipping the inference check")
        return

    model = load_model(
        model_path,
        model_type,
        is_peft=is_peft,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if model_type == "causal_lm":
        inputs = tokenizer(_SMOKE_PROMPT, return_tensors="pt")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        logger.info(f"Generating text for prompt: '{_SMOKE_PROMPT}'")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=_SMOKE_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=_SMOKE_TEMPERATURE,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")

        return

    test_text = _SMOKE_CLASSIFICATION_TEXT
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    logger.info(f"Running classification for: '{test_text}'")
    with torch.no_grad():
        outputs = model(**inputs)

    logger.info(f"Classification logits: {outputs.logits}")


def _cast_norms_to_fp32(model) -> None:
    """Keep normalization layers in fp32 after the bf16 load.

    Classified by the toolkit's norm predicate, the same one the save-dtype decisions use, rather
    than a second name rule that could disagree with it. Leaves only: a container that classifies as
    a norm must not drag its children up with it.
    """
    logger.info("Setting specific layers to appropriate precision...")
    for module in model.modules():
        if is_normalization_module(module) and next(module.children(), None) is None:
            module.to(torch.float32)


def _merge_adapter_to_bf16(
    adapter_dir,
    output_path,
    model_type,
    load_kwargs,
    *,
    device_map,
    max_shard_size,
    trust_remote_code,
) -> None:
    """``--peft --merge_adapter``: the shared merge, at this tool's dtype.

    :func:`merge_adapter_into_base` handles the gate order, the adapter's sidecars and the merged
    save; this tool supplies the dtype the base loads at and the fp32 norm leaves it is written
    with.
    """
    _reject_unsupported_peft_type(model_type)

    def load_base_model(base_model_path, *, excuse_task_head: bool):
        return _load_verified(base_model_path, model_type, excuse_task_head=excuse_task_head, **load_kwargs)

    merge_adapter_into_base(
        adapter_dir,
        output_path,
        load_base_model=load_base_model,
        tool="convert_to_bf16",
        device_map=device_map,
        max_shard_size=max_shard_size,
        trust_remote_code=trust_remote_code,
        prepare_for_save=_cast_norms_to_fp32,
    )


def _convert_checkpoint_to_bf16(
    model_path,
    output_path,
    model_type,
    load_kwargs,
    *,
    is_peft,
    device_map,
    max_shard_size,
    trust_remote_code,
) -> None:
    """A full checkpoint — or an adapter left unmerged — reloaded at bf16 and written back out."""
    # A per-rank EP/TP save reaches from_pretrained with expert keys under .shard_N: the real ones read
    # as missing and are randomly initialized (a warning, not a raise), which --verify's dtype count
    # cannot catch. Both guards run ahead of the load and of os.makedirs(output_path), so a refusal
    # leaves nothing behind.
    if os.path.isdir(model_path):
        reject_sharded_checkpoint(model_path)
    reject_in_place_conversion(model_path, output_path)

    # Full processor for VLMs (keeps processor_config.json), else a plain tokenizer. A PEFT adapter
    # dir usually carries only a tokenizer, so resolve it against the base model too.
    if is_peft:
        # A later merge_and_unload cannot fold native EP expert-LoRA, so refuse before re-saving one.
        assert_no_expert_lora_adapter(model_path)
        base_model_path = PeftConfig.from_pretrained(model_path).base_model_name_or_path
        # Under --peft the weights come from the base, so the guards above covered the wrong
        # directory: an --output_dir aimed at the base would overwrite the checkpoint being read.
        if base_model_path and os.path.isdir(base_model_path):
            reject_in_place_conversion(base_model_path, output_path)
            reject_sharded_checkpoint(base_model_path)
        weights_source = base_model_path
        processing_class = resolve_peft_processing_class(
            model_path, base_model_path, trust_remote_code=trust_remote_code
        )
    else:
        weights_source = model_path
        processing_class = load_processing_class(model_path, trust_remote_code=trust_remote_code)

    # Size preflight before the heavy load: from_pretrained lands the whole checkpoint in host RAM
    # unless --device_map routes it, and a full-model output is about the source's size on disk (an
    # unmerged --peft run writes adapter files only, so it has no full-model disk cost).
    preflight_model_load_resources(
        weights_source,
        output_path,
        tool="convert_to_bf16",
        device_map=device_map,
        writes_full_model=not is_peft,
    )

    if is_peft:
        logger.info("Loading PEFT adapter model...")
    model = load_model(model_path, model_type, is_peft=is_peft, **load_kwargs)

    if not is_peft:
        # Skipped for an adapter-only save: it writes no weights, so nothing re-applied here would
        # reach the output. The sidecars are copied across verbatim instead, for the later merge.
        for action in apply_training_sidecars(model, model_path):
            logger.info(action)

    _cast_norms_to_fp32(model)

    logger.info(f"Model converted to BF16. Saving to {output_path}...")

    os.makedirs(output_path, exist_ok=True)

    if is_peft:
        # Unmerged --peft writes adapter files only: sweeping here would delete the full-model weights
        # the directory holds, and restore_model_type would stamp a family this save never wrote.
        sanitize_generation_config(model)
        model.save_pretrained(output_path)
        if processing_class is not None:
            processing_class.save_pretrained(output_path)
        # Only this branch copies them by hand: the adapter dir's sidecars have to survive for the
        # later merge, and save_pretrained copies nothing.
        copy_training_sidecars(model_path, output_path)
        return

    save_full_checkpoint(
        model,
        output_path,
        processing_class=processing_class,
        source_dir=weights_source,
        max_shard_size=max_shard_size,
    )


def convert_to_bf16(
    model_path,
    output_path,
    model_type,
    is_peft=False,
    merge_adapter=False,
    device_map=None,
    verify=False,
    check_inference=False,
    trust_remote_code=True,
    max_shard_size=DEFAULT_MAX_SHARD_SIZE,
):
    # Refused before any I/O: merging is read only on the PEFT path, so without it the tool would exit
    # 0 having written an unmerged checkpoint.
    if merge_adapter and not is_peft:
        raise ValueError(
            "--merge_adapter is only used with --peft: a full-model source carries no adapter to fold in. "
            "Add --peft (the source is an adapter directory), or drop --merge_adapter."
        )

    logger.info(f"Loading model from {model_path}...")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
    }

    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if merge_adapter:
        _merge_adapter_to_bf16(
            model_path,
            output_path,
            model_type,
            load_kwargs,
            device_map=device_map,
            max_shard_size=max_shard_size,
            trust_remote_code=trust_remote_code,
        )
    else:
        _convert_checkpoint_to_bf16(
            model_path,
            output_path,
            model_type,
            load_kwargs,
            is_peft=is_peft,
            device_map=device_map,
            max_shard_size=max_shard_size,
            trust_remote_code=trust_remote_code,
        )

    logger.info(f"Model successfully converted and saved to {output_path}")

    # Raise so a failed conversion cannot report success to its caller.
    if verify and not verify_model_conversion(output_path):
        raise RuntimeError(f"BF16 verification failed for {output_path}: the saved model is not primarily bfloat16.")

    if check_inference:
        # An unmerged --peft save is an adapter dir with no config.json, which the plain loader rejects.
        run_test_inference(
            output_path, model_type, is_peft=is_peft and not merge_adapter, trust_remote_code=trust_remote_code
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to BF16 format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the source model directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the BF16 model (any model*.safetensors/index already there is removed first)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal_lm",
        choices=["causal_lm", "classifier", "base"],
        help="Type of the model: causal_lm (for LLMs), classifier, or base",
    )
    parser.add_argument("--peft", action="store_true", help="Whether the model is a PEFT adapter model")
    parser.add_argument(
        "--merge_adapter", action="store_true", help="If using PEFT, whether to merge the adapter with the base model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for model loading, e.g., 'auto' for automatic allocation",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Assert the saved checkpoint is stored in BF16 (reads the safetensors headers; raises on failure)",
    )
    parser.add_argument(
        "--check_inference",
        action="store_true",
        help="Run a test inference on the converted model to verify functionality",
    )
    add_max_shard_size_arg(parser, note="An unmerged --peft save writes adapter files only and ignores it.")
    add_trust_remote_code_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    # The shared model_preparation helpers log through accelerate's rank-aware logger, which needs
    # this state.
    PartialState()
    convert_to_bf16(
        args.input_dir,
        args.output_dir,
        args.model_type,
        is_peft=args.peft,
        merge_adapter=args.merge_adapter,
        device_map=args.device_map,
        verify=args.verify,
        check_inference=args.check_inference,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
