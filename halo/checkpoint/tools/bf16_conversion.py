"""Rewrite a full checkpoint as BF16 with every normalization leaf in fp32, or a PEFT adapter's tensors as BF16
streamed from its weights file, with the header-read ``--verify`` gate and the ``--check_inference`` smoke pass."""

import logging
import os
from collections import Counter
from collections.abc import Iterable

import torch
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoTokenizer

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, DEFAULT_MAX_SHARD_SIZE, SAFETENSORS_METADATA
from halo.checkpoint.reading import header_numel, iter_checkpoint_shard_entries, reject_partial_checkpoint
from halo.checkpoint.sidecars import apply_training_sidecars, copy_training_sidecars
from halo.checkpoint.tools.adapter_merge import adapter_input_gates, load_base_for_adapter
from halo.checkpoint.tools.offline_export import (
    ADAPTER_TASKS,
    TASK_MODEL_CLASSES,
    cast_norms_to_fp32,
    copy_checkpoint_aux_files,
    load_task_model,
    preflight_model_load_resources,
    reject_in_place_conversion,
    save_full_checkpoint,
)
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.tokenizer_setup import load_processing_class

logger = logging.getLogger(__name__)

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


def _load_verified(model_path, task, *, excuse_task_head: bool, text_only: bool = False, **load_kwargs):
    """:func:`load_task_model` with the loaded model's buffers finalized.

    ``--check_inference`` generates from what this returns, and transformers 5 hands back every
    non-persistent buffer as uninitialized memory, so the repair belongs on the load rather than on
    whichever caller happens to run a forward.
    """
    model = load_task_model(model_path, task, excuse_task_head=excuse_task_head, text_only=text_only, **load_kwargs)
    finalize_loaded_model(model)
    return model


def _load_adapter_model(adapter_dir, task, load_kwargs) -> PeftModel:
    """The adapter applied to its base, loaded through the class the adapter's keys address."""
    if task not in ADAPTER_TASKS:
        raise ValueError(f"PEFT is only supported for tasks {sorted(ADAPTER_TASKS)}, not {task!r}")
    peft_config = PeftConfig.from_pretrained(adapter_dir)

    def load_base_model(base_model_path, *, excuse_task_head: bool, text_only: bool = False):
        return _load_verified(
            base_model_path, task, excuse_task_head=excuse_task_head, text_only=text_only, **load_kwargs
        )

    base = load_base_for_adapter(
        adapter_dir,
        peft_config.base_model_name_or_path,
        load_base_model,
        excuse_task_head=bool(getattr(peft_config, "modules_to_save", None)),
        log=logger.info,
    )
    return PeftModel.from_pretrained(base.model, adapter_dir)


def load_model(model_path, task, is_peft=False, **load_kwargs):
    """Load ``model_path`` as ``task``, through the checkpoint-coverage gate.

    Raises on an unknown type, or a PEFT-unsupported one, rather than falling back to a different
    architecture head.

    ``is_peft`` loads the adapter's base and applies the adapter on top instead of handing the
    directory to PEFT's own auto-class: that class reaches ``from_pretrained`` for the base directly,
    where a truncated or key-mismatched checkpoint random-initializes the absent tensors with only a
    log line, and this tool would write them out as a finished bf16 model. A missing seq-cls head is
    legitimate only when the adapter declares one in ``modules_to_save`` (a classification adapter
    trained on a plain causal-LM base), so the exemption is read off that declaration.
    The base loads through the class the adapter's keys address (:func:`load_base_for_adapter`).
    """
    if task not in TASK_MODEL_CLASSES:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_MODEL_CLASSES)}")
    if not is_peft:
        return _load_verified(model_path, task, excuse_task_head=False, **load_kwargs)
    return _load_adapter_model(model_path, task, load_kwargs)


def _stored_bf16_share_passes(headers: Iterable, model_path: str) -> bool:
    """Whether the stored float parameters behind ``headers`` (safetensors slices) are primarily bfloat16,
    weighted by parameter count."""
    dtype_params: Counter[str] = Counter()
    for header in headers:
        dtype_params[header.get_dtype()] += header_numel(header)

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


def verify_model_conversion(model_path):
    """Whether the checkpoint's stored parameters are primarily bfloat16, read from the safetensors headers.

    Not a ``from_pretrained`` reload: that path casts every tensor to ``config.dtype`` on the way in,
    so a checkpoint still stored in fp32 comes back 100% bfloat16 and the check could never fail. The
    header carries each tensor's dtype and shape, so the parameter-weighted share costs no tensor
    reads.
    """
    logger.info(f"Verifying BF16 conversion for model at {model_path}...")
    headers = (reader.get_slice(key) for _shard, reader, key in iter_checkpoint_shard_entries(model_path))
    return _stored_bf16_share_passes(headers, model_path)


def verify_adapter_conversion(adapter_path):
    """Whether a converted adapter's weights file stores its float parameters primarily as bfloat16; an adapter
    carrying no tensors at all raises, since it loads as a no-op."""
    logger.info(f"Verifying BF16 conversion for adapter at {adapter_path}...")
    with safe_open(adapter_path, framework="pt") as reader:
        keys = list(reader.keys())
        if not keys:
            raise RuntimeError(f"{adapter_path} carries no tensors — the save produced an adapter with nothing in it.")
        return _stored_bf16_share_passes((reader.get_slice(key) for key in keys), adapter_path)


def run_test_inference(model_path, task, is_peft=False, trust_remote_code=True) -> None:
    """Run a simple inference test on the converted model and print what it produced.

    Diagnostic only: a broken conversion shows up in the printed text/logits, not as a return value,
    so the caller has nothing to branch on. ``--verify`` is the gate that raises. ``is_peft`` takes
    :func:`load_model`'s adapter path, the same split the ``--verify`` gate makes.
    """
    if task == "base":
        logger.warning("Test inference not implemented for base model type — skipping the inference check")
        return

    logger.info(f"Running test inference on model at {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    model = load_model(
        model_path,
        task,
        is_peft=is_peft,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    if task == "causal_lm":
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

    inputs = tokenizer(_SMOKE_CLASSIFICATION_TEXT, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    logger.info(f"Running classification for: '{_SMOKE_CLASSIFICATION_TEXT}'")
    with torch.no_grad():
        outputs = model(**inputs)

    logger.info(f"Classification logits: {outputs.logits}")


def _convert_adapter_to_bf16(adapter_dir, output_path) -> None:
    """Every floating tensor of the adapter's weights file rewritten as bf16, with no base load: a bf16 adapter
    keeps its weights as stored. Its other files (adapter config, tokenizer, training sidecars) carry across as
    they are, so its later merge resolves the processing class off the same directory; the card is tagged.

    No norm keeps fp32 here: an adapter has no module tree to single one out, and ``merge_peft_adapters.py
    --fp32_norms`` writes the fp32-norm layout of the merged model.
    """
    adapter_input_gates(adapter_dir, output_path)
    weights = os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE)
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"{adapter_dir} holds no {ADAPTER_SAFETENSORS_FILE}; the conversion streams that file. Re-save the "
            f"adapter with safe_serialization (PEFT's default) and convert again."
        )
    tensors = {}
    with safe_open(weights, framework="pt") as reader:
        for key in reader.keys():  # noqa: SIM118 - not a mapping
            tensor = reader.get_tensor(key)
            tensors[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
    logger.info(f"Adapter converted to BF16. Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    copy_checkpoint_aux_files(adapter_dir, output_path, include_resume_sidecars=False)
    copy_training_sidecars(adapter_dir, output_path)
    save_safetensors_file(tensors, os.path.join(output_path, ADAPTER_SAFETENSORS_FILE), metadata=SAFETENSORS_METADATA)


def _convert_checkpoint_to_bf16(
    model_path, output_path, task, load_kwargs, *, device_map, max_shard_size, trust_remote_code
) -> None:
    """A full checkpoint reloaded at bf16, norms cast back to fp32, and written back out."""
    # Guards before the load and os.makedirs: a per-rank EP/TP save loads its .shard_N expert keys as
    # missing and randomly initialized, which --verify's dtype count cannot catch.
    reject_partial_checkpoint(model_path)
    reject_in_place_conversion(model_path, output_path)
    processing_class = load_processing_class(model_path, trust_remote_code=trust_remote_code)

    # Size preflight before the heavy load: from_pretrained lands the whole checkpoint in host RAM
    # unless --device_map routes it, and the output is about the source's size on disk.
    preflight_model_load_resources(model_path, output_path, tool="convert_to_bf16", device_map=device_map)

    model = load_model(model_path, task, **load_kwargs)
    for action in apply_training_sidecars(model, model_path):
        logger.info(action)
    cast_norms_to_fp32(model)

    logger.info(f"Model converted to BF16. Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    save_full_checkpoint(
        model,
        output_path,
        processing_class=processing_class,
        source_dir=model_path,
        max_shard_size=max_shard_size,
    )


def convert_to_bf16(
    model_path,
    output_path,
    task,
    is_peft=False,
    device_map=None,
    verify=False,
    check_inference=False,
    trust_remote_code=True,
    max_shard_size=DEFAULT_MAX_SHARD_SIZE,
):
    logger.info(f"Loading model from {model_path}...")

    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
    }

    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if is_peft:
        _convert_adapter_to_bf16(model_path, output_path)
    else:
        _convert_checkpoint_to_bf16(
            model_path,
            output_path,
            task,
            load_kwargs,
            device_map=device_map,
            max_shard_size=max_shard_size,
            trust_remote_code=trust_remote_code,
        )

    logger.info(f"Model successfully converted and saved to {output_path}")

    # A --peft save writes no model weights, so its adapter is what gets verified: any
    # model*.safetensors in --output_dir is an earlier save's, which this one did not remove.
    if verify:
        verified = (
            verify_adapter_conversion(os.path.join(output_path, ADAPTER_SAFETENSORS_FILE))
            if is_peft
            else verify_model_conversion(output_path)
        )
        # Raise so a failed conversion cannot report success to its caller.
        if not verified:
            raise RuntimeError(
                f"BF16 verification failed for {output_path}: the saved model is not primarily bfloat16."
            )

    if check_inference:
        # A --peft save is an adapter dir with no config.json, which the plain loader rejects.
        run_test_inference(output_path, task, is_peft=is_peft, trust_remote_code=trust_remote_code)
