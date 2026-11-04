"""Rewrite a full checkpoint as BF16 with norm leaves in fp32, or an adapter's tensors as BF16 from its weights file,
with the header-read ``--verify`` gate and the ``--check_inference`` smoke pass."""

import logging
import os
from collections import Counter
from collections.abc import Iterable

import torch
from peft import PeftConfig, PeftModel
from safetensors import safe_open

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, DEFAULT_MAX_SHARD_SIZE, SAFETENSORS_METADATA
from halo.checkpoint.reading import header_numel, iter_checkpoint_shard_entries, reject_partial_checkpoint
from halo.checkpoint.sidecars import apply_training_sidecars, copy_training_sidecars
from halo.checkpoint.tools.adapter_merge import (
    adapter_base_trust,
    adapter_input_gates,
    apply_adapter,
    load_base_for_adapter,
)
from halo.checkpoint.tools.offline_export import (
    ADAPTER_TASKS,
    TASK_MODEL_CLASSES,
    carried_mtp_tail,
    cast_norms_to_fp32,
    copy_checkpoint_aux_files,
    load_task_model,
    preflight_model_load_resources,
    reject_in_place_conversion,
    save_full_checkpoint,
)
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.tokenizer_setup import load_processing_class, resolve_tokenizer

logger = logging.getLogger(__name__)

# --verify bar, weighted by numel: tiny fp32 norm leaves are expected, while an fp32 embedding or head still fails it.
_BF16_VERIFY_MIN_FRACTION = 0.99

# --check_inference: a handful of sampled tokens shows coherent text, at a cost of seconds on any host.
_SMOKE_PROMPT = "Hello, world! I am a"
_SMOKE_MAX_NEW_TOKENS = 20
_SMOKE_TEMPERATURE = 0.7
_SMOKE_CLASSIFICATION_TEXT = "This is a test sentence for classification."


def _load_verified(model_path, task, *, excuse_task_head: bool, text_only: bool = False, **load_kwargs):
    """:func:`load_task_model` with buffers finalized, since ``--check_inference`` generates from it and transformers 5
    hands back non-persistent buffers as uninitialized memory."""
    model = load_task_model(model_path, task, excuse_task_head=excuse_task_head, text_only=text_only, **load_kwargs)
    finalize_loaded_model(model)
    return model


def _reject_adapter_task(task) -> None:
    if task not in ADAPTER_TASKS:
        raise ValueError(f"PEFT is only supported for tasks {sorted(ADAPTER_TASKS)}, not {task!r}")


def _load_adapter_model(adapter_dir, task, load_kwargs) -> PeftModel:
    """The adapter applied to its base, loaded through the class the adapter's keys address."""
    _reject_adapter_task(task)
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
    trust_remote_code = load_kwargs.get("trust_remote_code", False)
    return apply_adapter(
        adapter_dir,
        base.model,
        adapter_processor=load_processing_class(adapter_dir, trust_remote_code=trust_remote_code),
        base_processor=load_processing_class(peft_config.base_model_name_or_path, trust_remote_code=trust_remote_code),
        log=logger.info,
    )


def load_model(model_path, task, is_peft=False, **load_kwargs):
    """Load ``model_path`` as ``task`` through the checkpoint-coverage gate, raising on an unknown task. ``is_peft``
    loads the adapter's base here, since PEFT's auto-class reaches ``from_pretrained`` directly and a truncated base
    would random-initialize its absent tensors. A missing seq-cls head is excused only if ``modules_to_save`` declares
    one."""
    if task not in TASK_MODEL_CLASSES:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_MODEL_CLASSES)}")
    if not is_peft:
        return _load_verified(model_path, task, excuse_task_head=False, **load_kwargs)
    return _load_adapter_model(model_path, task, load_kwargs)


def _stored_bf16_share_passes(headers: Iterable, model_path: str) -> bool:
    """Whether stored float parameters are primarily bfloat16, weighted by parameter count."""
    dtype_params: Counter[str] = Counter()
    for header in headers:
        dtype_params[header.get_dtype()] += header_numel(header)

    # Float tensors only, since an integer buffer can never be bf16; safetensors spells them F16/F32/F64/F8_*/BF16.
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
    """Whether stored parameters are primarily bfloat16, from the safetensors headers, since ``from_pretrained`` casts
    to ``config.dtype`` and a reload could never fail this check."""
    logger.info(f"Verifying BF16 conversion for model at {model_path}...")
    headers = (reader.get_slice(key) for _shard, reader, key in iter_checkpoint_shard_entries(model_path))
    return _stored_bf16_share_passes(headers, model_path)


def verify_adapter_conversion(adapter_path):
    """Whether an adapter's weights file is primarily bfloat16; no tensors at all raises, since it loads as a no-op."""
    logger.info(f"Verifying BF16 conversion for adapter at {adapter_path}...")
    with safe_open(adapter_path, framework="pt") as reader:
        keys = list(reader.keys())
        if not keys:
            raise RuntimeError(f"{adapter_path} carries no tensors — the save produced an adapter with nothing in it.")
        return _stored_bf16_share_passes((reader.get_slice(key) for key in keys), adapter_path)


def run_test_inference(model_path, task, is_peft=False, trust_remote_code=False) -> None:
    """Log what the converted model produces; diagnostic only, as ``--verify`` is the gate that raises."""
    # An adapter directory need not carry a tokenizer; its base does.
    sources = (
        (model_path, PeftConfig.from_pretrained(model_path).base_model_name_or_path) if is_peft else (model_path,)
    )
    processing_classes = (load_processing_class(source, trust_remote_code=trust_remote_code) for source in sources)
    processing_class = next((loaded for loaded in processing_classes if loaded is not None), None)
    if processing_class is None:
        logger.warning(f"No tokenizer at {' or '.join(sources)} — skipping the inference check")
        return
    tokenizer = resolve_tokenizer(processing_class)

    logger.info(f"Running test inference on model at {model_path}...")
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
    """Rewrite every float tensor of the adapter's weights file as bf16, with no base load, carrying its other files as
    they are. No norm keeps fp32, as an adapter has no module tree to single one out;
    ``merge_peft_adapters.py --fp32_norms`` writes that layout."""
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
    """Reload a full checkpoint at bf16, cast norms back to fp32, and write it out with the MTP head its class drops."""
    # Before the load, since a per-rank EP/TP save loads its .shard_N keys as missing and random-initialized, which
    # --verify's dtype count cannot catch.
    reject_partial_checkpoint(model_path)
    reject_in_place_conversion(model_path, output_path)
    processing_class = load_processing_class(model_path, trust_remote_code=trust_remote_code)

    # from_pretrained lands the whole checkpoint in host RAM unless --device_map routes it; output is source-sized.
    preflight_model_load_resources(model_path, output_path, tool="convert_to_bf16", device_map=device_map)

    model = load_model(model_path, task, **load_kwargs)
    mtp_tail = carried_mtp_tail(model, model_path, output_path, tool="convert_to_bf16", dtype=torch.bfloat16)
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
        carried_tensors=mtp_tail,
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
    trust_remote_code=None,
    max_shard_size=DEFAULT_MAX_SHARD_SIZE,
):
    """An unset ``trust_remote_code`` trusts a full checkpoint, which must be a local directory, and follows an
    adapter's base (:func:`adapter_base_trust`)."""
    if task == "base" and check_inference:
        raise ValueError("--check_inference has no inference path for --task base: drop it, or name the head's task.")
    if is_peft:
        _reject_adapter_task(task)
        if device_map is not None:
            raise ValueError("--peft rewrites the adapter file and loads no model, so --device_map places nothing.")
        trust_remote_code = adapter_base_trust(model_path, trust_remote_code)
    else:
        # A Hub repo here would run a third-party repo's code under this tool's local default.
        if not os.path.isdir(model_path):
            raise ValueError(
                f"--input_dir {model_path!r} is not a local directory. Download the repo first "
                f"(hf download {model_path} --local-dir <dir>) and point --input_dir at that directory."
            )
        if trust_remote_code is None:
            trust_remote_code = True

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

    # A --peft save writes no model weights; any model*.safetensors in --output_dir is an earlier save's.
    if verify:
        verified = (
            verify_adapter_conversion(os.path.join(output_path, ADAPTER_SAFETENSORS_FILE))
            if is_peft
            else verify_model_conversion(output_path)
        )
        if not verified:
            raise RuntimeError(
                f"BF16 verification failed for {output_path}: the saved model is not primarily bfloat16."
            )

    if check_inference:
        # A --peft save is an adapter dir with no config.json, which the plain loader rejects.
        run_test_inference(output_path, task, is_peft=is_peft, trust_remote_code=trust_remote_code)
