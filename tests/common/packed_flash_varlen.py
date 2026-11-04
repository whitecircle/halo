"""The packed-row varlen sweep: a family's forward fed the collator's varlen set vs one left to derive it, and a GC
training step. Kernels are CPU stand-ins and production patches are applied, so it rebinds process-global state: a
fresh interpreter per family (:func:`main`, :func:`report`), reporting through one ``FINDINGS:`` line."""

import functools
import os
import sys
from collections.abc import Callable, Mapping
from unittest.mock import MagicMock

import torch
from torch.overrides import TorchFunctionMode
from torch.utils.checkpoint import CheckpointError
from transformers import masking_utils
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from halo.data.collators.factory import select_data_collator
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.segment_markers import segment_markers_for
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    tiny_family_model,
)

FINDINGS_MARKER = "FINDINGS:"
# Report entries that are not failures.
SKIP_PREFIX = "skip:"
REACH_PREFIX = "reach:"
VARLEN_KEYS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
PAD_ID = 0
EOS_ID = 1
SEED = 1234
# The production flash backend on Blackwell; its int-seqlen patch is part of what the sweep holds.
SWEEP_ATTN_IMPLEMENTATION = "flash_attention_4"
# On a device each one waits for the device's queued work.
HOST_READS = frozenset({"__bool__", "__float__", "__index__", "__int__", "cpu", "item", "nonzero", "numpy", "tolist"})

_TORCH_DIR = os.path.dirname(torch.__file__)

MULTI_DOCUMENT_ROWS = (
    {"input_ids": [5, 6, 7, 8, 9, 10, 11], "seq_lengths": [3, 4]},
    {"input_ids": [12, 13, 14, 15, 16], "seq_lengths": [2, 2, 1]},
)

# Keyed as the GPU suites name them. DeepSeek-V4 refuses packing (compressed-KV windows), so it has no row to sweep.
PACKING_FAMILIES = {
    name: family
    for name, family in {
        **{
            f"dense:{name}": family
            for name, family in {**TINY_DENSE_FAMILIES, **TINY_DENSE_ATTENTION_FAMILIES}.items()
        },
        **TINY_MOE_FAMILIES,
        **TINY_MOE_VLM_FAMILIES,
    }.items()
    if name != "deepseek_v4"
}
# Where it is not the sweep's: a softcap stays off FA4's varlen kernels (Gemma 2), and the head_dim-256 kernel takes no
# window (Gemma 3).
PACKED_ATTN_IMPLEMENTATIONS = {"dense:gemma2": "flash_attention_2", "dense:gemma3_text": "flash_attention_2"}
# Off, or the checkpoint at that ``use_reentrant``.
GC_MODES = {"off": None, "non_reentrant": False, "reentrant": True}
# The model forward passes attention no extra kwargs, so the handed set never reaches it (Ling 2.0's remote forward).
SET_DROPPING_FAMILIES = frozenset({"bailing_moe"})


def padding_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = PAD_ID
    tok.eos_token_id = EOS_ID
    tok.padding_side = "right"

    def _pad(features, **kwargs):
        width = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            out["input_ids"].append(ids + [PAD_ID] * (width - len(ids)))
            out["attention_mask"].append([1] * len(ids) + [0] * (width - len(ids)))
        return {key: torch.tensor(value) for key, value in out.items()}

    tok.pad.side_effect = _pad
    return tok


def _attend(q, k, v, *, softmax_scale, causal, window_size, softcap):
    seq, heads = q.shape[0], q.shape[1]
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (softmax_scale or q.shape[-1] ** -0.5)
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    offset = torch.arange(seq)[:, None] - torch.arange(seq)[None, :]
    allowed = offset >= 0 if causal else torch.ones(seq, seq, dtype=torch.bool)
    if window_size[0] >= 0:
        allowed = allowed & (offset <= window_size[0])
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).to(q.dtype)


class ReferenceFlash:
    """CPU stand-ins for the flash kernels at FA2's signatures, recording varlen arguments, sliding window and
    softcap."""

    def __init__(self):
        self.calls: list[tuple] = []

    def varlen(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
    ):
        self.calls.append(
            ("varlen", cu_seqlens_q.tolist(), cu_seqlens_k.tolist(), max_seqlen_q, max_seqlen_k, window_size, softcap)
        )
        bounds = cu_seqlens_q.tolist()
        options = {"softmax_scale": softmax_scale, "causal": causal, "window_size": window_size, "softcap": softcap}
        return torch.cat(
            [_attend(q[a:b], k[a:b], v[a:b], **options) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        )

    def dense(
        self,
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        self.calls.append(("dense", tuple(q.shape), window_size, softcap))
        options = {"softmax_scale": softmax_scale, "causal": causal, "window_size": window_size, "softcap": softcap}
        return torch.stack([_attend(q[i], k[i], v[i], **options) for i in range(q.shape[0])])


class AttentionHostReads(TorchFunctionMode):
    """Records each :data:`HOST_READS` call inside a flash attention interface and outside its kernels: each is a host
    sync on a device."""

    def __init__(self):
        super().__init__()
        self.reads: list[str] = []
        self._depth = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if self._depth > 0 and getattr(func, "__name__", None) in HOST_READS:
            self.reads.append(func.__name__)
        return func(*args, **(kwargs or {}))

    def scoped(self, fn: Callable, depth: int) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self._depth += depth
            try:
                return fn(*args, **kwargs)
            finally:
                self._depth -= depth

        return wrapped

    def attending(self, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION) -> None:
        interface = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        interface = getattr(interface, "_recorded_original", interface)
        recorded = self.scoped(interface, 1)
        recorded._recorded_original = interface
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, recorded)


def _first_caller_outside_torch() -> str:
    frame = sys._getframe(1)
    while frame is not None and (
        frame.f_code.co_filename.startswith(_TORCH_DIR) or frame.f_code.co_filename == __file__
    ):
        frame = frame.f_back
    return "" if frame is None else frame.f_code.co_filename


class MaskHostReads(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", None) in HOST_READS and _first_caller_outside_torch() == masking_utils.__file__:
            self.count += 1
        return func(*args, **(kwargs or {}))


def count_derivations() -> dict[str, int]:
    """Call counts of the per-layer derivation a handed-over varlen set replaces, and of the padding-mask path before
    it."""
    counts = {"_is_packed_sequence": 0, "_prepare_from_posids": 0, "_upad_input": 0}
    for name in counts:
        original = getattr(flash_utils, name)
        original = getattr(original, "_counted_original", original)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        counted._counted_original = original
        setattr(flash_utils, name, counted)
    return counts


def hide_kernels(reads: AttentionHostReads) -> None:
    """Stop ``reads`` recording inside loaded kernels: the reference kernels read every offset back."""
    for name in ("_flash_fn", "_flash_varlen_fn"):
        kernel = getattr(flash_utils, name)
        kernel = getattr(kernel, "_hidden_original", kernel)
        hidden = reads.scoped(kernel, -1)
        hidden._hidden_original = kernel
        setattr(flash_utils, name, hidden)


def install_loaded_flash(reads: AttentionHostReads, attn_implementation: str) -> dict[str, int]:
    """The device counterpart of :func:`install_reference_flash`: the kernels transformers loads, recorded around."""
    flash_utils.lazy_import_flash_attention(attn_implementation)
    hide_kernels(reads)
    return count_derivations()


def install_reference_flash(
    reads: AttentionHostReads | None = None, *, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION
) -> tuple[ReferenceFlash, dict[str, int]]:
    """Fresh reference kernels where transformers caches its loaded ones, plus :func:`count_derivations`."""
    kernels = ReferenceFlash()
    flash_utils._loaded_implementation = attn_implementation
    flash_utils._flash_fn = kernels.dense
    flash_utils._flash_varlen_fn = kernels.varlen
    flash_utils._flash_with_kvcache_fn = None
    # Built after the family patches, as transformers builds it at the first forward after the load.
    flash_utils._process_flash_kwargs_fn = flash_utils._lazy_define_process_function(kernels.varlen)
    if reads is not None:
        hide_kernels(reads)
    return kernels, count_derivations()


def assert_transformers_derivation(varlen: Mapping, position_ids: torch.Tensor) -> None:
    """``varlen`` holds the set exactly where transformers takes its varlen kernel, equal to its derivation: int32
    offsets, host-int max lengths."""
    packed = bool(flash_utils._is_packed_sequence(position_ids, batch_size=position_ids.shape[0]))
    assert all((key in varlen) is packed for key in VARLEN_KEYS), (sorted(varlen), packed)
    if not packed:
        return
    (cu_q, cu_k), (max_q, max_k) = flash_utils.prepare_fa_kwargs_from_position_ids(position_ids)
    for key, derived in (("cu_seq_lens_q", cu_q), ("cu_seq_lens_k", cu_k)):
        assert varlen[key].dtype == derived.dtype == torch.int32, key
        assert torch.equal(varlen[key], derived), (key, varlen[key], derived)
    for key, derived in (("max_length_q", max_q), ("max_length_k", max_k)):
        assert type(varlen[key]) is int and varlen[key] == int(derived), (key, varlen[key], derived)


def set_attn_implementation(model_config, attn_implementation: str | None) -> None:
    for level in (model_config, model_config.get_text_config()):
        level._attn_implementation = attn_implementation


def use_flash(model_config, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION) -> None:
    """Resolve the decoder to ``attn_implementation`` and apply its production family patches."""
    set_attn_implementation(model_config, attn_implementation)
    apply_family_patches(model_config, attn_implementation)


def _packed_flash_batch(name: str, config) -> tuple[str, dict]:
    """Resolve ``config`` to ``name``'s flash label with its production patches; return the label and the collator's
    multi-document batch."""
    attn_implementation = PACKED_ATTN_IMPLEMENTATIONS.get(name, SWEEP_ATTN_IMPLEMENTATION)
    use_flash(config, attn_implementation)
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )
    return attn_implementation, collator.torch_call([dict(row) for row in MULTI_DOCUMENT_ROWS])


def family_findings(name: str) -> list[str]:
    """What breaks when ``name``'s packed batch hands flash its varlen set, plus a ``reach:`` entry; a lone ``skip:``
    if unreached."""
    if name in SET_DROPPING_FAMILIES:
        return [f"{SKIP_PREFIX} its model forward drops the varlen set, so its attention derives its own"]
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).eval()
    config = model.config
    if segment_markers_for(config).cu_seq_lens:
        return [f"{SKIP_PREFIX} its GatedDeltaNet mixers always take the set, which this path leaves unchanged"]
    if not getattr(model, "_supports_flash_attn", False):
        return [f"{SKIP_PREFIX} its model class declares no flash support, so its runs never resolve to flash"]
    attn_implementation, batch = _packed_flash_batch(name, config)
    if not all(key in batch for key in VARLEN_KEYS):
        return ["the flash run's multi-document batch carried no varlen set"]
    handed = {key: value for key, value in batch.items() if key != "labels"}
    derived_inputs = {key: value for key, value in handed.items() if key not in VARLEN_KEYS}

    def run(inputs):
        kernels, counts = install_reference_flash(attn_implementation=attn_implementation)
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
        return logits, kernels.calls, counts

    with_set, with_calls, with_counts = run(handed)
    derived, derived_calls, derived_counts = run(derived_inputs)

    findings = []
    if not torch.equal(with_set, derived):
        findings.append("the varlen set changed the logits")
    if with_calls != derived_calls:
        findings.append(f"the kernels saw different arguments: {with_calls} vs {derived_calls}")
    if with_counts["_is_packed_sequence"] or with_counts["_prepare_from_posids"]:
        findings.append(f"a layer still derived the set from position_ids: {with_counts}")
    # Every layer that derived the set now reads the handed-over one.
    kwargs_branch = sum(call[0] == "varlen" for call in with_calls) - with_counts["_upad_input"]
    if kwargs_branch != derived_counts["_prepare_from_posids"]:
        findings.append(
            f"{kwargs_branch} layers read the handed-over set where {derived_counts['_prepare_from_posids']} "
            f"derived it ({with_counts} / {derived_counts})"
        )
    if any(type(call[3]) is not int or type(call[4]) is not int for call in with_calls if call[0] == "varlen"):
        findings.append("a varlen kernel got a tensor max length")
    return [*findings, f"{REACH_PREFIX}{kwargs_branch}"]


def with_varlen_boundaries(batch: dict, cu_seq_lens: torch.Tensor) -> dict:
    """``batch`` handing ``cu_seq_lens`` as its varlen set; ``position_ids`` keep the true boundaries, so only a reader
    of the handed set sees these."""
    longest = int(cu_seq_lens.diff().max())
    return {
        **batch,
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": longest,
        "max_length_k": longest,
    }


def _shift_first_boundary(batch: dict) -> dict:
    """``batch`` with the set's first boundary one token later."""
    cu_seq_lens = batch["cu_seq_lens_q"].clone()
    cu_seq_lens[1] += 1
    return with_varlen_boundaries(batch, cu_seq_lens)


def tensors_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def recording_step(model, attn_implementation: str) -> Callable[[dict], tuple[dict, dict, dict]]:
    """One training step on ``model`` over the reference kernels: the loss and every gradient, and the derivation calls
    of its forward and of its backward."""

    def step(inputs):
        model.zero_grad(set_to_none=True)
        _kernels, counts = install_reference_flash(attn_implementation=attn_implementation)
        loss = model(**inputs, use_cache=False).loss
        forward = dict(counts)
        loss.backward()
        trained = {"loss": loss.detach()} | {
            param_name: param.grad.clone() for param_name, param in model.named_parameters() if param.grad is not None
        }
        return trained, forward, {key: counts[key] - forward[key] for key in counts}

    return step


def checkpointed_then_plain(model, step: Callable, batch: dict) -> tuple[dict | None, str | None, dict]:
    """What ``step`` trains on ``batch`` under ``model``'s checkpointing, or ``None`` and why the recompute was refused,
    then what it trains with the checkpointing off."""
    refused = None
    try:
        checkpointed = step(batch)[0]
    except CheckpointError as error:
        # Non-reentrant checkpointing refuses a recompute whose saved tensors differ from the forward's.
        checkpointed, refused = None, str(error).partition("\n")[0]
    model.gradient_checkpointing_disable()
    return checkpointed, refused, step(batch)[0]


def recompute_findings(name: str, gc_mode: str) -> list[str]:
    """What breaks when a GC training step hands flash its varlen set; a shifted boundary tells a recompute's
    re-derivation from reading the set."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).train()
    attn_implementation, batch = _packed_flash_batch(name, model.config)
    if not all(key in batch for key in VARLEN_KEYS):
        return ["the flash run's multi-document batch carried no varlen set"]
    reentrant = GC_MODES[gc_mode]
    if reentrant is not None:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": reentrant})

    step = recording_step(model, attn_implementation)
    with_set, _, set_backward = step(batch)
    derived, derived_forward, derived_backward = step(
        {key: value for key, value in batch.items() if key not in VARLEN_KEYS}
    )
    shifted, recompute_error, shifted_plain = checkpointed_then_plain(model, step, _shift_first_boundary(batch))

    findings = []
    if not tensors_equal(with_set, derived):
        findings.append("the varlen set changed the loss or a gradient")
    # Without the set, a recompute re-derives it in every layer whose forward did.
    derivations = derived_forward["_prepare_from_posids"]
    recomputed = derived_backward["_prepare_from_posids"]
    if not derivations or recomputed != (derivations if reentrant is not None else 0):
        findings.append(
            f"the backward re-derived the set in {recomputed} of the {derivations} attention layers whose forward "
            f"derived it, with gradient checkpointing {gc_mode}"
        )
    if set_backward["_is_packed_sequence"] or set_backward["_prepare_from_posids"]:
        findings.append(f"the recompute re-derived the varlen set from position_ids: {set_backward}")
    if recompute_error is not None or not tensors_equal(shifted, shifted_plain):
        findings.append(
            "the recompute attended over other boundaries than the set the forward was handed"
            + (f" ({recompute_error})" if recompute_error else "")
        )
    if tensors_equal(shifted_plain, with_set):
        findings.append("a shifted boundary trained the same step, so the recompute check cannot see a re-derivation")
    return findings


def mask_probe_findings(name: str) -> list[str]:
    """What breaks when a packed flash step builds masks without transformers' packed-sequence probe, which the family
    patches skip. A host read left in mask construction, a differing loss, logit or gradient, or an SDPA forward that
    no longer runs the probe it needs."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).train()
    if not getattr(model, "_supports_flash_attn", False):
        return [f"{SKIP_PREFIX} its model class declares no flash support, so its runs never resolve to flash"]
    if segment_markers_for(model.config).cu_seq_lens:
        return [f"{SKIP_PREFIX} its packed rows need the segment-aware GatedDeltaNet kernels, which no CPU runs"]
    attn_implementation, batch = _packed_flash_batch(name, model.config)
    patched = masking_utils._preprocess_mask_arguments
    if not getattr(patched, "_halo_packed_probe_skip", False):
        return ["the family patches left transformers' packed-sequence probe in place on a flash run"]

    def step():
        model.zero_grad(set_to_none=True)
        install_reference_flash(attn_implementation=attn_implementation)
        reads = MaskHostReads()
        with reads:
            outputs = model(**batch, use_cache=False)
        outputs.loss.backward()
        trained = {"loss": outputs.loss.detach(), "logits": outputs.logits.detach()} | {
            param_name: param.grad.clone() for param_name, param in model.named_parameters() if param.grad is not None
        }
        return trained, reads.count

    skipped, skipped_reads = step()
    masking_utils._preprocess_mask_arguments = patched.__wrapped__
    try:
        probed, probed_reads = step()
    finally:
        masking_utils._preprocess_mask_arguments = patched
    if not probed_reads:
        return [f"{SKIP_PREFIX} its masks never run transformers' packed-sequence probe"]

    findings = []
    if skipped_reads:
        findings.append(
            f"mask construction still read the device {skipped_reads} times ({probed_reads} with the probe)"
        )
    if not tensors_equal(skipped, probed):
        findings.append("skipping the probe changed the loss, the logits or a gradient")
    if getattr(model, "_supports_sdpa", False):
        set_attn_implementation(model.config, "sdpa")
        reads = MaskHostReads()
        with reads, torch.no_grad():
            model(**{key: value for key, value in batch.items() if key not in VARLEN_KEYS}, use_cache=False)
        if not reads.count:
            findings.append("an SDPA forward of the same rows skipped the probe its packed mask needs")
    return [*findings, f"{REACH_PREFIX}{probed_reads}"]


def report(findings: list[str]) -> None:
    print(FINDINGS_MARKER + "|".join(findings))


def main(name: str) -> None:
    report(family_findings(name))
