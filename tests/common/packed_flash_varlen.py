"""The packed-row varlen sweep: a tiny family's forward fed the packing collator's varlen set against the same
forward left to derive it from ``position_ids``, both through transformers' own flash dispatch; and a training
step under gradient checkpointing, whose backward recompute must read the same handed-over set.

The flash kernels are CPU reference stand-ins installed where transformers caches its loaded kernels, and the
family's production flash patches are applied, so the sweep rebinds process-global state: it runs in a fresh
interpreter per family (:func:`main`, :func:`report`) and reports through one ``FINDINGS:`` line.
"""

# The device-aware kernel-dispatch shim must land before the modeling modules bind transformers' hub-kernel
# fallback factory at import, or a CPU forward reaches the CUDA-only kernels.
import halo.kernels.hub_kernel_dispatch  # noqa: F401  # isort: skip

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
# Report entries that are not failures: a family the change does not reach, and how many layers it reached.
SKIP_PREFIX = "skip:"
REACH_PREFIX = "reach:"
VARLEN_KEYS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
PAD_ID = 0
EOS_ID = 1
SEED = 1234
# The production flash backend on Blackwell; its int-seqlen patch is part of what the sweep holds.
SWEEP_ATTN_IMPLEMENTATION = "flash_attention_4"
# The tensor reads that hand a value to the host: on a device each one waits for the device's queued work.
HOST_READS = frozenset({"__bool__", "__float__", "__index__", "__int__", "cpu", "item", "nonzero", "numpy", "tolist"})

_TORCH_DIR = os.path.dirname(torch.__file__)

# Rows as TRL's pack_dataset emits them, two packed rows of several documents each.
MULTI_DOCUMENT_ROWS = (
    {"input_ids": [5, 6, 7, 8, 9, 10, 11], "seq_lengths": [3, 4]},
    {"input_ids": [12, 13, 14, 15, 16], "seq_lengths": [2, 2, 1]},
)

# Every tiny family whose packed rows reach the packing collator, keyed as the GPU suites name them.
# DeepSeek-V4 refuses packing (compressed-KV windows), so it has no packed row to sweep.
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
# The flash label a family's packed rows train under where it is not the sweep's: a softcap stays off FA4's varlen
# kernels (Gemma 2), and its dedicated head_dim-256 kernel takes no sliding window (Gemma 3 at that width).
PACKED_ATTN_IMPLEMENTATIONS = {"dense:gemma2": "flash_attention_2", "dense:gemma3_text": "flash_attention_2"}
# Gradient checkpointing per recompute-check mode: off, or the checkpoint at that ``use_reentrant``.
GC_MODES = {"off": None, "non_reentrant": False, "reentrant": True}
# Families whose model forward passes its attention layers no extra kwargs, so the set the collator hands it never
# reaches attention, which derives its own from position_ids once per step (Ling 2.0's remote forward;
# tests/cpu/models/test_remote_flash_packed_isolation.py pins that).
SET_DROPPING_FAMILIES = frozenset({"bailing_moe"})


def padding_tokenizer() -> MagicMock:
    """The tokenizer surface the packing collators read: right padding with ``PAD_ID``."""
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
    """Reference attention over one ``[S, H, D]`` sequence: GQA by head repeat, a causal and a sliding mask."""
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
    """CPU stand-ins for the flash kernels at FA2's signatures, recording each call's varlen arguments, sliding
    window and softcap."""

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
    """While active, records each :data:`HOST_READS` call made inside a flash attention interface call and outside
    its kernels: the reads every attention layer of a forward makes, each a host sync on a device (a
    recompute in the backward runs outside the mode, which :func:`count_derivations` still sees).

    :func:`install_reference_flash` marks the kernels, :meth:`attending` the interface.
    """

    def __init__(self):
        super().__init__()
        self.reads: list[str] = []
        self._depth = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if self._depth > 0 and getattr(func, "__name__", None) in HOST_READS:
            self.reads.append(func.__name__)
        return func(*args, **(kwargs or {}))

    def scoped(self, fn: Callable, depth: int) -> Callable:
        """``fn`` shifting the recording depth by ``depth`` for the length of each call."""

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self._depth += depth
            try:
                return fn(*args, **kwargs)
            finally:
                self._depth -= depth

        return wrapped

    def attending(self, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION) -> None:
        """Record inside ``attn_implementation``'s attention interface, the call every attention layer makes."""
        interface = ALL_ATTENTION_FUNCTIONS[attn_implementation]
        interface = getattr(interface, "_recorded_original", interface)
        recorded = self.scoped(interface, 1)
        recorded._recorded_original = interface
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, recorded)


def _first_caller_outside_torch() -> str:
    """The file of the innermost frame on the stack outside torch and this module."""
    frame = sys._getframe(1)
    while frame is not None and (
        frame.f_code.co_filename.startswith(_TORCH_DIR) or frame.f_code.co_filename == __file__
    ):
        frame = frame.f_back
    return "" if frame is None else frame.f_code.co_filename


class MaskHostReads(TorchFunctionMode):
    """While active, counts the :data:`HOST_READS` calls transformers' mask construction (``masking_utils``) makes."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", None) in HOST_READS and _first_caller_outside_torch() == masking_utils.__file__:
            self.count += 1
        return func(*args, **(kwargs or {}))


def count_derivations() -> dict[str, int]:
    """Call counts of the per-layer derivation a handed-over varlen set replaces (and of the padding-mask path,
    which precedes it), counted from here on."""
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
    """Stop ``reads`` recording inside the flash kernels transformers holds loaded: a kernel's own reads are not the
    per-layer derivation's, and the reference kernels read every offset back."""
    for name in ("_flash_fn", "_flash_varlen_fn"):
        kernel = getattr(flash_utils, name)
        kernel = getattr(kernel, "_hidden_original", kernel)
        hidden = reads.scoped(kernel, -1)
        hidden._hidden_original = kernel
        setattr(flash_utils, name, hidden)


def install_loaded_flash(reads: AttentionHostReads, attn_implementation: str) -> dict[str, int]:
    """The device counterpart of :func:`install_reference_flash`: the kernels transformers loads for
    ``attn_implementation``, recorded around (:func:`hide_kernels`), plus :func:`count_derivations`."""
    flash_utils.lazy_import_flash_attention(attn_implementation)
    hide_kernels(reads)
    return count_derivations()


def install_reference_flash(
    reads: AttentionHostReads | None = None, *, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION
) -> tuple[ReferenceFlash, dict[str, int]]:
    """Fresh reference kernels where transformers caches its loaded ``attn_implementation`` kernels, plus
    :func:`count_derivations`; ``reads`` records around them (:func:`hide_kernels`)."""
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
    """``varlen`` holds the varlen set exactly where transformers takes its varlen kernel on ``position_ids``, equal
    to what it derives: the same int32 offsets, and host-int max lengths."""
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
    """Resolve ``model_config``'s decoder, at both config levels, to ``attn_implementation``."""
    for level in (model_config, model_config.get_text_config()):
        level._attn_implementation = attn_implementation


def use_flash(model_config, attn_implementation: str = SWEEP_ATTN_IMPLEMENTATION) -> None:
    """Resolve ``model_config``'s decoder to ``attn_implementation`` and apply its production family patches."""
    set_attn_implementation(model_config, attn_implementation)
    apply_family_patches(model_config, attn_implementation)


def _packed_flash_batch(name: str, config) -> tuple[str, dict]:
    """Resolve ``config`` to the flash label ``name``'s packed rows train under and apply its production family
    patches; return that label and the packing collator's multi-document batch."""
    attn_implementation = PACKED_ATTN_IMPLEMENTATIONS.get(name, SWEEP_ATTN_IMPLEMENTATION)
    use_flash(config, attn_implementation)
    collator = select_data_collator(
        padding_tokenizer(), train_on_completions_only=False, packing=True, model_config=config
    )
    return attn_implementation, collator.torch_call([dict(row) for row in MULTI_DOCUMENT_ROWS])


def family_findings(name: str) -> list[str]:
    """What breaks when ``name``'s packed batch hands flash its varlen set, then a ``reach:`` entry counting
    the attention calls that read it; a lone ``skip:`` entry for a family the change does not reach."""
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
    # Every layer that derived the set from position_ids now reads the handed-over one: none dropped it.
    kwargs_branch = sum(call[0] == "varlen" for call in with_calls) - with_counts["_upad_input"]
    if kwargs_branch != derived_counts["_prepare_from_posids"]:
        findings.append(
            f"{kwargs_branch} layers read the handed-over set where {derived_counts['_prepare_from_posids']} "
            f"derived it ({with_counts} / {derived_counts})"
        )
    if any(type(call[3]) is not int or type(call[4]) is not int for call in with_calls if call[0] == "varlen"):
        findings.append("a varlen kernel got a tensor max length")
    return [*findings, f"{REACH_PREFIX}{kwargs_branch}"]


def _shift_first_boundary(batch: dict) -> dict:
    """``batch`` with its varlen set's first document boundary one token later. ``position_ids`` keep the true
    boundary, so only a pass that reads the handed-over set sees the shift."""
    cu_seq_lens = batch["cu_seq_lens_q"].clone()
    cu_seq_lens[1] += 1
    longest = int(cu_seq_lens.diff().max())
    return {
        **batch,
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": longest,
        "max_length_k": longest,
    }


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def recompute_findings(name: str, gc_mode: str) -> list[str]:
    """What breaks when a training step on ``name``'s packed batch, checkpointed per ``gc_mode``
    (:data:`GC_MODES`), hands flash its varlen set: the backward recompute must re-run attention on the handed-over
    set rather than re-derive it from ``position_ids``. A set with a shifted boundary tells the two apart, since
    the derivation would restore the true boundary in the recompute alone."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PACKING_FAMILIES[name]).train()
    attn_implementation, batch = _packed_flash_batch(name, model.config)
    if not all(key in batch for key in VARLEN_KEYS):
        return ["the flash run's multi-document batch carried no varlen set"]
    reentrant = GC_MODES[gc_mode]
    if reentrant is not None:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": reentrant})

    def step(inputs):
        """The loss and every gradient of one step on ``inputs``, and the derivation calls its forward and its
        backward made."""
        model.zero_grad(set_to_none=True)
        _kernels, counts = install_reference_flash(attn_implementation=attn_implementation)
        loss = model(**inputs, use_cache=False).loss
        forward = dict(counts)
        loss.backward()
        trained = {"loss": loss.detach()} | {
            param_name: param.grad.clone() for param_name, param in model.named_parameters() if param.grad is not None
        }
        return trained, forward, {key: counts[key] - forward[key] for key in counts}

    with_set, _, set_backward = step(batch)
    derived, derived_forward, derived_backward = step(
        {key: value for key, value in batch.items() if key not in VARLEN_KEYS}
    )
    shifted_batch = _shift_first_boundary(batch)
    recompute_error = None
    try:
        shifted = step(shifted_batch)[0]
    except CheckpointError as error:
        # Non-reentrant checkpointing refuses a recompute whose saved tensors differ from the forward's.
        shifted, recompute_error = None, str(error).partition("\n")[0]
    model.gradient_checkpointing_disable()
    shifted_plain = step(shifted_batch)[0]

    findings = []
    if not _same(with_set, derived):
        findings.append("the varlen set changed the loss or a gradient")
    # Anti-vacuity: without the set, a checkpoint's recompute re-derives it in every layer whose forward did.
    derivations = derived_forward["_prepare_from_posids"]
    recomputed = derived_backward["_prepare_from_posids"]
    if not derivations or recomputed != (derivations if reentrant is not None else 0):
        findings.append(
            f"the backward re-derived the set in {recomputed} of the {derivations} attention layers whose forward "
            f"derived it, with gradient checkpointing {gc_mode}"
        )
    if set_backward["_is_packed_sequence"] or set_backward["_prepare_from_posids"]:
        findings.append(f"the recompute re-derived the varlen set from position_ids: {set_backward}")
    if recompute_error is not None or not _same(shifted, shifted_plain):
        findings.append(
            "the recompute attended over other boundaries than the set the forward was handed"
            + (f" ({recompute_error})" if recompute_error else "")
        )
    if _same(shifted_plain, with_set):
        findings.append("a shifted boundary trained the same step, so the recompute check cannot see a re-derivation")
    return findings


def mask_probe_findings(name: str) -> list[str]:
    """What breaks when ``name``'s packed flash training step builds its masks without transformers' packed-sequence
    probe, which the family patches skip on a flash label: a host read left in mask construction, or a loss, logit or
    gradient that differs from the same step with the probe; an SDPA forward of the same rows that no longer runs the
    probe its packed mask needs. A ``reach:`` entry counts the reads the probe makes; a lone ``skip:`` entry marks a
    family whose masks never run it."""
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
    if not _same(skipped, probed):
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
    """Print ``findings`` as the one line :func:`tests.common.repo.probe_findings` reads."""
    print(FINDINGS_MARKER + "|".join(findings))


def main(name: str) -> None:
    report(family_findings(name))
