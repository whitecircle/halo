#!/usr/bin/env python
"""A tree row on the real kernels matches every example run alone.

The CPU tier proves the plans and the seams against reference kernels; this pins what only the CUDA
kernels can break. The gather form calls the flash varlen kernel with more keys than queries per
sequence and relies on its bottom-right causal and window alignment, in forward and backward — a
shape neither upstream nor the stock packed path exercises. The split form reads the fused-sink
log-sum-exp and backpropagates through it. A GatedDeltaNet layer runs its delta-rule kernel with one
initial state per varlen sequence and a gradient through both states, and its convolution over a
gathered row. Each model is finalized as a loaded one is (``finalize_loaded_model``), so the delta-rule kernel
autotunes under the FLA Blackwell restriction a training run gets.

One tiny bf16 model per kernel shape — GPT-OSS (live sinks, a window shorter than the branches),
Qwen3 MoE (no sink, no window), Qwen3.5 MoE at its release head shapes (FA4's head_dim-256 kernel, gather only,
and FA2; GatedDeltaNet layers) — runs a three-level tree row in every form its backends allow. Checked against the fp32
stock forward of each example on its own: hidden states and mixer gradients; the same row without
its plan as a negative control; root tokens of the split form bit-identical to the plain packed
forward after the first layer (the stock call is untouched for them); the recurrence warm-up running
every state variant a training row compiles; and the key gather's fp32 gradient accumulation against
the bf16 scatter at a fan-out where the latter visibly drifts. Every admitted backend's kernel is also
compared directly with the plain-torch kernel the CPU tier uses in its place, on sequences with more
keys than queries under a window. The models route every token to all of their experts: top-k routing flips on bf16 noise,
which would measure the router, not the mixers.
"""

import copy
import functools

import torch
from transformers import modeling_flash_attention_utils as flash_utils

from halo.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN
from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.models.attention.backend_setup import patch_transformers_flash_varlen_int_seqlen
from halo.models.attention.tree import (
    TREE_ATTN_IMPLEMENTATIONS,
    TREE_FAMILIES,
    TreeFamily,
    _GatherTokens,
    _on_device,
    install_tree_attention,
    resolve_tree_plan_options,
    warm_recurrent_mixer_kernels,
    warm_split_form_kernels,
)
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY, build_node_table, build_tree_attention_plan
from tests.common.harness import gpu_test_main
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.trees import concat_columns, example_rows, randomize_sinks, reference_varlen, tree_columns

WINDOW = 16
VOCAB = 1024
HIDDEN_REL_TOL = 4e-2  # bf16 through four layers against fp32
GRAD_COSINE_FLOOR = 0.999
LEAK_REL_FLOOR = 0.3  # a row without its plan must be far from the reference, or the checks compare nothing
# Config overrides per family: four layers, every expert active.
_ALL_EXPERTS = {"num_hidden_layers": 4, "num_experts_per_tok": 2}
FAMILIES = {
    "gpt_oss": {**_ALL_EXPERTS, "num_local_experts": 2, "sliding_window": WINDOW},
    "qwen3_moe": {**_ALL_EXPERTS, "num_experts": 2},
    "qwen3_5_moe_text": {
        **_ALL_EXPERTS,
        "num_experts": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    },
}
# The forms checked per backend. FA2 returns no log-sum-exp gradient and drops GPT-OSS's live sinks;
# FA4's head_dim-256 kernel (Qwen3.5) takes no log-sum-exp gradient either.
BACKENDS = {
    "gpt_oss": {"flash_attention_4": ("split", "gather")},
    "qwen3_moe": {"flash_attention_4": ("split", "gather"), "flash_attention_2": ("gather",)},
    "qwen3_5_moe_text": {"flash_attention_4": ("gather",), "flash_attention_2": ("gather",)},
}
_MIXERS = ("self_attn", "linear_attn")
# (takes a state, returns one, normalizes query and key itself) of the recurrence calls along a
# chain: its root, its middle, its leaf. A tree row is normalized once, ahead of the depths.
RECURRENCE_STATE_VARIANTS = {(False, True, False), (True, True, False), (True, False, False)}


def _tree_row(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Trees of several shapes in one row: trunks shorter and longer than the window, a second level,
    branches from 2 to 70 tokens, and a plain document."""

    def tokens(n: int) -> list[int]:
        return torch.randint(0, VOCAB, (n,), generator=generator).tolist()

    trees = []
    for trunk_len, fan_out in ((5, 3), (40, 5), (300, 6), (90, 1)):
        trunk, paths, trained = tokens(trunk_len), [], []
        for branch in range(fan_out):
            middle = tokens(30) if branch % 2 else []
            for tail_len in (2, 20, 70)[: 1 + branch % 3]:
                answer = 1 + branch % 4
                path = trunk + middle + tokens(tail_len + answer)
                paths.append(path)
                trained.append([False] * (len(path) - answer) + [True] * answer)
        trees.append(tree_columns(paths, trained))
    return {name: column.cuda() for name, column in concat_columns(trees).items()}


def _hidden(model, row, **kwargs) -> torch.Tensor:
    return model.model(
        input_ids=row["input_ids"][None], position_ids=row[POSITION_COLUMN][None], use_cache=False, **kwargs
    ).last_hidden_state[0]


def _loss(model, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = torch.nonzero(labels != LABEL_IGNORE_INDEX).flatten()
    logits = model.lm_head(hidden[targets - 1]).float()
    return torch.nn.functional.cross_entropy(logits, labels[targets], reduction="sum")


def _mixer_gradients(model) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.float().flatten()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and "sinks" not in name and any(mixer in name for mixer in _MIXERS)
        ]
    )


def _alone(model, row, examples) -> tuple[torch.Tensor, torch.Tensor]:
    """Hidden states of every example run on its own by the stock forward, laid back on the row, and
    the mixer gradients of the summed loss over their trained tokens."""
    model.zero_grad(set_to_none=True)
    hidden = torch.zeros(len(row["input_ids"]), model.config.hidden_size, device="cuda")
    for rows in examples:
        states = model.model(input_ids=row["input_ids"][rows][None], use_cache=False).last_hidden_state[0]
        hidden[rows] = states.detach().float()
        _loss(model, states, row["labels"][rows]).backward()
    return hidden, _mixer_gradients(model)


def _record_recurrence_states(family: TreeFamily) -> set[tuple[bool, bool, bool]]:
    """Note, for every call of the family's recurrence from here on, whether it took a state, whether
    it returned one and whether it normalized. Installed before tree attention, so it sees what the
    kernel is asked."""
    seen: set[tuple[bool, bool, bool]] = set()
    stock = getattr(family.module, family.stateful.recurrence)

    @functools.wraps(stock)
    def recording(*args, initial_state=None, output_final_state=False, **kwargs):
        seen.add((initial_state is not None, output_final_state, kwargs["use_qk_l2norm_in_kernel"]))
        return stock(*args, initial_state=initial_state, output_final_state=output_final_state, **kwargs)

    setattr(family.module, family.stateful.recurrence, recording)
    return seen


def _kernel_contract_errors(backend: str) -> tuple[float, float]:
    """The backend's varlen kernel against :func:`~tests.common.trees.reference_varlen`, output and
    key gradient: sequences with more keys than queries, the keys ahead of the queries shorter and
    longer than the window."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    heads, kv_heads, head_dim = 4, 2, 64
    query_lengths, ahead = (40, 7, 130, 25), (9, WINDOW + 20, 3, 0)
    key_lengths = [queries + keys for queries, keys in zip(query_lengths, ahead, strict=True)]

    def states(tokens: int, n_heads: int) -> torch.Tensor:
        return torch.randn(tokens, n_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

    q, k, v = states(sum(query_lengths), heads), states(sum(key_lengths), kv_heads), states(sum(key_lengths), kv_heads)
    probe = states(sum(query_lengths), heads).float()
    (_, varlen, *_), (flash_kwargs, _) = flash_utils.lazy_import_flash_attention(backend)
    call = {
        "cu_seqlens_q": torch.tensor([0, *query_lengths], dtype=torch.int32, device="cuda").cumsum(
            0, dtype=torch.int32
        ),
        "cu_seqlens_k": torch.tensor([0, *key_lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32),
        **flash_kwargs(
            query_length=sum(query_lengths),
            key_length=sum(key_lengths),
            is_causal=True,
            softmax_scale=head_dim**-0.5,
            sliding_window=WINDOW,
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(key_lengths),
        ),
    }
    results = []
    for kernel, dtype in ((varlen, torch.bfloat16), (reference_varlen([]), torch.float32)):
        keys = k.to(dtype, copy=True).requires_grad_(True)
        out = kernel(q.to(dtype), keys, v.to(dtype), **call)
        out = out[0] if isinstance(out, tuple) else out
        (out.float() * probe).sum().backward()
        results.append((out.detach().float(), keys.grad.float()))
    (out, grad), (reference_out, reference_grad) = results
    return max_abs_rel_err(out, reference_out), max_abs_rel_err(grad, reference_grad)


def _gather_accumulation_errors() -> tuple[float, float]:
    """Gradient of a key shared by sixty sequences: bf16 scatter and fp32 accumulation against fp32 truth."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    keys = torch.randn(64, 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    index = torch.arange(64, device="cuda").repeat(60)
    upstream = torch.randn(len(index), 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    truth = torch.zeros(keys.shape, device="cuda").index_add_(0, index, upstream.float())
    errors = []
    for gather in (lambda: keys.index_select(0, index), lambda: _GatherTokens.apply(keys, index, 0)):
        keys.grad = None
        gather().backward(upstream)
        errors.append(max_abs_rel_err(keys.grad, truth))
    return errors[0], errors[1]


def _family_checks(name: str, row: dict[str, torch.Tensor]) -> dict[str, bool]:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[name], overrides=FAMILIES[name])
    finalize_loaded_model(model)
    randomize_sinks(model)
    family = TREE_FAMILIES[name]
    recurrence_states = _record_recurrence_states(family) if family.stateful else None
    table = build_node_table(row[POSITION_COLUMN].cpu(), row[DEPTH_COLUMN].cpu())
    examples = [rows.cuda() for rows in example_rows(table)]
    reference, truth = _alone(copy.deepcopy(model).float().cuda().train(), row, examples)
    model = model.to(torch.bfloat16).cuda()
    options = resolve_tree_plan_options(model.config)
    plans = {
        form: {
            TREE_ATTENTION_KEY: _on_device(
                build_tree_attention_plan(table, full_form=form, **options), row["labels"].device
            )
        }
        for form in ("split", "gather")
    }

    checks = {}
    for backend, forms in BACKENDS[name].items():
        tag = f"{name}_fa{backend[-1]}"
        model.config._attn_implementation = backend
        install_tree_attention(model, forms[0])
        model.eval()
        with torch.no_grad():
            for form in forms:
                rel = max_abs_rel_err(_hidden(model, row, **plans[form]), reference)
                log(f"  [{tag}, {form}] hidden vs each example alone (fp32) rel {rel:.2e}")
                checks[f"{tag}_{form}_hidden_matches_examples_alone"] = rel < HIDDEN_REL_TOL
            leak = max_abs_rel_err(_hidden(model, row), reference)
            log(f"  [{tag}, no plan] hidden vs each example alone (fp32) rel {leak:.2e}")
            checks[f"{tag}_row_without_plan_is_far_from_examples_alone"] = leak > LEAK_REL_FLOOR
        model.train()
        for form in forms:
            model.zero_grad(set_to_none=True)
            _loss(model, _hidden(model, row, **plans[form]), row["labels"]).backward()
            cosine = torch.nn.functional.cosine_similarity(_mixer_gradients(model), truth, dim=0).item()
            log(f"  [{tag}, {form}] mixer gradient cosine vs examples alone (fp32) {cosine:.6f}")
            checks[f"{tag}_{form}_gradients_match_examples_alone"] = cosine > GRAD_COSINE_FLOOR

    if recurrence_states is not None:
        recurrence_states.clear()
        warm_recurrent_mixer_kernels(model, dtype=torch.bfloat16)
        checks[f"{name}_recurrence_warm_up_runs_every_state_variant"] = recurrence_states == RECURRENCE_STATE_VARIANTS
    if "split" not in BACKENDS[name].get("flash_attention_4", ()):
        return checks

    # A root has no ancestors, so the split form leaves its tokens to the stock packed call: the
    # first full layer's attention gives them what it gives the same roots packed on their own.
    model.config._attn_implementation = "flash_attention_4"
    model.eval()
    with torch.no_grad():
        attention = next(layer.self_attn for layer in model.model.layers if layer.self_attn.sliding_window is None)
        positions = row[POSITION_COLUMN][None]
        states = model.model.embed_tokens(row["input_ids"][None])
        rotary = model.model.rotary_emb(states, positions)
        roots = torch.from_numpy(table.parents == -1).repeat_interleave(torch.from_numpy(table.lengths)).cuda()
        tree_out, _ = attention(states, rotary, None, position_ids=positions, **plans["split"])
        packed_out, _ = attention(
            states[:, roots], tuple(part[:, roots] for part in rotary), None, position_ids=positions[:, roots]
        )
        checks[f"{name}_split_root_tokens_bit_identical_to_packed"] = torch.equal(tree_out[0, roots], packed_out[0])
        checks[f"{name}_split_branch_tokens_differ_from_packed"] = not torch.equal(
            tree_out, attention(states, rotary, None, position_ids=positions)[0]
        )
    warm_split_form_kernels(model.config, dtype=torch.bfloat16)
    return checks


@gpu_test_main(exact_world_size=1, prefix="fa4_tree_attention")
def run(ctx):
    ensure_fa4_kernel_cache_env()
    patch_transformers_flash_varlen_int_seqlen()
    row = _tree_row(torch.Generator().manual_seed(0))
    table = build_node_table(row[POSITION_COLUMN].cpu(), row[DEPTH_COLUMN].cpu())
    log(f"  row: {len(row['input_ids'])} tokens, {len(table.starts)} nodes, depth {int(row[DEPTH_COLUMN].max())}")

    checks = {}
    for name in FAMILIES:
        checks.update(_family_checks(name, row))

    for backend in TREE_ATTN_IMPLEMENTATIONS:
        out_error, grad_error = _kernel_contract_errors(backend)
        log(
            f"  [{backend}] windowed kernel vs the reference kernel: out rel {out_error:.2e}, key grad {grad_error:.2e}"
        )
        checks[f"{backend}_kernel_matches_the_reference_kernel"] = max(out_error, grad_error) < HIDDEN_REL_TOL

    scatter_error, accumulated_error = _gather_accumulation_errors()
    log(
        f"  shared-key gradient vs fp32 truth: bf16 scatter {scatter_error:.2e}, fp32 accumulation {accumulated_error:.2e}"
    )
    checks["gather_backward_accumulates_in_fp32"] = accumulated_error < 5e-3 < scatter_error
    return {"checks": checks}


if __name__ == "__main__":
    run()
