"""The keyed random init a from-scratch run draws (issue #146):
- every family's tensors are all drawn, constants exactly as ``from_config`` sets them and random ones from the same
  distribution,
- the draw depends only on the seed and the model, not on the global random state, which it leaves as it found it,
- an expert drawn alone, as a rank holding a slice of the bank will draw it, equals its slice of the whole draw,
- no expert or layer starts as a copy of another,
- a tied pair is drawn once, by the embedding, as ``from_config`` leaves it,
- every MoE family's draw, through the gathered save's hub layout and back through ``from_pretrained``, is the draw,
- two gloo ranks that enter ``load_whole_model`` with different global seeds hold the same weights.
Tiny models on CPU."""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.writing import normalize_gathered_state_dict
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.keyed_init import (
    build_keyed_model,
    draw_module,
    fused_expert_banks,
    init_order,
)
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.whole_model import load_whole_model
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES

FAMILIES = {**TINY_MOE_FAMILIES, **{f"dense_{name}": family for name, family in TINY_DENSE_FAMILIES.items()}}
BANKED = ("gpt_oss", "qwen3_moe", "deepseek_v4", "zaya")
SEED = 1234
# Below this many elements a sample's mean and std say little; such a tensor is only checked to be random too.
MIN_STAT_ELEMENTS = 64
# Both sides are samples, so a gap of this many standard errors of their difference fails.
SIGMAS = 5.0


def _config(name: str):
    return FAMILIES[name].build({}).config


def _keyed(name: str, seed: int = SEED) -> nn.Module:
    family = FAMILIES[name]
    return build_keyed_model(
        family.load_class, _config(name), torch.float32, seed=seed, trust_remote_code=family.trust_remote_code
    )


def _from_config(name: str) -> nn.Module:
    family = FAMILIES[name]
    factory = getattr(family.load_class, "from_config", None) or family.load_class._from_config
    kwargs = {"trust_remote_code": family.trust_remote_code} if hasattr(family.load_class, "from_config") else {}
    return factory(_config(name), dtype=torch.float32, **kwargs)


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_tensor_is_drawn_as_from_config_draws_it(name):
    """Constants (zeros, ones, ``A_log``-style fills) exactly, random tensors within a few standard errors in mean and
    std. A family whose ``_init_weights`` left a tensor undrawn would raise in the build."""
    keyed = _keyed(name).state_dict()
    reference = _from_config(name).state_dict()
    assert keyed.keys() == reference.keys()
    off = []
    for key, ref in reference.items():
        ref, got = ref.float(), keyed[key].float()
        if not ref.is_floating_point() or ref.numel() < 2:
            continue
        if ref.std() == 0:
            if not torch.equal(got, ref):
                off.append(f"{key}: constant {ref.flatten()[0].item():.4g}, keyed mean {got.mean().item():.4g}")
        elif got.std() == 0:
            off.append(f"{key}: random in from_config, constant when keyed")
        elif ref.numel() >= MIN_STAT_ELEMENTS:
            n, sigma = ref.numel(), ref.std().item()
            if abs(got.std().item() - sigma) > SIGMAS * sigma / n**0.5:
                off.append(f"{key}: std {got.std().item():.4g} against {sigma:.4g} (n={n})")
            if abs(got.mean().item() - ref.mean().item()) > SIGMAS * sigma * (2 / n) ** 0.5:
                off.append(f"{key}: mean {got.mean().item():.4g} against {ref.mean().item():.4g} (n={n})")
    assert not off, off


def _family_init_skipping(monkeypatch, suffix: str) -> None:
    """Qwen3-MoE's ``_init_weights`` with the modules whose class name ends in ``suffix`` left out."""
    # The class defining it: the backbone is a PreTrainedModel of its own and dispatches through it too.
    owner = next(cls for cls in type(FAMILIES["qwen3_moe"].build({})).__mro__ if "_init_weights" in cls.__dict__)
    family_init = owner._init_weights

    def skipping(self, module):
        if not type(module).__name__.endswith(suffix):
            family_init(self, module)

    monkeypatch.setattr(owner, "_init_weights", skipping)


def test_a_tensor_no_init_draws_is_refused_by_name(monkeypatch):
    """Its NaN fill would otherwise train. The router has no ``reset_parameters`` and transformers' default init knows
    no router."""
    _family_init_skipping(monkeypatch, "TopKRouter")
    with pytest.raises(RuntimeError, match=r"parameter\(s\) undrawn.*mlp\.gate\.weight"):
        _keyed("qwen3_moe")


def test_an_embedding_the_family_init_skips_takes_its_own_reset_parameters(monkeypatch):
    """The init its constructor runs, N(0, 1), as a remote-code router's ``reset_parameters`` draws it, not the default
    init's N(0, initializer_range)."""
    _family_init_skipping(monkeypatch, "Embedding")
    embedding = _keyed("qwen3_moe").get_input_embeddings().weight
    assert torch.isfinite(embedding).all() and 0.8 < embedding.std().item() < 1.2


def test_a_norm_the_family_init_skips_takes_the_default_init(monkeypatch):
    """As a remote-code family's override can skip its norms, which ``from_config`` leaves at their constructor ones."""
    _family_init_skipping(monkeypatch, "RMSNorm")
    norms = {name: p for name, p in _keyed("qwen3_moe").named_parameters() if "norm" in name}
    assert norms and all(torch.equal(p, torch.ones_like(p)) for p in norms.values())


def _keyed_from(config, seed: int = SEED) -> nn.Module:
    family = FAMILIES["qwen3_moe"]
    return build_keyed_model(family.load_class, config, torch.float32, seed=seed, trust_remote_code=False)


def test_the_draw_ignores_the_global_random_state_and_leaves_it_as_it_was():
    config = _config("qwen3_moe")
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    first = _state_hash(_keyed_from(config))
    assert torch.equal(torch.random.get_rng_state(), before), "the build moved the global CPU generator"
    torch.manual_seed(99)
    torch.rand(1000)
    assert _state_hash(_keyed_from(config)) == first


def test_another_seed_draws_every_random_tensor_anew():
    first, second = _keyed("qwen3_moe", seed=1).state_dict(), _keyed("qwen3_moe", seed=2).state_dict()
    same = [key for key, tensor in first.items() if tensor.std() > 0 and torch.equal(tensor, second[key])]
    assert not same, same


@pytest.mark.parametrize("name", BANKED)
def test_an_expert_drawn_alone_equals_its_slice_of_the_whole_draw(name):
    """The draw a rank holding the upper half of every bank makes (ep2's rank 1): only those experts, keyed by their
    global index. The lower half stays as it was."""
    model = _keyed(name)
    banks = fused_expert_banks(model)
    assert banks, f"{name} declares no fused expert bank"
    order = {module_name: (module, init_fn) for module_name, module, init_fn in init_order(model)}
    for bank, num_experts in banks.items():
        module, init_fn = order[bank]
        whole = {n: p.detach().clone() for n, p in module.named_parameters(recurse=False)}
        with torch.no_grad():
            for param in module.parameters(recurse=False):
                param.fill_(float("nan"))
        upper = range(num_experts // 2, num_experts)
        with torch.no_grad():
            draw_module(module, init_fn, seed=SEED, name=bank, num_experts=num_experts, experts=upper)
        for param_name, param in module.named_parameters(recurse=False):
            if param.shape[0] != num_experts:
                continue
            assert torch.equal(param[upper.start :], whole[param_name][upper.start :]), f"{bank}.{param_name}"
            assert torch.isnan(param[: upper.start]).all(), f"{bank}.{param_name}: drew experts outside the slice"


@pytest.mark.parametrize("name", BANKED)
def test_no_expert_or_layer_starts_as_a_copy(name):
    model = _keyed(name)
    copies = []
    for bank, num_experts in fused_expert_banks(model).items():
        for param_name, param in model.get_submodule(bank).named_parameters(recurse=False):
            if not param.is_floating_point() or param.shape[0] != num_experts or param[0].std() == 0:
                continue
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    if torch.equal(param[i], param[j]):
                        copies.append(f"{bank}.{param_name}: experts {i} and {j}")
    by_suffix: dict[str, list[tuple[str, torch.Tensor]]] = {}
    # Parameters only: a buffer may hold the same computed value in every layer by design.
    for key, tensor in model.named_parameters():
        if ".layers." in key and tensor.is_floating_point() and tensor.std() > 0:
            prefix, _, rest = key.partition(".layers.")
            _layer, _, suffix = rest.partition(".")
            by_suffix.setdefault(f"{prefix}|{suffix}", []).append((key, tensor))
    for entries in by_suffix.values():
        for i, (key_i, t_i) in enumerate(entries):
            for key_j, t_j in entries[i + 1 :]:
                if t_i.shape == t_j.shape and torch.equal(t_i, t_j):
                    copies.append(f"{key_i} == {key_j}")
    assert not copies, copies


@pytest.mark.parametrize("name", sorted(TINY_MOE_FAMILIES))
def test_a_gathered_save_of_the_draw_reloads_as_the_draw(name, tmp_path):
    """The GPU gate's CPU half for every MoE family: a training checkpoint's gathered save writes the draw in the
    family's hub layout beside its config and remote code, and a stock reload reads back the draw bit for bit."""
    family = TINY_MOE_FAMILIES[name]
    model = _keyed(name)
    drawn = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    on_disk = normalize_gathered_state_dict(model, model.state_dict(), keep_live_dtype=True)
    save_model_config(model, str(tmp_path))
    save_file(
        {key: tensor.detach().clone().contiguous() for key, tensor in on_disk.items()}, tmp_path / "model.safetensors"
    )
    reloaded = family.load_class.from_pretrained(
        tmp_path, dtype=torch.float32, trust_remote_code=family.trust_remote_code
    ).state_dict()
    assert set(reloaded) == set(drawn), sorted(set(reloaded) ^ set(drawn))[:8]
    differ = sorted(key for key in drawn if not torch.equal(reloaded[key], drawn[key]))
    assert not differ, differ[:8]


def test_a_tied_pair_is_drawn_once_as_from_config_leaves_it():
    """``from_config`` draws the embedding (its padding row zeroed), then ties the head to it, so the head's own
    draw never reaches the pair."""
    tiny = {**TINY_QWEN3_CONFIG, "pad_token_id": 0}
    tied = build_keyed_model(
        Qwen3ForCausalLM,
        Qwen3Config(**{**tiny, "tie_word_embeddings": True}),
        torch.float32,
        seed=SEED,
        trust_remote_code=False,
    )
    untied = build_keyed_model(
        Qwen3ForCausalLM, Qwen3Config(**tiny), torch.float32, seed=SEED, trust_remote_code=False
    )
    embedding = tied.get_input_embeddings().weight
    assert tied.get_output_embeddings().weight is embedding
    assert torch.equal(embedding[0], torch.zeros_like(embedding[0])), "the head's draw overwrote the padding row"
    assert torch.equal(embedding, untied.get_input_embeddings().weight)


def _load_with_rank_seed(rank: int, config) -> None:
    """Each rank reseeds its global generators differently, then loads through the whole-model route."""
    torch.manual_seed(1000 + rank)
    model = load_whole_model(
        TINY_MOE_FAMILIES["qwen3_moe"].load_class,
        ModelSource("tiny", None, "/nonexistent", random_init_seed=SEED),
        ParallelismConfig(),
        load_phase="Random init",
        ep_config=None,
        config=config,
        dtype=torch.float32,
        trust_remote_code=False,
        place_on_device=False,
    )
    digest = torch.tensor(list(bytes.fromhex(_state_hash(model))), dtype=torch.uint8)
    gathered = [torch.empty_like(digest) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, digest)
    assert all(torch.equal(g, gathered[0]) for g in gathered), "ranks drew different weights"


def test_two_ranks_with_different_global_seeds_load_the_same_weights():
    run_gloo_ranks(_load_with_rank_seed, 2, _config("qwen3_moe"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
