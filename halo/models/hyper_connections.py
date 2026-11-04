"""Upstream's hyper-connection forwards on the fused kernels, for the families whose mHC residual mixes through a
Sinkhorn-projected matrix. Idempotent; a moved upstream signature fails loud, and an input off CUDA or in fp64, which the
kernels compute in fp32, runs upstream's own forward. ``HALO_HC_KERNELS=0`` leaves upstream's classes as they are."""

import functools
import inspect
import logging

import torch
import transformers.models.deepseek_v4.modeling_deepseek_v4 as deepseek_v4_modeling
import transformers.models.glm5_next.modeling_glm5_next as glm5_next_modeling

from halo.env import env_flag
from halo.kernels.hyper_connection import collapse_streams, mix_streams, project_streams, sinkhorn_knopp
from halo.models.config_access import model_type_matches
from halo.models.moe.family_specs import DEEPSEEK_V4, GLM5_NEXT
from halo.models.upstream_guard import verify_upstream_parameters
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_halo_hc_kernels"


def _hyper_connection_forward(self, hidden_streams):
    """``*HyperConnection.forward`` (transformers ~5.20) with its projection, Sinkhorn loop and collapse on the kernels."""
    hc = self.hc_mult
    projected = project_streams(hidden_streams, self.fn, self.input_norm.eps)
    pre_w, post_w, comb_w = projected.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = self.scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    return post, sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps), collapse_streams(hidden_streams, pre)


def _mixed(block, residual, post, comb):
    """Upstream casts both weights to the streams' run dtype ahead of the mix, so they round first here too."""
    return mix_streams(block, residual, post.to(residual.dtype), comb.to(residual.dtype))


def _glm5_next_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    prev_topk_indices=None,
    **kwargs,
):
    """``Glm5NextTextDecoderLayer.forward`` (transformers ~5.20) with its mixes on the kernel."""
    post, comb, collapsed = self.attn_hc(hidden_states)
    collapsed = self.input_layernorm(collapsed)
    topk_indices = None
    if self.block_type == "linear_attention":
        block = self.self_attn(
            hidden_states=collapsed,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    else:
        block, _, topk_indices = self.self_attn(
            hidden_states=collapsed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
    hidden_states = _mixed(block, hidden_states, post, comb)
    post, comb, collapsed = self.ffn_hc(hidden_states)
    block = self.mlp(self.post_attention_layernorm(collapsed))
    return _mixed(block, hidden_states, post, comb), topk_indices


def _deepseek_v4_layer_forward(self, hidden_states, input_ids=None, **kwargs):
    """``DeepseekV4DecoderLayer.forward`` (transformers ~5.20) with its mixes on the kernel."""
    post, comb, collapsed = self.attn_hc(hidden_states)
    block, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    hidden_states = _mixed(block, hidden_states, post, comb)
    post, comb, collapsed = self.ffn_hc(hidden_states)
    block = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
    return _mixed(block, hidden_states, post, comb)


def _deepseek_v4_head_forward(self, x):
    """``DeepseekV4HyperHead.forward`` (transformers ~5.20) with its projection and collapse on the kernels. Outside any
    decoder layer, so its fp32 copies of the streams would outlast per-layer checkpointing to the backward."""
    mixes = project_streams(x, self.hc_fn, self.input_norm.eps)
    return collapse_streams(x, torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps)


# Each family's classes that read the streams, with the forward that replaces upstream's. HY-V4's hyper-connection has
# no mixing matrix, and it is off the roster.
_FAMILIES = {
    GLM5_NEXT: (
        (glm5_next_modeling.Glm5NextTextHyperConnection, _hyper_connection_forward),
        (glm5_next_modeling.Glm5NextTextDecoderLayer, _glm5_next_layer_forward),
    ),
    DEEPSEEK_V4: (
        (deepseek_v4_modeling.DeepseekV4HyperConnection, _hyper_connection_forward),
        (deepseek_v4_modeling.DeepseekV4DecoderLayer, _deepseek_v4_layer_forward),
        (deepseek_v4_modeling.DeepseekV4HyperHead, _deepseek_v4_head_forward),
    ),
}


def _install(owner: type, fused) -> None:
    stock = owner.forward
    if getattr(stock, _PATCHED_ATTR, False):
        return
    verify_upstream_parameters(stock, tuple(inspect.signature(fused).parameters), "apply_hyper_connection_patches")

    @functools.wraps(stock)
    def forward(self, hidden_states, *args, **kwargs):
        on_kernels = hidden_states.is_cuda and hidden_states.dtype != torch.float64
        return (fused if on_kernels else stock)(self, hidden_states, *args, **kwargs)

    setattr(forward, _PATCHED_ATTR, True)
    owner.forward = forward
    if is_global_main_process():
        logger.info(f"Patched {owner.__name__}.forward onto the hyper-connection kernels")


def apply_hyper_connection_patches(model_config) -> None:
    if not env_flag("HALO_HC_KERNELS", True):
        return
    for family, patches in _FAMILIES.items():
        if model_type_matches(model_config, *family.model_types):
            for owner, fused in patches:
                _install(owner, fused)
