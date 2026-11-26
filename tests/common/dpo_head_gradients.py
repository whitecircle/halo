"""Pinned head operands and gradients for diagnosing DPO compute-dtype differences."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from unittest.mock import patch

import torch
import torch.nn.functional as F

import halo.trainers.preference.chunked_dpo as chunked_dpo
from halo.models.introspection.unwrapping import base_transformers_model


def gradient_difference(actual, expected):
    error = (actual.double() - expected.double()).norm()
    norm = expected.double().norm()
    return {"absolute_l2": float(error), "relative_l2": float(error / norm) if norm else None}


def full_head_derivative(
    hidden, weight, targets, upstream, *, quantize_logits_gradient=False, round_forward_logits=False
):
    """Full-plane FP32 head on rounded operands, with optional backward or incorrect forward rounding."""
    with torch.enable_grad(), torch.autocast(hidden.device.type, enabled=False):
        h = hidden.detach().float().requires_grad_()
        w = weight.detach().to(hidden.dtype).float().requires_grad_()
        logits = F.linear(h, w)
        if round_forward_logits:
            logits = logits.to(hidden.dtype).float()
        if quantize_logits_gradient:
            dtype = hidden.dtype if quantize_logits_gradient is True else quantize_logits_gradient
            logits.register_hook(lambda gradient: gradient.to(dtype).float())
        logps = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
        dh, dw, dz = torch.autograd.grad(logps, (h, w, logits), upstream.float())
    return {"logps": logps.detach(), "hidden": dh.to(hidden.dtype), "weight": dw.to(weight.dtype), "logits": dz}


@dataclass
class PinnedHeadGradients:
    """Detached copies captured while the FSDP root owns the live head; hooks never replace gradients."""

    bounded: dict = field(default_factory=dict)
    full: dict = field(default_factory=dict)
    hooks: list = field(default_factory=list)

    def _capture(self, state, name, tensor):
        state[name] = tensor.detach().clone()

    def _watch(self, state, name, tensor):
        self._capture(state, name, tensor)
        self.hooks.append(tensor.register_hook(lambda gradient: self._capture(state, f"grad_{name}", gradient)))

    @contextmanager
    def record(self, full_trainer):
        original = chunked_dpo.chunked_head_logprobs
        head = base_transformers_model(full_trainer.model).get_output_embeddings()
        original_head = head.forward

        def bounded(hidden, weight, targets, bias, temperature, **kwargs):
            if bias is not None or temperature != 1 or kwargs.get("logit_scale") or kwargs.get("softcap"):
                raise AssertionError("Pinned Qwen3 head diagnosis requires its identity transform")
            self._watch(self.bounded, "hidden", hidden)
            self._watch(self.bounded, "weight", weight)
            self._capture(self.bounded, "targets", targets)
            logps, entropy = original(hidden, weight, targets, bias, temperature, **kwargs)
            self._watch(self.bounded, "logps", logps)
            return logps, entropy

        def full(module, hidden):
            self._watch(self.full, "hidden", hidden)
            self._watch(self.full, "weight", module.weight)
            logits = original_head(hidden)
            self._watch(self.full, "logits", logits)
            return logits

        with patch.object(chunked_dpo, "chunked_head_logprobs", bounded):
            head.forward = MethodType(full, head)
            try:
                yield self
            finally:
                head.forward = original_head
                for hook in self.hooks:
                    hook.remove()
                self.hooks.clear()

    def report(self, actual_gradients, expected_gradients):
        actual, full = self.bounded, self.full
        active = full["grad_logits"].ne(0).any(-1)
        full_hidden = full["hidden"][active]
        if full_hidden.shape != actual["hidden"].shape:
            raise AssertionError("Pinned full-head supervised rows do not align with bounded rows")
        exact = full_head_derivative(actual["hidden"], actual["weight"], actual["targets"], actual["grad_logps"])
        rounded = full_head_derivative(
            actual["hidden"],
            actual["weight"],
            actual["targets"],
            actual["grad_logps"],
            quantize_logits_gradient=True,
        )
        parameter_errors = {
            name: gradient_difference(actual_gradients[name], want)["relative_l2"]
            for name, want in expected_gradients.items()
        }
        return {
            "qualification": "DIAGNOSTIC_NOT_ACCEPTANCE",
            "hidden_dtype": str(actual["hidden"].dtype),
            "live_head_weight_dtype": str(actual["weight"].dtype),
            "supervised_rows": actual["hidden"].shape[0],
            "identical_live_weight": torch.equal(actual["weight"], full["weight"]),
            "identical_supervised_hidden": torch.equal(actual["hidden"], full_hidden),
            "logps_vs_fp32_head": gradient_difference(actual["logps"], exact["logps"]),
            "full_logits_gradient_vs_pinned_upstream": gradient_difference(
                full["grad_logits"][active], exact["logits"]
            ),
            "native_hidden_vs_fp32_derivative": gradient_difference(actual["grad_hidden"], exact["hidden"]),
            "native_weight_vs_fp32_derivative": gradient_difference(actual["grad_weight"], exact["weight"]),
            "native_hidden_vs_bf16_dlogits_derivative": gradient_difference(actual["grad_hidden"], rounded["hidden"]),
            "native_weight_vs_bf16_dlogits_derivative": gradient_difference(actual["grad_weight"], rounded["weight"]),
            "full_hidden_vs_fp32_derivative": gradient_difference(full["grad_hidden"][active], exact["hidden"]),
            "full_weight_vs_fp32_derivative": gradient_difference(full["grad_weight"], exact["weight"]),
            "parameter_relative_l2": parameter_errors,
        }
