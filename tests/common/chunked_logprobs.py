"""The position-free embedding backbone the chunked-log-probs suites score through."""

import torch
from transformers import PretrainedConfig

from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM, HeadTransform
from halo.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin


class EmbeddingBackboneHarness(ChunkedGRPOLogprobsMixin):
    """Position-free embedding backbone, so a trimmed row and its padded twin agree exactly on every real completion
    position. Records each backbone call's input shape and whether it ran mask-free."""

    temperature = 1.0

    def __init__(self, attn_impl: str, vocab: int = 40, hidden: int = 8):
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.model = torch.nn.Module()
        self.model.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.model.config = PretrainedConfig()
        self.model.config._attn_implementation = attn_impl
        self.model.get_output_embeddings = lambda: self.model.lm_head
        self.forward_calls: list[tuple[tuple[int, ...], bool]] = []

    def _head_transform(self, unwrapped_model) -> HeadTransform:
        return IDENTITY_HEAD_TRANSFORM

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        self.forward_calls.append((tuple(input_ids.shape), attention_mask is None))
        full = self.embed(input_ids)[:, :-1, :]
        return full[:, -logits_to_keep:, :]
