"""sentence-transformers adapters for embedding training.

:class:`PreloadedTransformer` wraps an already-loaded backbone in ST's ``Transformer`` interface (ST's own
reloads from a path, discarding the EP/TP patches); :func:`patch_sentence_transformer_gradient_checkpointing`
gives ST's ``gradient_checkpointing_*`` the keywords transformers' trainer passes.
"""

import inspect

import torch
from sentence_transformers.base.model import BaseModel
from sentence_transformers.base.modules.input_module import InputModule
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.checkpoint.atomic import apply_umask_mode
from src.checkpoint.config_export import finalize_exported_config
from src.checkpoint.weights_source import config_schema_source


class PreloadedTransformer(InputModule):
    """sentence-transformers' Transformer module over a pre-loaded backbone, held as ``auto_model``."""

    config_file_name: str = "sentence_bert_config.json"
    config_keys: list = ["max_seq_length", "do_lower_case"]

    def __init__(
        self,
        auto_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 512,
        do_lower_case: bool = False,
    ) -> None:
        super().__init__()
        self.auto_model = auto_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case

        forward_params = set(inspect.signature(self.auto_model.forward).parameters)
        self.model_forward_params = forward_params | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
        }

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Run the backbone and add ``token_embeddings`` to *features*."""
        trans_features = {k: v for k, v in features.items() if k in self.model_forward_params}
        outputs = self.auto_model(**trans_features, **kwargs, return_dict=True)

        token_embeddings = outputs[0]  # last_hidden_state
        features["token_embeddings"] = token_embeddings

        if self.auto_model.config.output_hidden_states and "hidden_states" in outputs:
            features["all_layer_embeddings"] = outputs["hidden_states"]

        return features

    def preprocess(
        self,
        inputs: list[str] | list[dict] | list[tuple[str, str]],
        prompt: str | None = None,
        padding: str | bool = True,
        max_length: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Tokenize texts as sentence-transformers' ``Transformer.preprocess`` does: ``prompt`` prepended to each
        text (a pair's first), the collator's per-batch ``max_length`` over ``max_seq_length``, other hints ignored."""
        if prompt:
            inputs = [(prompt + text[0], *text[1:]) if isinstance(text, tuple) else prompt + text for text in inputs]
        output: dict = {}
        if isinstance(inputs[0], str):
            to_tokenize = [inputs]
        elif isinstance(inputs[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in inputs:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in inputs:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=max_length if max_length is not None else self.max_seq_length,
            )
        )
        return output

    def get_embedding_dimension(self) -> int:
        return self.auto_model.config.get_text_config().hidden_size

    def save(self, output_path: str, **kwargs) -> None:
        """Save backbone and tokenizer. For EP/TP, the trainer handles saving."""
        self.auto_model.save_pretrained(output_path)
        apply_umask_mode(output_path)
        finalize_exported_config(self.auto_model.config, output_path, source=config_schema_source(self.auto_model))
        self.tokenizer.save_pretrained(output_path)
        self.save_config(output_path)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        raise NotImplementedError(
            "PreloadedTransformer cannot be loaded standalone. "
            "Use load_distributed_model() and construct the SentenceTransformer manually."
        )

    def __repr__(self) -> str:
        return (
            f"PreloadedTransformer("
            f"{{max_seq_length: {self.max_seq_length}, "
            f"architecture: {self.auto_model.__class__.__name__}}})"
        )


def _transformers_model(self) -> PreTrainedModel:
    model = self.transformers_model
    if model is None:
        raise ValueError("gradient_checkpointing is on, but this SentenceTransformer holds no transformers model")
    return model


def _gradient_checkpointing_enable(
    self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
) -> None:
    _transformers_model(self).gradient_checkpointing_enable(
        gradient_checkpointing_kwargs, every_n_layers=every_n_layers, offload=offload
    )


def _gradient_checkpointing_disable(self) -> None:
    _transformers_model(self).gradient_checkpointing_disable()


def patch_sentence_transformer_gradient_checkpointing() -> None:
    """Route ST's ``gradient_checkpointing_enable`` / ``_disable``, which lack transformers' ``every_n_layers``
    and ``offload``, to the pipeline's transformers model; idempotent."""
    BaseModel.gradient_checkpointing_enable = _gradient_checkpointing_enable
    BaseModel.gradient_checkpointing_disable = _gradient_checkpointing_disable
