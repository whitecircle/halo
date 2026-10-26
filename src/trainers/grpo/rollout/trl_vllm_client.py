"""TRL's ``VLLMClient`` for online GRPO: TRL's generation contract over the vLLM weight-sync client.

TRL's own client imports vLLM, which the training image does not carry, so online GRPO hands TRL this
class in its place.
"""

import logging

from src.env import env_positive_float, nccl_timeout, watchdog_bounded_seconds
from src.weight_sync.clients.base import POST_TIMEOUT_S
from src.weight_sync.clients.vllm import VLLMWeightSyncClient

logger = logging.getLogger(__name__)


class TrlVLLMClient(VLLMWeightSyncClient):
    """The surface TRL's server-mode generation calls: the parent's weight sync plus ``generate``."""

    def __init__(self, *args, **kwargs):
        # Derived rather than a literal: a generation request blocks every peer at the next collective,
        # so it must stay under the watchdog, a lowered one included.
        self._generation_timeout = env_positive_float(
            "HALO_VLLM_GENERATION_TIMEOUT_SECONDS", watchdog_bounded_seconds()
        )
        watchdog = nccl_timeout().total_seconds()
        if self._generation_timeout >= watchdog:
            logger.warning(
                f"HALO_VLLM_GENERATION_TIMEOUT_SECONDS={self._generation_timeout:.0f}s is at or above the "
                f"NCCL collective watchdog ({watchdog:.0f}s): a slow generation holds every peer at the "
                f"next collective until the watchdog aborts the run instead. Lower it, or raise "
                f"DIST_NCCL_TIMEOUT_MINUTES."
            )
        super().__init__(*args, **kwargs)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        images: list | None = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        logprobs: int | None = 0,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
    ) -> dict[str, list]:
        """Generate completions via native vLLM ``/v1/completions``.

        Matches TRL's ``VLLMClient.generate`` contract, returning ``{prompt_ids, completion_ids,
        logprobs, logprob_token_ids}``. Only the sampled token's logprob is available here (num_logprobs=1).
        """
        if images is not None and any(image is not None for image in images):
            raise NotImplementedError(
                "This client's /v1/completions path is text-only — VLM online GRPO with images "
                "is unsupported (generations would silently be image-blind)."
            )
        if not prompts:
            return {"prompt_ids": [], "completion_ids": [], "logprobs": [], "logprob_token_ids": []}
        # Every filter is stated, an off one included, since the server fills an omitted one from the
        # model's generation_config.json; TRL's top_k 0 is vLLM's own off value. The completions route
        # fills token_logprobs only from logprobs 1, and GRPO asks for 0.
        sampling = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "logprobs": max(int(logprobs or 0), 1),
            **(generation_kwargs or {}),
            "return_token_ids": True,
        }
        if structured_outputs_regex:
            sampling["structured_outputs"] = {"regex": structured_outputs_regex}

        prompt_ids = []
        for prompt in prompts:
            if isinstance(prompt, (list, tuple)):
                prompt_ids.append(list(prompt))
            else:
                reply = self.request("POST", "/tokenize", retry=True, timeout=POST_TIMEOUT_S, json={"prompt": prompt})
                prompt_ids.append(reply.json()["tokens"])

        # One batched request: continuous batching schedules all prompts together (wall clock ~max).
        # Single-shot: a retrying session's read timeout would re-queue a full generation server-side.
        data = self.request(
            "POST",
            "/v1/completions",
            retry=False,
            timeout=self._generation_timeout,
            json={"prompt": list(prompts), **sampling},
        ).json()
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float]]] = []
        all_logprob_token_ids: list[list[list[int]]] = []
        for choice in sorted(data["choices"], key=lambda c: c.get("index", 0)):
            token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            completion_ids = self._completion_ids(choice)
            all_completion_ids.append(completion_ids)
            all_logprobs.append([[v] for v in token_logprobs])
            all_logprob_token_ids.append([[tid] for tid in completion_ids[: len(token_logprobs)]])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "logprob_token_ids": all_logprob_token_ids,
        }

    def _completion_ids(self, choice: dict) -> list[int]:
        """The sampled token IDs the server returned for one choice.

        Not reconstructed from the decoded text: a BPE re-merge need not reproduce the sampled stream,
        and ids that disagree with the logprobs desync the GRPO importance ratio. A missing field means
        the server ignored ``return_token_ids``, which raises.
        """
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError(
                f"vLLM server {self.base_url} returned a completion without `token_ids` even though "
                f"the request set `return_token_ids`. Recovering the ids by re-tokenizing the text "
                f"would silently desync the GRPO importance-sampling ratio, so the rollout is "
                f"refused — serve this model on vLLM >= 0.10.2, which honors the flag."
            )
        return list(token_ids)
