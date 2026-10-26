"""TRL's ``VLLMClient`` contract over the vLLM weight-sync client, installed in its place by online GRPO since
TRL's own client imports vLLM, which the training image does not carry."""

import logging

from halo.env import env_positive_float, nccl_timeout, watchdog_bounded_seconds
from halo.weight_sync.clients.base import POST_TIMEOUT_S
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient

logger = logging.getLogger(__name__)

_TEXT_ONLY = (
    "This client's /v1/completions path is text-only — VLM online GRPO with images is unsupported "
    "(generations would silently be image-blind)."
)


class TrlVLLMClient(VLLMWeightSyncClient):
    """The surface TRL's server-mode generation calls: the vLLM weight sync plus ``generate``."""

    def __init__(self, *args, **kwargs):
        # Under the collective watchdog: a generation blocks every peer at the next collective.
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

    def image_features(
        self, images: list[list | None], max_concurrent_requests: int | None = None
    ) -> list[dict | None]:
        """TRL's server-side image processing, which feeds :meth:`generate`'s ``features``: refused for any image."""
        if any(images):
            raise NotImplementedError(_TEXT_ONLY)
        return [None] * len(images)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        features: list[dict | None] | None = None,
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
        """TRL's ``VLLMClient.generate`` over ``/v1/completions``: ``{prompt_ids, completion_ids, logprobs,
        logprob_token_ids}``, with the sampled token's logprob only."""
        if features is not None and any(feature is not None for feature in features):
            raise NotImplementedError(_TEXT_ONLY)
        if not prompts:
            return {"prompt_ids": [], "completion_ids": [], "logprobs": [], "logprob_token_ids": []}
        # Every filter is stated, since the server fills an omitted one from generation_config.json; the route
        # fills token_logprobs only from logprobs 1, where GRPO asks for 0.
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

        prompt_ids = self._prompt_ids(prompts)
        # One batched, single-shot request: a session retry would re-queue a full generation server-side.
        data = self.request(
            "POST",
            "/v1/completions",
            retry=False,
            timeout=self._generation_timeout,
            json={"prompt": list(prompts), **sampling},
        ).json()
        return {"prompt_ids": prompt_ids, **self._parse_choices(data["choices"])}

    def _prompt_ids(self, prompts: list[str] | list[list[int]]) -> list[list[int]]:
        """Each prompt's token ids: a token list as given, a string through the server's ``/tokenize``."""
        prompt_ids = []
        for prompt in prompts:
            if isinstance(prompt, (list, tuple)):
                prompt_ids.append(list(prompt))
            else:
                reply = self.request("POST", "/tokenize", retry=True, timeout=POST_TIMEOUT_S, json={"prompt": prompt})
                prompt_ids.append(reply.json()["tokens"])
        return prompt_ids

    def _parse_choices(self, choices: list[dict]) -> dict[str, list]:
        """``{completion_ids, logprobs, logprob_token_ids}`` of the completion choices, in prompt order."""
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float]]] = []
        all_logprob_token_ids: list[list[list[int]]] = []
        for choice in sorted(choices, key=lambda c: c.get("index", 0)):
            token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            completion_ids = choice.get("token_ids")
            if completion_ids is None:
                raise RuntimeError(
                    f"vLLM server {self.base_url} returned a completion without `token_ids` even though "
                    f"the request set `return_token_ids`. Recovering the ids by re-tokenizing the text "
                    f"would silently desync the GRPO importance-sampling ratio, so the rollout is "
                    f"refused — serve this model on vLLM >= 0.10.2, which honors the flag."
                )
            completion_ids = list(completion_ids)
            all_completion_ids.append(completion_ids)
            all_logprobs.append([[v] for v in token_logprobs])
            all_logprob_token_ids.append([[tid] for tid in completion_ids[: len(token_logprobs)]])
        return {
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "logprob_token_ids": all_logprob_token_ids,
        }
