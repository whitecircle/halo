"""A served Bradley-Terry or sequence-classification reward model, scored over its engine's HTTP
classify route: vLLM (``--runner pooling``) or SGLang (``--is-embedding``)."""

import asyncio
import logging
import math
from collections.abc import Sequence

import httpx
from transformers import AutoTokenizer

from src.rewards.samples import ScoringSample, scored_messages
from src.rewards.scoring import Scorer, ScoreResult
from src.rewards.spec import RewardModelTerm

logger = logging.getLogger(__name__)

_PROBE_SAMPLE = ScoringSample(
    prompt=[{"role": "user", "content": "Reply with the single word: ready"}],
    completion=[{"role": "assistant", "content": "ready"}],
)


class ServedRewardModel(Scorer):
    """The scorer behind a :class:`RewardModelTerm`.

    Every sample is rendered with the reward model's own chat template and tokenized without added
    special tokens — the exact tokens TRL's in-process reward-model path scores — so both engines
    score the same sequence and a template that spells its own BOS is not given a second one. vLLM's
    ``/classify`` takes the rendered text with ``add_special_tokens`` off and returns the head's outputs
    per input (``use_activation`` off keeps them raw logits); SGLang's ``/classify`` takes the token
    ids and returns the outputs as an ``embedding`` vector. The logit at ``label_index`` goes through
    the term's logistic map.
    """

    term: RewardModelTerm

    def __init__(self, term: RewardModelTerm):
        super().__init__(term)
        self._tokenizer = None
        self._client: httpx.AsyncClient | None = None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return (f"reward_model/{self.term.name}/logit",)

    def _tokenizer_for_rendering(self):
        """The reward model's tokenizer, loaded on first use; a load failure raises through the caller
        (the launch probe surfaces it) rather than scoring every batch as a request failure."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.term.tokenizer or self.term.model)
        return self._tokenizer

    def _render(self, samples: Sequence[ScoringSample]) -> list[str]:
        tokenizer = self._tokenizer_for_rendering()
        return [
            tokenizer.apply_chat_template(scored_messages(s, self.term.transcript), tokenize=False) for s in samples
        ]

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.term.request_timeout)

    def _client_for_requests(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    async def score(self, samples: Sequence[ScoringSample]) -> list[ScoreResult]:
        """Verdicts in sample order, from batches of the term's ``batch_size``."""
        client = self._client_for_requests()
        batches = [samples[i : i + self.term.batch_size] for i in range(0, len(samples), self.term.batch_size)]
        results = await self._gather_batches(client, batches)
        return [result for batch in results for result in batch]

    async def _gather_batches(self, client: httpx.AsyncClient, batches: list[Sequence[ScoringSample]]):
        return await asyncio.gather(*(self._bounded(self._score_batch, client, batch) for batch in batches))

    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        return (await self._score_batch(self._client_for_requests(), [sample]))[0]

    async def _score_batch(self, client: httpx.AsyncClient, samples: Sequence[ScoringSample]) -> list[ScoreResult]:
        term = self.term
        # The load stays loud (the launch probe surfaces a bad tokenizer); the render does not. A
        # ``full`` transcript hands the template tool turns and tool_calls, which a reward model's
        # template may refuse — and a raise here escapes every guard up to the actor's catch-all,
        # masking the whole episode for the rest of the run.
        self._tokenizer_for_rendering()
        try:
            texts = self._render(samples)
        except Exception as e:
            logger.warning("reward model %r render failed: %s: %s", term.name, type(e).__name__, e)
            error = f"render failed: {type(e).__name__}: {e}"
            return [ScoreResult(None, error=error) for _ in samples]
        try:
            logits = await self._logits(client, texts)
        except Exception as e:
            logger.warning("reward model %r request failed: %s: %s", term.name, type(e).__name__, e)
            error = f"request failed: {type(e).__name__}: {e}"
            return [ScoreResult(None, error=error) for _ in samples]
        return [ScoreResult(term.normalize(logit), {f"reward_model/{term.name}/logit": logit}) for logit in logits]

    async def _logits(self, client: httpx.AsyncClient, texts: list[str]) -> list[float]:
        """The head output at ``label_index`` for each text, in order, from the engine's classify route."""
        term = self.term
        url = f"{term.server_root}/classify"
        if term.backend == "vllm":
            body = {"model": term.model, "input": texts, "use_activation": False, "add_special_tokens": False}
            response = await client.post(url, json=body)
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [row["probs"] for row in rows]
        else:
            input_ids = self._tokenizer_for_rendering()(texts, add_special_tokens=False)["input_ids"]
            response = await client.post(url, json={"input_ids": input_ids})
            response.raise_for_status()
            payload = response.json()
            vectors = [item["embedding"] for item in (payload if isinstance(payload, list) else [payload])]
        if len(vectors) != len(texts):
            raise ValueError(f"reward model returned {len(vectors)} outputs for {len(texts)} inputs")
        logits = []
        for vector in vectors:
            if term.label_index >= len(vector):
                raise ValueError(
                    f"reward model head has {len(vector)} output(s); label_index {term.label_index} is out of range"
                )
            logit = float(vector[term.label_index])
            if not math.isfinite(logit):
                raise ValueError(f"reward model returned a non-finite output: {logit}")
            logits.append(logit)
        return logits

    async def verify(self) -> None:
        """Score a one-line probe through a fresh client; a request, shape or template failure raises."""
        client = self._create_client()
        try:
            result = (await self._score_batch(client, [_PROBE_SAMPLE]))[0]
        finally:
            await client.aclose()
        if result.score is None:
            raise RuntimeError(f"reward_model term {self.term.name!r} probe failed: {result.error}")

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
