"""The environmental trainer's real batch build over stub rows, for CPU tests of what a step trains.

``_build_training_tensors`` runs for real; tokenization, the recompute forward and the reference
forward are stubs, so a test sets the rows, the policy's log-probs and the knobs, and reads the batch,
the world metrics and the completions record. Single process: every gather is the identity.
"""

import types
from collections import defaultdict

import torch

from src.configs.async_training_config import AsyncTrainingConfig, ISMaskConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import BatchBuildFence, DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.rollout.completions_logging import unbounded_completion_logs
from src.trainers.grpo.rollout.trajectory_tokenize import TurnRow
from tests.common.grpo_metrics import attach_world_metrics


def episode(reward: float, *, valid: bool = True) -> RolloutResult:
    """One collected episode of one assistant turn; ``valid=False`` is an infra-errored one."""
    trajectory = Trajectory(messages=[Message.user("p"), Message.assistant("a")])
    return RolloutResult(prompt="p", trajectory=trajectory, total_reward=reward, error=None if valid else "boom")


def row(
    completion: list[int],
    sampling: list[float] | None = None,
    loss_mask: list[int] | None = None,
    negative_only: bool = False,
) -> TurnRow:
    """A one-token-prompt row; ``sampling`` carries the engine's per-token log-probs, ``negative_only`` tags
    an untrainable turn's row."""
    return TurnRow(
        torch.tensor([5]),
        torch.tensor(completion),
        torch.tensor(loss_mask if loss_mask is not None else [1] * len(completion)),
        None if sampling is None else torch.tensor(sampling),
        None,
        negative_only,
    )


def batch_host(
    rows: list[TurnRow],
    policy: torch.Tensor,
    *,
    training: bool,
    num_generations: int = 2,
    scale_rewards: str = "group",
    std_floor: float = 0.0,
    is_correction: bool = False,
    beta: float = 0.0,
    reference: torch.Tensor | None = None,
    forced_close_ids: tuple[int, ...] | None = None,
    is_mask_config: ISMaskConfig | None = None,
    skip_update_masked_frac: float | None = None,
    balance_token_mass: bool = False,
    save_completions: bool = False,
):
    """A trainer stand-in whose ``_build_training_tensors`` is the real one; the recompute forward returns
    ``policy`` (``[rows, completion width]``) and the reference forward ``reference``. ``scale_rewards`` and
    ``std_floor`` are the advantage's ``scale_rewards`` and ``scale_rewards_std_floor``."""
    host = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    host.model = types.SimpleNamespace(training=training)
    host.args = types.SimpleNamespace(
        scale_rewards=scale_rewards,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        mask_truncated_completions=False,
    )
    host.accelerator = types.SimpleNamespace(gather=lambda x: x)
    host.async_config = AsyncTrainingConfig()
    host._carry_reasoning = False
    host._train_on_sampled_tokens = False
    host._is_correction = is_correction
    host._isr_engine_reference = False
    host._forced_close_ids = forced_close_ids
    host._is_mask_config = is_mask_config or ISMaskConfig()
    host._skip_update_masked_frac = skip_update_masked_frac
    host._breaker_tripped_this_step = False
    host._balance_token_mass = balance_token_mass
    host._drop_degenerate_groups = False
    host._empty_rollout_steps = 0
    host._routing_injector = None
    host._batch_errors = BatchBuildFence()
    host.vllm_importance_sampling_clip_max = 2.0
    host.num_iterations = 1
    host.beta = beta
    host.num_generations = host.num_generations_eval = num_generations
    host._scale_rewards_std_floor = std_floor
    host.pad_token_id, host.eos_token_id = 0, 2
    host._tokenize_step_rows = lambda results: [[r] for r in rows]
    host._get_per_token_logps_and_entropies = lambda model, ids, mask, keep, compute_entropy: (policy, None)
    host._compute_ref_logps = lambda ids, mask, keep: reference
    host._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    host._save_completions, host.log_completions = save_completions, False
    host._logs = unbounded_completion_logs()
    return host


def build(host, episodes: list[RolloutResult]) -> dict[str, torch.Tensor]:
    """One step's batch, in the host's mode, without eval padding."""
    mode = "train" if host.model.training else "eval"
    return host._build_training_tensors(episodes, torch.device("cpu"), mode, 0, [[] for _ in episodes])
