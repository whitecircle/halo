"""Reward-model rejection sampling: generate N hypotheses per prompt, score them, keep the extremes.

Generates against an OpenAI-compatible API and scores with a local reward model. ``--output_format``
picks the shape: ``preference`` (default; DPO/SMPO chosen/rejected pairs) or ``offline_grpo``
(completions list + rewards).

Usage:
    # Preference format (DPO/SMPO)
    python scripts/inference/reward_model/rm_rejection_sampling.py \\
        --model my-model \\
        --dataset data/prompts.jsonl \\
        --rm_model_id path/to/reward-model

    # Offline GRPO format
    python scripts/inference/reward_model/rm_rejection_sampling.py \\
        --model my-model \\
        --dataset data/prompts.jsonl \\
        --rm_model_id path/to/reward-model \\
        --output_format offline_grpo

Input JSONL format:
    {"id": "...", "prompt": [{"role": "user", "content": "..."}], ...}

Output formats:
    preference:    {"id": "...", "prompt": [...], "chosen": [...], "rejected": [...],
                    "chosen_score": 0.9, "rejected_score": 0.1, "all_scores": [...]}
    offline_grpo:  {"prompt": [...], "completions": [[...], [...]], "rewards": [0.9, 0.7]}
"""

from halo.entrypoints.rejection_sampling import OUTPUT_FORMATS, run_rejection_sampling
from halo.entrypoints.resumable_generation import run_async_cli
from scripts.inference.reward_model.reward_model_arguments import build_generation_parser


def parse_args():
    parser = build_generation_parser("Async generation with RM rejection sampling", temperature_default=0.8)
    parser.add_argument("--n_hypos", type=int, default=5, help="Hypotheses per prompt (default: 5, at least 2)")
    parser.add_argument(
        "--output_format",
        type=str,
        default="preference",
        choices=list(OUTPUT_FORMATS),
        help="Output format: preference (DPO/SMPO) or offline_grpo (default: preference)",
    )
    parser.add_argument(
        "--rm_max_batch_size",
        type=int,
        default=8,
        help="Hypotheses the reward model scores per forward (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.n_hypos < 2:
        parser.error(
            f"--n_hypos must be at least 2 (got {args.n_hypos}): one hypothesis cannot separate a best from a worst"
        )
    return args


async def main():
    await run_rejection_sampling(parse_args())


if __name__ == "__main__":
    run_async_cli(main)
