"""Reward-model rejection sampling: generate N hypotheses per prompt, score them, keep the extremes.

``--output_format`` picks ``preference`` (DPO/SMPO pairs) or ``offline_grpo`` (completions list + rewards)."""

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
