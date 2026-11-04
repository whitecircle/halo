"""Score one generated response per prompt with a local reward model, printing summary statistics."""

from halo.entrypoints.resumable_generation import run_async_cli
from halo.entrypoints.reward_scoring import run_reward_scoring
from scripts.inference.reward_model.reward_model_arguments import build_reward_model_parser


def parse_args():
    parser = build_reward_model_parser("Async generation with RM scoring", temperature_default=0.0)
    return parser.parse_args()


async def main():
    await run_reward_scoring(parse_args())


if __name__ == "__main__":
    run_async_cli(main)
