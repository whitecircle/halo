#!/usr/bin/env python
"""Gradio Environment Playground — Interactive UI for testing GRPO environments.

Runs multi-turn episodes with any registered environment against a live rollout server, through the
same driver the offline eval uses (:func:`src.environments.eval_runner.run_episode`), so an episode
here is generated and graded exactly as one collected by the trainer.

Usage:
    # 1. Start the rollout server
    docker compose -f docker-compose.vllm.yml up vllm-server

    # 2. Launch playground
    python scripts/inference/playground/gradio_environment_playground.py
    python scripts/inference/playground/gradio_environment_playground.py --port 7861
"""

import argparse
import asyncio
import json
import time
from typing import Any

import gradio as gr

from scripts.inference._common import add_gradio_server_args, launch_gradio
from src.configs.rollout_config import RolloutConfig
from src.environments.base import Trajectory
from src.environments.eval_runner import run_episode
from src.environments.registry import get_registered_environments, resolve_environment
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, create_openai_client, resolve_local_api_key

# Per non-assistant message, in the transcript panel: a full tool observation (a failing test log, a
# search dump) is capped at 16k chars by the env itself and would dominate the turn it belongs to.
_MAX_RENDERED_CHARS = 500


def build_context(expected_answer: str) -> dict[str, Any]:
    """Build the episode context, parsing structured answers (code_contests test cases) as JSON."""
    context: dict[str, Any] = {}
    if expected_answer:
        try:
            context["answer"] = json.loads(expected_answer)
        except (json.JSONDecodeError, TypeError):
            context["answer"] = expected_answer
    return context


def render_transcript(trajectory: Trajectory | None) -> list[dict[str, str]]:
    """The episode's messages as Gradio chat turns.

    Gradio renders only ``user``/``assistant``; a tool or system turn is shown as a labelled user
    turn rather than dropped, since the observation is what the next turn conditions on.
    """
    rendered: list[dict[str, str]] = []
    for message in trajectory.messages if trajectory else []:
        content = message.content or ""
        if message.role == "assistant":
            if message.tool_calls:
                calls = "\n".join(
                    f"  `{call['function']['name']}({call['function']['arguments']})`" for call in message.tool_calls
                )
                content = f"{content}\n\n**Tool calls:**\n{calls}"
            rendered.append({"role": "assistant", "content": content})
        else:
            prefix = "" if message.role == "user" else f"**[{message.role}]** "
            rendered.append({"role": "user", "content": f"{prefix}{content[:_MAX_RENDERED_CHARS]}"})
    return rendered


def render_summary(env_type: str, trajectory: Trajectory | None, elapsed: float, max_turns: int) -> str:
    """Episode verdict for the summary panel, read off the driver's own ``_eval_stats``."""
    if trajectory is None:
        return f"**Environment:** {env_type}\n\n**No episode** — the environment produced no trajectory."

    stats = trajectory.info.get("_eval_stats", {})
    parts = [
        f"**Environment:** {env_type}",
        f"**Turns:** {trajectory.num_turns}/{max_turns}",
        f"**Reward:** {trajectory.total_reward:.3f}",
        f"**Done:** {trajectory.done}",
        f"**Truncated:** {trajectory.truncated}",
        f"**Generations:** {stats.get('generations', 0)}",
        f"**Tokens generated:** {stats.get('completion_tokens', 0)}",
        f"**Latency:** {elapsed:.1f}s",
    ]
    # A turn the engine cut off at its token cap is a common cause of a zero-scoring episode.
    if stats.get("length_capped"):
        parts.append(f"**Length-capped turns:** {trajectory.info.get('length_cutoff_turns', 0)}")
    if trajectory.info.get("final_answer"):
        parts.append(f"**Final answer:** {trajectory.info['final_answer']}")
    if trajectory.info.get("total_tool_calls"):
        parts.append(
            f"**Tool calls:** {trajectory.info['total_tool_calls']} "
            f"(successful: {trajectory.info.get('successful_tool_calls', 0)})"
        )
    return "\n".join(parts)


def run_playground_episode(
    env_type: str,
    prompt: str,
    expected_answer: str,
    server_url: str,
    api_key: str | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_turns: int,
    success_reward: float,
) -> tuple[list[dict[str, str]], str]:
    """Run one episode through the shared eval driver and render it for the UI.

    The shared driver rather than a local copy of the loop: it stamps ``finish_reason`` (so a
    length-cut turn is recovered here as in training rather than graded as a final answer), binds the
    episode's reasoning-effort level and token budget, and finalizes a truncated episode.
    """
    env = resolve_environment(env_type, {"max_turns": int(max_turns), "success_reward": success_reward})
    # A hand-edited "localhost:8000/v1" is not a URL the SDK can route. The field is prefilled with a
    # full one, so the scheme is restored rather than refused.
    url = server_url if server_url.startswith("http") else f"http://{server_url}"
    client = create_openai_client(base_url=url, api_key_override=api_key)

    start_time = time.time()
    trajectory = asyncio.run(
        run_episode(
            env,
            prompt,
            build_context(expected_answer),
            client,
            rollout=RolloutConfig(
                # None rather than "": an unset model name is dropped from the body and a
                # single-model server answers with what it serves, while an empty string 404s.
                model_name=model_name or None,
                temperature=temperature,
                max_tokens=int(max_tokens),
                top_p=top_p,
            ),
        )
    )
    elapsed = time.time() - start_time
    return render_transcript(trajectory), render_summary(env_type, trajectory, elapsed, int(max_turns))


def create_demo(default_base_url: str = DEFAULT_LOCAL_BASE_URL, api_key: str | None = None):
    available_envs = get_registered_environments()
    default_env = "react_math" if "react_math" in available_envs else available_envs[0]

    def on_run(
        env_type,
        prompt,
        expected_answer,
        server_url,
        model_name,
        temperature,
        max_tokens,
        top_p,
        max_turns,
        success_reward,
    ):
        if not prompt.strip():
            return [], "Please enter a prompt."
        try:
            return run_playground_episode(
                env_type,
                prompt,
                expected_answer,
                server_url,
                api_key,
                model_name,
                temperature,
                max_tokens,
                top_p,
                int(max_turns),
                success_reward,
            )
        except Exception as e:
            return [], f"**Error:** {e}"

    with gr.Blocks(title="Environment Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Environment Playground\nTest GRPO environments interactively against a rollout server.")

        with gr.Row():
            # Left sidebar
            with gr.Column(scale=1):
                env_type = gr.Dropdown(
                    choices=available_envs,
                    value=default_env,
                    label="Environment Type",
                )
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="The task the episode opens with")
                expected_answer = gr.Textbox(
                    label="Expected Answer (optional)",
                    placeholder='Plain text, or JSON for a structured answer (code_contests: {"test_cases": [...]})',
                )

                gr.Markdown("### Rollout Server")
                # The API key is not a component: a Gradio value, hidden or not, is serialized into
                # every browser's page config. It stays in the create_demo closure.
                server_url = gr.Textbox(label="Rollout Server Base URL", value=default_base_url)
                model_name = gr.Textbox(label="Model Name (optional)", value="")

                gr.Markdown("### Generation")
                temperature = gr.Slider(0, 2, step=0.05, value=0.7, label="Temperature")
                max_tokens = gr.Slider(64, 8192, step=64, value=1024, label="Max Tokens")
                top_p = gr.Slider(0.1, 1.0, step=0.05, value=0.95, label="Top P")

                gr.Markdown("### Environment")
                max_turns = gr.Slider(1, 30, step=1, value=10, label="Max Turns")
                success_reward = gr.Slider(0, 5, step=0.1, value=1.0, label="Success Reward")

                run_btn = gr.Button("Run Episode", variant="primary")

            # Main content
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Episode", type="messages", height=500)
                summary = gr.Markdown(label="Episode Summary", value="*Run an episode to see results*")

        run_btn.click(
            on_run,
            inputs=[
                env_type,
                prompt,
                expected_answer,
                server_url,
                model_name,
                temperature,
                max_tokens,
                top_p,
                max_turns,
                success_reward,
            ],
            outputs=[chatbot, summary],
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    """The playground's CLI: the rollout endpoint it prefills, plus the shared server block."""
    parser = argparse.ArgumentParser(description="Gradio Environment Playground")
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_LOCAL_BASE_URL,
        help=f"Rollout server base URL, prefilled into the UI (default: {DEFAULT_LOCAL_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=resolve_local_api_key(),
        help="API key for the rollout server (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the "
        "placeholder a keyless local server accepts). A real key belongs in the environment. Kept "
        "server-side.",
    )
    add_gradio_server_args(parser, port_default=7860)
    return parser


def main():
    args = build_parser().parse_args()

    demo = create_demo(default_base_url=args.vllm_url, api_key=args.api_key)
    launch_gradio(demo, args)


if __name__ == "__main__":
    main()
