#!/usr/bin/env python
"""Gradio Environment Playground: interactive UI for testing GRPO environments.

Episodes run against a live rollout server through the offline eval's driver
(:func:`halo.environments.eval_runner.run_episode`), so one is generated and graded exactly as the trainer collects it.
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Any

import gradio as gr

from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.eval_runner import EVAL_STATS_KEY, reject_answerless_examples, run_episode
from halo.environments.messages import Trajectory
from halo.environments.registry import build_environment, get_registered_environments
from halo.environments.rollout_config import RolloutConfig
from halo.inference.endpoints import DEFAULT_LOCAL_BASE_URL
from halo.inference.openai_client import build_openai_client
from scripts.cli_arguments import add_openai_endpoint_args
from scripts.inference.inference_arguments import add_gradio_server_args, launch_gradio

logger = logging.getLogger(__name__)

# Per non-assistant message: a tool observation is capped at 16k chars by the env and would dominate its turn.
_MAX_RENDERED_CHARS = 500


def build_context(expected_answer: str) -> dict[str, Any]:
    """Parses structured answers (code_contests test cases) as JSON."""
    context: dict[str, Any] = {}
    if expected_answer:
        try:
            context[ANSWER_KEY] = json.loads(expected_answer)
        except json.JSONDecodeError:
            context[ANSWER_KEY] = expected_answer
    return context


def render_transcript(trajectory: Trajectory | None) -> list[dict[str, str]]:
    """Gradio renders only ``user``/``assistant``; a tool or system turn is shown as a labelled user turn, since the
    observation is what the next turn conditions on."""
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
    """Episode verdict for the summary panel, read off the driver's own stats (:data:`EVAL_STATS_KEY`)."""
    if trajectory is None:
        return f"**Environment:** {env_type}\n\n**No episode** — the environment produced no trajectory."

    stats = trajectory.info.get(EVAL_STATS_KEY, {})
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
    # A turn the engine cut off at its token cap, or one the model ended on nothing, is a common
    # cause of a zero-scoring episode.
    if stats.get("length_capped"):
        parts.append(f"**Length-capped turns:** {trajectory.info.get('length_cutoff_turns', 0)}")
    if trajectory.info.get("empty_turns"):
        parts.append(f"**Empty turns:** {trajectory.info['empty_turns']}")
    # ReAct records its answer as ``final_answer``, the native protocol as ``final_response``.
    final = trajectory.info.get("final_answer") or trajectory.info.get("final_response")
    if final:
        parts.append(f"**Final answer:** {final}")
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
) -> tuple[list[dict[str, str]], str]:
    """Through the shared driver: it stamps ``finish_reason`` (so a length-cut or empty turn recovers as in training),
    binds the reasoning-effort level and token budget, and finalizes a truncated episode."""
    env = build_environment(env_type, {"max_turns": int(max_turns)})
    try:
        context = build_context(expected_answer)
        reject_answerless_examples(env, [{"context": context}], "the Expected Answer box")
        # A hand-edited "localhost:8000/v1" is not a URL the SDK can route, so the scheme is restored.
        url = server_url if server_url.startswith("http") else f"http://{server_url}"
        client = build_openai_client(base_url=url, api_key_override=api_key)

        start_time = time.time()
        trajectory = asyncio.run(
            run_episode(
                env,
                prompt,
                context,
                client,
                rollout=RolloutConfig(
                    # None: an unset model name is dropped and a single-model server answers; an empty string 404s.
                    model_name=model_name or None,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                    top_p=top_p,
                ),
            )
        )
        elapsed = time.time() - start_time
        return render_transcript(trajectory), render_summary(env_type, trajectory, elapsed, int(max_turns))
    finally:
        # One environment per click: its sandbox sessions, server connections and scorer clients go with it.
        env.close()


def _episode_handler(api_key: str | None):
    """The Run button's callback. The API key is bound here: a Gradio value, hidden or not, is serialized into every
    browser's page config."""

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
            )
        except Exception as e:
            logger.exception("Playground episode failed")
            return [], f"**Error:** {e}"

    return on_run


def _build_sidebar(available_envs: list[str], default_base_url: str) -> tuple[list, gr.Button]:
    """The sidebar's inputs, in the order :func:`_episode_handler`'s callback takes them, and its Run button."""
    env_type = gr.Dropdown(
        choices=available_envs,
        value="react_math" if "react_math" in available_envs else available_envs[0],
        label="Environment Type",
    )
    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="The task the episode opens with")
    expected_answer = gr.Textbox(
        label="Expected Answer (required where the environment grades against one)",
        placeholder='Plain text, or JSON for a structured answer (code_contests: {"test_cases": [...]})',
    )

    gr.Markdown("### Rollout Server")
    server_url = gr.Textbox(label="Rollout Server Base URL", value=default_base_url)
    model_name = gr.Textbox(label="Model Name (optional)", value="")

    gr.Markdown("### Generation")
    temperature = gr.Slider(0, 2, step=0.05, value=0.7, label="Temperature")
    max_tokens = gr.Slider(64, 8192, step=64, value=1024, label="Max Tokens")
    top_p = gr.Slider(0.1, 1.0, step=0.05, value=0.95, label="Top P")

    gr.Markdown("### Environment")
    max_turns = gr.Slider(1, 30, step=1, value=10, label="Max Turns")

    inputs = [env_type, prompt, expected_answer, server_url, model_name, temperature, max_tokens, top_p, max_turns]
    return inputs, gr.Button("Run Episode", variant="primary")


def create_demo(default_base_url: str = DEFAULT_LOCAL_BASE_URL, api_key: str | None = None):
    with gr.Blocks(title="Environment Playground") as demo:
        gr.Markdown("# Environment Playground\nTest GRPO environments interactively against a rollout server.")
        with gr.Row():
            with gr.Column(scale=1):
                inputs, run_btn = _build_sidebar(get_registered_environments(), default_base_url)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Episode", height=500)
                summary = gr.Markdown(label="Episode Summary", value="*Run an episode to see results*")
        run_btn.click(_episode_handler(api_key), inputs=inputs, outputs=[chatbot, summary])
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gradio Environment Playground")
    add_openai_endpoint_args(parser)
    add_gradio_server_args(parser, port_default=7860)
    return parser


def main():
    args = build_parser().parse_args()

    demo = create_demo(default_base_url=args.base_url, api_key=args.api_key)
    launch_gradio(demo, args, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
