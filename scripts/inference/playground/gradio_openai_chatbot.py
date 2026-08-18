"""Gradio chat UI for a model served behind an OpenAI-compatible API (vLLM, ...).

Async streaming, configurable generation parameters, and system prompts.

Usage:
    python scripts/inference/playground/gradio_openai_chatbot.py \
        --model my-model --model-url http://localhost:8000/v1

    # Serving the model:
    VLLM_MODEL=Qwen/Qwen3-8B docker compose -f docker-compose.vllm.yml up vllm-server
"""

import argparse
from collections.abc import AsyncGenerator

import gradio as gr
from openai import NOT_GIVEN, AsyncOpenAI

from scripts.inference._common import add_gradio_server_args, launch_gradio
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, create_openai_client, resolve_local_api_key


def build_parser() -> argparse.ArgumentParser:
    """The chat UI's CLI: the served model it talks to, plus the shared server block."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible chat interface")
    parser.add_argument(
        "--model-url",
        type=str,
        default=DEFAULT_LOCAL_BASE_URL,
        help=f"API base URL (default: {DEFAULT_LOCAL_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=resolve_local_api_key(),
        help="API key for the served model (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the placeholder "
        "a keyless local server accepts). A real key belongs in the environment.",
    )
    # No stand-in default: a made-up name 404s on every request, while omitting the field lets a
    # single-model server answer with whatever it serves.
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Served model name (default: none — the server's only model answers; a multi-model server needs it)",
    )
    parser.add_argument(
        "--stop-token-ids", type=str, default="", help="Comma-separated stop token IDs (vLLM-specific)"
    )
    add_gradio_server_args(parser, port_default=8730)
    return parser


def build_messages(
    history: list[dict[str, str]],
    message: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build OpenAI messages list from Gradio chat history."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    return messages


def create_demo(client: AsyncOpenAI, model: str | None, stop_token_ids: list[int]):
    """Create Gradio chat demo with async streaming; ``model=None`` sends no model field."""

    async def predict(
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> AsyncGenerator[str, None]:
        messages = build_messages(history, message, system_prompt)

        extra_body = {}
        if top_k > 0:
            extra_body["top_k"] = top_k
        if repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = repetition_penalty
        if stop_token_ids:
            extra_body["stop_token_ids"] = stop_token_ids

        try:
            stream = await client.chat.completions.create(
                model=model or NOT_GIVEN,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                extra_body=extra_body if extra_body else None,
            )

            partial = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                partial += delta
                yield partial
        except Exception as e:
            yield f"Error: {e}"

    demo = gr.ChatInterface(
        predict,
        type="messages",
        title=f"Chat: {model or 'served model'}",
        additional_inputs=[
            gr.Textbox(
                label="System prompt",
                lines=2,
                placeholder="Enter system prompt...",
                render=False,
            ),
            gr.Slider(0, 2, step=0.05, value=0.7, label="Temperature", render=False),
            gr.Slider(64, 16384, step=64, value=4096, label="Max tokens", render=False),
            gr.Slider(0.1, 1.0, step=0.05, value=1.0, label="Top P", render=False),
            gr.Slider(0, 100, step=1, value=0, label="Top K (0 = disabled)", render=False),
            gr.Slider(1.0, 2.0, step=0.05, value=1.0, label="Repetition penalty", render=False),
        ],
    )
    return demo


def main():
    args = build_parser().parse_args()

    client = create_openai_client(base_url=args.model_url, api_key_override=args.api_key)

    stop_ids = []
    if args.stop_token_ids:
        stop_ids = [int(x.strip()) for x in args.stop_token_ids.split(",") if x.strip()]

    demo = create_demo(client, args.model, stop_ids)
    launch_gradio(demo, args)


if __name__ == "__main__":
    main()
