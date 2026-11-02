"""Register a Qwen model with the unchanged source project utilities.

The paper code looks up models by name in util.llm_map. This module adds an
entry that calls the local vLLM server with the paper's settings: the
temperature from the notebook and seed 42. Thinking mode is off, so the model
answers directly, as the paper's models do.
"""

from __future__ import annotations

import os

DEFAULT_NAME = "qwen35-27b"


def register_qwen(name: str = DEFAULT_NAME) -> None:
    from langchain_openai import ChatOpenAI

    import util

    def create_chat(temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-27B"),
            temperature=temperature,
            base_url=os.environ.get("QWEN_API_BASE", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("QWEN_API_KEY", "EMPTY"),
            timeout=600,
            max_retries=3,
            model_kwargs={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "seed": 42,
            },
        )

    util.llm_map[name] = create_chat
