"""Register Qwen3.5-4B with the unchanged source project utilities."""

from __future__ import annotations

import os


def register_qwen() -> None:
    from langchain_openai import ChatOpenAI

    import util

    def create_chat(temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
            temperature=temperature,
            base_url=os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
            timeout=600,
            max_retries=3,
            model_kwargs={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "seed": 42,
            },
        )

    util.llm_map["qwen35"] = create_chat
