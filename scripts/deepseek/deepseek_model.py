"""Register DeepSeek-R1 (served by Ollama) with the unchanged source project utilities.

The paper code looks up models by name in util.llm_map. This module adds an
entry that calls the local Ollama server through its OpenAI-compatible API,
with the paper's settings: the temperature from the notebook and seed 42.

DeepSeek-R1 is a reasoning model. With Ollama 0.6.0 its answer always starts
with a block of reasoning text between <think> and </think>, and this cannot
be turned off. The paper code finds the class label with a substring search
over the whole answer, so the reasoning text would decide the label.
ThinkStrippingChatOpenAI removes the reasoning block and passes on only the
text after </think>. Adapted from scripts/qwen_model.py (branch qwen3.5-27b-jobs).
"""

from __future__ import annotations

import os

DEFAULT_NAME = "deepseek-r1-7b"
DEFAULT_MODEL = "deepseek-r1-7b-ctx8192"
DEFAULT_API_BASE = "http://127.0.0.1:11434/v1"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Tries for one request when the answer is empty after the reasoning is removed.
# Only used with a temperature above 0: at temperature 0 with a fixed seed a
# second try gives the same text.
EMPTY_ANSWER_TRIES = 3


def strip_thinking(text):
    """Return the answer without the reasoning block.

    - "<think>...</think>answer" gives "answer".
    - "<think>..." without the closing tag gives "": the model reached the
      token limit while it was still reasoning, so there is no answer.
    - Text without the tags is returned unchanged.
    """
    if not isinstance(text, str):
        return text
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    if THINK_OPEN in text:
        return ""
    return text


def _make_chat_class():
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    def clean(result: ChatResult) -> ChatResult:
        generations = []
        for generation in result.generations:
            message = generation.message
            message = message.copy(update={"content": strip_thinking(message.content)})
            generations.append(ChatGeneration(message=message, generation_info=generation.generation_info))
        return ChatResult(generations=generations, llm_output=result.llm_output)

    def is_empty(result: ChatResult) -> bool:
        return any(not str(generation.message.content).strip() for generation in result.generations)

    class ThinkStrippingChatOpenAI(ChatOpenAI):
        """ChatOpenAI that removes the <think>...</think> block from every answer."""

        def _tries(self) -> int:
            return EMPTY_ANSWER_TRIES if (self.temperature or 0) > 0 else 1

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = None
            for _ in range(self._tries()):
                result = clean(super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs))
                if not is_empty(result):
                    break
            return result

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            result = None
            for _ in range(self._tries()):
                result = clean(await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))
                if not is_empty(result):
                    break
            return result

    return ThinkStrippingChatOpenAI


def register_deepseek(name: str = DEFAULT_NAME) -> None:
    import util

    chat_class = _make_chat_class()

    def create_chat(temperature: float):
        return chat_class(
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            temperature=temperature,
            base_url=os.environ.get("DEEPSEEK_API_BASE", DEFAULT_API_BASE),
            api_key=os.environ.get("DEEPSEEK_API_KEY", "EMPTY"),
            # Requests wait in the Ollama queue, so the time limit is long.
            timeout=3600,
            max_retries=3,
            # Limit for reasoning plus answer. Without it a request can run on for a long time.
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "2048")),
            model_kwargs={"seed": 42},
        )

    util.llm_map[name] = create_chat
