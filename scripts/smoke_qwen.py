#!/usr/bin/env python3
"""Check a local OpenAI-compatible Qwen server with one classification call."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    args = parser.parse_args()

    models_url = args.base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            available = request_json(models_url)
            break
        except (URLError, TimeoutError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                print(f"Model server did not become ready: {models_url}", file=sys.stderr)
                return 1
            time.sleep(5)

    model_ids = [item["id"] for item in available.get("data", [])]
    if not model_ids:
        print("The model server returned no model IDs.", file=sys.stderr)
        return 1
    model_id = args.model if args.model in model_ids else model_ids[0]
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Classify the question. Return one class only: Number, Location, Person, Description, Entity, or Abbreviation."},
            {"role": "user", "content": "What is the capital of Denmark?"},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = request_json(args.base_url.rstrip("/") + "/chat/completions", payload)
    answer = result["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        print("The model returned an empty answer.", file=sys.stderr)
        return 1

    report = {
        "model": model_id,
        "prompt": payload["messages"][1]["content"],
        "answer": answer.strip(),
        "temperature": 0,
        "seed": 42,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
