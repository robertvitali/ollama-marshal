"""OpenAI-compatible endpoint translation for Ollama."""

from __future__ import annotations

import time
import uuid
from typing import Any


def parse_openai_chat_request(body: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    """Parse an OpenAI /v1/chat/completions request into Ollama format.

    Args:
        body: The OpenAI-format request body.

    Returns:
        Tuple of (model_name, ollama_request_body, is_streaming).
    """
    model = body.get("model", "")
    stream = body.get("stream", False)

    ollama_body: dict[str, Any] = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": stream,
    }

    # Map common OpenAI parameters to Ollama options
    options: dict[str, Any] = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "stop" in body:
        options["stop"] = body["stop"]
    if "seed" in body:
        options["seed"] = body["seed"]
    if "frequency_penalty" in body:
        options["frequency_penalty"] = body["frequency_penalty"]
    if "presence_penalty" in body:
        options["presence_penalty"] = body["presence_penalty"]

    if options:
        ollama_body["options"] = options

    return model, ollama_body, stream


def parse_openai_completion_request(
    body: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    """Parse an OpenAI /v1/completions request into Ollama format.

    Args:
        body: The OpenAI-format request body.

    Returns:
        Tuple of (model_name, ollama_request_body, is_streaming).
    """
    model = body.get("model", "")
    stream = body.get("stream", False)

    ollama_body: dict[str, Any] = {
        "model": model,
        "prompt": body.get("prompt", ""),
        "stream": stream,
    }

    options: dict[str, Any] = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "stop" in body:
        options["stop"] = body["stop"]

    if options:
        ollama_body["options"] = options

    return model, ollama_body, stream


def parse_openai_embedding_request(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse an OpenAI /v1/embeddings request into Ollama format.

    Args:
        body: The OpenAI-format request body.

    Returns:
        Tuple of (model_name, ollama_request_body).
    """
    model = body.get("model", "")
    input_text = body.get("input", "")

    # OpenAI allows input to be a string or list of strings
    if isinstance(input_text, list):
        input_text = input_text[0] if input_text else ""

    ollama_body: dict[str, Any] = {
        "model": model,
        "prompt": input_text,
    }

    return model, ollama_body


def ollama_chat_to_openai(
    ollama_response: dict[str, Any], model: str
) -> dict[str, Any]:
    """Convert an Ollama chat response to OpenAI format.

    Args:
        ollama_response: The response from Ollama's /api/chat.
        model: The model name used.

    Returns:
        OpenAI-formatted response dict.
    """
    message = ollama_response.get("message", {})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
            "completion_tokens": ollama_response.get("eval_count", 0),
            "total_tokens": (
                ollama_response.get("prompt_eval_count", 0)
                + ollama_response.get("eval_count", 0)
            ),
        },
    }


def ollama_generate_to_openai(
    ollama_response: dict[str, Any], model: str
) -> dict[str, Any]:
    """Convert an Ollama generate response to OpenAI format.

    Args:
        ollama_response: The response from Ollama's /api/generate.
        model: The model name used.

    Returns:
        OpenAI-formatted response dict.
    """
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": ollama_response.get("response", ""),
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
            "completion_tokens": ollama_response.get("eval_count", 0),
            "total_tokens": (
                ollama_response.get("prompt_eval_count", 0)
                + ollama_response.get("eval_count", 0)
            ),
        },
    }


def ollama_embedding_to_openai(
    ollama_response: dict[str, Any], model: str
) -> dict[str, Any]:
    """Convert an Ollama embedding response to OpenAI format.

    Args:
        ollama_response: The response from Ollama's /api/embeddings.
        model: The model name used.

    Returns:
        OpenAI-formatted response dict.
    """
    embedding = ollama_response.get("embedding", [])

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": embedding,
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
            "total_tokens": ollama_response.get("prompt_eval_count", 0),
        },
    }


def ollama_chat_stream_to_openai(chunk: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a streaming Ollama chat chunk to OpenAI SSE format.

    Args:
        chunk: A single NDJSON line from Ollama's streaming response.
        model: The model name.

    Returns:
        OpenAI-formatted streaming chunk dict.
    """
    message = chunk.get("message", {})
    done = chunk.get("done", False)

    delta: dict[str, str] = {}
    if content := message.get("content"):
        delta["content"] = content
    if role := message.get("role"):
        delta["role"] = role

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": "stop" if done else None,
            }
        ],
    }
