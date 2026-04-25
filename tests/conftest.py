"""Shared test fixtures for ollama-marshal tests."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def sample_ps_response() -> dict[str, Any]:
    """Mock response from Ollama /api/ps endpoint."""
    return {
        "models": [
            {
                "name": "llama3:latest",
                "model": "llama3:latest",
                "size": 4_661_224_676,
                "digest": "a6990ed6be41",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8B",
                    "quantization_level": "Q4_0",
                },
                "expires_at": "2026-04-24T23:59:59Z",
                "size_vram": 4_661_224_676,
            }
        ]
    }


@pytest.fixture
def sample_tags_response() -> dict[str, Any]:
    """Mock response from Ollama /api/tags endpoint."""
    return {
        "models": [
            {
                "name": "llama3:latest",
                "model": "llama3:latest",
                "modified_at": "2026-04-20T10:00:00Z",
                "size": 4_661_224_676,
                "digest": "a6990ed6be41",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8B",
                    "quantization_level": "Q4_0",
                },
            },
            {
                "name": "codellama:latest",
                "model": "codellama:latest",
                "modified_at": "2026-04-19T10:00:00Z",
                "size": 3_825_819_519,
                "digest": "b1234567890a",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0",
                },
            },
            {
                "name": "mistral:latest",
                "model": "mistral:latest",
                "modified_at": "2026-04-18T10:00:00Z",
                "size": 4_109_865_159,
                "digest": "c9876543210b",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "mistral",
                    "families": ["mistral"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0",
                },
            },
        ]
    }


@pytest.fixture
def sample_chat_request() -> dict[str, Any]:
    """Sample Ollama /api/chat request body."""
    return {
        "model": "llama3:latest",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "stream": True,
    }


@pytest.fixture
def sample_generate_request() -> dict[str, Any]:
    """Sample Ollama /api/generate request body."""
    return {
        "model": "llama3:latest",
        "prompt": "Why is the sky blue?",
        "stream": False,
    }


@pytest.fixture
def sample_embeddings_request() -> dict[str, Any]:
    """Sample Ollama /api/embeddings request body."""
    return {
        "model": "llama3:latest",
        "prompt": "Hello, world!",
    }


@pytest.fixture
def sample_chat_response() -> dict[str, Any]:
    """Sample non-streaming Ollama /api/chat response."""
    return {
        "model": "llama3:latest",
        "created_at": "2026-04-24T12:00:00Z",
        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
        "done": True,
        "total_duration": 1_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200_000_000,
        "eval_count": 8,
        "eval_duration": 300_000_000,
    }


@pytest.fixture
def sample_show_response() -> dict[str, Any]:
    """Mock response from Ollama /api/show endpoint."""
    return {
        "modelfile": "FROM llama3",
        "parameters": "stop [INST]\nstop [/INST]",
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8B",
            "quantization_level": "Q4_0",
        },
        "model_info": {
            "general.file_type": 2,
            "general.parameter_count": 8_030_261_248,
        },
    }
