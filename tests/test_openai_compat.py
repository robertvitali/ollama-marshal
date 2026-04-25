from __future__ import annotations

import time

from ollama_marshal.openai_compat import (
    ollama_chat_stream_to_openai,
    ollama_chat_to_openai,
    ollama_embedding_to_openai,
    ollama_generate_to_openai,
    parse_openai_chat_request,
    parse_openai_completion_request,
    parse_openai_embedding_request,
)

# ---------------------------------------------------------------------------
# parse_openai_chat_request
# ---------------------------------------------------------------------------


class TestParseOpenaiChatRequest:
    def test_basic_request(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        model, ollama_body, stream = parse_openai_chat_request(body)

        assert model == "gpt-4"
        assert ollama_body["model"] == "gpt-4"
        assert ollama_body["messages"] == [{"role": "user", "content": "Hello"}]
        assert ollama_body["stream"] is False
        assert stream is False

    def test_streaming_flag(self):
        body = {
            "model": "gpt-4",
            "messages": [],
            "stream": True,
        }
        _model, ollama_body, stream = parse_openai_chat_request(body)

        assert stream is True
        assert ollama_body["stream"] is True

    def test_maps_temperature(self):
        body = {"model": "m", "messages": [], "temperature": 0.7}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["temperature"] == 0.7

    def test_maps_top_p(self):
        body = {"model": "m", "messages": [], "top_p": 0.9}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["top_p"] == 0.9

    def test_maps_max_tokens_to_num_predict(self):
        body = {"model": "m", "messages": [], "max_tokens": 256}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["num_predict"] == 256

    def test_maps_stop(self):
        body = {"model": "m", "messages": [], "stop": ["\n", "END"]}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["stop"] == ["\n", "END"]

    def test_maps_seed(self):
        body = {"model": "m", "messages": [], "seed": 42}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["seed"] == 42

    def test_maps_frequency_penalty(self):
        body = {"model": "m", "messages": [], "frequency_penalty": 0.5}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["frequency_penalty"] == 0.5

    def test_maps_presence_penalty(self):
        body = {"model": "m", "messages": [], "presence_penalty": 0.3}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["presence_penalty"] == 0.3

    def test_multiple_options(self):
        body = {
            "model": "m",
            "messages": [],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
            "seed": 7,
        }
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["options"]["temperature"] == 0.5
        assert ollama_body["options"]["top_p"] == 0.9
        assert ollama_body["options"]["num_predict"] == 100
        assert ollama_body["options"]["seed"] == 7

    def test_no_options_when_none_provided(self):
        body = {"model": "m", "messages": []}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert "options" not in ollama_body

    def test_missing_model_defaults_to_empty_string(self):
        body = {"messages": []}
        model, ollama_body, _ = parse_openai_chat_request(body)

        assert model == ""
        assert ollama_body["model"] == ""

    def test_missing_messages_defaults_to_empty_list(self):
        body = {"model": "m"}
        _, ollama_body, _ = parse_openai_chat_request(body)

        assert ollama_body["messages"] == []


# ---------------------------------------------------------------------------
# parse_openai_completion_request
# ---------------------------------------------------------------------------


class TestParseOpenaiCompletionRequest:
    def test_basic_request(self):
        body = {"model": "gpt-3.5", "prompt": "Once upon a time"}
        model, ollama_body, stream = parse_openai_completion_request(body)

        assert model == "gpt-3.5"
        assert ollama_body["model"] == "gpt-3.5"
        assert ollama_body["prompt"] == "Once upon a time"
        assert ollama_body["stream"] is False
        assert stream is False

    def test_streaming_flag(self):
        body = {"model": "m", "prompt": "hi", "stream": True}
        _, _, stream = parse_openai_completion_request(body)

        assert stream is True

    def test_maps_options(self):
        body = {
            "model": "m",
            "prompt": "hi",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 512,
            "stop": ["END"],
        }
        _, ollama_body, _ = parse_openai_completion_request(body)

        assert ollama_body["options"]["temperature"] == 0.8
        assert ollama_body["options"]["top_p"] == 0.95
        assert ollama_body["options"]["num_predict"] == 512
        assert ollama_body["options"]["stop"] == ["END"]

    def test_no_options_when_none_provided(self):
        body = {"model": "m", "prompt": "hi"}
        _, ollama_body, _ = parse_openai_completion_request(body)

        assert "options" not in ollama_body

    def test_missing_prompt_defaults_to_empty_string(self):
        body = {"model": "m"}
        _, ollama_body, _ = parse_openai_completion_request(body)

        assert ollama_body["prompt"] == ""

    def test_missing_model_defaults_to_empty_string(self):
        body = {"prompt": "hi"}
        model, _, _ = parse_openai_completion_request(body)

        assert model == ""


# ---------------------------------------------------------------------------
# parse_openai_embedding_request
# ---------------------------------------------------------------------------


class TestParseOpenaiEmbeddingRequest:
    def test_string_input(self):
        body = {"model": "embed-model", "input": "Hello world"}
        model, ollama_body = parse_openai_embedding_request(body)

        assert model == "embed-model"
        assert ollama_body["model"] == "embed-model"
        assert ollama_body["prompt"] == "Hello world"

    def test_list_input_uses_first_element(self):
        body = {"model": "embed-model", "input": ["first", "second", "third"]}
        _, ollama_body = parse_openai_embedding_request(body)

        assert ollama_body["prompt"] == "first"

    def test_empty_list_input(self):
        body = {"model": "embed-model", "input": []}
        _, ollama_body = parse_openai_embedding_request(body)

        assert ollama_body["prompt"] == ""

    def test_missing_input_defaults_to_empty_string(self):
        body = {"model": "embed-model"}
        _, ollama_body = parse_openai_embedding_request(body)

        assert ollama_body["prompt"] == ""

    def test_missing_model(self):
        body = {"input": "hi"}
        model, _ = parse_openai_embedding_request(body)

        assert model == ""


# ---------------------------------------------------------------------------
# ollama_chat_to_openai
# ---------------------------------------------------------------------------


class TestOllamaChatToOpenai:
    def test_correct_structure(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = ollama_chat_to_openai(ollama_resp, "llama3")

        assert result["object"] == "chat.completion"
        assert result["model"] == "llama3"
        assert result["id"].startswith("chatcmpl-")
        assert isinstance(result["created"], int)
        assert abs(result["created"] - int(time.time())) < 5

    def test_choices(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = ollama_chat_to_openai(ollama_resp, "llama3")

        assert len(result["choices"]) == 1
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello!"
        assert choice["finish_reason"] == "stop"

    def test_usage_counts(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hi"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        result = ollama_chat_to_openai(ollama_resp, "llama3")

        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 8
        assert result["usage"]["total_tokens"] == 23

    def test_missing_counts_default_to_zero(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hi"},
        }
        result = ollama_chat_to_openai(ollama_resp, "llama3")

        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0

    def test_missing_message_defaults(self):
        ollama_resp = {}
        result = ollama_chat_to_openai(ollama_resp, "llama3")

        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == ""

    def test_unique_ids(self):
        ollama_resp = {"message": {"role": "assistant", "content": ""}}
        r1 = ollama_chat_to_openai(ollama_resp, "m")
        r2 = ollama_chat_to_openai(ollama_resp, "m")
        assert r1["id"] != r2["id"]


# ---------------------------------------------------------------------------
# ollama_generate_to_openai
# ---------------------------------------------------------------------------


class TestOllamaGenerateToOpenai:
    def test_correct_structure(self):
        ollama_resp = {
            "response": "The sky is blue because...",
            "prompt_eval_count": 12,
            "eval_count": 20,
        }
        result = ollama_generate_to_openai(ollama_resp, "llama3")

        assert result["object"] == "text_completion"
        assert result["model"] == "llama3"
        assert result["id"].startswith("cmpl-")
        assert isinstance(result["created"], int)

    def test_text_field(self):
        ollama_resp = {
            "response": "The answer is 42.",
            "prompt_eval_count": 5,
            "eval_count": 4,
        }
        result = ollama_generate_to_openai(ollama_resp, "llama3")

        assert len(result["choices"]) == 1
        assert result["choices"][0]["text"] == "The answer is 42."
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage(self):
        ollama_resp = {
            "response": "hi",
            "prompt_eval_count": 7,
            "eval_count": 3,
        }
        result = ollama_generate_to_openai(ollama_resp, "llama3")

        assert result["usage"]["prompt_tokens"] == 7
        assert result["usage"]["completion_tokens"] == 3
        assert result["usage"]["total_tokens"] == 10

    def test_missing_response_defaults_to_empty(self):
        ollama_resp = {}
        result = ollama_generate_to_openai(ollama_resp, "m")

        assert result["choices"][0]["text"] == ""

    def test_missing_counts_default_to_zero(self):
        ollama_resp = {"response": "hi"}
        result = ollama_generate_to_openai(ollama_resp, "m")

        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0

    def test_unique_ids(self):
        ollama_resp = {"response": ""}
        r1 = ollama_generate_to_openai(ollama_resp, "m")
        r2 = ollama_generate_to_openai(ollama_resp, "m")
        assert r1["id"] != r2["id"]


# ---------------------------------------------------------------------------
# ollama_embedding_to_openai
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingToOpenai:
    def test_correct_structure(self):
        ollama_resp = {
            "embedding": [0.1, 0.2, 0.3],
            "prompt_eval_count": 4,
        }
        result = ollama_embedding_to_openai(ollama_resp, "embed-model")

        assert result["object"] == "list"
        assert result["model"] == "embed-model"

    def test_embedding_data(self):
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        ollama_resp = {
            "embedding": embedding,
            "prompt_eval_count": 3,
        }
        result = ollama_embedding_to_openai(ollama_resp, "embed-model")

        assert len(result["data"]) == 1
        data_item = result["data"][0]
        assert data_item["object"] == "embedding"
        assert data_item["index"] == 0
        assert data_item["embedding"] == embedding

    def test_usage(self):
        ollama_resp = {
            "embedding": [0.1],
            "prompt_eval_count": 10,
        }
        result = ollama_embedding_to_openai(ollama_resp, "m")

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["total_tokens"] == 10

    def test_missing_embedding_defaults_to_empty_list(self):
        ollama_resp = {}
        result = ollama_embedding_to_openai(ollama_resp, "m")

        assert result["data"][0]["embedding"] == []

    def test_missing_counts_default_to_zero(self):
        ollama_resp = {"embedding": [0.1]}
        result = ollama_embedding_to_openai(ollama_resp, "m")

        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0


# ---------------------------------------------------------------------------
# ollama_chat_stream_to_openai
# ---------------------------------------------------------------------------


class TestOllamaChatStreamToOpenai:
    def test_chunk_with_content(self):
        chunk = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }
        result = ollama_chat_stream_to_openai(chunk, "llama3")

        assert result["object"] == "chat.completion.chunk"
        assert result["model"] == "llama3"
        assert result["id"].startswith("chatcmpl-")
        assert isinstance(result["created"], int)

        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["delta"]["content"] == "Hello"
        assert choice["delta"]["role"] == "assistant"
        assert choice["finish_reason"] is None

    def test_done_chunk(self):
        chunk = {
            "message": {},
            "done": True,
        }
        result = ollama_chat_stream_to_openai(chunk, "llama3")

        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["delta"] == {}

    def test_chunk_without_content(self):
        chunk = {
            "message": {"role": "assistant"},
            "done": False,
        }
        result = ollama_chat_stream_to_openai(chunk, "llama3")

        delta = result["choices"][0]["delta"]
        assert "content" not in delta
        assert delta["role"] == "assistant"

    def test_chunk_with_only_content_no_role(self):
        chunk = {
            "message": {"content": "world"},
            "done": False,
        }
        result = ollama_chat_stream_to_openai(chunk, "llama3")

        delta = result["choices"][0]["delta"]
        assert delta["content"] == "world"
        assert "role" not in delta

    def test_empty_content_not_included(self):
        chunk = {
            "message": {"content": "", "role": ""},
            "done": False,
        }
        result = ollama_chat_stream_to_openai(chunk, "m")

        # Empty strings are falsy, so walrus operator skips them
        assert result["choices"][0]["delta"] == {}

    def test_unique_ids_per_chunk(self):
        chunk = {"message": {"content": "a"}, "done": False}
        r1 = ollama_chat_stream_to_openai(chunk, "m")
        r2 = ollama_chat_stream_to_openai(chunk, "m")
        assert r1["id"] != r2["id"]

    def test_missing_message_key(self):
        chunk = {"done": True}
        result = ollama_chat_stream_to_openai(chunk, "m")

        assert result["choices"][0]["delta"] == {}
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_missing_done_defaults_to_false(self):
        chunk = {"message": {"content": "hi"}}
        result = ollama_chat_stream_to_openai(chunk, "m")

        assert result["choices"][0]["finish_reason"] is None
