from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

MAX_TOKENS = 4096

Message = dict[str, str]
BatchMessages = Sequence[Sequence[Message]]


def _messages_are_batch(messages: Any) -> bool:
    return (
        isinstance(messages, Sequence)
        and len(messages) > 0
        and isinstance(messages[0], Sequence)
        and len(messages[0]) > 0
        and isinstance(messages[0][0], dict)
    )


def _detect_provider(model_name: str) -> str:
    lower = model_name.lower()
    if "grok" in lower:
        return "azure_grok"
    if "gemini" in lower:
        return "gemini"
    if "claude" in lower or "sonnet" in lower:
        return "anthropic"
    return "azure_openai"


def _anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            continue
        formatted.append(
            {
                "role": message.get("role", "user"),
                "content": [{"type": "text", "text": str(message.get("content", ""))}],
            }
        )
    return formatted


def _messages_to_text(messages: Sequence[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


class Model:
    """LLM client wrapper for Azure OpenAI, Azure Anthropic, and Gemini."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = _detect_provider(model_name)
        self.temperature = 0.1 if temperature is None else float(temperature)
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.model_id = model_name
        self.client: Any = None
        self.embedding_client: AzureOpenAI | None = None
        self._init_client()
        self.reset()

    def _init_client(self) -> None:
        if self.provider == "azure_openai":
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.model_name)
            if not endpoint or not api_key:
                raise ValueError("AZURE_OPENAI_ENDPOINTとOPENAI_API_KEYを設定してください。")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            self.embedding_client = self.client
            self.model_id = deployment
        elif self.provider == "anthropic":
            base_url = os.environ.get("AZURE_CLAUDE_ENDPOINT")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
            deployment = os.environ.get("AZURE_CLAUDE_DEPLOYMENT", self.model_name)
            if not base_url or not api_key:
                raise ValueError("AZURE_CLAUDE_ENDPOINTとANTHROPIC_API_KEYを設定してください。")
            self.client = anthropic.Anthropic(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "anthropic-version": version,
                },
            )
            self.model_id = deployment
        elif self.provider == "azure_grok":
            endpoint = os.environ.get("AZURE_GROK_ENDPOINT")
            api_key = os.environ.get("GROK_API_KEY")
            deployment = os.environ.get("AZURE_GROK_DEPLOYMENT", self.model_name)
            if not endpoint or not api_key:
                raise ValueError("AZURE_GROK_ENDPOINTとGROK_API_KEYを設定してください。")
            self.client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
            )
            self.model_id = deployment
        elif self.provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEYを設定してください。")
            genai.configure(api_key=api_key)
            base_model = os.environ.get("GEMINI_MODEL_NAME", self.model_name)
            if not base_model.startswith("models/"):
                base_model = f"models/{base_model}"
            self.model_id = base_model
            self.client = genai.GenerativeModel(self.model_id)
        else:
            raise ValueError(f"Unsupported provider for model {self.model_name}")

    def reset(self) -> None:
        self.message_history: list[Message] = [dict(role="system", content=self.system_prompt)]
        self.history: list[dict[str, Any]] = []

    @property
    def last_output_text(self) -> str:
        return self.history[-1]["output_text"]

    @property
    def config(self) -> dict[str, Any]:
        return dict(
            model_name=self.model_name,
            provider=self.provider,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
        )

    def _prepend_system(self, messages: Sequence[Message]) -> list[Message]:
        if messages and messages[0].get("role") == "system":
            return list(messages)
        return [dict(role="system", content=self.system_prompt), *messages]

    def _send_messages(
        self, messages: Sequence[Message], max_tokens: int = MAX_TOKENS, max_retries: int = 3
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                if self.provider == "azure_openai":
                    params: dict[str, Any] = dict(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                    )
                    # GPT-5系はtemperature固定のため送らない
                    if not self.model_name.startswith("gpt-5"):
                        params["temperature"] = self.temperature
                    resp = self.client.chat.completions.create(**params)
                    return resp.choices[0].message.content or ""

                if self.provider == "azure_grok":
                    # grok-4はreasoning modelのため、max_completion_tokensを使用し、temperatureは送らない
                    params = dict(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                    )
                    resp = self.client.chat.completions.create(**params)
                    return resp.choices[0].message.content or ""

                if self.provider == "anthropic":
                    resp = self.client.messages.create(
                        model=self.model_id,
                        system=self.system_prompt,
                        messages=_anthropic_messages(messages),
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                    )
                    parts = []
                    for block in resp.content:
                        text = getattr(block, "text", None)
                        if text:
                            parts.append(text)
                    return "".join(parts)

                user_text = messages[-1].get("content", "") if messages else ""
                prompt_text = f"{self.system_prompt}\n\n{user_text}".strip()
                response = self.client.generate_content(
                    prompt_text,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": max(max_tokens, 256),
                    },
                )
                try:
                    if hasattr(response, "text") and response.text:
                        return response.text
                except Exception:
                    pass
                candidates = getattr(response, "candidates", None) or []
                if not candidates:
                    return ""
                parts = getattr(candidates[0].content, "parts", []) or []  # type: ignore[attr-defined]
                outputs = []
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        outputs.append(part_text)
                if not outputs and candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", "")
                    return f"[no content returned; finish_reason={finish_reason}]"
                return " ".join(outputs)
            except Exception as error:  # pragma: no cover - transient API errors
                last_error = error
                if attempt == max_retries - 1:
                    break
                sleep_s = (1.5**attempt) + random.random() * 0.25
                time.sleep(sleep_s)
        raise RuntimeError(f"Generation failed after retries: {last_error}") from last_error

    def generate(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.message_history.append(dict(role="user", content=input_text))
        output_text = self._send_messages(
            self.message_history,
            max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
            max_retries=kwargs.get("max_retries", 3),
        )
        self.message_history.append(dict(role="assistant", content=output_text))
        completion: dict[str, Any] = {}
        self.history.append(
            dict(
                input_text=input_text,
                output_text=output_text,
                completion=completion,
            )
        )
        return dict(input_text=input_text, output_text=output_text)

    def generate_with_messages(
        self, messages_or_batches: Sequence[Message] | BatchMessages, **kwargs: Any
    ) -> Any:
        max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
        max_retries = kwargs.get("max_retries", 3)
        if not _messages_are_batch(messages_or_batches):
            messages = self._prepend_system(messages_or_batches)  # type: ignore[arg-type]
            return self._send_messages(
                messages,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

        batches: BatchMessages = messages_or_batches  # type: ignore[assignment]
        results: list[str] = ["" for _ in batches]

        def worker(idx: int, msgs: Sequence[Message]) -> tuple[int, str]:
            try:
                normalized = self._prepend_system(msgs)
                return idx, self._send_messages(
                    normalized,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                )
            except Exception:
                return idx, ""

        max_workers = max(1, min(len(batches), 8))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, i, batch) for i, batch in enumerate(batches)]
            for fut in as_completed(futures):
                try:
                    idx, text = fut.result()
                except Exception:
                    continue
                if 0 <= idx < len(results):
                    results[idx] = text
        return results

    def embedding(
        self,
        text: str,
        model_name: str | None = None,
        normalize: bool = True,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> dict[str, Any]:
        if not self.embedding_client:
            raise ValueError("Azure OpenAIクライアントが設定されていません。")
        if not text:
            return {"embedding": [], "model": model_name or "", "dim": 0}
        embed_model = model_name or os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        )
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.embedding_client.embeddings.create(model=embed_model, input=text)
                vec = resp.data[0].embedding
                if normalize and vec:
                    s = sum(v * v for v in vec)
                    if s > 0:
                        norm = s**0.5
                        vec = [v / norm for v in vec]
                return {"embedding": vec, "model": embed_model, "dim": len(vec)}
            except Exception as error:
                last_err = error
                if attempt == max_retries - 1:
                    break
                sleep_s = (backoff**attempt) + random.random() * 0.25
                time.sleep(sleep_s)
        raise RuntimeError(f"Embedding failed: {last_err}")  # pragma: no cover
