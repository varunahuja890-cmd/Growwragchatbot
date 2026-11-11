"""Phase 7 — LLM adapters (llm/factory.py).

One chat call at temperature 0; the provider string comes from config
(default Groq, OpenAI-compatible chat). The query pipeline never imports a
provider SDK directly — it depends on the ``ChatClient`` protocol below.

Every provider:

- implements ``complete(system, user, temperature, max_tokens) -> str``;
- retries once internally on timeout / 5xx / 429, then raises (the pipeline
  turns that into the template fallback). Non-retryable client errors
  (401 bad key, 400 bad request) fail immediately;
- implements ``check_model() -> (status, message)`` where status is one of:

  - ``"ok"`` — the model id exists at the provider;
  - ``"bad_model"`` — the id is not accepted (for example
    ``qwen/qwen3.8-27b`` is not a valid Groq id). This must stop the app
    with that id in the error;
  - ``"unverified"`` — auth or network problem; the app degrades to the
    warning banner + cited templates instead of stopping.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from groww_faq.config import LlmSettings, Secrets

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str: ...
    def check_model(self) -> tuple[str, str]: ...


class _OpenAICompatClient:
    """Shared implementation for Groq / OpenAI / Mistral chat endpoints."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s),
        )

    # ---------------------------------------------------------------- call
    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error = "unknown"
        for attempt in (1, 2):  # "LLM timeout twice -> template fallback"
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"http_{resp.status_code}"
                    if attempt == 1:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    break
                if resp.status_code >= 400:  # bad key / bad request — not retryable
                    last_error = f"http_{resp.status_code}: {resp.text[:160]}"
                    break
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except (httpx.HTTPError, KeyError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise RuntimeError(f"LLM call failed after 2 attempts ({last_error})")

    def check_model(self) -> tuple[str, str]:
        """(status, message) — ok | bad_model | unverified."""
        try:
            resp = self._client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            ids = [m.get("id", "") for m in resp.json().get("data", [])]
            if self.model in ids:
                return "ok", ""
            sample = ", ".join(ids[:8])
            return "bad_model", (
                f"model id '{self.model}' is not accepted by the provider "
                f"(valid ids include: {sample})"
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (401, 403):
                return "unverified", (
                    f"API key was rejected by the provider (HTTP {code}) — "
                    "check the key in .env. Model answers fall back to cited "
                    "templates until it is fixed."
                )
            return "unverified", f"could not verify model id (HTTP {code})"
        except httpx.HTTPError as exc:
            return "unverified", f"could not verify model id: {exc}"


class GroqClient(_OpenAICompatClient):
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        super().__init__("https://api.groq.com/openai/v1", api_key, model, timeout_s)


class OpenAIClient(_OpenAICompatClient):
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        super().__init__("https://api.openai.com/v1", api_key, model, timeout_s)


class MistralClient(_OpenAICompatClient):
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        super().__init__("https://api.mistral.ai/v1", api_key, model, timeout_s)


class AnthropicClient:
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self.model = model
        self._client = httpx.Client(
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=httpx.Timeout(timeout_s),
        )

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        last_error = "unknown"
        for attempt in (1, 2):
            try:
                resp = self._client.post("https://api.anthropic.com/v1/messages", json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"http_{resp.status_code}"
                    if attempt == 1:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    break
                if resp.status_code >= 400:  # bad key / bad request — not retryable
                    last_error = f"http_{resp.status_code}: {resp.text[:160]}"
                    break
                resp.raise_for_status()
                data = resp.json()
                parts = data.get("content", [])
                if not parts:
                    raise KeyError("empty content")
                return str(parts[0].get("text", "")).strip()
            except (httpx.HTTPError, KeyError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise RuntimeError(f"LLM call failed after 2 attempts ({last_error})")

    def check_model(self) -> tuple[str, str]:
        """(status, message) — ok | bad_model | unverified."""
        try:
            resp = self._client.get("https://api.anthropic.com/v1/models")
            resp.raise_for_status()
            ids = [m.get("id", "") for m in resp.json().get("data", [])]
            if self.model in ids:
                return "ok", ""
            sample = ", ".join(ids[:8])
            return "bad_model", (
                f"model id '{self.model}' not found (valid ids include: {sample})"
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (401, 403):
                return "unverified", (
                    f"API key was rejected by the provider (HTTP {code}) — "
                    "check the key in .env. Model answers fall back to cited "
                    "templates until it is fixed."
                )
            return "unverified", f"could not verify model id (HTTP {code})"
        except httpx.HTTPError as exc:
            return "unverified", f"could not verify model id: {exc}"


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self.model = model
        self._client = httpx.Client(
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(timeout_s),
        )

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        last_error = "unknown"
        for attempt in (1, 2):
            try:
                resp = self._client.post(url, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"http_{resp.status_code}"
                    if attempt == 1:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    break
                if resp.status_code >= 400:  # bad key / bad request — not retryable
                    last_error = f"http_{resp.status_code}: {resp.text[:160]}"
                    break
                resp.raise_for_status()
                data = resp.json()
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise RuntimeError(f"LLM call failed after 2 attempts ({last_error})")

    def check_model(self) -> tuple[str, str]:
        """(status, message) — ok | bad_model | unverified."""
        try:
            resp = self._client.get(
                "https://generativelanguage.googleapis.com/v1beta/models"
            )
            resp.raise_for_status()
            names = [
                m.get("name", "").removeprefix("models/")
                for m in resp.json().get("models", [])
            ]
            if self.model in names:
                return "ok", ""
            sample = ", ".join(n for n in names[:8] if n)
            return "bad_model", (
                f"model id '{self.model}' not found (valid ids include: {sample})"
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (400, 401, 403):  # Gemini reports a bad key as 400
                return "unverified", (
                    f"API key was rejected by the provider (HTTP {code}) — "
                    "check the key in .env. Model answers fall back to cited "
                    "templates until it is fixed."
                )
            return "unverified", f"could not verify model id (HTTP {code})"
        except httpx.HTTPError as exc:
            return "unverified", f"could not verify model id: {exc}"


def create_chat_client(llm: LlmSettings, secrets: Secrets) -> ChatClient:
    """Provider string -> adapter. The API key comes from .env only."""
    provider = (llm.provider or "groq").lower()
    key = _api_key(provider, secrets)
    if provider == "groq":
        return GroqClient(key, llm.model, llm.timeout_s)
    if provider == "openai":
        return OpenAIClient(key, llm.model, llm.timeout_s)
    if provider == "mistral":
        return MistralClient(key, llm.model, llm.timeout_s)
    if provider == "anthropic":
        return AnthropicClient(key, llm.model, llm.timeout_s)
    if provider == "gemini":
        return GeminiClient(key, llm.model, llm.timeout_s)
    raise RuntimeError(
        f"unknown llm.provider {provider!r} — expected groq | gemini | openai "
        "| anthropic | mistral"
    )


def _api_key(provider: str, secrets: Secrets) -> str:
    pairs = {
        "groq": (secrets.groq_api_key, "GROQ_API_KEY"),
        "openai": (secrets.openai_api_key, "OPENAI_API_KEY"),
        "anthropic": (secrets.anthropic_api_key, "ANTHROPIC_API_KEY"),
        "gemini": (secrets.gemini_api_key, "GEMINI_API_KEY"),
        "mistral": (secrets.mistral_api_key, "MISTRAL_API_KEY"),
    }
    key, env_name = pairs[provider]
    if not key:
        raise RuntimeError(
            f"{env_name} is missing — add it to .env (copy .env.example). "
            "Without it no LLM answer can be generated."
        )
    return key
