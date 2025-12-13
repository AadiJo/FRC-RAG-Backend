"""OpenRouter client for chat completions.

Keeps interface similar to the existing Chutes client.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional

import requests

from .config import get_config

Config = get_config()
logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for interacting with OpenRouter's OpenAI-compatible API."""

    def __init__(self):
        self.base_url = getattr(Config, "OPENROUTER_BASE_URL", None) or "https://openrouter.ai/api/v1"
        self.api_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

        # Prefer OPENROUTER_API_KEY, but allow legacy CHUTES_API_TOKEN as fallback
        self.api_token = (getattr(Config, "OPENROUTER_API_KEY", "") or "").strip() or (
            getattr(Config, "CHUTES_API_TOKEN", "") or ""
        ).strip()

        self.model = getattr(Config, "OPENROUTER_DEFAULT_MODEL", None) or "openrouter/auto"

        self.http_referer = (getattr(Config, "OPENROUTER_HTTP_REFERER", "") or "").strip()
        self.app_title = (getattr(Config, "OPENROUTER_APP_TITLE", "") or "").strip()

        if not self.api_token:
            logger.warning("OpenRouter API key not configured (set OPENROUTER_API_KEY)")

    def _headers(self, api_key: str) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Optional but recommended by OpenRouter
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def chat_completion(
        self,
        prompt: str,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        show_reasoning: Optional[bool] = None,
        custom_api_key: Optional[str] = None,
        custom_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if stream:
            return "".join(
                self.chat_completion_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    show_reasoning=show_reasoning,
                    custom_api_key=custom_api_key,
                    custom_model=custom_model,
                    system_prompt=system_prompt,
                )
            )

        api_key = (custom_api_key or self.api_token or "").strip()
        model = (custom_model or self.model or "").strip()
        if not api_key:
            raise ValueError("OpenRouter API key not configured")
        if not model:
            raise ValueError("OpenRouter model not configured")

        include_reasoning = show_reasoning if show_reasoning is not None else Config.SHOW_MODEL_REASONING

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            logger.info("Sending request to OpenRouter with model %s", model)
            response = requests.post(self.api_url, headers=self._headers(api_key), json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # OpenAI-compatible format
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}

            content = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""

            if include_reasoning and reasoning:
                if content:
                    return f"> *Reasoning: {reasoning}*\n\n{content}"
                return f"> *Reasoning: {reasoning}*"

            if content:
                return content

            if reasoning and not include_reasoning:
                logger.warning("Only reasoning available, but show_reasoning disabled")
                return "I apologize, but I'm having trouble generating a proper response. Please try again."

            logger.error("No content in OpenRouter response: %s", result)
            raise ValueError("No content in OpenRouter response")

        except requests.exceptions.RequestException as e:
            logger.error("OpenRouter request error: %s", e)
            raise ConnectionError(f"Failed to connect to OpenRouter: {e}")

    def chat_completion_stream(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        show_reasoning: Optional[bool] = None,
        custom_api_key: Optional[str] = None,
        custom_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        api_key = (custom_api_key or self.api_token or "").strip()
        model = (custom_model or self.model or "").strip()
        if not api_key:
            raise ValueError("OpenRouter API key not configured")
        if not model:
            raise ValueError("OpenRouter model not configured")

        include_reasoning = show_reasoning if show_reasoning is not None else Config.SHOW_MODEL_REASONING

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            logger.info("Sending streaming request to OpenRouter with model %s", model)
            response = requests.post(
                self.api_url,
                headers=self._headers(api_key),
                json=payload,
                stream=True,
                timeout=90,
            )
            response.raise_for_status()

            reasoning_started = False
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data_json = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data_json.get("choices") or []
                if not choices:
                    continue

                delta = (choices[0].get("delta") or {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""

                if include_reasoning and reasoning:
                    formatted_reasoning = reasoning.replace("\n", "\n> ")
                    if not reasoning_started:
                        yield f"> *Reasoning:*\n> {formatted_reasoning}"
                        reasoning_started = True
                    else:
                        yield formatted_reasoning

                if content:
                    if reasoning_started:
                        yield "\n\n"
                        reasoning_started = False
                    yield content

            if reasoning_started:
                yield "\n\n"

        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            try:
                if getattr(e, "response", None) is not None:
                    error_detail = f"{e} - Response: {e.response.text}"  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.error("OpenRouter streaming request error: %s", error_detail)
            raise ConnectionError(f"Failed to connect to OpenRouter for streaming: {error_detail}")

    def list_models(self, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        key = (api_key or self.api_token or "").strip()
        if not key:
            raise ValueError("OpenRouter API key not configured")

        response = requests.get(self.models_url, headers=self._headers(key), timeout=30)
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or []

        models: List[Dict[str, Any]] = []
        for m in data:
            model_id = m.get("id")
            name = m.get("name") or model_id
            pricing = m.get("pricing") or {}

            def _is_zero(v: Any) -> bool:
                try:
                    return float(v) == 0.0
                except Exception:
                    return str(v).strip() in {"0", "0.0", "0.00"}

            free = _is_zero(pricing.get("prompt")) and _is_zero(pricing.get("completion"))
            if model_id:
                models.append({"id": model_id, "name": name, "free": free})

        return models

    def get_quota(self, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve quota/usage information for the provided API key.

        This method tries several low-cost endpoints and inspects response headers
        for common rate-limit/quota headers. If no quota info is discoverable,
        returns None.
        """
        key = (api_key or self.api_token or "").strip()
        if not key:
            raise ValueError("OpenRouter API key not configured")

        try:
            # First, try a cheap models request and inspect headers
            resp = requests.get(self.models_url, headers=self._headers(key), timeout=20)
            resp.raise_for_status()

            headers = {k.lower(): v for k, v in resp.headers.items()}

            def _get_header(*names):
                for n in names:
                    v = headers.get(n.lower())
                    if v is not None:
                        return v
                return None

            remaining = _get_header('x-ratelimit-remaining', 'x-ratelimit-remaining-requests', 'x-openrouter-remaining')
            limit = _get_header('x-ratelimit-limit', 'x-ratelimit-limit-requests', 'x-openrouter-limit')
            reset = _get_header('x-ratelimit-reset', 'x-ratelimit-reset-seconds')

            if remaining is not None or limit is not None:
                try:
                    rem = int(remaining) if remaining is not None else None
                except Exception:
                    rem = None
                try:
                    lim = int(limit) if limit is not None else None
                except Exception:
                    lim = None
                try:
                    reset_secs = int(reset) if reset is not None else None
                except Exception:
                    reset_secs = None

                return {
                    'remaining': rem,
                    'limit': lim,
                    'reset_seconds': reset_secs,
                    'source': 'headers'
                }

            # If no headers, try a dedicated usage/quotas endpoint if available
            # (some OpenRouter deployments expose /usage or /quota)
            for ep in ('/usage', '/quota', '/v1/usage'):
                try:
                    url = self.base_url.rstrip('/') + ep
                    r2 = requests.get(url, headers=self._headers(key), timeout=15)
                    if r2.status_code == 200:
                        body = r2.json()
                        # Try to parse common fields
                        rem = body.get('remaining') or body.get('quota_remaining') or body.get('daily_remaining')
                        lim = body.get('limit') or body.get('quota_limit') or body.get('daily_limit')
                        return {
                            'remaining': rem,
                            'limit': lim,
                            'raw': body,
                            'source': url
                        }
                except Exception:
                    continue

            # Nothing discovered
            return None

        except requests.exceptions.HTTPError as e:
            # Propagate HTTP errors to caller for explicit handling
            raise
        except Exception as e:
            logger.debug("Could not determine OpenRouter quota: %s", e)
            return None

    def check_health(self) -> bool:
        if not self.api_token:
            return False
        try:
            # Models endpoint is cheap and validates auth.
            _ = self.list_models(self.api_token)
            return True
        except Exception as e:
            logger.error("OpenRouter health check failed: %s", e)
            return False
