"""Small, bounded client for the arena's OpenAI-compatible LLM gateway."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class Gateway:
    PRICE_IN = 0.10 / 1_000_000
    PRICE_CACHED = 0.01 / 1_000_000
    PRICE_OUT = 0.50 / 1_000_000
    BUDGET_USD = 0.75
    RESERVE_USD = 0.02

    def __init__(self) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", "http://gateway:4000/v1").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "haiku")
        self._lock = threading.RLock()
        self._generation = 0
        self._local_spend = 0.0
        self._remote_spend: float | None = None
        self._local_at_remote = 0.0
        self._reserved = 0.0
        self._disabled = False
        self._refreshing = False
        self._last_refresh = 0.0

    def update(self, msg: dict[str, Any]) -> None:
        """Accept credentials supplied by prep/start/turn without losing the last good ones."""
        cfg = msg.get("llm") if isinstance(msg, dict) else None
        if not isinstance(cfg, dict):
            return
        base = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else None
        key = cfg.get("api_key") if isinstance(cfg.get("api_key"), str) else None
        model = cfg.get("model") if isinstance(cfg.get("model"), str) else None
        with self._lock:
            if key and key != self.api_key:
                self.api_key = key
                self._generation += 1
                self._local_spend = 0.0
                self._remote_spend = None
                self._local_at_remote = 0.0
                self._reserved = 0.0
                self._disabled = False
                self._last_refresh = 0.0
            if base:
                self.base_url = base.rstrip("/")
            if model:
                self.model = model
        self._refresh_spend_async()

    def _spent_locked(self) -> float:
        if self._remote_spend is None:
            return self._local_spend
        return self._remote_spend + max(0.0, self._local_spend - self._local_at_remote)

    @property
    def budget_left(self) -> float:
        with self._lock:
            return max(0.0, self.BUDGET_USD - self._spent_locked() - self._reserved)

    @property
    def available(self) -> bool:
        with self._lock:
            try:
                scheme = urllib.parse.urlsplit(self.base_url).scheme
            except ValueError:
                return False
            return bool(
                self.api_key
                and scheme in ("http", "https")
                and not self._disabled
                and self.BUDGET_USD - self._spent_locked() - self._reserved > self.RESERVE_USD
            )

    def _refresh_spend_async(self) -> None:
        with self._lock:
            now = time.monotonic()
            if not self.api_key or self._refreshing or now - self._last_refresh < 20.0:
                return
            self._refreshing = True
            self._last_refresh = now
            generation = self._generation
            base = self.base_url
            key = self.api_key
            local_at_start = self._local_spend

        def refresh() -> None:
            try:
                root = re.sub(r"/v1/?$", "", base)
                req = urllib.request.Request(
                    f"{root}/key/info", headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=2.0) as response:
                    data = json.loads(response.read(1_000_001))
                if not isinstance(data, dict):
                    return
                info = data.get("info") if isinstance(data.get("info"), dict) else data
                spend = info.get("spend")
                if isinstance(spend, (int, float)) and spend >= 0:
                    with self._lock:
                        if generation == self._generation:
                            self._remote_spend = float(spend)
                            self._local_at_remote = local_at_start
            except (OSError, ValueError, TypeError, urllib.error.URLError):
                pass
            finally:
                with self._lock:
                    self._refreshing = False

        threading.Thread(target=refresh, daemon=True, name="gateway-spend").start()

    def refresh(self) -> None:
        """Sync shared-key spend in the background, at most once every 20 seconds."""
        self._refresh_spend_async()

    @staticmethod
    def _text_from_response(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    value = part.get("text")
                    if isinstance(value, str):
                        parts.append(value)
            return "".join(parts) or None
        return None

    @classmethod
    def _cost(cls, data: Any) -> float:
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return 0.0
        try:
            prompt = max(0, int(usage.get("prompt_tokens") or 0))
            completion = max(0, int(usage.get("completion_tokens") or 0))
            details = usage.get("prompt_tokens_details")
            cached = max(0, int(details.get("cached_tokens") or 0)) if isinstance(details, dict) else 0
            cached = min(cached, prompt)
        except (TypeError, ValueError):
            return 0.0
        return (prompt - cached) * cls.PRICE_IN + cached * cls.PRICE_CACHED + completion * cls.PRICE_OUT

    @staticmethod
    def _is_budget_error(status: int, body: bytes) -> bool:
        if status in (401, 402, 403):
            return True
        msg = body[:1200].decode("utf-8", errors="ignore").lower()
        return "budget" in msg and any(word in msg for word in ("exceed", "limit", "exhaust"))

    def complete(
        self,
        system: str,
        user: str,
        *,
        timeout_s: float,
        max_tokens: int = 500,
        thinking: int = 0,
        tag: str = "",
    ) -> str | None:
        """Return model text or None, never waiting past timeout_s for the network."""
        del tag  # Caller labels are deliberately excluded from protocol and logs.
        try:
            timeout_s = float(timeout_s)
            max_tokens = int(max_tokens)
            thinking = int(thinking)
        except (TypeError, ValueError, OverflowError):
            return None
        if timeout_s <= 0 or max_tokens <= 0 or not isinstance(system, str) or not isinstance(user, str):
            return None
        deadline = time.monotonic() + timeout_s
        if thinking > 0:
            thinking = max(1024, thinking)
            max_tokens = max(max_tokens, thinking + 1500)
            thinking_body: dict[str, Any] = {"type": "enabled", "budget_tokens": thinking}
        else:
            thinking_body = {"type": "disabled"}

        # Reserve a conservative amount while a request is in flight. The gateway's
        # reported usage replaces this estimate when the worker finishes.
        estimate = ((len(system) + len(user)) / 2) * self.PRICE_IN + max_tokens * self.PRICE_OUT
        with self._lock:
            try:
                scheme = urllib.parse.urlsplit(self.base_url).scheme
            except ValueError:
                return None
            if (
                not self.api_key
                or scheme not in ("http", "https")
                or self._disabled
                or self.BUDGET_USD - self._spent_locked() - self._reserved - estimate <= self.RESERVE_USD
            ):
                return None
            self._reserved += estimate
            generation = self._generation
            url = self.base_url + "/chat/completions"
            key = self.api_key
            model = self.model

        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "thinking": thinking_body,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        done = threading.Event()
        box: dict[str, Any] = {}

        def request() -> None:
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=max(0.1, deadline - time.monotonic())) as response:
                    raw = response.read(5_000_001)
                if len(raw) > 5_000_000:
                    return
                data = json.loads(raw)
                box["text"] = self._text_from_response(data)
                with self._lock:
                    if generation == self._generation:
                        self._local_spend += self._cost(data)
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read(1200)
                except OSError:
                    error_body = b""
                if self._is_budget_error(exc.code, error_body):
                    with self._lock:
                        if generation == self._generation:
                            self._disabled = True
            except (OSError, ValueError, TypeError, urllib.error.URLError):
                pass
            finally:
                with self._lock:
                    if generation == self._generation:
                        self._reserved = max(0.0, self._reserved - estimate)
                done.set()

        if time.monotonic() >= deadline:
            with self._lock:
                if generation == self._generation:
                    self._reserved = max(0.0, self._reserved - estimate)
            return None
        try:
            threading.Thread(target=request, daemon=True, name="gateway-complete").start()
        except RuntimeError:
            with self._lock:
                if generation == self._generation:
                    self._reserved = max(0.0, self._reserved - estimate)
            return None
        if not done.wait(max(0.0, deadline - time.monotonic())):
            return None
        return box.get("text")

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: float,
        max_tokens: int = 500,
        thinking: int = 0,
        tag: str = "",
    ) -> dict[str, Any] | None:
        raw = self.complete(system, user, timeout_s=timeout_s, max_tokens=max_tokens, thinking=thinking, tag=tag)
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
        return None
