"""Exercise the gateway client against a local OpenAI-compatible HTTP stub."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from harness.gateway import Gateway  # noqa: E402


def reply(content: object, *, prompt: int = 0, cached: int = 0, completion: int = 0) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


class GatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.requests: queue.Queue[dict] = queue.Queue()
        self.responses: queue.Queue[tuple[dict, float]] = queue.Queue()
        requests = self.requests
        responses = self.responses

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _send(self, data: dict) -> None:
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def do_GET(self) -> None:
                requests.put({"method": "GET", "path": self.path, "authorization": self.headers.get("Authorization")})
                self._send({"info": {"spend": 0.0}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                requests.put({"method": "POST", "path": self.path,
                              "authorization": self.headers.get("Authorization"), "body": body})
                try:
                    data, delay = responses.get_nowait()
                except queue.Empty:
                    data, delay = reply("{}"), 0.0
                if delay:
                    time.sleep(delay)
                self._send(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def gateway(self, key: str = "key-one", model: str = "haiku-test") -> Gateway:
        client = Gateway()
        client.update({"llm": {"base_url": self.base_url, "api_key": key, "model": model}})
        return client

    def next_post(self) -> dict:
        while True:
            item = self.requests.get(timeout=2)
            if item["method"] == "POST":
                return item

    def test_json_request_auth_disabled_thinking_and_cost(self) -> None:
        self.responses.put((reply('```json\n{"action":"left"}\n```', prompt=100, cached=25, completion=40), 0.0))
        client = self.gateway()
        result = client.complete_json("system rules", "live position", timeout_s=1.0, max_tokens=300)
        request = self.next_post()

        self.assertEqual(result, {"action": "left"})
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["authorization"], "Bearer key-one")
        self.assertEqual(request["body"]["model"], "haiku-test")
        self.assertEqual(request["body"]["messages"], [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "live position"},
        ])
        self.assertEqual(request["body"]["thinking"], {"type": "disabled"})
        self.assertEqual(request["body"]["max_tokens"], 300)
        expected_cost = 75 * Gateway.PRICE_IN + 25 * Gateway.PRICE_CACHED + 40 * Gateway.PRICE_OUT
        self.assertAlmostEqual(client.budget_left, 0.75 - expected_cost, places=8)

    def test_latest_key_rotation_and_thinking_headroom(self) -> None:
        self.responses.put((reply("first", prompt=1_000_000), 0.0))
        client = self.gateway()
        self.assertEqual(client.complete("sys", "one", timeout_s=1.0), "first")
        first = self.next_post()
        self.assertEqual(first["authorization"], "Bearer key-one")
        self.assertAlmostEqual(client.budget_left, 0.65, places=5)

        client.update({"llm": {"base_url": self.base_url, "api_key": "key-two", "model": "haiku-new"}})
        self.assertAlmostEqual(client.budget_left, 0.75, places=5)
        self.responses.put((reply("second"), 0.0))
        self.assertEqual(client.complete("sys", "two", timeout_s=1.0, max_tokens=100, thinking=200), "second")
        second = self.next_post()
        self.assertEqual(second["authorization"], "Bearer key-two")
        self.assertEqual(second["body"]["model"], "haiku-new")
        self.assertEqual(second["body"]["thinking"], {"type": "enabled", "budget_tokens": 1024})
        self.assertGreaterEqual(second["body"]["max_tokens"], 1024 + 1500)

    def test_non_json_content_returns_none(self) -> None:
        self.responses.put((reply("There is no JSON answer."), 0.0))
        client = self.gateway()
        self.assertIsNone(client.complete_json("sys", "user", timeout_s=1.0))
        self.next_post()

    def test_hard_timeout_returns_before_slow_gateway(self) -> None:
        self.responses.put((reply("too late"), 0.35))
        client = self.gateway()
        started = time.monotonic()
        result = client.complete("sys", "user", timeout_s=0.08)
        elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 0.25)
        self.next_post()


if __name__ == "__main__":
    unittest.main()
