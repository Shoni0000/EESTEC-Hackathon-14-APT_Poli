"""Content-addressed prep artifacts and append-only opponent observations."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


ARTIFACT_VERSION = 3


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _safe(value: str) -> str:
    label = "".join(c if c.isascii() and (c.isalnum() or c in "-_.") else "_" for c in value)
    return (label.strip("._") or "unknown")[:64]


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class MemoryStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("MEMORY_DIR", "/memory"))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.root = Path("./memory")
            self.root.mkdir(parents=True, exist_ok=True)

    def _game_dir(self, game_id: str) -> Path:
        return self.root / "games" / f"{_safe(game_id)}-{_digest(game_id)[:8]}"

    def save_artifact(
        self,
        game_id: str,
        variants: list[dict[str, Any]],
        metadata: dict[str, Any],
        playbook: str,
        solver_code: str,
    ) -> dict[str, Any]:
        """Each content revision has its own path; no shared mutable index is needed."""
        rules = [{"label": str(v.get("label", "default")), "rules": str(v.get("rules", ""))} for v in variants]
        key = _digest({"version": ARTIFACT_VERSION, "game_id": game_id, "variants": rules})
        artifact = {
            "version": ARTIFACT_VERSION,
            "game_id": game_id,
            "key": key,
            "created_at": time.time(),
            "variants": rules,
            "metadata": metadata,
            "playbook": playbook,
            "solver_code": solver_code,
        }
        _atomic_json(self._game_dir(game_id) / f"{key}.json", artifact)
        return artifact

    def load_artifact(self, game_id: str, label: str = "", rules: str = "") -> dict[str, Any] | None:
        base = self._game_dir(game_id)
        if not base.exists():
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for path in base.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("version") != ARTIFACT_VERSION or item.get("game_id") != game_id:
                    continue
                variants = item.get("variants") or []
                if label and not any(v.get("label") == label for v in variants):
                    continue
                score = float(item.get("created_at") or 0) * 1e-10
                if rules:
                    for variant in variants:
                        if label and variant.get("label") != label:
                            continue
                        prepared = str(variant.get("rules") or "")
                        if prepared == rules:
                            score += 1000
                        elif prepared and (prepared in rules or rules in prepared):
                            score += 500
                        else:
                            common = os.path.commonprefix((prepared, rules))
                            score += 100 * len(common) / max(len(prepared), len(rules), 1)
                if best is None or score > best[0]:
                    best = score, item
            except (OSError, ValueError, TypeError):
                continue
        return best[1] if best else None

    def _opponent_dir(self, opponent: str, game_id: str) -> Path:
        return self.root / "opponents" / _digest(opponent) / _digest(game_id)

    def load_profile(self, opponent: str, game_id: str) -> dict[str, Any]:
        profile: dict[str, Any] = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "recent_chat": []}
        if not opponent:
            return profile
        directory = self._opponent_dir(opponent, game_id)
        if not directory.exists():
            return profile
        paths = sorted(directory.glob("*.json"), reverse=True)[:100]
        recent: list[str] = []
        for path in paths:
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            profile["games"] += 1
            result = event.get("result")
            key = {"win": "wins", "loss": "losses", "draw": "draws"}.get(result)
            if key:
                profile[key] += 1
            if len(recent) < 4:
                recent.extend(str(t)[:160] for t in event.get("recent_chat", [])[-2:] if isinstance(t, str))
        profile["recent_chat"] = recent[:4]
        return profile

    def save_result(self, opponent: str, game_id: str, result: str, recent_chat: list[str]) -> None:
        if not opponent or result not in {"win", "loss", "draw"}:
            return
        directory = self._opponent_dir(opponent, game_id)
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
        _atomic_json(directory / name, {"result": result, "recent_chat": recent_chat[-3:]})
