"""Guard chat against game-specific speech restrictions.

The rules are data, so discovery only accepts an explicitly named ban.  In
particular, ordinary instructions about illegal moves or private information
must not become a list of forbidden chat words.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


_CONFUSABLES = str.maketrans(
    {
        # Frequent Greek and Cyrillic lookalikes used to disguise Latin words.
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "і": "i", "ј": "j", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
        "Α": "a", "Β": "b", "Ε": "e", "Ζ": "z", "Η": "h", "Ι": "i", "Κ": "k",
        "Μ": "m", "Ν": "n", "Ο": "o", "Ρ": "p", "Τ": "t", "Υ": "y", "Χ": "x",
        "α": "a", "β": "b", "γ": "y", "δ": "d", "ε": "e", "ι": "i", "κ": "k",
        "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    }
)
# Cyrillic dze resembles Latin s and is used to disguise short banned words.
_CONFUSABLES.update(str.maketrans({"\u0455": "s"}))
_LEET_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "+": "t", "|": "i"})
_LEET_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "+": "t", "|": "l"})

_READY_LINES = (
    "Ready to play.",
    "Good luck.",
    "Let's begin.",
    "All set.",
    "I'm ready.",
)
_TURN_LINES = (
    "Let's see how this unfolds.",
    "The game continues.",
    "I have made my choice.",
    "Onward.",
    "That is an interesting turn.",
    "We'll see where this leads.",
    "A thoughtful round so far.",
)

_LABEL = r"(?:forbidden|banned|prohibited|taboo|restricted|off[ -]limits|disallowed)"
_KIND = r"(?:word|term|subject|topic|phrase|name|concept)s?"
_BAN_LINE = re.compile(
    rf"\b(?:the\s+)?{_LABEL}\s+{_KIND}\s*(?::|=|\bis\b|\bare\b|\binclude(?:s)?\b)\s*(?P<items>[^\n]{{1,120}})",
    re.IGNORECASE,
)
_REVERSED_BAN_LINE = re.compile(
    rf"\b(?:the\s+)?{_KIND}\s+(?:you\s+)?(?:must\s+not|may\s+not|cannot|can't|should\s+not)\s+(?:say|mention|write|use|discuss)\s*(?::|=|\bis\b)?\s*(?P<items>[^\n]{{1,120}})",
    re.IGNORECASE,
)
_NO_MENTION_LINE = re.compile(
    rf"\b(?:never|do\s+not|don't|must\s+not|may\s+not|cannot|can't)\s+(?:say|mention|write|use|discuss|talk\s+about)\s+(?:the\s+)?{_KIND}\s*(?::|=|\bis\b|\bof\b)?\s*(?P<items>[^\n]{{1,120}})",
    re.IGNORECASE,
)
_DIRECT_QUOTED = re.compile(
    r"\b(?:never|do\s+not|don't|must\s+not|may\s+not|cannot|can't)\s+"
    r"(?:say|mention|write|discuss|talk\s+about)\s+"
    r"(?:the\s+)?(?:word\s+|term\s+|topic\s+|subject\s+|phrase\s+)?"
    r"[\"'`“‘]([^\"'`”’\n]{1,60})[\"'`”’]",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"'`“‘]([^\"'`”’\n]{1,60})[\"'`”’]")
_GENERIC = {
    "none", "nothing", "not specified", "no", "n/a", "it", "this", "that",
    "them", "anything", "everything", "the subject", "the topic", "the word",
    "forbidden subject", "forbidden topic", "forbidden word", "opponent",
    "your opponent", "private information", "secret information",
}


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold().translate(_CONFUSABLES)
    return "".join(ch for ch in value if not unicodedata.category(ch).startswith("M") and unicodedata.category(ch) != "Cf")


def _compact(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def _items(fragment: str) -> list[str]:
    # Rule prose after a label is not part of the forbidden expression.
    fragment = re.split(r"[.;]|\s+(?:even\s+if|including\s+in|during\s+the|in\s+(?:your|any|the)\s+(?:chat|answer|response|message))\b", fragment, maxsplit=1, flags=re.IGNORECASE)[0]
    quoted = [match.group(1).strip() for match in _QUOTED.finditer(fragment)]
    if quoted:
        return quoted
    fragment = re.sub(r"^(?:the\s+|any\s+)", "", fragment.strip(), flags=re.IGNORECASE)
    return [piece.strip(" \t\r\n,:'\"`“”‘’()[]") for piece in re.split(r"\s*,\s*|\s+or\s+|\s+and\s+", fragment)]


class SpeechGuard:
    """Sanitize each outgoing chat line before emitting a protocol response."""

    def __init__(self, terms: Iterable[str] = (), strict: bool = False) -> None:
        self.strict = bool(strict)
        self.terms: list[str] = []
        self._keys: list[str] = []
        seen: set[str] = set()
        for value in terms:
            if not isinstance(value, str):
                continue
            term = value.strip().strip("\"'`“”‘’.,:;!?()[]")
            key = _compact(_fold(term))
            if key and key not in seen:
                seen.add(key)
                self.terms.append(term)
                self._keys.append(key)

    @staticmethod
    def discover(rules: str) -> list[str]:
        """Extract only explicit forbidden expressions from a rules document."""
        if not isinstance(rules, str) or not rules:
            return []
        found: list[str] = []
        for pattern in (_BAN_LINE, _REVERSED_BAN_LINE, _NO_MENTION_LINE):
            for match in pattern.finditer(rules):
                found.extend(_items(match.group("items")))
        found.extend(match.group(1).strip() for match in _DIRECT_QUOTED.finditer(rules))
        result: list[str] = []
        seen: set[str] = set()
        for item in found:
            item = re.sub(r"\s+", " ", item).strip()
            folded = _fold(item)
            key = _compact(folded)
            if (
                not key or len(item) > 60 or len(item.split()) > 6
                or folded in _GENERIC or folded.startswith(("no ", "not "))
                or key in seen
            ):
                continue
            seen.add(key)
            result.append(item)
        return result

    def hits(self, text: str) -> bool:
        if not isinstance(text, str) or not text or not self._keys:
            return False
        folded = _fold(text)
        variants = (folded, folded.translate(_LEET_I), folded.translate(_LEET_L))
        for variant in variants:
            compact = _compact(variant)
            for key in self._keys:
                if len(key) >= 4 and key in compact:
                    return True
                if len(key) < 4:
                    # Match short words even when their letters are separated,
                    # while avoiding e.g. a banned "ai" within "said".
                    pattern = r"(?<!\w)" + r"[^\w]*".join(map(re.escape, key)) + r"(?!\w)"
                    if re.search(pattern, variant, flags=re.UNICODE):
                        return True
        return False

    def neutral(self, *, ready: bool = False, turn: int = 0) -> str:
        pool = _READY_LINES if ready else _TURN_LINES
        safe = [candidate for candidate in pool if not self.hits(candidate)]
        return safe[(turn if isinstance(turn, int) else 0) % len(safe)] if safe else "..."

    def clean(self, text: Any, *, ready: bool = False, turn: int = 0) -> str:
        if self.strict:
            pool = _READY_LINES if ready else _TURN_LINES
            if isinstance(text, str) and text in pool and not self.hits(text):
                return text
            return self.neutral(ready=ready, turn=turn)
        if not isinstance(text, str):
            return self.neutral(ready=ready, turn=turn)
        value = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text)
        value = re.sub(r"\s+", " ", value).strip()[:200]
        if value and not self.hits(value):
            return value
        return self.neutral(ready=ready, turn=turn)
