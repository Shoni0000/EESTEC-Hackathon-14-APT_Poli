"""Small, independently checked tactics for released tournament games.

These helpers use only the live observation and exact legal actions. A parser
failure returns None so the generic specialist can take over.
"""

from __future__ import annotations

import re
import secrets
import time
from functools import lru_cache
from heapq import heappop, heappush
from itertools import permutations, product
from pathlib import Path
from typing import Any


def validate_known(action: str, state: dict[str, Any]) -> bool | None:
    """Validate null-list action formats that the released rules define exactly."""
    observation = str(state.get("observation") or "")
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    game = str(state.get("game_id") or "")
    if game == "wordle_duel" or observation.startswith("WORDLE DUEL"):
        length = variant.get("length")
        if not isinstance(length, int):
            match = re.search(r"WORDLE DUEL\s*\((\d+) letters", observation)
            length = int(match.group(1)) if match else None
        if isinstance(length, int) and 2 <= length <= 20:
            return bool(re.fullmatch(rf"(?i:guess) [A-Za-z]{{{length}}}", action))
    if game == "colonel_blotto" or observation.startswith("COLONEL BLOTTO"):
        setup = re.search(r"must place exactly (\d+) soldiers on (\d+) battlefields", observation, re.I)
        if setup:
            soldiers, fields = int(setup.group(1)), int(setup.group(2))
            # The arena accepts either spaces or commas between allocations.
            if not re.fullmatch(r"[0-9]+(?:(?:[ \t]+|,[ \t]*)[0-9]+)*", action.strip()):
                return False
            numbers = [int(piece) for piece in re.findall(r"[0-9]+", action)]
            return len(numbers) == fields and sum(numbers) == soldiers
    if game == "guess_number" or observation.startswith("GUESS THE NUMBER"):
        match = re.fullmatch(r"guess ([0-9]+)", action)
        if not match:
            return False
        low, high = variant.get("lo"), variant.get("hi")
        if not isinstance(low, int) or not isinstance(high, int):
            bounds = re.search(r"GUESS THE NUMBER\s*\((\d+)-(\d+)\)", observation)
            if not bounds:
                return None
            low, high = int(bounds.group(1)), int(bounds.group(2))
        return low <= int(match.group(1)) <= high
    if game == "nim" or observation.startswith("NIM"):
        variant_piles = _nim_piles(observation)
        if variant_piles is None:
            return None
        if int(variant.get("max_piles") or 1) != 1:
            return None
        move = re.fullmatch(r"take ([0-9]+) from (?:pile )?([0-9]+)", action, re.I)
        if not move:
            return False
        amount, pile = int(move.group(1)), int(move.group(2)) - 1
        cap = int(variant.get("max_take") or 0)
        return 0 <= pile < len(variant_piles) and 1 <= amount <= variant_piles[pile] and (cap == 0 or amount <= cap)
    if game == "bulls_and_cows" or observation.startswith("BULLS AND COWS"):
        parameters = _bulls_parameters(state)
        if parameters is None:
            return None
        alphabet, length, repeats = parameters
        match = re.fullmatch(r"(?:(?i:guess)[ \t]+)?([A-Za-z0-9]+)", action.strip())
        if not match:
            return False
        code = match.group(1).upper()
        return len(code) == length and all(char in alphabet for char in code) and (repeats or len(set(code)) == length)
    if game == "battleship" or observation.startswith("BATTLESHIP"):
        variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
        size = variant.get("size")
        board = _shots_board(observation, size) if isinstance(size, int) else None
        if board is None:
            return None
        match = re.fullmatch(r"(?i:fire)[ \t]+([A-Za-z])[ \t]*([0-9]+)", action.strip())
        if not match:
            return False
        row, col = int(match.group(2)) - 1, ord(match.group(1).lower()) - 97
        return 0 <= row < size and 0 <= col < size and board[row][col] == "."
    return None


def _nim_piles(observation: str) -> tuple[int, ...] | None:
    lines = re.findall(r"Piles now:([^\n]+)", observation, re.I)
    if not lines:
        return None
    pairs = [(int(index), int(size)) for index, size in
             re.findall(r"pile\s+(\d+)\s*=\s*(\d+)", lines[-1], re.I)]
    if not pairs or [index for index, _ in pairs] != list(range(1, len(pairs) + 1)):
        return None
    return tuple(size for _, size in pairs)


def _nim(state: dict[str, Any]) -> dict[str, Any] | None:
    observation = str(state.get("observation") or "")
    piles = _nim_piles(observation)
    if piles is None:
        return None
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    if int(variant.get("max_piles") or 1) != 1:
        return None
    cap = int(variant.get("max_take") or 0)
    misere = bool(variant.get("misere")) or "MISERE PLAY" in observation.upper()
    legal = state.get("legal_actions")
    rejected = set(state.get("rejected_actions") or [])
    options: list[tuple[str, int, int]] = []
    if isinstance(legal, list) and legal:
        for action in legal:
            if not isinstance(action, str) or action in rejected:
                continue
            match = re.fullmatch(r"take ([0-9]+) from (?:pile )?([0-9]+)", action, re.I)
            if match:
                options.append((action, int(match.group(1)), int(match.group(2)) - 1))
    else:
        for pile, size in enumerate(piles):
            for amount in range(1, min(size, cap or size) + 1):
                action = f"take {amount} from pile {pile + 1}"
                if action not in rejected:
                    options.append((action, amount, pile))
    options = [(action, amount, pile) for action, amount, pile in options
               if 0 <= pile < len(piles) and 1 <= amount <= piles[pile] and (cap == 0 or amount <= cap)]
    if not options:
        return None

    @lru_cache(maxsize=10000)
    def misere_winning(position: tuple[int, ...]) -> bool:
        if not any(position):
            return True  # The previous player took the last token and lost.
        for pile, size in enumerate(position):
            for amount in range(1, min(size, cap or size) + 1):
                following = list(position)
                following[pile] -= amount
                if not misere_winning(tuple(following)):
                    return True
        return False

    ranked: list[tuple[bool, int, str]] = []
    for action, amount, pile in options:
        following = list(piles)
        following[pile] -= amount
        if misere:
            wins = not misere_winning(tuple(following))
        else:
            nim_sum = 0
            for size in following:
                nim_sum ^= size % (cap + 1) if cap else size
            wins = nim_sum == 0
        ranked.append((wins, amount, action))
    winning = [item for item in ranked if item[0]]
    chosen = max(winning or ranked, key=lambda item: (item[1], item[2]))
    return {"action": chosen[2], "confidence": 1.0 if chosen[0] else 0.65,
            "proof": bool(chosen[0]), "tactic": "forced_win" if chosen[0] else "hard_position",
            "reason": "Exact Nim winning-state calculation."}


def _bulls_parameters(state: dict[str, Any]) -> tuple[str, int, bool] | None:
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    alphabet = str(variant.get("alphabet") or "").upper()
    length = variant.get("length")
    repeats = variant.get("repeats")
    if not isinstance(length, int) or not 2 <= length <= 6 or not 2 <= len(alphabet) <= 10:
        return None
    if len(set(alphabet)) != len(alphabet) or not all(char.isascii() and char.isalnum() for char in alphabet):
        return None
    if not isinstance(repeats, bool) or (not repeats and length > len(alphabet)):
        return None
    return alphabet, length, repeats


def _bulls_feedback(guess: str, secret: str) -> tuple[int, int]:
    bulls = sum(a == b for a, b in zip(guess, secret))
    cows = sum(min(guess.count(char), secret.count(char)) for char in set(guess)) - bulls
    return bulls, cows


def _bulls_history(observation: str, length: int, alphabet: str) -> list[tuple[str, int, int]]:
    # The opponent's guesses are evidence about our code only. Stop before
    # that section so their feedback cannot corrupt the candidate set.
    own = re.split(r"opponent'?s? (?:previous )?guesses|their guesses|rival'?s? guesses",
                   observation, maxsplit=1, flags=re.I)[0]
    own = re.split(r"your (?:previous )?guesses", own, maxsplit=1, flags=re.I)[-1]
    results: list[tuple[str, int, int]] = []
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])([A-Za-z0-9]{{{length}}})(?![A-Za-z0-9])"
        r"[^\n]{0,35}?(\d+)\s*bulls?\b[^\n]{0,20}?(\d+)\s*cows?\b", re.I,
    )
    for match in pattern.finditer(own):
        code = match.group(1).upper()
        bulls, cows = int(match.group(2)), int(match.group(3))
        if all(char in alphabet for char in code) and bulls + cows <= length:
            results.append((code, bulls, cows))
    return results[-12:]


def _bulls(state: dict[str, Any]) -> dict[str, Any] | None:
    parameters = _bulls_parameters(state)
    if parameters is None:
        return None
    alphabet, length, repeats = parameters
    observation = str(state.get("observation") or "")
    history = _bulls_history(observation, length, alphabet)
    rejected = set(state.get("rejected_actions") or [])
    if not history:
        opening = ("AABC" if alphabet == "ABCDEF" and length == 4 else
                   alphabet[:length] if length <= len(alphabet) else
                   (alphabet * ((length + len(alphabet) - 1) // len(alphabet)))[:length])
        action = f"guess {opening}"
        if action not in rejected:
            return {"action": action, "confidence": 0.8, "proof": False,
                    "tactic": "information", "reason": "Broad first code probe."}
    started = time.monotonic()
    codes = permutations(alphabet, length) if not repeats else product(alphabet, repeat=length)
    candidates: list[str] = []
    for index, symbols in enumerate(codes):
        if index % 4096 == 0 and time.monotonic() - started > 0.85:
            return None
        code = "".join(symbols)
        if f"guess {code}" in rejected or any(_bulls_feedback(guess, code) != (bulls, cows)
                                                   for guess, bulls, cows in history):
            continue
        candidates.append(code)
    if not candidates:
        return None
    if len(candidates) == 1:
        return {"action": f"guess {candidates[0]}", "confidence": 1.0,
                "proof": True, "tactic": "deduced_code", "reason": "Only code matching our feedback."}
    if len(candidates) > 15000:
        chosen = candidates[len(candidates) // 2]
    else:
        sample = candidates[::max(1, len(candidates) // 250)][:250]
        probes = candidates[::max(1, len(candidates) // 90)][:90]
        if time.monotonic() - started > 0.6:
            chosen = candidates[0]
        else:
            def split_score(probe: str) -> int:
                buckets: dict[tuple[int, int], int] = {}
                for secret in sample:
                    key = _bulls_feedback(probe, secret)
                    buckets[key] = buckets.get(key, 0) + 1
                return sum(count * count for count in buckets.values())
            chosen = min(probes, key=lambda code: (split_score(code), code))
    return {"action": f"guess {chosen}", "confidence": 0.85,
            "proof": False, "tactic": "candidate_filter", "reason": f"{len(candidates)} codes fit our feedback."}


def _rps(state: dict[str, Any]) -> dict[str, Any] | None:
    legal = state.get("legal_actions")
    if not isinstance(legal, list) or not legal:
        return None
    names = {str(action).casefold(): action for action in legal if isinstance(action, str)}
    signs = ("rock", "paper", "scissors")
    available = [sign for sign in signs if sign in names and names[sign] not in set(state.get("rejected_actions") or [])]
    if not available:
        return None
    observation = str(state.get("observation") or "")
    history: list[tuple[str, str]] = []
    for line in observation.splitlines():
        own = re.search(r"\byou\s*(?:threw|played|chose|:)?\s*(rock|paper|scissors)\b", line, re.I)
        rival = re.search(r"\bopponent\s*(?:threw|played|chose|:)?\s*(rock|paper|scissors)\b", line, re.I)
        if own and rival:
            history.append((own.group(1).casefold(), rival.group(1).casefold()))
    history = history[-16:]
    weights = {sign: 1.0 for sign in signs}
    for age, (_, rival) in enumerate(reversed(history)):
        weights[rival] += 0.85 ** age
    if len(history) >= 2:
        previous_own, previous_rival = history[-1]
        for earlier, later in zip(history[:-1], history[1:]):
            if earlier[1] == previous_rival:
                weights[later[1]] += 1.5
            if earlier[0] == previous_own:
                weights[later[1]] += 1.0
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    doubled = variant.get("double")
    double = ("rock wins count double" in str(state.get("rules") or "").casefold()
              or isinstance(doubled, list) and "rock" in doubled)
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    scores = {}
    for sign in available:
        win_value = 2 if double and sign == "rock" else 1
        scores[sign] = sum(weight * (win_value if beats[sign] == rival else
                                     -(2 if double and rival == "rock" else 1)
                                     if beats[rival] == sign else 0)
                           for rival, weight in weights.items())
    strongest = max(available, key=lambda sign: scores[sign])
    # Keep a mixed component so our own deterministic history is harder to
    # exploit. Strong observed bias gets a higher exploitation rate.
    spread = max(weights.values()) / max(0.01, sum(weights.values()))
    explore = 18 if spread > 0.6 else 35
    chosen = available[secrets.randbelow(len(available))] if secrets.randbelow(100) < explore else strongest
    return {"action": names[chosen], "confidence": 0.72 if history else 0.55,
            "proof": False, "tactic": "adaptive_mixed", "reason": "Mixed response to revealed throws."}


def _dots_exact(candidates: dict[str, frozenset[str]], undrawn: set[frozenset[str]],
                sides: list[set[frozenset[str]]]) -> tuple[str, int] | None:
    """Solve small Dots endgames, accounting for the extra turn on a capture."""
    if not 2 <= len(undrawn) <= 18:
        return None
    edges = tuple(sorted(undrawn, key=lambda item: tuple(sorted(item))))
    indices = {edge: index for index, edge in enumerate(edges)}
    boxes = tuple(sum(1 << indices[edge] for edge in box if edge in indices) for box in sides)
    touching = tuple(tuple(mask for mask in boxes if mask & (1 << index))
                     for index in range(len(edges)))
    deadline = time.monotonic() + 1.25
    nodes = 0

    @lru_cache(maxsize=300000)
    def value(mask: int) -> int:
        nonlocal nodes
        nodes += 1
        if nodes % 1024 == 0 and time.monotonic() > deadline:
            raise TimeoutError
        if mask == 0:
            return 0
        best = -1000
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining -= bit
            index = bit.bit_length() - 1
            claimed = sum(mask & box == bit for box in touching[index])
            following = mask ^ bit
            result = claimed + value(following) if claimed else -value(following)
            if result > best:
                best = result
        return best

    initial = (1 << len(edges)) - 1
    best_action = ""
    best_value = -1000
    try:
        for action, edge in candidates.items():
            bit = 1 << indices[edge]
            claimed = sum(initial & box == bit for box in touching[indices[edge]])
            following = initial ^ bit
            result = claimed + value(following) if claimed else -value(following)
            if result > best_value:
                best_action, best_value = action, result
    except TimeoutError:
        return None
    return (best_action, best_value) if best_action else None


def _dots(state: dict[str, Any]) -> dict[str, Any] | None:
    legal = state.get("legal_actions")
    if not isinstance(legal, list) or not legal:
        return None
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    label = str(variant.get("label") or "")
    dimensions = re.search(r"(\d+)\s*x\s*(\d+)\s*boxes", label, re.I)
    if not dimensions:
        dimensions = re.search(r"(\d+)\s*x\s*(\d+)\s*boxes", str(state.get("rules") or ""), re.I)
    if not dimensions:
        return None
    height, width = int(dimensions.group(1)), int(dimensions.group(2))
    if not 1 <= height <= 8 or not 1 <= width <= 8:
        return None
    def edge(a: str, b: str) -> frozenset[str]:
        return frozenset((a, b))
    def dot(col: int, row: int) -> str:
        return f"{chr(97 + col)}{row + 1}"
    candidates: dict[str, frozenset[str]] = {}
    undrawn: set[frozenset[str]] = set()
    rejected = set(state.get("rejected_actions") or [])
    for action in legal:
        if not isinstance(action, str):
            continue
        match = re.fullmatch(r"line\s+([a-z]\d+)\s*[- ]\s*([a-z]\d+)", action, re.I)
        if not match:
            continue
        normalized = edge(match.group(1).lower(), match.group(2).lower())
        undrawn.add(normalized)
        if action not in rejected:
            candidates[action] = normalized
    if not candidates:
        return None
    sides: list[set[frozenset[str]]] = []
    for row in range(height):
        for col in range(width):
            a, b, c, d = dot(col, row), dot(col + 1, row), dot(col, row + 1), dot(col + 1, row + 1)
            sides.append({edge(a, b), edge(c, d), edge(a, c), edge(b, d)})
    exact = _dots_exact(candidates, undrawn, sides)
    if exact:
        return {"action": exact[0], "confidence": 1.0, "proof": True,
                "tactic": "endgame_search", "reason": f"Exact remaining box differential {exact[1]}."}
    remaining = [(box, len(box & undrawn)) for box in sides]
    ranked = []
    for action, drawn in candidates.items():
        captures = sum(drawn in box and count == 1 for box, count in remaining)
        gifts = sum(drawn in box and count == 2 for box, count in remaining)
        # Capturing a box grants another turn; giving a third side allows the
        # opponent to capture it. Prefer safe lines when no capture exists.
        ranked.append((captures, -gifts, action))
    captures, safety, action = max(ranked)
    bad_reply = ""
    if not captures:
        after = undrawn - {candidates[action]}
        for reply, edge_to_draw in candidates.items():
            if reply == action:
                continue
            gives_us_box = any(edge_to_draw in box and len(box & after) == 2 for box in sides)
            captures_for_them = any(edge_to_draw in box and len(box & after) == 1 for box in sides)
            if gives_us_box and not captures_for_them:
                bad_reply = reply
                break
    return {"action": action, "confidence": 0.94 if captures or safety == 0 else 0.65,
            "proof": bool(captures), "tactic": "capture" if captures else "safe_line",
            "bad_reply": bad_reply,
            "reason": f"Completes {captures} boxes; exposes {-safety}."}


def _hex_board(observation: str, size: int) -> tuple[tuple[str, ...], ...] | None:
    rows: dict[int, tuple[str, ...]] = {}
    for line in observation.splitlines():
        match = re.match(r"^\s*(\d+)\s+([.XOxo\s]+?)\s+(\d+)\s*$", line)
        if not match or int(match.group(1)) != int(match.group(3)):
            continue
        cells = tuple(re.findall(r"[.XO]", match.group(2).upper()))
        if len(cells) == size and 1 <= int(match.group(1)) <= size:
            rows[int(match.group(1))] = cells
    return tuple(rows[row] for row in range(1, size + 1)) if len(rows) == size else None


def _hex_distance(board: tuple[tuple[str, ...], ...], mark: str) -> int:
    size = len(board)
    rival = "O" if mark == "X" else "X"
    distance = [[10000] * size for _ in range(size)]
    queue: list[tuple[int, int, int]] = []
    starts = ((0, col) for col in range(size)) if mark == "X" else ((row, 0) for row in range(size))
    for row, col in starts:
        if board[row][col] != rival:
            cost = int(board[row][col] == ".")
            distance[row][col] = cost
            heappush(queue, (cost, row, col))
    while queue:
        cost, row, col = heappop(queue)
        if cost != distance[row][col]:
            continue
        if (mark == "X" and row == size - 1) or (mark == "O" and col == size - 1):
            return cost
        for dr, dc in ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)):
            nr, nc = row + dr, col + dc
            if not 0 <= nr < size or not 0 <= nc < size or board[nr][nc] == rival:
                continue
            new_cost = cost + int(board[nr][nc] == ".")
            if new_cost < distance[nr][nc]:
                distance[nr][nc] = new_cost
                heappush(queue, (new_cost, nr, nc))
    return 10000


def _hex(state: dict[str, Any]) -> dict[str, Any] | None:
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    # A pie-rule swap may transfer the X/O objective without changing the
    # original seat number. The native search has no reliable role mapping.
    if variant.get("swap"):
        return None
    size = variant.get("size")
    if not isinstance(size, int) or not 3 <= size <= 9:
        return None
    legal = state.get("legal_actions")
    if not isinstance(legal, list) or not legal or "swap" in legal:
        return None
    board = _hex_board(str(state.get("observation") or ""), size)
    if board is None:
        return None
    mark = "X" if int(state.get("seat") or 0) == 0 else "O"
    other = "O" if mark == "X" else "X"
    rejected = set(state.get("rejected_actions") or [])
    candidates = []
    for action in legal:
        match = re.fullmatch(r"place ([a-z])(\d+)", action, re.I) if isinstance(action, str) else None
        if match and action not in rejected:
            row, col = int(match.group(2)) - 1, ord(match.group(1).lower()) - 97
            if 0 <= row < size and 0 <= col < size and board[row][col] == ".":
                candidates.append((action, row, col))
    if not candidates:
        return None
    def placed(row: int, col: int, symbol: str) -> tuple[tuple[str, ...], ...]:
        next_board = list(board)
        line = list(board[row])
        line[col] = symbol
        next_board[row] = tuple(line)
        return tuple(next_board)
    start = time.monotonic()
    opponent_threats = {(row, col) for _, row, col in candidates
                        if _hex_distance(placed(row, col, other), other) == 0}
    ranked = []
    for action, row, col in candidates:
        if time.monotonic() - start > 0.55:
            return None
        after = placed(row, col, mark)
        ours = _hex_distance(after, mark)
        theirs = _hex_distance(after, other)
        center = size - abs(row - (size - 1) / 2) - abs(col - (size - 1) / 2)
        score = (10000 if ours == 0 else 0) + (1000 if (row, col) in opponent_threats else 0)
        score += 2.5 * min(theirs, size + 1) - 4 * ours + 0.07 * center
        ranked.append((score, action, ours, (row, col) in opponent_threats))
    _, action, distance, blocked = max(ranked)
    tactic = "immediate_win" if distance == 0 else "forced_defense" if blocked else "bridge"
    return {"action": action, "confidence": 1.0 if distance == 0 else 0.93 if blocked else 0.76,
            "proof": distance == 0, "tactic": tactic, "reason": "Hex shortest-path connection search."}


def _shots_board(observation: str, size: int) -> tuple[tuple[str, ...], ...] | None:
    if not 3 <= size <= 10:
        return None
    section = re.search(r"YOUR SHOTS[^\n]*\n(.*?)(?:YOUR WATERS|RECENT SHOTS|HISTORY|$)", observation, re.I | re.S)
    if not section:
        return None
    rows: dict[int, tuple[str, ...]] = {}
    for line in section.group(1).splitlines():
        match = re.match(r"^\s*(\d+)\s+([.oOX#\s]+?)\s*(?:\d+)?\s*$", line)
        if not match:
            continue
        cells = tuple(re.findall(r"[.oX#]", match.group(2).replace("O", "o")))
        row = int(match.group(1))
        if len(cells) == size and 1 <= row <= size:
            rows[row] = cells
    return tuple(rows[row] for row in range(1, size + 1)) if len(rows) == size else None


def _battleship(state: dict[str, Any]) -> dict[str, Any] | None:
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    size = variant.get("size")
    if not isinstance(size, int) or bool(variant.get("salvo")):
        return None
    observation = str(state.get("observation") or "")
    board = _shots_board(observation, size)
    if board is None:
        return None
    legal = state.get("legal_actions")
    allowed = set(legal) if isinstance(legal, list) and legal else None
    rejected = set(state.get("rejected_actions") or [])
    choices = {(row, col): f"fire {chr(97 + col)}{row + 1}"
               for row in range(size) for col in range(size) if board[row][col] == "."}
    choices = {cell: action for cell, action in choices.items()
               if action not in rejected and (allowed is None or action in allowed)}
    if not choices:
        return None
    lengths = [int(value) for value in re.findall(r"\((\d+) cells?\b", str(state.get("rules") or ""), re.I)]
    lengths = [length for length in lengths if 2 <= length <= size][:8] or [3, 2]
    active_hits = {(row, col) for row in range(size) for col in range(size) if board[row][col] == "X"}
    scores = {cell: 0.0 for cell in choices}
    target_scores = {cell: 0.0 for cell in choices}
    for length in lengths:
        for dr, dc in ((0, 1), (1, 0)):
            for row in range(size):
                for col in range(size):
                    cells = [(row + step * dr, col + step * dc) for step in range(length)]
                    if any(not (0 <= r < size and 0 <= c < size) or board[r][c] in "o#" for r, c in cells):
                        continue
                    hits = len(active_hits.intersection(cells))
                    for cell in cells:
                        if cell in scores:
                            scores[cell] += 1.0
                            if hits:
                                target_scores[cell] += 10.0 * hits * hits
    if active_hits and max(target_scores.values(), default=0) > 0:
        priority = target_scores
        tactic = "finish_ship"
    else:
        priority = scores
        tactic = "hunt"
    middle = (size - 1) / 2
    cell = max(choices, key=lambda item: (priority[item], scores[item],
                                         -abs(item[0] - middle) - abs(item[1] - middle)))
    return {"action": choices[cell], "confidence": 0.83 if tactic == "finish_ship" else 0.68,
            "proof": False, "tactic": tactic, "reason": "Ship-placement density on unexplored cells."}


@lru_cache(maxsize=1)
def _common_words() -> tuple[str, ...]:
    try:
        return tuple(Path(__file__).with_name("common_words.txt").read_text(encoding="utf-8").split())
    except OSError:
        return ()


def _wordle_marks(guess: str, secret: str) -> str:
    marks = ["X"] * len(guess)
    remaining: dict[str, int] = {}
    for index, (letter, target) in enumerate(zip(guess, secret)):
        if letter == target:
            marks[index] = "G"
        else:
            remaining[target] = remaining.get(target, 0) + 1
    for index, letter in enumerate(guess):
        if marks[index] == "X" and remaining.get(letter, 0):
            marks[index] = "Y"
            remaining[letter] -= 1
    return "".join(marks)


def _wordle_history(observation: str, length: int) -> list[tuple[str, str]]:
    own = re.split(r"Opponent'?s? guesses", observation, maxsplit=1, flags=re.I)[0]
    own = re.split(r"Your guesses[^\n]*:\s*", own, maxsplit=1, flags=re.I)[-1]
    result = []
    for line in own.splitlines():
        match = re.match(rf"^\s*\d+[.)]\s*([A-Za-z]{{{length}}})\s+((?:[GYX]\s*){{{length}}})", line, re.I)
        if match:
            guess = match.group(1).lower()
            marks = "".join(re.findall(r"[GYX]", match.group(2).upper()))
            if len(marks) == length:
                result.append((guess, marks))
    return result


def _wordle(state: dict[str, Any]) -> dict[str, Any] | None:
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    length = variant.get("length")
    if not isinstance(length, int) or length not in (4, 5, 6):
        return None
    observation = str(state.get("observation") or "")
    history = _wordle_history(observation, length)
    rejected = set(state.get("rejected_actions") or [])
    if not history:
        opening = {4: "sale", 5: "crane", 6: "orates"}[length]
        if int(state.get("turn") or 1) <= 1 and f"guess {opening}" not in rejected:
            return {"action": f"guess {opening}", "confidence": 0.8,
                    "proof": False, "tactic": "information", "reason": "High-coverage opening."}
        return None
    words = [word for word in _common_words() if len(word) == length]
    used = {guess for guess, _ in history}
    candidates = [word for word in words if word not in used and f"guess {word}" not in rejected
                  and all(_wordle_marks(guess, word) == marks for guess, marks in history)]
    if not candidates:
        return None
    if len(candidates) == 1:
        chosen = candidates[0]
    elif len(candidates) <= 8 or int(state.get("turn") or 1) >= int(variant.get("max_guesses") or 6) - 1:
        chosen = candidates[0]
    else:
        sample = candidates[:250]
        probes = list(dict.fromkeys(candidates[:75] + [word for word in words[:400] if word not in used][:75]))
        def split_score(probe: str) -> int:
            buckets: dict[str, int] = {}
            for secret in sample:
                key = _wordle_marks(probe, secret)
                buckets[key] = buckets.get(key, 0) + 1
            return sum(count * count for count in buckets.values())
        rank = {word: index for index, word in enumerate(words)}
        chosen = min(probes, key=lambda word: (split_score(word), word not in candidates,
                                               rank.get(word, 10000)))
    return {"action": f"guess {chosen}", "confidence": 0.91 if len(candidates) <= 8 else 0.76,
            "proof": False, "tactic": "candidate_filter",
            "reason": f"{len(candidates)} common words fit exact color feedback."}


def _poker(state: dict[str, Any]) -> dict[str, Any] | None:
    legal = state.get("legal_actions")
    if not isinstance(legal, list) or not legal:
        return None
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    style = str(variant.get("style") or "").casefold()
    if style not in {"kuhn", "leduc"}:
        return None
    observation = str(state.get("observation") or "")
    held = re.search(r"\b(?:your (?:private )?card|you hold|your hand)\s*[:=-]?\s*([JQK])[shdc]?\b", observation, re.I)
    if not held:
        return None
    card = held.group(1).upper()
    board = re.search(r"\b(?:community|board) card\s*[:=-]?\s*([JQK])[shdc]?\b", observation, re.I)
    legal_names = {str(action).casefold(): action for action in legal if isinstance(action, str)
                   and action not in set(state.get("rejected_actions") or [])}
    strength = (style == "kuhn" and card == "K") or (style == "leduc" and board is not None
                                                       and card == board.group(1).upper())
    if strength:
        for action in ("raise", "bet", "call", "check"):
            if action in legal_names:
                return {"action": legal_names[action], "confidence": 0.94,
                        "proof": False, "tactic": "value_bet", "reason": "Top-rank or unique paired hand."}
    if style == "kuhn" and card == "J" and "fold" in legal_names:
        return {"action": legal_names["fold"], "confidence": 0.88,
                "proof": False, "tactic": "save_chips", "reason": "Bottom card facing a wager."}
    return None


def _board(observation: str, variant: dict[str, Any]) -> tuple[tuple[str, ...], ...] | None:
    lines = re.findall(r"^\|([.XOxo ]+)\|\s*$", observation, re.M)
    rows = [tuple(re.findall(r"[.XO]", line.upper())) for line in lines]
    if not rows or len({len(row) for row in rows}) != 1:
        return None
    if isinstance(variant.get("rows"), int) and len(rows) != variant["rows"]:
        return None
    if isinstance(variant.get("cols"), int) and len(rows[0]) != variant["cols"]:
        return None
    return tuple(rows)


def _drop(board: tuple[tuple[str, ...], ...], col: int, mark: str) -> tuple[tuple[tuple[str, ...], ...], int] | None:
    if col < 0 or col >= len(board[0]):
        return None
    for row in range(len(board) - 1, -1, -1):
        if board[row][col] == ".":
            next_board = list(board)
            changed = list(board[row])
            changed[col] = mark
            next_board[row] = tuple(changed)
            return tuple(next_board), row
    return None


def _won(board: tuple[tuple[str, ...], ...], row: int, col: int, mark: str, needed: int) -> bool:
    height, width = len(board), len(board[0])
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            r, c = row + dr * direction, col + dc * direction
            while 0 <= r < height and 0 <= c < width and board[r][c] == mark:
                count += 1
                r += dr * direction
                c += dc * direction
        if count >= needed:
            return True
    return False


def _possible(board: tuple[tuple[str, ...], ...]) -> list[int]:
    return [col for col, cell in enumerate(board[0]) if cell == "."]


def _winning_columns(board: tuple[tuple[str, ...], ...], mark: str, needed: int) -> list[int]:
    result = []
    for col in _possible(board):
        dropped = _drop(board, col, mark)
        if dropped and _won(dropped[0], dropped[1], col, mark, needed):
            result.append(col)
    return result


def _heuristic(board: tuple[tuple[str, ...], ...], mark: str, needed: int) -> float:
    opponent = "O" if mark == "X" else "X"
    height, width = len(board), len(board[0])
    score = 0.0
    middle = (width - 1) / 2
    for row in board:
        for col, cell in enumerate(row):
            if cell != ".":
                score += (1 if cell == mark else -1) * (width - abs(col - middle)) * 0.15
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(height):
            for col in range(width):
                end_r, end_c = row + dr * (needed - 1), col + dc * (needed - 1)
                if not (0 <= end_r < height and 0 <= end_c < width):
                    continue
                line = [board[row + dr * offset][col + dc * offset] for offset in range(needed)]
                own, rival = line.count(mark), line.count(opponent)
                if own and rival:
                    continue
                if own:
                    score += 3 ** own
                elif rival:
                    score -= 3.4 ** rival
    return score


def _search(board: tuple[tuple[str, ...], ...], mark: str, needed: int,
            depth: int, alpha: float, beta: float, deadline: float,
            cache: dict[tuple[Any, ...], float]) -> float:
    if time.monotonic() >= deadline:
        raise TimeoutError
    key = (board, mark, depth)
    if key in cache:
        return cache[key]
    if depth == 0:
        return _heuristic(board, mark, needed)
    options = _possible(board)
    if not options:
        return 0.0
    other = "O" if mark == "X" else "X"
    best = -1_000_000.0
    ordered = sorted(options, key=lambda col: abs(col - (len(board[0]) - 1) / 2))
    for col in ordered:
        dropped = _drop(board, col, mark)
        if not dropped:
            continue
        next_board, row = dropped
        value = 100_000 + depth if _won(next_board, row, col, mark, needed) else -_search(
            next_board, other, needed, depth - 1, -beta, -alpha, deadline, cache,
        )
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    # A beta cutoff is a bound, not an exact value. Reusing it as an exact
    # transposition score can select a losing line at a different search window.
    if alpha < beta:
        cache[key] = best
    return best


def _connect_four(state: dict[str, Any]) -> dict[str, Any] | None:
    legal = state.get("legal_actions")
    if not isinstance(legal, list) or not legal:
        return None
    actions = {int(match.group(1)) - 1: item for item in legal if isinstance(item, str)
               if (match := re.fullmatch(r"drop (\d+)", item))}
    variant = state.get("variant") if isinstance(state.get("variant"), dict) else {}
    board = _board(str(state.get("observation") or ""), variant)
    if not board or not actions:
        return None
    needed = variant.get("connect")
    if not isinstance(needed, int):
        match = re.search(r"CONNECT\s+(\d+)", str(state.get("observation") or ""))
        needed = int(match.group(1)) if match else 4
    if not 3 <= needed <= len(board[0]):
        return None
    mark = "X" if int(state.get("seat") or 0) == 0 else "O"
    other = "O" if mark == "X" else "X"
    options = [col for col in actions if col in _possible(board)]
    if not options:
        return None
    wins = [col for col in options if col in _winning_columns(board, mark, needed)]
    if wins:
        col = min(wins, key=lambda value: abs(value - (len(board[0]) - 1) / 2))
        return {"action": actions[col], "confidence": 1.0, "proof": True,
                "tactic": "immediate_win", "reason": "Completes a line now."}
    safe = []
    for col in options:
        dropped = _drop(board, col, mark)
        if dropped and not _winning_columns(dropped[0], other, needed):
            safe.append(col)
    choices = safe or options
    if len(safe) == 1:
        return {"action": actions[safe[0]], "confidence": 0.99, "proof": True,
                "tactic": "forced_defense", "reason": "Only move preventing an immediate loss."}
    deadline = time.monotonic() + 0.35
    best_col = min(choices, key=lambda value: abs(value - (len(board[0]) - 1) / 2))
    best_score = -1_000_000.0
    cache: dict[tuple[Any, ...], float] = {}
    for depth in (2, 3, 4):
        scores: dict[int, float] = {}
        try:
            for col in choices:
                dropped = _drop(board, col, mark)
                if not dropped:
                    continue
                scores[col] = -_search(dropped[0], other, needed, depth - 1,
                                       -1_000_000.0, 1_000_000.0, deadline, cache)
        except TimeoutError:
            break
        if scores:
            best_col = max(scores, key=lambda col: (scores[col], -abs(col - (len(board[0]) - 1) / 2)))
            best_score = scores[best_col]
    dropped = _drop(board, best_col, mark)
    bad_reply = ""
    if dropped:
        for reply in _possible(dropped[0]):
            next_board = _drop(dropped[0], reply, other)
            if next_board and not _won(next_board[0], next_board[1], reply, other, needed) and _winning_columns(next_board[0], mark, needed):
                bad_reply = f"drop {reply + 1}"
                break
    return {"action": actions[best_col], "confidence": 0.82, "proof": False,
            "tactic": "position", "bad_reply": bad_reply,
            "reason": f"Bounded board search score {best_score:.1f}."}


def choose_known(state: dict[str, Any]) -> dict[str, Any] | None:
    game = str(state.get("game_id") or "")
    if game == "nim":
        return _nim(state)
    if game == "bulls_and_cows":
        return _bulls(state)
    if game == "rps":
        return _rps(state)
    if game == "dots_and_boxes":
        return _dots(state)
    if game == "hex":
        return _hex(state)
    if game == "battleship":
        return _battleship(state)
    if game == "mini_poker":
        return _poker(state)
    if game == "connect_four":
        return _connect_four(state)
    if game == "wordle_duel":
        return _wordle(state)
    return None
