"""Run generated game solvers with limited authority and a hard deadline.

The source is checked before it reaches a fresh Python child. The child receives
only JSON data, has a small set of builtins and imports, and returns JSON data.
On Linux, address space and CPU limits provide a second line of defence. This is
an execution boundary for generated algorithms, not a general Python security
boundary against deliberately crafted interpreter exploits.
"""

from __future__ import annotations

import ast
import builtins
import functools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


MAX_SOURCE_BYTES = 80_000
MAX_RESULT_BYTES = 2_000_000
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

# Keep this list in sync with the solver generation prompt. In particular, do
# not add modules which expose reflection, file handles or network clients.
SAFE_MODULES = frozenset({
    "bisect", "collections", "fractions", "functools", "heapq", "itertools",
    "json", "math", "random", "re", "statistics", "time",
})

_BANNED_NAMES = frozenset({
    "breakpoint", "compile", "delattr", "dir", "eval", "exec", "exit",
    "getattr", "globals", "hasattr", "help", "input", "locals", "open",
    "print", "quit", "setattr", "vars", "__import__", "__builtins__",
})

# str.format and format_map can resolve object attributes named inside a
# dynamically assembled format string, bypassing the AST attribute check.
_BANNED_ATTRIBUTES = frozenset({"format", "format_map"})

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "bytes", "bytearray", "callable",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "hash", "hex", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "ArithmeticError", "AssertionError",
    "AttributeError", "Exception", "IndexError", "KeyError", "LookupError",
    "NotImplementedError", "OverflowError", "RecursionError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)


def _unsafe_identifier(name: str) -> bool:
    return "__" in name or name in _BANNED_NAMES


@functools.lru_cache(maxsize=32)
def validate_source(source: str) -> str | None:
    """Return a concise rejection reason, or ``None`` when source is allowed."""
    if not isinstance(source, str) or not source.strip():
        return "empty solver source"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return f"solver source exceeds {MAX_SOURCE_BYTES} bytes"
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError) as exc:
        return f"Python syntax error: {exc}"

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        where = f" on line {line}" if line else ""

        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            return f"classes and async code are not allowed{where}"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return f"global and nonlocal declarations are not allowed{where}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in SAFE_MODULES:
                    return f"import {alias.name!r} is not allowed{where}"
                if alias.asname and _unsafe_identifier(alias.asname):
                    return f"unsafe import alias {alias.asname!r}{where}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in SAFE_MODULES:
                return f"import from {node.module!r} is not allowed{where}"
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    return f"private or star import is not allowed{where}"
                if alias.asname and _unsafe_identifier(alias.asname):
                    return f"unsafe import alias {alias.asname!r}{where}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _BANNED_ATTRIBUTES:
                return f"attribute {node.attr!r} is not allowed{where}"
        elif isinstance(node, ast.Name):
            if _unsafe_identifier(node.id):
                return f"name {node.id!r} is not allowed{where}"
        elif isinstance(node, (ast.FunctionDef, ast.arg)):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
            if _unsafe_identifier(name):
                return f"name {name!r} is not allowed{where}"
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and "__" in value:
                return f"double underscore in string is not allowed{where}"
            if isinstance(value, bytes) and b"__" in value:
                return f"double underscore in bytes is not allowed{where}"
        elif isinstance(node, ast.keyword):
            if node.arg and _unsafe_identifier(node.arg):
                return f"keyword {node.arg!r} is not allowed{where}"

    return None


def _safe_import(name: str, globals_: Any = None, locals_: Any = None,
                 fromlist: Any = (), level: int = 0) -> Any:
    if level or name not in SAFE_MODULES:
        raise ImportError(f"import {name!r} is not allowed")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    result = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    result["__import__"] = _safe_import
    return result


def _apply_linux_limits(timeout_s: float) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource

        cpu_limit = max(1, math.ceil(timeout_s) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except (ImportError, OSError, ValueError):
        # Parent still enforces the wall-clock deadline if a platform refuses
        # one of these optional resource controls.
        pass


def _child_reply(result: Any = None, error: str | None = None) -> None:
    try:
        encoded = json.dumps({"result": result, "error": error}, ensure_ascii=True,
                             allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
            raise ValueError("solver result is too large")
    except (TypeError, ValueError, OverflowError) as exc:
        encoded = json.dumps({"result": None, "error": f"result is not JSON data: {exc}"})
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _child_main() -> None:
    try:
        request = json.load(sys.stdin)
        source = request["source"]
        function = request["function"]
        args = request["args"]
        timeout_s = float(request["timeout_s"])
        error = validate_source(source)
        if error:
            _child_reply(error=error)
            return
        _apply_linux_limits(timeout_s)
        namespace: dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "__name__": "generated_solver",
        }
        # Generated algorithms never get the protocol stream. A stray write
        # through a library goes to stderr; stdout is reserved for our reply.
        stdout = sys.stdout
        try:
            sys.stdout = sys.stderr
            exec(compile(source, "<generated solver>", "exec"), namespace)
            target = namespace.get(function)
            if not callable(target):
                raise ValueError(f"function {function!r} is missing")
            result = target(*args)
        finally:
            sys.stdout = stdout
        _child_reply(result=result)
    except BaseException as exc:
        _child_reply(error=f"{type(exc).__name__}: {str(exc)[:300]}")


def _minimal_child_env() -> dict[str, str]:
    # Do not expose gateway keys or match data through environment variables.
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_function(source: str, function: str, args: tuple,
                 timeout_s: float) -> tuple[Any | None, str | None]:
    """Execute one generated function in a disposable child process.

    Results and arguments must be JSON data. The error is ``None`` only when
    the function completed successfully, including when its result is ``None``.
    """
    error = validate_source(source)
    if error:
        return None, error
    if not isinstance(function, str) or not function.isidentifier() or function.startswith("_"):
        return None, "invalid function name"
    if not isinstance(args, tuple):
        return None, "args must be a tuple"
    try:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            return None, "timeout_s must be positive and finite"
        request = json.dumps({"source": source, "function": function, "args": args,
                              "timeout_s": timeout_s}, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"invalid sandbox input: {str(exc)[:160]}"

    command = [sys.executable, "-I", "-S", "-B", "-X", "utf8",
               str(Path(__file__).resolve()), "--sandbox-child"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            env=_minimal_child_env(), cwd=tempfile.gettempdir(),
            creationflags=flags,
        )
        try:
            stdout, stderr = process.communicate(request, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return None, f"solver timed out after {timeout_s:g}s"
    except (OSError, ValueError) as exc:
        return None, f"could not start solver child: {str(exc)[:160]}"

    if process.returncode != 0 and not stdout.strip():
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no error output"
        return None, f"solver child exited {process.returncode}: {detail[:160]}"
    try:
        reply = json.loads(stdout)
        if not isinstance(reply, dict) or not isinstance(reply.get("error"), (str, type(None))):
            raise ValueError("malformed child reply")
        return reply.get("result"), reply.get("error")
    except (ValueError, TypeError) as exc:
        return None, f"invalid solver child reply: {str(exc)[:160]}"


if __name__ == "__main__" and sys.argv[1:] == ["--sandbox-child"]:
    _child_main()
