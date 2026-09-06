"""Minimal structured model calls through an ephemeral Codex CLI session."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)
_LOG_LOCK = threading.Lock()
_MARKER_LOCK = threading.Lock()
_TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_TEXT = (
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "econnreset",
    "timed out",
    "timeout",
)
_NAMED_SUBSCHEMA_KEYS = {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
_SUBSCHEMA_KEYS = {
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_SUBSCHEMA_LIST_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_DISABLED_FEATURES = (
    "multi_agent",
    "multi_agent_v2",
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "code_mode",
    "code_mode_buffered_exec",
    "code_mode_only",
    "apps",
    "plugins",
    "plugin_sharing",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "auth_elicitation",
    "default_mode_request_user_input",
    "deferred_executor",
    "enable_mcp_apps",
    "executor_capability_discovery",
    "goals",
    "guardian_approval",
    "hooks",
    "personality",
    "request_permissions_tool",
    "shell_snapshot",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "workspace_dependencies",
)
_CONTEXT_ENVIRONMENT_KEYS = (
    "CODEX_THREAD_ID",
    "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
    "CODEX_CI",
    "CODEX_SANDBOX_NETWORK_DISABLED",
)
_REASONING_SUMMARY_MODES = {"auto", "concise", "detailed", "none"}


class RunnerError(RuntimeError):
    """Base class for model-transport failures."""


class TransientRunnerError(RunnerError):
    """A provider or transport failure that a supervisor may retry."""


class PermanentRunnerError(RunnerError):
    """A configuration or protocol failure that an unchanged retry cannot repair."""


class StructuredOutputError(PermanentRunnerError):
    """The provider output remained invalid after bounded correction attempts."""


@dataclass(frozen=True)
class CallMetadata:
    """Applied settings and immutable identifiers for one completed model call."""

    call_id: str
    prompt_version: str
    prompt_sha256: str
    request_sha256: str
    response_sha256: str
    model: str | None
    effort: str | None
    thinking: bool
    backend: str
    attempts: int
    elapsed_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> str:
    """Return the first balanced JSON object, tolerating a Markdown fence."""

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
        elif character == '"':
            in_string = not in_string
        elif not in_string:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError("unbalanced JSON object in response")


def _reduced_schema(value: Any) -> Any:
    """Drop JSON Schema annotations that do not affect validation but consume input."""

    if not isinstance(value, dict):
        return value
    reduced: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"title", "default"}:
            continue
        if key in _NAMED_SUBSCHEMA_KEYS and isinstance(item, dict):
            reduced[key] = {name: _reduced_schema(schema) for name, schema in item.items()}
        elif key in _SUBSCHEMA_KEYS:
            reduced[key] = [_reduced_schema(schema) for schema in item] if isinstance(item, list) else _reduced_schema(item)
        elif key in _SUBSCHEMA_LIST_KEYS and isinstance(item, list):
            reduced[key] = [_reduced_schema(schema) for schema in item]
        else:
            reduced[key] = item
    return reduced


def _jsonl_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _strict_codex_schema(output_schema: dict[str, Any]) -> dict[str, Any] | None:
    """Return a strict Codex-compatible schema, or None for unsupported free-form mappings."""

    schema = json.loads(json.dumps(output_schema))
    supported = True

    def normalize(node: Any) -> None:
        nonlocal supported
        if isinstance(node, list):
            for item in node:
                normalize(item)
            return
        if not isinstance(node, dict):
            return
        node.pop("default", None)
        if node.get("type") == "object" or isinstance(node.get("properties"), dict):
            properties = node.get("properties")
            additional = node.get("additionalProperties")
            if isinstance(properties, dict) and not isinstance(additional, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            elif additional not in (None, False):
                supported = False
            elif additional is None:
                supported = False
        for value in node.values():
            normalize(value)

    normalize(schema)
    return schema if supported else None


def _event_error(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "turn.failed" and isinstance(event.get("error"), dict):
            message = event["error"].get("message")
            if isinstance(message, str):
                return message
        if event.get("type") == "error" and isinstance(event.get("message"), str):
            return event["message"]
    return ""


def _disabled_skills_config(environment: dict[str, str]) -> str | None:
    home = Path(environment.get("HOME") or Path.home()).expanduser()
    codex_home = Path(environment.get("CODEX_HOME") or home / ".codex").expanduser()
    roots = (codex_home / "skills", home / ".agents" / "skills", Path("/etc/codex/skills"))
    skill_paths: set[str] = set()
    for root in roots:
        try:
            skill_paths.update(str(path) for path in root.rglob("SKILL.md") if path.is_file())
        except OSError:
            continue
    if not skill_paths:
        return None
    entries = ",".join(f"{{path={json.dumps(path)},enabled=false}}" for path in sorted(skill_paths))
    return f"skills.config=[{entries}]"


def _reasoning_summary_mode(environment: dict[str, str], thinking: bool) -> str:
    value = environment.get("GLEE_CODEX_REASONING_SUMMARY") or ("detailed" if thinking else "none")
    normalized = value.strip().lower()
    if normalized not in _REASONING_SUMMARY_MODES:
        supported = ", ".join(sorted(_REASONING_SUMMARY_MODES))
        raise PermanentRunnerError(f"GLEE_CODEX_REASONING_SUMMARY must be one of: {supported}")
    return normalized


def _reasoning_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event["item"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "reasoning"
    ]


def _provider_receipt(error: BaseException) -> dict[str, Any]:
    value = getattr(error, "provider_receipt", None)
    return value if isinstance(value, dict) else {}


class CodexCliRunner:
    """Call GPT through `codex exec` with agent capabilities suppressed."""

    name = "codex-cli"
    default_model = "gpt-5.6-terra"

    def __init__(
        self,
        *,
        prompts_dir: Path | None = None,
        executable: str | None = None,
        timeout_s: int | None = None,
        validation_retries: int | None = None,
        log_path: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.prompts_dir = prompts_dir or Path(os.environ.get("GLEE_PROMPTS_DIR") or project_root / "prompts")
        self.executable = executable or os.environ.get("GLEE_CODEX_BIN") or "codex"
        self.timeout_s = timeout_s if timeout_s is not None else int(os.environ.get("GLEE_CODEX_TIMEOUT_S", "3600"))
        self.validation_retries = validation_retries if validation_retries is not None else int(os.environ.get("GLEE_CODEX_VALIDATION_RETRIES", "4"))
        self.log_path = log_path or Path(os.environ.get("GLEE_CODEX_LLM_LOG") or project_root / "runs" / "llm_calls.jsonl")
        self.session_dir = session_dir or Path(os.environ.get("GLEE_CODEX_TMP_ROOT") or tempfile.gettempdir())
        if self.timeout_s <= 0:
            raise PermanentRunnerError("GLEE_CODEX_TIMEOUT_S must be positive")
        if self.validation_retries < 0:
            raise PermanentRunnerError("GLEE_CODEX_VALIDATION_RETRIES cannot be negative")

    def _prompt(self, role: str) -> tuple[str, str, str]:
        path = self.prompts_dir / f"{role}.md"
        if not path.is_file():
            raise PermanentRunnerError(f"missing prompt for role {role!r}: {path}")
        text = path.read_text(encoding="utf-8")
        digest = _sha(text)
        return text, f"{role}@{digest[:8]}", digest

    def _settings(self, role: str, model: str | None, effort: str | None) -> tuple[str, str, bool]:
        suffix = re.sub(r"[^A-Za-z0-9]", "_", role).upper()
        resolved_model = model or os.environ.get(f"GLEE_CODEX_MODEL_{suffix}") or os.environ.get("GLEE_CODEX_MODEL") or self.default_model
        resolved_effort = effort or os.environ.get(f"GLEE_CODEX_EFFORT_{suffix}") or os.environ.get("GLEE_CODEX_EFFORT") or "max"
        if resolved_effort == "ultra":
            raise PermanentRunnerError("reasoning effort 'ultra' is forbidden because it can launch subagents")
        return resolved_model, resolved_effort, True

    @staticmethod
    def _prompt_schema_system_text(template: str, output_schema: dict[str, Any]) -> str:
        schema_text = json.dumps(_reduced_schema(output_schema), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{template.rstrip()}\n\nReturn only JSON matching:\n{schema_text}"

    @staticmethod
    def _system_text(template: str, output_schema: dict[str, Any]) -> str:
        if _strict_codex_schema(output_schema) is not None:
            return template.rstrip()
        return CodexCliRunner._prompt_schema_system_text(template, output_schema)

    def _mark_transient(self, detail: str) -> None:
        marker_value = os.environ.get("GLEE_CODEX_RETRY_MARKER")
        if not marker_value:
            return
        marker = Path(marker_value)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            with _MARKER_LOCK:
                marker.write_text(json.dumps({"ts": _now(), "error": detail}, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _is_transient(stdout: str, stderr: str) -> bool:
        blob = f"{stdout}\n{stderr}".lower()
        for event in _jsonl_events(stdout):
            raw_status = event.get("status") or event.get("status_code")
            try:
                if int(raw_status) in _TRANSIENT_STATUSES:
                    return True
            except (TypeError, ValueError):
                pass
        return any(token in blob for token in _TRANSIENT_TEXT)

    def _raise_provider_error(self, message: str, receipt: dict[str, Any], transient: bool) -> None:
        error_type = TransientRunnerError if transient else PermanentRunnerError
        error = error_type(message)
        error.provider_receipt = receipt
        if transient:
            self._mark_transient(message)
        raise error

    def _complete(
        self,
        system_text: str,
        user_text: str,
        output_schema: dict[str, Any] | None,
        model: str | None,
        effort: str | None,
        thinking: bool,
    ) -> tuple[str, dict[str, Any]]:
        environment = dict(os.environ)
        reasoning_summary = _reasoning_summary_mode(environment, thinking)
        skills_config = _disabled_skills_config(environment)
        for key in _CONTEXT_ENVIRONMENT_KEYS:
            environment.pop(key, None)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="deeprmm-codex-", dir=self.session_dir) as temporary_value:
            temporary = Path(temporary_value)
            schema_path = temporary / "schema.json"
            system_path = temporary / "system.txt"
            output_path = temporary / "last-message.json"
            strict_schema = _strict_codex_schema(output_schema) if output_schema is not None else None
            system_path.write_text(system_text, encoding="utf-8")
            if strict_schema is not None:
                schema_path.write_text(json.dumps(strict_schema, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--strict-config",
                "-c",
                'approval_policy="never"',
                "-c",
                'personality="none"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                f"model_instructions_file={json.dumps(str(system_path))}",
                "-c",
                f'model_reasoning_summary="{reasoning_summary}"',
                "-c",
                'model_verbosity="low"',
            ]
            for feature in _DISABLED_FEATURES:
                command.extend(["--disable", feature])
            command.extend(["-c", 'web_search="disabled"'])
            if skills_config is not None:
                command.extend(["-c", skills_config])
            if model:
                command.extend(["--model", model])
            if effort:
                command.extend(["-c", f'model_reasoning_effort="{effort}"'])
            if strict_schema is not None:
                command.extend(["--output-schema", str(schema_path)])
            command.extend(["--output-last-message", str(output_path), "--json", "-"])
            try:
                process = subprocess.run(command, input=user_text, capture_output=True, text=True, timeout=self.timeout_s, env=environment, cwd=temporary)
            except subprocess.TimeoutExpired as error:
                receipt = {
                    "command": command,
                    "timeout_s": self.timeout_s,
                    "reasoning_summary_mode": reasoning_summary,
                    "output_schema": strict_schema or output_schema,
                    "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or "",
                    "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "",
                }
                self._raise_provider_error(f"Codex CLI timed out after {self.timeout_s}s", receipt, transient=True)
                raise AssertionError("unreachable") from error
            except FileNotFoundError as error:
                self._raise_provider_error(f"Codex CLI executable not found: {self.executable!r}", {"command": command, "reasoning_summary_mode": reasoning_summary, "output_schema": strict_schema or output_schema}, transient=False)
                raise AssertionError("unreachable") from error
            except OSError as error:
                self._raise_provider_error(f"could not start Codex CLI: {error}", {"command": command, "reasoning_summary_mode": reasoning_summary, "output_schema": strict_schema or output_schema, "error": str(error)}, transient=False)
                raise AssertionError("unreachable") from error
            stdout = process.stdout or ""
            stderr = process.stderr or ""
            events = _jsonl_events(stdout)
            receipt = {
                "command": command,
                "returncode": process.returncode,
                "event_stream": stdout,
                "stderr": stderr,
                "reasoning_summary_mode": reasoning_summary,
                "reasoning_items": _reasoning_items(events),
                "output_schema": strict_schema or output_schema,
                "output_schema_mode": "none" if output_schema is None else ("codex-strict" if strict_schema is not None else "prompt-and-pydantic"),
            }
            if process.returncode != 0:
                diagnostic = " | ".join(value for value in (_event_error(events).strip(), stderr.strip()) if value)
                detail = f"Codex CLI failed with rc={process.returncode}: {diagnostic[:2000] or '(no stdout/stderr)'}"
                self._raise_provider_error(detail, receipt, transient=self._is_transient(stdout, stderr))
            if not output_path.is_file():
                self._raise_provider_error("Codex CLI completed without writing its final message", receipt, transient=False)
            raw = output_path.read_text(encoding="utf-8")
            if not raw.strip():
                self._raise_provider_error("Codex CLI wrote an empty final message", receipt, transient=False)
            usage: dict[str, Any] = {}
            thread_id: str | None = None
            for event in events:
                if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                    thread_id = event["thread_id"]
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    usage = event["usage"]
            return raw, {
                "input_chars": len(system_text) + len(user_text),
                "system_chars": len(system_text),
                "user_chars": len(user_text),
                "output_chars": len(raw),
                "tokens_in": usage.get("input_tokens") or 0,
                "tokens_out": usage.get("output_tokens") or 0,
                "cache_read": usage.get("cached_input_tokens") or 0,
                "cache_write": 0,
                "reasoning_tokens": usage.get("reasoning_output_tokens") or 0,
                "cost_usd": 0.0,
                "thread_id": thread_id,
                **receipt,
            }

    def _log(self, record: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with _LOG_LOCK, self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            pass

    def call_text(self, role: str, body: str, *, model: str | None = None, effort: str | None = None) -> tuple[str, CallMetadata]:
        """Call one role once and preserve its free-form text verbatim."""

        template, prompt_version, prompt_sha = self._prompt(role)
        resolved_model, resolved_effort, thinking = self._settings(role, model, effort)
        system_text = template.rstrip()
        call_id = uuid.uuid4().hex
        call_started = time.monotonic()
        attempt_started = time.monotonic()
        try:
            raw, provider_meta = self._complete(system_text, body, None, resolved_model, resolved_effort, thinking)
        except (TransientRunnerError, PermanentRunnerError) as error:
            self._log({"schema_version": 1, "ts": _now(), "call_id": call_id, "role": role, "prompt_version": prompt_version, "model": resolved_model, "effort": resolved_effort, "thinking": thinking, "backend": self.name, "attempt": 1, "ok": False, "transient": isinstance(error, TransientRunnerError), "elapsed_s": round(time.monotonic() - attempt_started, 6), "request": {"system": system_text, "user": body, "system_sha256": _sha(system_text), "user_sha256": _sha(body)}, "provider": _provider_receipt(error), "output_format": "text", "error": f"{type(error).__name__}: {error}"})
            raise
        self._log({"schema_version": 1, "ts": _now(), "call_id": call_id, "role": role, "prompt_version": prompt_version, "model": resolved_model, "effort": resolved_effort, "thinking": thinking, "backend": self.name, "attempt": 1, "ok": True, "transient": False, "elapsed_s": round(time.monotonic() - attempt_started, 6), "request": {"system": system_text, "user": body, "system_sha256": _sha(system_text), "user_sha256": _sha(body)}, "response": {"raw": raw, "sha256": _sha(raw)}, "provider": provider_meta, "output_format": "text"})
        return raw, CallMetadata(call_id=call_id, prompt_version=prompt_version, prompt_sha256=prompt_sha, request_sha256=_sha(body), response_sha256=_sha(raw), model=resolved_model, effort=resolved_effort, thinking=thinking, backend=self.name, attempts=1, elapsed_s=time.monotonic() - call_started, tokens_in=int(provider_meta.get("tokens_in") or 0), tokens_out=int(provider_meta.get("tokens_out") or 0), reasoning_tokens=int(provider_meta.get("reasoning_tokens") or 0), cache_read=int(provider_meta.get("cache_read") or 0), cache_write=int(provider_meta.get("cache_write") or 0), cost_usd=float(provider_meta.get("cost_usd") or 0.0))

    def call_structured(self, role: str, body: str, model_cls: type[T], *, model: str | None = None, effort: str | None = None) -> tuple[T, CallMetadata]:
        """Call one role, validate its JSON, and preserve every provider attempt."""

        template, prompt_version, prompt_sha = self._prompt(role)
        resolved_model, resolved_effort, thinking = self._settings(role, model, effort)
        output_schema = model_cls.model_json_schema()
        system_text = self._system_text(template, output_schema)
        call_id = uuid.uuid4().hex
        call_started = time.monotonic()
        feedback = ""
        last_error: BaseException | None = None
        for attempt in range(1, self.validation_retries + 2):
            user_text = body + feedback
            attempt_started = time.monotonic()
            raw = ""
            provider_meta: dict[str, Any] = {}
            try:
                raw, provider_meta = self._complete(system_text, user_text, output_schema, resolved_model, resolved_effort, thinking)
                parsed = model_cls.model_validate_json(_extract_json(raw))
            except (TransientRunnerError, PermanentRunnerError) as error:
                self._log({"schema_version": 1, "ts": _now(), "call_id": call_id, "role": role, "prompt_version": prompt_version, "model": resolved_model, "effort": resolved_effort, "thinking": thinking, "backend": self.name, "attempt": attempt, "ok": False, "transient": isinstance(error, TransientRunnerError), "elapsed_s": round(time.monotonic() - attempt_started, 6), "request": {"system": system_text, "user": user_text, "system_sha256": _sha(system_text), "user_sha256": _sha(user_text)}, "provider": _provider_receipt(error), "error": f"{type(error).__name__}: {error}"})
                raise
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                self._log({"schema_version": 1, "ts": _now(), "call_id": call_id, "role": role, "prompt_version": prompt_version, "model": resolved_model, "effort": resolved_effort, "thinking": thinking, "backend": self.name, "attempt": attempt, "ok": False, "transient": False, "elapsed_s": round(time.monotonic() - attempt_started, 6), "request": {"system": system_text, "user": user_text, "system_sha256": _sha(system_text), "user_sha256": _sha(user_text)}, "response": {"raw": raw, "sha256": _sha(raw)}, "provider": provider_meta, "error": f"{type(error).__name__}: {str(error)[:2000]}"})
                feedback = f"\n\nYour previous response failed validation:\n{str(error)[:1200]}\nReturn ONLY a corrected JSON object."
                continue
            self._log({"schema_version": 1, "ts": _now(), "call_id": call_id, "role": role, "prompt_version": prompt_version, "model": resolved_model, "effort": resolved_effort, "thinking": thinking, "backend": self.name, "attempt": attempt, "ok": True, "transient": False, "elapsed_s": round(time.monotonic() - attempt_started, 6), "request": {"system": system_text, "user": user_text, "system_sha256": _sha(system_text), "user_sha256": _sha(user_text)}, "response": {"raw": raw, "sha256": _sha(raw)}, "provider": provider_meta})
            return parsed, CallMetadata(call_id=call_id, prompt_version=prompt_version, prompt_sha256=prompt_sha, request_sha256=_sha(user_text), response_sha256=_sha(raw), model=resolved_model, effort=resolved_effort, thinking=thinking, backend=self.name, attempts=attempt, elapsed_s=time.monotonic() - call_started, tokens_in=int(provider_meta.get("tokens_in") or 0), tokens_out=int(provider_meta.get("tokens_out") or 0), reasoning_tokens=int(provider_meta.get("reasoning_tokens") or 0), cache_read=int(provider_meta.get("cache_read") or 0), cache_write=int(provider_meta.get("cache_write") or 0), cost_usd=float(provider_meta.get("cost_usd") or 0.0))
        raise StructuredOutputError(f"{role}: output failed validation after {self.validation_retries + 1} attempts") from last_error
