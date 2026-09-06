"""Codex transport specialized for the released arena roles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .codex_transport import CodexCliRunner, PermanentRunnerError
from .immutable_blob import externalize_call_request


_GLEE_FAMILY_ROLES = {
    "glee_nommd_bargaining": "glee_nommd_bargaining.md",
    "glee_nommd_negotiation": "glee_nommd_negotiation.md",
    "glee_nommd_persuasion": "glee_nommd_persuasion.md",
}

_GLEE_STANDALONE_ROLES = {
    "glee_bargaining_cloud_predict": "glee_bargaining_cloud_predict.md",
    "glee_global_tactic_map": "glee_global_tactic_map.md",
    "glee_global_tactic_reduce": "glee_global_tactic_reduce.md",
    "glee_named_opponent_digest": "glee_named_opponent_digest.md",
    "glee_named_opponent_incremental": "glee_named_opponent_incremental.md",
    "glee_named_opponent_synthesis": "glee_named_opponent_synthesis.md",
    "glee_persuasion_buyer_continuation_selector": "glee_persuasion_buyer_continuation_selector.md",
}

_GLEE_META_CONTROLLER_ROLES = {
    f"glee_meta_controller_v2_15_{stage}_{family}": (
        "glee_meta_controller_common.md",
        "glee_meta_controller_planner.md" if stage == "planner" else "glee_meta_controller_selector.md",
        f"glee_meta_controller_{stage}_{family}.md",
    )
    for stage in ("planner", "selector")
    for family in ("bargaining", "negotiation", "persuasion")
}


class _ArenaRunnerMixin:
    """Share final prompt composition and immutable request logging."""

    def _configure_arena(self, *, blob_root: Path | None) -> None:
        self.blob_root = blob_root

    def _prompt(self, role: str) -> tuple[str, str, str]:
        """Compose the shared arena contract with one stage and family module."""

        staged_names = _GLEE_META_CONTROLLER_ROLES.get(role)
        if staged_names is not None:
            paths = tuple(self.prompts_dir / name for name in staged_names)
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise PermanentRunnerError(f"missing prompt for role {role!r}: {missing[0]}")
            text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            return text, f"{role}@{digest[:8]}", digest
        standalone_name = _GLEE_STANDALONE_ROLES.get(role)
        if standalone_name is not None:
            path = self.prompts_dir / standalone_name
            if not path.is_file():
                raise PermanentRunnerError(f"missing prompt for role {role!r}: {path}")
            text = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            return text, f"{role}@{digest[:8]}", digest
        module_name = _GLEE_FAMILY_ROLES.get(role)
        if module_name is None:
            return super()._prompt(role)
        paths = (self.prompts_dir / "glee_nommd_core.md", self.prompts_dir / module_name)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise PermanentRunnerError(f"missing composed prompt for role {role!r}: {missing[0]}")
        text = "\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in paths) + "\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, f"{role}@{digest[:8]}", digest

    def _log(self, record: dict[str, Any]) -> None:
        """Store exact prompts once as immutable blobs and log verified references."""

        try:
            reduced = externalize_call_request(record, log_path=self.log_path, blob_root=self.blob_root)
        except OSError:
            reduced = record
        super()._log(reduced)


class ArenaCodexRunner(_ArenaRunnerMixin, CodexCliRunner):
    """Use the minimal Codex transport with arena-owned role composition."""

    def __init__(
        self,
        *,
        prompts_dir: Path,
        log_path: Path,
        session_dir: Path,
        timeout_s: int | None = None,
        validation_retries: int = 2,
        blob_root: Path | None = None,
    ) -> None:
        self._configure_arena(blob_root=blob_root)
        CodexCliRunner.__init__(
            self,
            prompts_dir=prompts_dir,
            log_path=log_path,
            session_dir=session_dir,
            timeout_s=timeout_s,
            validation_retries=validation_retries,
        )

    def _settings(self, role: str, model: str | None, effort: str | None) -> tuple[str, str, bool]:
        del role
        resolved_effort = effort or "max"
        if resolved_effort == "ultra":
            raise PermanentRunnerError("reasoning effort 'ultra' is forbidden because it can launch subagents")
        return model or "gpt-5.6-terra", resolved_effort, True
