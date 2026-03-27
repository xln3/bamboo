"""Claude Code agent adapter."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import AgentAdapter


class ClaudeCodeAdapter(AgentAdapter):
    @property
    def _base_agent_id(self) -> str:
        return "claude-code"

    def build_command(
        self, prompt: str, workdir: Path, timeout_s: int
    ) -> list[str]:
        claude_bin = shutil.which("claude") or "claude"
        return [
            claude_bin,
            "-p",
            prompt,
            "--output-format",
            "text",
            "--max-turns",
            "50",
        ]
