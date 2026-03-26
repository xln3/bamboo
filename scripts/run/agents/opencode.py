"""OpenCode agent adapter."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import AgentAdapter


class OpenCodeAdapter(AgentAdapter):
    @property
    def agent_id(self) -> str:
        return "opencode"

    def build_command(
        self, prompt: str, workdir: Path, timeout_s: int
    ) -> list[str]:
        opencode_bin = shutil.which("opencode") or "opencode"
        model = os.environ.get("OPENCODE_MODEL", "openai/gpt-4o-mini")
        return [
            opencode_bin,
            "run",
            "-m",
            model,
            prompt,
        ]

    def env_overrides(self) -> dict[str, str]:
        return {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
        }
