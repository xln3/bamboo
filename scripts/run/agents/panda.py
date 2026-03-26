"""Panda agent adapter."""

from __future__ import annotations

import os
from pathlib import Path

from .base import AgentAdapter

PANDA_ROOT = Path(os.environ.get(
    "PANDA_ROOT", Path.home() / "panda2026" / "panda"
))
DENO = Path(os.environ.get("DENO_BIN", Path.home() / ".deno" / "bin" / "deno"))


class PandaAdapter(AgentAdapter):
    @property
    def agent_id(self) -> str:
        return "panda"

    def build_command(
        self, prompt: str, workdir: Path, timeout_s: int
    ) -> list[str]:
        return [
            str(DENO),
            "run",
            "--allow-all",
            str(PANDA_ROOT / "mod.ts"),
            "-p",
            prompt,
            "--profile",
            "coding",
            "--max-tokens",
            "64000",
        ]

    def env_overrides(self) -> dict[str, str]:
        # Panda reads .env from its own directory, but also honor explicit env
        overrides = {
            "PANDA_API_KEY": os.environ.get("PANDA_API_KEY", ""),
            "PANDA_BASE_URL": os.environ.get("PANDA_BASE_URL", "https://aihubmix.com/v1"),
            "PANDA_PROVIDER": os.environ.get("PANDA_PROVIDER", "openai"),
            "PANDA_MODEL": os.environ.get("PANDA_MODEL", "claude-sonnet-4-20250514"),
        }
        # Disable proxy for Deno — aihubmix needs direct connection
        overrides["HTTP_PROXY"] = ""
        overrides["HTTPS_PROXY"] = ""
        overrides["http_proxy"] = ""
        overrides["https_proxy"] = ""
        overrides["NO_PROXY"] = "*"
        return overrides
