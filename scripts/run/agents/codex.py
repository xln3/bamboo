"""OpenAI Codex CLI agent adapter.

NOTE: Codex uses the OpenAI Responses API (/v1/responses), NOT Chat Completions.
It requires a direct OpenAI API key — OpenAI-compatible proxies like aihubmix
do not support the /responses endpoint. Set OPENAI_API_KEY to a real OpenAI key.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import AgentAdapter


class CodexAdapter(AgentAdapter):
    @property
    def _base_agent_id(self) -> str:
        return "codex"

    def build_command(
        self, prompt: str, workdir: Path, timeout_s: int
    ) -> list[str]:
        codex_bin = shutil.which("codex")
        if codex_bin:
            cmd = [codex_bin]
        else:
            cmd = ["npx", "@openai/codex"]
        mc = self._model_config
        model = mc.get("model") or os.environ.get("CODEX_MODEL", "o4-mini")
        return [
            *cmd,
            "exec",
            "--full-auto",
            "-m",
            model,
            prompt,
        ]

    def env_overrides(self) -> dict[str, str]:
        mc = self._model_config
        return {
            "OPENAI_API_KEY": mc.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": mc.get("base_url") or os.environ.get("OPENAI_BASE_URL", ""),
        }
