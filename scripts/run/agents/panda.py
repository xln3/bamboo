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
    def _base_agent_id(self) -> str:
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
            "32000",
        ]

    def env_overrides(self) -> dict[str, str]:
        mc = self._model_config
        if not mc.get("api_key"):
            raise ValueError(
                "PandaAdapter requires model config with api_key. "
                "Use --model <profile> or set PANDA_API_KEY env var.\n"
                "See configs/models.example.json for config format."
            )

        overrides = {
            "PANDA_API_KEY": mc["api_key"],
            "PANDA_BASE_URL": mc["base_url"],
            "PANDA_PROVIDER": mc.get("provider", "openai"),
            "PANDA_MODEL": mc["model"],
        }

        # Handle proxy settings
        no_proxy_extra = mc.get("no_proxy", "")
        if no_proxy_extra:
            existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
            no_proxy = f"{existing},{no_proxy_extra}" if existing else no_proxy_extra
            overrides["NO_PROXY"] = no_proxy
            overrides["no_proxy"] = no_proxy

        return overrides
