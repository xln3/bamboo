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
        mc = self._model_config
        cmd = [
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
        cost_limit = mc.get("cost_limit")
        if cost_limit:
            cmd.extend(["--cost-limit", str(cost_limit)])
        return cmd

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

        # Auto-extract domain from base_url for NO_PROXY
        from urllib.parse import urlparse
        api_domain = urlparse(mc["base_url"]).hostname or ""
        no_proxy_extra = mc.get("no_proxy", "")
        domains = {d.strip() for d in no_proxy_extra.split(",") if d.strip()}
        if api_domain:
            domains.add(api_domain)

        if domains:
            existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
            all_domains = {d.strip() for d in existing.split(",") if d.strip()} | domains
            no_proxy = ",".join(sorted(all_domains))
            overrides["NO_PROXY"] = no_proxy
            overrides["no_proxy"] = no_proxy

        return overrides
