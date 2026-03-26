"""Base agent adapter interface."""

from __future__ import annotations

import abc
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Result of running an agent on a single paper."""

    agent_id: str
    paper_id: str
    exit_code: int
    wall_time_ms: int
    stdout: str = ""
    stderr: str = ""
    result_json: dict | None = None
    error: str | None = None


class AgentAdapter(abc.ABC):
    """Interface that each agent adapter must implement."""

    @property
    @abc.abstractmethod
    def agent_id(self) -> str:
        ...

    @abc.abstractmethod
    def build_command(
        self,
        prompt: str,
        workdir: Path,
        timeout_s: int,
    ) -> list[str]:
        """Return the shell command to invoke this agent."""
        ...

    def env_overrides(self) -> dict[str, str]:
        """Extra environment variables for the agent process."""
        return {}

    def run(
        self,
        prompt: str,
        workdir: Path,
        result_path: Path,
        timeout_s: int = 1800,
    ) -> RunResult:
        """Invoke the agent and collect its result."""
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(prompt, workdir, timeout_s)

        env = {**os.environ, **self.env_overrides()}
        start = time.monotonic_ns()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s + 60,  # grace period beyond agent timeout
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as e:
            wall_ms = int((time.monotonic_ns() - start) / 1_000_000)
            return RunResult(
                agent_id=self.agent_id,
                paper_id="",
                exit_code=-1,
                wall_time_ms=wall_ms,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                error=f"Process timed out after {timeout_s + 60}s",
            )

        wall_ms = int((time.monotonic_ns() - start) / 1_000_000)

        # Try to load result JSON written by the agent
        result_json = None
        if result_path.exists():
            try:
                result_json = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        return RunResult(
            agent_id=self.agent_id,
            paper_id="",
            exit_code=exit_code,
            wall_time_ms=wall_ms,
            stdout=stdout,
            stderr=stderr,
            result_json=result_json,
        )
