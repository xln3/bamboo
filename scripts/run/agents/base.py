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
    """Interface that each agent adapter must implement.

    Args:
        model_config: Optional dict from configs/models.json with keys:
            provider, model, base_url, api_key, and optional no_proxy.
            When provided, agent_id is suffixed with the model profile name
            and env_overrides uses these values instead of defaults.
    """

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        self._model_config = model_config or {}

    @property
    @abc.abstractmethod
    def _base_agent_id(self) -> str:
        """The agent's base name (e.g. 'panda')."""
        ...

    @property
    def agent_id(self) -> str:
        """Agent ID, suffixed with model profile name when model_config is set."""
        base = self._base_agent_id
        profile = self._model_config.get("_profile_name", "")
        return f"{base}-{profile}" if profile else base

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
        log_dir: Path | None = None,
    ) -> RunResult:
        """Invoke the agent and collect its result.

        If log_dir is provided, stdout/stderr are streamed to files in
        real time (via tee), so you can ``tail -f`` the logs while the
        agent is still running.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(prompt, workdir, timeout_s)

        env = {**os.environ, **self.env_overrides()}
        start = time.monotonic_ns()

        # When log_dir is given, stream to files in real time
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / "stdout.txt"
            stderr_path = log_dir / "stderr.txt"
            stdout_f = open(stdout_path, "w")
            stderr_f = open(stderr_path, "w")
        else:
            stdout_f = None
            stderr_f = None

        try:
            if stdout_f:
                # Stream mode: pipe to files in real time
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workdir),
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                )
                try:
                    proc.wait(timeout=timeout_s + 60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    wall_ms = int((time.monotonic_ns() - start) / 1_000_000)
                    stdout_f.close()
                    stderr_f.close()
                    stdout = stdout_path.read_text()[-200000:]
                    stderr = stderr_path.read_text()[-100000:]
                    result_json = None
                    if result_path.exists():
                        try:
                            result_json = json.loads(result_path.read_text())
                        except (json.JSONDecodeError, OSError):
                            pass
                    return RunResult(
                        agent_id=self.agent_id,
                        paper_id="",
                        exit_code=-1,
                        wall_time_ms=wall_ms,
                        stdout=stdout,
                        stderr=stderr,
                        result_json=result_json,
                        error=f"Process timed out after {timeout_s + 60}s",
                    )
                exit_code = proc.returncode
                stdout_f.close()
                stderr_f.close()
                stdout = stdout_path.read_text()[-200000:]
                stderr = stderr_path.read_text()[-100000:]
            else:
                # Capture mode (legacy)
                proc = subprocess.run(
                    cmd,
                    cwd=str(workdir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s + 60,
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
