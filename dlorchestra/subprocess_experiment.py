"""
Framework-agnostic experiment base: run stages as subprocesses.

For tools that aren't PyTorch (TensorFlow, a different conda env, a
compiled binary), mixing runtimes into one Python process is fragile.
The contract instead: run this command (optionally in a named conda env),
collect results from a directory. No foreign-framework import ever touches
this process.

Subclasses implement ``run(config)`` using ``run_stage()``; there's no fixed
fold-loop skeleton here (unlike ``TorchExperiment``) because subprocess-based
tools rarely share one.
"""

import logging
import subprocess
from pathlib import Path

from .experiment import BaseExperiment

logger = logging.getLogger(__name__)

_CONDA_INSTALL_DIRS = ("miniforge3", "miniconda3", "anaconda3")


class SubprocessExperiment(BaseExperiment):
    """Base for experiments that shell out to another environment/tool."""

    @staticmethod
    def find_conda() -> str:
        """Locate the conda executable without relying on a sourced shell hook."""
        for name in _CONDA_INSTALL_DIRS:
            for subdir in ("condabin", "bin"):
                candidate = Path.home() / name / subdir / "conda"
                if candidate.exists():
                    return str(candidate)
        return "conda"  # assume it's already on PATH

    def run_stage(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        conda_env: str | None = None,
    ) -> None:
        """Run one pipeline stage as a subprocess, raising on failure.

        When ``conda_env`` is given, the command runs via
        ``conda run -n <conda_env>`` so it executes in that env's
        interpreter/dependencies without activating it in this process.
        """
        if conda_env:
            cmd = [self.find_conda(), "run", "-n", conda_env, "--no-capture-output", *cmd]
        logger.info("$ %s%s", " ".join(cmd), f"  (cwd={cwd})" if cwd else "")
        subprocess.run(cmd, cwd=cwd, check=True)
