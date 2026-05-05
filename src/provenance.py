"""Lightweight run-provenance capture: git SHA, SLURM job ID, host, timestamp.

Stamped into every results.json so a future reader can answer
"what code produced this?" and "what scheduler run was this?".
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import subprocess


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def capture_provenance() -> dict:
    """Snapshot enough environment info to reproduce a run later."""
    return {
        "git_sha": _git(["git", "rev-parse", "HEAD"]),
        "git_branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--porcelain"])),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "hostname": socket.gethostname(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python_version": platform.python_version(),
    }
