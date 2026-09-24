"""Secret lookup: environment variable first, then macOS Keychain, else None.

Never logs, prints, or returns a secret through anything but its return value. On non-macOS or
when the `security` tool is missing, the Keychain step is silently skipped.
"""

from __future__ import annotations

import subprocess
import sys


def secret(name_env: str, keychain_service: str) -> str | None:
    """Resolve a secret: `$name_env`, then the macOS Keychain entry for `keychain_service`."""
    value = _env(name_env)
    if value:
        return value
    if sys.platform != "darwin":
        return None
    return _keychain(keychain_service)


def _env(name_env: str) -> str | None:
    import os

    value = os.environ.get(name_env)
    return value.strip() if value and value.strip() else None


def _keychain(service: str) -> str | None:
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-a", "sam", "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None
