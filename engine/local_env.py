"""Read MASSIVE_API_KEY (and other settings) from the environment or the project's .env.local.

The key is never logged or printed. The server loads .env.local the same way (server.js).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_env(path=None):
    """Copy KEY=VALUE pairs from .env.local into os.environ without overriding existing values."""
    env_file = Path(path) if path else PROJECT_ROOT / ".env.local"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def env_value(name, default=""):
    """Return a local setting. Never log or print the value (keys, tokens)."""
    load_local_env()
    return (os.environ.get(name) or default).strip()


def massive_key():
    key = env_value("MASSIVE_API_KEY")
    if not key:
        raise ValueError("Set MASSIVE_API_KEY in the environment or .env.local.")
    return key
