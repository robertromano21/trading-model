"""Structured JSON-lines logging for the engine scripts, in the same format as the dashboard's winston logs.

Each line: {"timestamp": ISO-UTC, "level": "info|warn|error|debug", "message": ..., "source": "engine:<script>",
"pid": ..., "requestId"/"jobId" (when started by the dashboard), ...fields}. Files: logs/engine.YYYY-MM-DD.log (UTC
date; same folder and naming as server.*.log so the dashboard's log viewer reads them). A file over 10 MB continues in
engine.YYYY-MM-DD.log.1, .2, ...; files older than 14 days are deleted. Secrets (API keys, tokens, Schwab account
numbers) are redacted before writing.

    from structured_log import get_logger, log_run
    log = get_logger("paper_trader")
    log.info("run finished", equity=10_000)
"""

import json
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.environ.get("TRADING_LOG_DIR") or PROJECT_ROOT / "logs")
MAX_BYTES = 10 * 1024 * 1024
RETENTION_DAYS = 14
MAX_MESSAGE = 16384
SECRET_ENV = ("MASSIVE_API_KEY", "SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "OPENAI_API_KEY", "XAI_API_KEY", "GROQ_API_KEY",
              "ANTHROPIC_API_KEY")
PATTERNS = [
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[redacted]"),
    (re.compile(r"(?i)((?:api[_-]?key|apikey|access_token|refresh_token|id_token|client_secret|code|token|secret)[\"']?\s*[:=]\s*[\"']?)[^\"'&\s,}]{6,}"), r"\1[redacted]"),
    (re.compile(r"\b(sk|xai|gsk)-[A-Za-z0-9_-]{10,}"), r"\1-[redacted]"),
    (re.compile(r"(?i)(account(?:number|_number|hash|id)?[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9]{4,}([A-Za-z0-9]{4})\b"), r"\1…\2"),
]
SECRET_KEY = re.compile(r"(?i)(token|secret|password|api_?key|authorization|cookie|hash_?value|account_?number)")
LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


def redact(text):
    text = str(text)
    for name in SECRET_ENV:
        value = os.environ.get(name, "").strip()
        if len(value) >= 6:
            text = text.replace(value, "[redacted]")
    for pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    if len(text) > MAX_MESSAGE:
        text = text[: MAX_MESSAGE - 2048] + f"\n[truncated {len(text)} chars]\n" + text[-2048:]
    return text


def _clean(value):
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): "[redacted]" if SECRET_KEY.search(str(k)) else _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value][:200]
    return redact(value)


_pruned = False


def _prune():
    global _pruned
    if _pruned:
        return
    _pruned = True
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).date().isoformat()
    for path in LOG_DIR.glob("engine.*.log*"):
        match = re.match(r"engine\.(\d{4}-\d{2}-\d{2})\.log", path.name)
        if match and match.group(1) < cutoff:
            try:
                path.unlink()
            except OSError:
                pass


def _target():
    day = datetime.now(timezone.utc).date().isoformat()
    path = LOG_DIR / f"engine.{day}.log"
    part = 0
    while path.exists() and path.stat().st_size > MAX_BYTES:
        part += 1
        path = LOG_DIR / f"engine.{day}.log.{part}"
    return path


class EngineLogger:
    def __init__(self, source, min_level=None):
        self.source = f"engine:{source}"
        self.min_level = LEVELS.get((min_level or os.environ.get("TRADING_LOG_LEVEL") or "info").lower(), 20)
        self.context = {k: v for k, v in {"requestId": os.environ.get("TM_REQUEST_ID"),
                                           "jobId": os.environ.get("TM_JOB_ID")}.items() if v}

    def log(self, level, message, **fields):
        if LEVELS.get(level, 20) < self.min_level:
            return
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                  "level": level, "message": redact(message), "source": self.source, "pid": os.getpid(),
                  **self.context, **_clean(fields)}
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _prune()
            with open(_target(), "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as error:   # logging must never break a run
            print(f"[engine log] {error}", file=sys.stderr)

    def debug(self, message, **fields): self.log("debug", message, **fields)
    def info(self, message, **fields): self.log("info", message, **fields)
    def warn(self, message, **fields): self.log("warn", message, **fields)
    def error(self, message, **fields): self.log("error", message, **fields)
    warning = warn


def get_logger(source):
    return EngineLogger(source)


@contextmanager
def log_run(source, command=None, **fields):
    """Log start/finish/failure (with traceback) of a script's main command."""
    log = get_logger(source)
    started = time.time()
    log.info("started", command=command or " ".join(sys.argv[1:]), **fields)
    try:
        yield log
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        (log.info if code == 0 else log.error)("exited", code=code, detail=None if isinstance(stop.code, int) else stop.code,
                                                seconds=round(time.time() - started, 2))
        raise
    except BaseException as error:
        log.error("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=12),
                  seconds=round(time.time() - started, 2))
        raise
    else:
        log.info("finished", seconds=round(time.time() - started, 2))
