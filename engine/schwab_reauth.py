"""Re-authorize the Schwab API token with schwab-py, the same call Robert used interactively:

    auth.easy_client(api_key, app_secret, "https://127.0.0.1:8182", "schwab_token.json")

Run by the dashboard's "Click here to reauthorize" button (or by hand). It needs schwab-py, so it must run with a
Python that has it installed (the dashboard probes SCHWAB_AUTH_PYTHON, the repo .venv, then the "py -3" launcher).
The existing token file is moved aside first so a fresh browser login always happens; it is restored if the login
fails. A browser window opens: log in to Schwab and approve, then accept the 127.0.0.1 certificate warning if shown.
App key and secret come from SCHWAB_APP_KEY / SCHWAB_APP_SECRET (environment or .env.local) and are never printed.
After login only a market-data quote (SPY) is requested to verify the token; no account or order endpoints.

  python schwab_reauth.py [--timeout 300]
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from local_env import PROJECT_ROOT, load_local_env


def say(message):
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)
    try:
        from structured_log import get_logger
        get_logger("schwab_reauth").log("error" if message.startswith("ERROR") else "info", message)
    except Exception:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for the browser login")
    args = parser.parse_args(argv)
    load_local_env()
    key = os.environ.get("SCHWAB_APP_KEY", "").strip()
    secret = os.environ.get("SCHWAB_APP_SECRET", "").strip()
    if not key or not secret:
        say("ERROR: SCHWAB_APP_KEY and SCHWAB_APP_SECRET are not set (environment or .env.local).")
        return 3
    try:
        from schwab import auth
    except ImportError:
        say(f"ERROR: schwab-py is not installed for {sys.executable}. Set SCHWAB_AUTH_PYTHON to a Python that has it.")
        return 4
    callback = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
    token_path = Path(os.environ.get("SCHWAB_TOKEN_PATH") or PROJECT_ROOT / "schwab_token.json")
    backup = None
    if token_path.is_file():
        backup = token_path.with_name(f"{token_path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        token_path.rename(backup)
        say(f"Moved the old token aside ({backup.name}).")
    say("Opening the Schwab login in your browser. Log in, approve the app, then accept the 127.0.0.1 "
        "certificate warning if the browser shows one.")
    started = time.time()
    try:
        client = auth.easy_client(key, secret, callback, str(token_path), interactive=False,
                                  callback_timeout=args.timeout)
    except Exception as error:   # restore the previous token so nothing is lost
        say(f"ERROR: login did not complete: {type(error).__name__}: {error}")
        if backup and backup.is_file() and not token_path.is_file():
            backup.rename(token_path)
            say("Restored the previous token file.")
        return 1
    say(f"New token saved to {token_path.name} after {time.time() - started:.0f}s.")
    try:
        response = client.get_quote("SPY")
        say(f"Verified with a market-data quote request: HTTP {response.status_code}.")
    except Exception as error:
        say(f"WARNING: token saved but the test quote failed: {error}")
    say("DONE")
    return 0


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("schwab_reauth"):
        sys.exit(main())
