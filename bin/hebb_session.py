#!/usr/bin/env python3
"""SessionStart hook for the Hebb plugin: make sure this machine is signed in, and say so if not.

On the first session after the plugin is installed there is no key yet, so this opens the sign-in
(the browser, or a link and code where there is no browser) and tells the user in one line. That
is the whole setup: install, sign in once, done. It does this at most once a day and only when a
session starts fresh, so ignoring it is not punished with a browser tab every time.
HEBB_NO_AUTO_LOGIN=1 turns the automatic part off; /hebb:login always works.

It also copies the login script to ~/.hebb/hebb_login.py, which is how /hebb:login finds it: a
skill's text cannot name the plugin's own folder, but it can name a file in the user's home.

Never blocks and never fails the session: every error is swallowed and the hook prints nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~/.hebb")
KEY_FILE = os.path.join(HOME_DIR, "key")
LOGIN = os.path.join(HOME_DIR, "hebb_login.py")
STAMP = os.path.join(os.path.expanduser(os.environ.get("HEBB_HOOK_STATE", "~/.hebb/state")), "auto-login-at")
AUTO_EVERY_S = 24 * 3600


def has_key() -> bool:
    if os.environ.get("HEBB_KEY", "").strip():
        return True
    try:
        with open(KEY_FILE) as f:
            return bool(f.read().strip())
    except OSError:
        return False


def install_login_script() -> None:
    src = os.path.join(HERE, "hebb_login.py")
    try:
        with open(src, "rb") as a:
            new = a.read()
        try:
            with open(LOGIN, "rb") as b:
                if b.read() == new:
                    return
        except OSError:
            pass
        os.makedirs(HOME_DIR, mode=0o700, exist_ok=True)
        shutil.copyfile(src, LOGIN)
    except OSError:
        pass


def duplicate_install() -> str:
    """The plugin replaces the old copy-paste setup. Running both means every hook fires twice
    (two injections, every lesson learned twice) and every tool is listed twice."""
    found = []
    try:
        with open(os.path.expanduser("~/.claude/settings.json")) as f:
            hooks = json.load(f).get("hooks") or {}
        if "hebb_hooks.py" in json.dumps(hooks):
            found.append("the Hebb hooks in ~/.claude/settings.json")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        with open(os.path.expanduser("~/.claude.json")) as f:
            servers = json.load(f).get("mcpServers") or {}
        if "hebb_mcp.py" in json.dumps(servers):
            found.append("the `hebb` MCP server added with `claude mcp add`")
    except (OSError, ValueError, AttributeError):
        pass
    if not found:
        return ""
    return ("Hebb is installed twice: the plugin and " + " and ".join(found) + ". Remove the old setup "
            "(delete those entries, or run `claude mcp remove hebb`) so hooks and tools do not run twice.")


def auto_login_due(source: str) -> bool:
    if os.environ.get("HEBB_NO_AUTO_LOGIN") or source not in ("startup", ""):
        return False
    try:
        return time.time() - os.path.getmtime(STAMP) > AUTO_EVERY_S
    except OSError:
        return True


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        event = {}
    source = str(event.get("source") or "")
    install_login_script()

    notes = []
    dup = duplicate_install()
    if dup:
        notes.append(dup)

    context = ""
    if not has_key():
        context = ("Hebb (memory for Claude Code) is installed but this machine is not signed in, so it "
                   "cannot learn or recall anything yet. If the user asks about Hebb or memory, tell them "
                   "to type /hebb:login.")
        if auto_login_due(source):
            try:
                os.makedirs(os.path.dirname(STAMP), exist_ok=True)
                with open(STAMP, "w") as f:
                    f.write(str(int(time.time())))
                r = subprocess.run([sys.executable, LOGIN, "--start"], capture_output=True, text=True, timeout=8)
                notes.append("Hebb: finish connecting by signing in. " + r.stdout.strip())
            except Exception:
                notes.append("Hebb is installed but not signed in. Type /hebb:login to connect.")
        else:
            notes.append("Hebb is installed but not signed in. Type /hebb:login to connect.")

    out = {}
    if notes:
        out["systemMessage"] = "\n".join(notes)
    if context:
        out["hookSpecificOutput"] = {"hookEventName": "SessionStart", "additionalContext": context}
    if out:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        raise SystemExit(0)
