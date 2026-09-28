#!/usr/bin/env python3
"""Sign Claude Code in to Hebb: a browser tab opens, you sign in, the key lands in ~/.hebb/key.

    python3 hebb_login.py            sign in and wait here until it is done
    python3 hebb_login.py --start    open the sign-in and return at once; a background helper
                                     finishes the job (what /hebb:login and the plugin use, because
                                     Claude Code's tools cannot sit waiting on a browser)
    python3 hebb_login.py --device   no browser on this machine (SSH, a container): print a link
                                     and a code to open on any other device
    python3 hebb_login.py --switch   sign in again, as someone else
    python3 hebb_login.py --status   say who this machine is signed in as
    python3 hebb_login.py --logout   forget the key on this machine

WHY A FILE. The hooks and the MCP server both read ~/.hebb/key, and both start fresh on every use,
so the moment this file exists everything is connected: no restart, no config to edit, no key to
paste. The key it holds is an ordinary Hebb API key named "Connected: Claude Code on <machine>",
so it shows up in the dashboard and revoking it there disconnects this machine.

HOW. The browser flow is OAuth with PKCE against a one-shot server on 127.0.0.1 that exists only
until the redirect arrives (the way `gh auth login --web` works). Where there is no browser it
falls back to the device flow (RFC 8628). Standard library only, like the rest of the plugin.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

API = os.environ.get("HEBB_BASE", "https://hebb-site.pages.dev/v1").rstrip("/")
ORIGIN = API[:-3] if API.endswith("/v1") else API
KEY_FILE = os.path.expanduser("~/.hebb/key")
CLIENT_ID = "hebb-cli"
WAIT_S = 600


def machine() -> str:
    return (socket.gethostname().split(".")[0] or "this machine")[:40]


def call(url: str, data: dict | None = None, key: str | None = None):
    """POST a form when `data` is given, else GET. Returns (status, parsed JSON or {})."""
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    headers = {"User-Agent": "hebb-login/1", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except ValueError:
            return e.code, {}
    except Exception as e:                                        # offline, DNS, timeout
        return 0, {"error_description": f"could not reach {ORIGIN}: {e}"}


def read_key() -> str:
    k = os.environ.get("HEBB_KEY", "").strip()
    if k:
        return k
    try:
        with open(KEY_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def save_key(key: str) -> None:
    # Written to a temporary file created 600 and then renamed over the old one, so the key is never
    # readable by anyone else, not even for the moment between creating the file and chmod-ing it.
    os.makedirs(os.path.dirname(KEY_FILE), mode=0o700, exist_ok=True)
    tmp = KEY_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key.strip() + "\n")
    os.replace(tmp, KEY_FILE)
    os.chmod(KEY_FILE, 0o600)


def who(key: str):
    """(True, email) if the key works; (False, None) if Hebb rejected it; (None, None) if offline."""
    s, d = call(f"{API}/auth/me", key=key)
    if s == 200:
        return True, (d.get("account") or {}).get("email")
    return (False, None) if s in (401, 403) else (None, None)


def has_browser() -> bool:
    if os.environ.get("BROWSER"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    # Linux with no display: webbrowser would fall back to a text browser (lynx, w3m) that takes
    # over the terminal, or to nothing at all. The device flow is the right answer there.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def detach_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": 0x00000008 | 0x00000200}           # DETACHED_PROCESS | NEW_PROCESS_GROUP
    return {"start_new_session": True}


# ---------------------------------------------------------------------- browser flow

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hebb</title><body style="margin:0;min-height:100vh;display:grid;place-items:center;background:#0A0E17;color:#F4F1EA;
font:16px/1.5 system-ui,sans-serif"><main style="max-width:420px;padding:24px;text-align:center">
<p style="font:700 13px monospace;letter-spacing:.2em;color:#F2A93B">HEBB</p><h1 style="font-size:22px">{title}</h1>
<p style="color:#A9B4C6">{body}</p></main></body></html>"""


def pkce():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def authorize_url(port: int, challenge: str, state: str) -> str:
    return f"{ORIGIN}/oauth/authorize?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": CLIENT_ID, "redirect_uri": f"http://127.0.0.1:{port}/callback",
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        "resource": f"{ORIGIN}/v1/mcp", "device_name": machine()})


class Callback:
    """The one-shot server. Answers exactly one valid redirect, then stops."""

    def __init__(self, verifier: str, state: str):
        self.verifier, self.state, self.result = verifier, state, None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):                              # keep the terminal clean
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                if u.path != "/callback" or not secrets.compare_digest(q.get("state", ""), outer.state):
                    return self.reply(400, "That link is not for this sign-in", "Nothing was changed. You can close this tab.")
                if q.get("error"):
                    outer.result = ("denied", None)
                    return self.reply(200, "Not connected", "Sign-in was cancelled. You can close this tab.")
                s, d = call(f"{ORIGIN}/oauth/token", {
                    "grant_type": "authorization_code", "code": q.get("code", ""), "client_id": CLIENT_ID,
                    "redirect_uri": f"http://127.0.0.1:{outer.port}/callback", "code_verifier": outer.verifier})
                key = d.get("access_token", "")
                if s != 200 or not key.startswith("hebb_live_"):
                    outer.result = ("failed", d.get("error_description") or f"HTTP {s}")
                    return self.reply(200, "Not connected", "Hebb did not accept the sign-in. Run /hebb:login again.")
                save_key(key)
                outer.result = ("ok", None)
                self.reply(200, "You're connected", "Go back to Claude Code. Hebb is learning from your sessions now. You can close this tab.")

            def reply(self, status, title, body):
                html = PAGE.format(title=title, body=body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.server.timeout = 1
        self.port = self.server.server_address[1]

    def wait(self, seconds: float = WAIT_S):
        deadline = time.time() + seconds
        while self.result is None and time.time() < deadline:
            self.server.handle_request()
        self.server.server_close()
        return self.result or ("timeout", None)


def serve_loopback_child() -> int:
    """Background helper: reads its secrets from stdin (never argv, which other users can see),
    reports its port on stdout, then lets go of the terminal and waits for the browser."""
    job = json.loads(sys.stdin.readline())
    cb = Callback(job["verifier"], job["state"])
    print(json.dumps({"port": cb.port}), flush=True)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    cb.wait()
    return 0


def browser_login(background: bool) -> int:
    verifier, challenge = pkce()
    state = secrets.token_urlsafe(16)
    if background:
        try:
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve-loopback"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, **detach_kwargs())
            child.stdin.write(json.dumps({"verifier": verifier, "state": state}) + "\n")
            child.stdin.close()
            port = json.loads(child.stdout.readline())["port"]
            child.stdout.close()
        except (OSError, ValueError, KeyError):
            # No local port to listen on (a locked-down machine): a link and a code still work.
            return device_login(background)
        url = authorize_url(port, challenge, state)
        opened = webbrowser.open(url)
        print(("Opened your browser to sign in to Hebb." if opened else "Open this link to sign in to Hebb:") +
              f"\n{url}\nOnce you sign in, Claude Code is connected. Nothing to restart.")
        return 0

    cb = Callback(verifier, state)
    url = authorize_url(cb.port, challenge, state)
    print(f"Opening your browser to sign in to Hebb. If it does not open, visit:\n{url}", flush=True)
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    status, detail = cb.wait()
    return finish(status, detail)


# ---------------------------------------------------------------------- device flow

def poll_device(job: dict) -> tuple[str, str | None]:
    interval = max(1, int(job.get("interval", 5)))
    deadline = time.time() + min(WAIT_S, int(job.get("expires_in", WAIT_S)))
    while time.time() < deadline:
        time.sleep(interval)
        s, d = call(f"{ORIGIN}/oauth/token", {"grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                              "device_code": job["device_code"], "client_id": CLIENT_ID})
        if s == 200 and str(d.get("access_token", "")).startswith("hebb_live_"):
            save_key(d["access_token"])
            return "ok", None
        err = d.get("error")
        if err == "slow_down":
            interval += 5
        elif err != "authorization_pending" and s != 0:
            return "failed", d.get("error_description") or err or f"HTTP {s}"
    return "timeout", None


def poll_device_child() -> int:
    job = json.loads(sys.stdin.readline())
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    poll_device(job)
    return 0


def device_login(background: bool) -> int:
    s, d = call(f"{ORIGIN}/oauth/device_authorization", {"client_id": CLIENT_ID, "device_name": machine()})
    if s != 200 or "device_code" not in d:
        print(f"Could not start the sign-in: {d.get('error_description') or s}")
        return 1
    link = d.get("verification_uri_complete") or d["verification_uri"]
    print(f"To connect Hebb, open this link on any device and sign in:\n{link}\n"
          f"Check the code matches: {d['user_code']}", flush=True)
    job = {k: d[k] for k in ("device_code", "interval", "expires_in") if k in d}
    if background:
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--poll-device"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 text=True, **detach_kwargs())
        child.stdin.write(json.dumps(job) + "\n")
        child.stdin.close()
        print("Claude Code connects by itself as soon as you finish. Nothing to restart.")
        return 0
    return finish(*poll_device(job))


# ---------------------------------------------------------------------- entry

def finish(status: str, detail: str | None) -> int:
    if status == "ok":
        ok, email = who(read_key())
        print(f"Connected{' as ' + email if email else ''}. Hebb is now learning from your Claude Code sessions.")
        return 0
    print({"denied": "Sign-in was cancelled. Nothing was changed.",
           "timeout": "Timed out waiting for the sign-in. Run it again when you are ready."}.get(
        status, f"Sign-in failed: {detail}"))
    return 1


def main(argv: list[str]) -> int:
    if "--serve-loopback" in argv:
        return serve_loopback_child()
    if "--poll-device" in argv:
        return poll_device_child()

    if "--logout" in argv:
        try:
            os.remove(KEY_FILE)
            print(f"Signed out on this machine ({KEY_FILE} removed). Revoke the key in the dashboard to "
                  f"cut it off everywhere: {ORIGIN}/dashboard")
        except FileNotFoundError:
            print("Not signed in on this machine.")
        return 0

    key = read_key()
    if key and "--switch" not in argv:
        ok, email = who(key)
        if ok:
            print(f"Already connected to Hebb{' as ' + email if email else ''}. "
                  f"To use a different account: python3 {os.path.abspath(__file__)} --switch")
            return 0
        if ok is None and "--status" not in argv:
            print(f"Could not reach Hebb at {ORIGIN} to check the saved key. Try again when online.")
            return 1
        if "--status" in argv:
            print("Offline: cannot check the saved key." if ok is None else
                  "The saved key was revoked or is no longer valid. Run /hebb:login to connect again.")
            return 1
    elif "--status" in argv:
        print("Not signed in to Hebb on this machine. Run /hebb:login to connect.")
        return 1

    background = "--start" in argv
    if "--device" in argv or not has_browser():
        return device_login(background)
    return browser_login(background)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
