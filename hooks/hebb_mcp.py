#!/usr/bin/env python3
"""Hebb's MCP server, as the Claude Code plugin runs it.

    hebb_mcp.py <plugin data folder> <key, or nothing>

WITH A KEY it is a pipe to the Hebb service's own MCP endpoint: every message from Claude Code is
posted there and every answer handed back, so the tools are the service's (remember, recall,
list_memories, share, forget, never_run) and nothing here has to keep up with them.

WITHOUT A KEY it answers the core tools itself from one file on this machine, the same file the
hooks read and write, so the plugin works the moment it is installed, with no account. A key the
service rejects falls back to the same local mode rather than leaving Claude with broken tools.

JSON-RPC 2.0 over stdin/stdout, one message per line. Standard library only. stdout is the protocol
channel: nothing else is ever printed to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request

try:
    import fcntl
except ImportError:
    fcntl = None

URL = os.environ.get("HEBB_MCP_URL", "https://hebb-site.pages.dev/v1/mcp")
VERSION = "0.6.0"
DASHBOARD = "https://hebb-site.pages.dev/dashboard.html"


def arg(i):
    v = sys.argv[i].strip() if len(sys.argv) > i else ""
    return "" if v.startswith("${") else v       # an unset option arrives as its placeholder


KEY = arg(2)
DATA = arg(1) or os.environ.get("CLAUDE_PLUGIN_DATA", "")
# One folder for every client, the same one the hooks use, so Claude Code and Codex share one memory.
# DATA is only where Claude Code kept memories before 0.8, read once to move them over.
STATE = os.path.expanduser(os.environ.get("HEBB_HOOK_STATE") or "~/.hebb/state")
STORE = os.path.join(STATE, "memories.json")

# The hooks' rule, kept identical: anything that looks like a credential is never stored.
SECRETISH = re.compile(
    r"sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY|Bearer\s+[A-Za-z0-9._\-]{16,}|"
    r"(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S{6,}", re.I)


# ---------------------------------------------------------------------- history
#
# The same tamper-evident log the hooks write (hebb_hooks.py, `audit`): one line per event, each
# carrying the SHA-256 of the one before. Keep the two `audit` functions identical.

GENESIS = "0" * 64
HISTORY = os.path.join(STATE, "history.log")


def host():
    try:
        return socket.gethostname().split(".")[0]
    except Exception:
        return "this machine"


def audit(event, slot="", detail=""):
    try:
        os.makedirs(STATE, exist_ok=True)
        with open(HISTORY, "a+") as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = [l for l in f.read().splitlines() if l.strip()]
            prev = json.loads(tail[-1])["hash"] if tail else GENESIS
            e = {"t": round(time.time(), 3), "event": event, "slot": str(slot)[:200],
                 "detail": str(detail)[:300], "host": host(), "prev": prev}
            e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
            f.write(json.dumps(e, sort_keys=True) + "\n")
    except Exception:
        pass


def read_history():
    """(entries, index of the first entry that breaks the chain or None)."""
    try:
        with open(HISTORY) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except FileNotFoundError:
        return [], None
    entries, prev, broken = [], GENESIS, None
    for i, line in enumerate(lines):
        try:
            e = json.loads(line)
            h = e.pop("hash")
            ok = e.get("prev") == prev and hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() == h
        except Exception:
            e, h, ok = {"event": "unreadable", "slot": "", "detail": "", "t": 0}, "", False
        if not ok and broken is None:
            broken = i
        entries.append(e)
        prev = h
    return entries, broken


def history(limit=20):
    entries, broken = read_history()
    if not entries:
        return text("No history yet. Hebb records here everything it learns, blocks, corrects or forgets.")
    first = time.strftime("%Y-%m-%d", time.localtime(entries[0].get("t", 0)))
    seal = (f"Record intact: {len(entries)} entries, each sealed to the one before, since {first}."
            if broken is None else
            f"Record BROKEN at entry {broken + 1} of {len(entries)}: an entry was edited, removed or "
            "inserted there. Entries from that point on cannot be trusted.")
    week = [e for e in entries if e.get("t", 0) > time.time() - 7 * 86400]
    counts = {k: sum(e.get("event") == k for e in week) for k in ("blocked", "corrected", "learned")}
    summary = (f"Last 7 days: {counts['blocked']} banned commands blocked, {counts['corrected']} known "
               f"mistakes corrected before they ran, {counts['learned']} lessons learned.")
    rows = []
    for e in entries[-max(1, min(int(limit or 20), 200)):]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("t", 0)))
        rows.append(f"{when}  {e.get('event', '?'):<10} {e.get('slot', '')}"
                    + (f"  ({e['detail']})" if e.get("detail") else ""))
    return text("\n".join([seal, summary, ""] + rows))


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------- with a key

def forward(line, mid, patch=None):
    """Post one message to the service and hand back whatever it answers. Returns False when the
    key was refused, so the caller can answer from the local file instead."""
    global KEY
    req = urllib.request.Request(URL, data=line.encode(), method="POST", headers={
        "Authorization": f"Bearer {KEY}", "content-type": "application/json",
        "accept": "application/json, text/event-stream", "user-agent": f"hebb-plugin/{VERSION}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            KEY = ""
            return False
        body, err = "", f"The Hebb service answered {e.code}."
    except Exception:
        body, err = "", "The Hebb service could not be reached."
    else:
        err = ""
    if err:
        if mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": err}})
        return True
    # Plain JSON, or server-sent events whose data lines are the messages.
    chunks = [body] if body.lstrip().startswith(("{", "[")) else \
        [l[5:].strip() for l in body.splitlines() if l.startswith("data:")]
    for c in chunks:
        try:
            m = json.loads(c)
        except Exception:
            continue
        for one in (m if isinstance(m, list) else [m]):
            send(patch(one) if patch else one)
    return True


def with_history(msg):
    tools = (msg.get("result") or {}).get("tools")
    if isinstance(tools, list) and not any(t.get("name") == "history" for t in tools):
        tools.append(HISTORY_TOOL)
    return msg


SYNCED = {"remember": "remembered", "forget": "forgot", "never_run": "banned", "share": "shared"}


# ---------------------------------------------------------------------- without a key

class memory_lock:
    """Same lock file as the hooks: a read-modify-write here never undoes one made there."""
    def __enter__(self):
        os.makedirs(STATE, exist_ok=True)
        self.f = open(os.path.join(STATE, "memories.lock"), "a")
        if fcntl:
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.f.close()


def migrate(old_state):
    """Move pre-0.8 Claude Code memories into the shared folder once (same rules as the hooks)."""
    try:
        old = os.path.expanduser(old_state or "")
        if not old or not os.path.isdir(old) or os.path.realpath(old) == os.path.realpath(STATE):
            return
        marker = os.path.join(old, "moved-to-shared")
        if os.path.exists(marker):
            return
        with memory_lock():
            by = {r.get("slot"): r for r in load() if isinstance(r, dict)}
            try:
                with open(os.path.join(old, "memories.json")) as f:
                    came = json.load(f)
            except Exception:
                came = []
            for r in came if isinstance(came, list) else []:
                cur = by.get(r.get("slot")) if isinstance(r, dict) else None
                if isinstance(r, dict) and (cur is None or float(r.get("updated_at") or 0) > float(cur.get("updated_at") or 0)):
                    by[r.get("slot")] = r
            save(list(by.values()))
        old_log = os.path.join(old, "history.log")
        if os.path.exists(old_log):
            shutil.copyfile(old_log, HISTORY if not os.path.exists(HISTORY)
                            else os.path.join(STATE, "history-before-sharing.log"))
        with open(marker, "w") as f:
            f.write(STATE + "\n")
    except Exception:
        pass


def load():
    try:
        with open(STORE) as f:
            rules = json.load(f)
        return rules if isinstance(rules, list) else []
    except Exception:
        return []


def save(rules):
    os.makedirs(STATE, exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rules, f)
    os.replace(tmp, STORE)


def put(slot, value):
    slot = slot.strip()[:180]
    with memory_lock():
        rules = [r for r in load() if r.get("slot") != slot]
        rules.append({"slot": slot, "value": value.strip(), "updated_at": time.time()})
        save(rules)


def words(text):
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.split(r"[^a-z0-9.\-_/]+", (text or "").lower()) if len(w) > 2}


def line(r):
    return f"- {r.get('slot')}: {r.get('value')}"


def text(t, error=False):
    return {"content": [{"type": "text", "text": t}], **({"isError": True} if error else {})}


def call(name, a):
    if name == "remember":
        slot, value = str(a.get("name", "")), str(a.get("value", ""))
        if not slot.strip() or not value.strip():
            return text("remember needs a name and a value.", True)
        if SECRETISH.search(slot + " " + value):
            return text("Not stored: it looks like a credential, and Hebb never stores those.", True)
        put(slot, value)
        audit("remembered", slot.strip(), value.strip())
        return text(f"Remembered on this machine: {slot.strip()} -> {value.strip()}")
    if name == "recall":
        rules = load()
        want = str(a.get("name", "")).strip()
        hit = [r for r in rules if want and r.get("slot") == want]
        if not hit:
            q = words(want + " " + str(a.get("question", "")))
            scored = sorted(((len(q & words(f"{r.get('slot')} {r.get('value')}")), r) for r in rules),
                            key=lambda x: -x[0])
            hit = [r for s, r in scored if s][:5]
        return text("\n".join(line(r) for r in hit) if hit else "Nothing is stored for that.")
    if name == "list_memories":
        rules = load()
        return text("\n".join(line(r) for r in rules) if rules else "No memories yet.")
    if name == "forget":
        slot = str(a.get("name", "")).strip()
        with memory_lock():
            rules = load()
            keep = [r for r in rules if r.get("slot") != slot]
            if len(keep) == len(rules):
                return text(f"Nothing is stored under {slot!r}.")
            save(keep)
        audit("forgot", slot)
        return text(f"Forgot {slot!r}.")
    if name == "never_run":
        cmd, why = str(a.get("command", "")).strip(), str(a.get("reason", "")).strip()
        if not cmd:
            return text("never_run needs the command to block.", True)
        put("never: " + cmd, why or "the user asked never to run this")
        audit("banned", "never: " + cmd, why)
        return text(f"Blocked on this machine: {cmd}")
    if name == "history":
        return history(a.get("limit", 20))
    return text(f"Unknown tool {name!r}.", True)


def schema(props, required=()):
    return {"type": "object", "properties": {k: {"type": v[0], "description": v[1]} for k, v in props.items()},
            "required": list(required)}


HISTORY_TOOL = {
    "name": "history", "description": "Show Hebb's tamper-evident history on this machine: what it "
    "learned, blocked, corrected and forgot, with weekly counts, and whether the record is intact.",
    "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer",
                    "description": "how many recent entries to show (default 20)"}}, "required": []}}

TOOLS = [
    {"name": "remember", "description": "Save a fact, preference or decision the user states, so it is "
     "known in every later session. Use a short stable name. Never store secrets.",
     "inputSchema": schema({"name": ("string", "short stable name"), "value": ("string", "the fact")},
                           ("name", "value"))},
    {"name": "recall", "description": "Look up saved memories by exact name or by a question. Check here "
     "before asking the user something they may have said before.",
     "inputSchema": schema({"name": ("string", "exact name, if known"),
                            "question": ("string", "what you want to know")})},
    {"name": "list_memories", "description": "List everything saved.", "inputSchema": schema({})},
    {"name": "forget", "description": "Delete a saved memory by name, when the user asks or it is wrong.",
     "inputSchema": schema({"name": ("string", "exact name")}, ("name",))},
    {"name": "never_run", "description": "Permanently refuse a shell command. Only when the user asks, "
     "or right after a command caused damage and they confirm. Never infer this.",
     "inputSchema": schema({"command": ("string", "the command text to block"),
                            "reason": ("string", "why; shown when it is blocked")}, ("command", "reason"))},
]

TOOLS.append(HISTORY_TOOL)

INSTRUCTIONS = ("Hebb is the user's memory for Claude Code, in local mode: memories are kept on this "
                "machine only. Recall before asking the user something they may have told you; "
                "remember what they state without being asked. Never store secrets. To share memories "
                f"with a team or across computers, the user can add a free key from {DASHBOARD}.")


def local(msg):
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return                                   # a notification: nothing to answer
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion") or "2025-06-18",
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "hebb", "version": VERSION},
                  "instructions": INSTRUCTIONS}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            result = call(params.get("name"), params.get("arguments") or {})
        except Exception as e:
            result = text(f"Hebb could not do that: {e}", True)
    else:
        return send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"No method {method}"}})
    send({"jsonrpc": "2.0", "id": mid, "result": result})


def main():
    if DATA and not os.environ.get("HEBB_HOOK_STATE"):
        migrate(os.path.join(DATA, "state"))
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        one = msg if isinstance(msg, dict) else {}
        name = (one.get("params") or {}).get("name") if one.get("method") == "tools/call" else None
        if KEY and name != "history":
            patch = with_history if one.get("method") == "tools/list" else None
            if forward(raw, one.get("id"), patch):
                if name in SYNCED:
                    a = (one.get("params") or {}).get("arguments") or {}
                    audit(SYNCED[name], str(a.get("name") or a.get("command") or ""), "sent to the Hebb service")
                continue
        for m in (msg if isinstance(msg, list) else [msg]):
            if isinstance(m, dict):
                local(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
