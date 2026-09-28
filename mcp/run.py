#!/usr/bin/env python3
"""Starts the Hebb MCP server for the plugin, signed in or not.

hebb_mcp.py, the MCP server, reads HEBB_KEY once and exits if it is missing,
which is right for a hand-written config and wrong for a plugin: straight after installing there
is no key yet, and a server that failed to start stays failed until Claude Code restarts. So this
runs the same tools with the key read on every call, from HEBB_KEY or ~/.hebb/key. Before sign-in,
a call says how to sign in; the call after /hebb:login succeeds. No restart either way.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hebb_mcp as m  # noqa: E402

BASE = os.environ.get("HEBB_BASE", m.DEFAULT_BASE).strip() or m.DEFAULT_BASE
KEY_FILE = os.path.expanduser("~/.hebb/key")
NOT_SIGNED_IN = ("Hebb is not signed in on this machine yet, so nothing can be saved or recalled. "
                 "Ask the user to type /hebb:login (it opens a browser once), then try again.")


class Hebb(m.Hebb):
    """Notices a 401, so a revoked key gets told apart from an ordinary failure."""
    rejected = False

    def call(self, *a, **kw):
        s, d = super().call(*a, **kw)
        self.rejected = self.rejected or s == 401
        return s, d


def key() -> str:
    k = os.environ.get("HEBB_KEY", "").strip()
    if k:
        return k
    try:
        with open(KEY_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def main() -> int:
    m.log(f"serving {len(m.TOOLS)} tools against {BASE} ({'signed in' if key() else 'not signed in yet'})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue
        if method == "initialize":
            result = {"protocolVersion": m.PROTOCOL_VERSION, "capabilities": {"tools": {}},
                      "serverInfo": {"name": m.NAME, "version": m.VERSION}}
        elif method == "tools/list":
            result = {"tools": m.TOOLS}
        elif method == "tools/call":
            p = msg.get("params") or {}
            k = key()
            if not k:
                result = m.fail(NOT_SIGNED_IN)
            else:
                h = Hebb(k, BASE)
                try:
                    result = m.run_tool(h, p.get("name", ""), p.get("arguments") or {})
                except Exception as e:                           # never take the server down
                    result = m.fail(f"Unexpected error: {e}")
                if h.rejected:
                    result = m.fail("Hebb rejected this machine's key (it was revoked or replaced). "
                                    "Ask the user to type /hebb:login to connect again.")
        elif method == "ping":
            result = {}
        elif method in ("resources/list", "prompts/list"):
            result = {"resources": []} if method == "resources/list" else {"prompts": []}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": mid,
                              "error": {"code": -32601, "message": f"unknown method {method}"}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
