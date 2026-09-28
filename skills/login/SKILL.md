---
description: Sign this machine in to Hebb (opens the browser once). Use when the user asks to connect, log in to, sign in to, or switch accounts on Hebb, or when a Hebb tool says it is not signed in.
allowed-tools: Bash(python3 ~/.hebb/hebb_login.py *)
---

Connect Claude Code to the user's Hebb memory.

Run this with the Bash tool, exactly:

```bash
python3 ~/.hebb/hebb_login.py --start
```

Add `--switch` if the user wants a different account, or `--device` if they say the browser did not
open or they are on a remote machine (SSH, a container). Use `--status` to answer "am I connected?"
and `--logout` if they ask to sign out.

The command returns at once. Relay what it printed in one or two plain sentences, including any link
and code word for word. The user signs in in the browser; Hebb connects by itself when they finish,
with nothing to restart. Do not wait, poll, or run it again unless they ask.

If `~/.hebb/hebb_login.py` does not exist, the Hebb plugin has not started yet in this session: ask
the user to restart Claude Code once, then try again.
