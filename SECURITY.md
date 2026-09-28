# Security

The hooks in this plugin see the shell commands Claude Code runs, so it matters exactly what they do.
Everything that runs on your computer is in this repository, in plain Python with no dependencies:

- `hooks/hebb_hooks.py`: the hooks (learn from failed commands, fix commands, refuse banned ones)
- `mcp/hebb_mcp.py` and `mcp/run.py`: the memory tools Claude uses
- `bin/hebb_login.py` and `bin/hebb_session.py`: sign-in and the start-of-session check

What leaves your computer is described in the README and in the
[privacy policy](https://hebb-site.pages.dev/privacy.html).

## Reporting a problem

If you find a security issue, email **neilgilani@gmail.com** with "Hebb security" in the subject
instead of opening a public issue. You will get a reply within a few days.
