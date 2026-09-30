---
name: memory
description: Show and manage what Hebb has learned. Use when the user asks what Hebb knows or remembers, asks you to remember, forget, or never run something, asks whether Hebb is working, or when Hebb blocked or corrected a command. The user can also run it as /hebb:memory to see everything Hebb has learned.
---

# Hebb

Hebb learns from what actually happens in Claude Code. When a command fails and a different one
works, it keeps that as a lesson and stops the failing command the next time, handing you the one
that works. It also keeps what the user tells you to remember, and commands they have banned. The
tools come from the `hebb` MCP server: `list_memories`, `recall`, `remember`, `forget`, `never_run`,
and `share` when the user has connected a key.

## When the user runs /hebb:memory

1. Call `list_memories`.
2. Show the result in three short groups, one line each:
   - **Learned on this machine**: entries named `missing: ...` or `command: ...`
   - **Banned commands**: entries named `never: ...`, with the reason
   - **Things you told me**: everything else
3. If there is nothing yet, say so, and that Hebb learns by itself: the first time a command fails
   here and something else works, it keeps the fix.
4. End with two or three things they can say, for example "remember that we deploy with
   `make ship`", "never run `git push --force`", "forget the python lesson".

## When Hebb stops a command

- **"Not run: ... Run this instead: ..."** means Hebb has seen this exact mistake on this machine.
  Run the command it gives you, as given, and mention it in one line. Do not retry the original.
- **"Blocked by your saved rule ..."** means the user banned it. Do not reach the same result another
  way. Tell the user it was blocked and why, and that they can lift the ban by asking you to forget
  that rule.

## Remember, recall, forget

- When the user states a preference, a convention, a decision, or a fact about their setup, call
  `remember` without being asked. Use a short, stable name (`deploy command`, `test runner`) so a
  later update replaces it instead of adding a duplicate. Say what you saved in one line.
- Before asking the user something they may have told you before, call `recall` first.
- Never store passwords, keys, tokens or other secrets, or personal details about other people.
  Hebb refuses anything that looks like a credential.
- When a memory is wrong or out of date (a tool Hebb says is missing is now installed, a convention
  changed), call `forget` with its exact name. Use `list_memories` to find the name.

## Never run

Call `never_run` only when the user explicitly asks you to ban a command, or confirms it right after
a command did damage. Pass the exact command text and their reason. Never infer a ban on your own:
it becomes a hard block on every future session.

## Sharing with a team

`share` exists only when the user has added a Hebb key. If they ask to share something and the tool
is not there, Hebb is in local mode: memories stay on this computer. Tell them to get a free key
from https://hebb-site.pages.dev/dashboard.html and add it with `/plugin`, then pick hebb and
update its settings.

## "Is Hebb working?"

Call `list_memories`. If it answers, Hebb is on. If the `hebb` tools are missing, the plugin is not
loaded: ask them to run `/plugin`, check that hebb is enabled, and start a new session.
