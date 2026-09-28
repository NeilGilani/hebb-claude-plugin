#!/usr/bin/env python3
"""Hebb hooks: Claude Code learns from what actually happened, without being told.

    hebb_hooks.py inject     # UserPromptSubmit   -- put what it knows in front of it
    hebb_hooks.py guard      # PreToolUse         -- refuse a banned command, catch a known mistake
    hebb_hooks.py observe    # PostToolUse(Failure) -- remember what broke, pair it with the fix
    hebb_hooks.py session    # SessionStart       -- in the plugin, say so if there is no key yet

WHY THIS IS NOT A MEMORY STORE WITH EXTRA STEPS. Everything else in this space -- a vault, a
CLAUDE.md, an MCP tool -- requires somebody to decide a fact is worth keeping and then write it
down. That is the part people do not do. These hooks sit in the loop where the work happens and
learn from the one signal nobody has to author: a command failed, then a different command
succeeded. That is a correction, it happened by itself, and it is exactly what an assistant
repeats forever otherwise.

The other half is that `inject` costs the model NOTHING to use. An MCP tool has to be chosen; a
hook is simply there, the way CLAUDE.md is there, except it is scoped to what you are actually
asking about rather than to the folder you happen to be in.

WHAT IT REFUSES TO LEARN, which matters more than what it learns. A memory that fills with junk is
worse than no memory, because the junk is injected into every prompt afterwards. So a lesson is
only stored when the failure and the fix are recognisably the same intent, the failure is recent,
and neither command looks like it carries a secret. Everything else is dropped silently.

Install: see `clients/hooks/README.md`, or install the Claude Code plugin, whose copy of this file
is made by `clients/claude-plugin/sync.sh`. That copy leaves out every block between the
"standalone" markers below: inside the plugin the key comes only from Claude Code (see
`read_key`), and the plugin never edits Claude Code's settings. Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("HEBB_BASE", "https://hebb-site.pages.dev/v1").rstrip("/")
# What the hooks keep between calls: recent failures to pair with a fix, and a one-minute cache of
# the memories. In the plugin that is Claude Code's data folder for it, removed on uninstall.
STATE = os.path.expanduser(os.environ.get("HEBB_HOOK_STATE") or (
    os.path.join(os.environ["CLAUDE_PLUGIN_DATA"], "state") if os.environ.get("CLAUDE_PLUGIN_DATA")
    else "~/.hebb/state"))


def read_key():
    """Where the key comes from.

    In the Claude Code plugin, Claude Code passes it as the argument after the action: it is the
    plugin's `hebb_key` userConfig value, which Claude Code asks for when the plugin is enabled,
    keeps in the system's credential store, and substitutes into the hook's `args`. The plugin's
    copy of this file reads the key from nowhere else.
    """
    k = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[1] != "install" else ""
    if k.startswith("${"):          # the option was never set, so Claude Code left the placeholder
        k = ""
    return k


KEY = read_key()

# How long a failure stays eligible to be paired with a fix. Long enough for a person to think,
# short enough that an unrelated command ten minutes later is not called a correction.
PAIR_WINDOW_S = 180
MAX_INJECT = 3
CACHE_TTL_S = 60

# Anything that looks like a credential is never written to memory, whatever the model decides.
# Case-insensitive throughout, which over-matches slightly -- and over-matching is the safe
# direction here, since the only cost is declining to learn one lesson.
SECRETISH = re.compile(
    r"sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY|Bearer\s+[A-Za-z0-9._\-]{16,}|"
    r"(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S{6,}", re.I)

STOP = {"the", "and", "for", "what", "when", "where", "which", "how", "are", "was", "with",
        "that", "this", "then", "from", "into", "our", "your", "you", "use", "using", "get",
        "not", "but", "its", "can", "does", "did", "has", "have", "run", "all", "any", "should"}


def api(path, method="GET", body=None, timeout=6):
    """Short timeout on purpose: `inject` runs before every prompt, and a memory service having a
    slow day must never be the reason somebody's prompt hangs."""
    if not KEY:
        return 0, {}
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "User-Agent": "hebb-hooks/1",
                 **({"content-type": "application/json"} if body is not None else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except Exception:
        return 0, {}


def words(text):
    """Tokens, depluralised so `tests` and `test` match. Crude, but it is applied to both sides of
    every comparison, so being wrong about English matters less than being consistent."""
    out = []
    for w in re.split(r"[^a-z0-9.\-_/]+", (text or "").lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) > 2 and w not in STOP:
            out.append(w)
    return out


def state_path(name):
    os.makedirs(STATE, exist_ok=True)
    return os.path.join(STATE, name)


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


def out(payload):
    print(json.dumps(payload))
    return 0


# ---------------------------------------------------------------------- inject

def rules_cached():
    """All memories, cached briefly. Without the cache this is a network round trip on every
    keystroke-to-enter, which is the difference between a hook nobody notices and one people
    uninstall."""
    p = state_path("rules.json")
    c = read_json(p, {})
    if c.get("at", 0) + CACHE_TTL_S > time.time():
        return c.get("rules", [])
    s, d = api("/rules")
    if s != 200:
        return c.get("rules", [])          # stale beats empty
    rules = d.get("rules", [])
    write_json(p, {"at": time.time(), "rules": rules})
    return rules


# Facts in these namespaces are injected on EVERY prompt, without any relevance test.
#
# WHY, and it is the most important decision in this file. Lexical matching cannot connect
# "run the tests" to "`pnpm` is not installed" -- they share no words, and no threshold fixes
# that, because the gap is semantic. A measured A/B showed the consequence: the memory was
# learned correctly and then never retrieved, so it changed nothing. But what the hooks learn is
# a MACHINE PROFILE -- which tools are absent, which command is the right one here -- and that is
# small, bounded by the number of distinct tools, and potentially relevant to any prompt that
# runs anything. So it is treated the way CLAUDE.md is treated: always present, cheap, and true.
# Everything else still has to earn its place by matching.
ALWAYS = ("missing:", "command:", "never:")
MAX_ALWAYS = 8


def describe(r):
    age = ""
    ts = r.get("updated_at")
    if ts:
        days = int((time.time() - float(ts)) / 86400)
        if days >= 90:
            age = f" (saved {days} days ago -- may be out of date, say so)"
        elif days >= 1:
            age = f" (saved {days} days ago)"
    # Attribution travels with a shared rule. "Who decided this" is the first thing anyone asks when
    # a rule they did not write starts governing their work.
    who = f" (shared by {r['shared_by']})" if r.get("shared_by") else (
        " (shared by your team)" if r.get("scope") == "team" else "")
    return f"- {r.get('slot')}: {r.get('value')}{who}{age}"


def inject(event):
    prompt = event.get("prompt") or event.get("user_input") or ""
    rules = rules_cached()
    qs = set(words(prompt))

    # A rule somebody chose to SHARE with their team is always in force too. Sharing is a deliberate
    # act and teams keep few of them, so they are treated like the machine profile rather than made
    # to match the wording of a prompt.
    always = [r for r in rules
              if str(r.get("slot", "")).startswith(ALWAYS) or r.get("scope") == "team"][:MAX_ALWAYS]
    chosen = list(always)

    if qs:
        scored = []
        for r in rules:
            if r in always:
                continue
            ns, vs = set(words(str(r.get("slot", "")))), set(words(str(r.get("value", ""))))
            # A name match counts double: a rule called `deploy command` answers a deploy question
            # better than one that merely mentions deploying.
            score = (2 * len(qs & ns) + len(qs & vs)) / max(1, len(qs))
            if score >= 0.34:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        chosen += [r for _s, r in scored[:MAX_INJECT]]

    if not chosen:
        return 0
    return out({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext":
            "Known about this machine and this user, from previous sessions:\n"
            + "\n".join(describe(r) for r in chosen) +
            "\nTrust these over guessing, and do not re-discover what is already stated here.",
    }})


# ---------------------------------------------------------------------- learn

def host():
    try:
        return socket.gethostname().split(".")[0]
    except Exception:
        return "this machine"


# A missing tool is the highest-signal thing a shell error contains: it is a fact about the
# machine, it is stated outright by the error, and it does not need a second command to confirm.
MISSING = [
    (re.compile(r"command not found:\s*([A-Za-z0-9_.+-]+)"), "command"),
    (re.compile(r"([A-Za-z0-9_.+-]+):\s*command not found"), "command"),
    (re.compile(r"No module named '?([A-Za-z0-9_.]+)'?"), "module"),
]


# A shell probe -- `which x`, `command -v x`, `type x` -- reporting "x not found" is as definitive
# as running x and failing, and in practice Claude Code probes far more often than it blunders into
# a 127. But "not found" is everywhere ("user not found", "404 not found"), so this is only trusted
# when the command itself shows a probe was what ran.
PROBE = re.compile(r"\b(which|command\s+-v|type)\b")
PROBE_MISSING = re.compile(r"^([A-Za-z0-9_.+-]+) not found$", re.M)


def missing_dep(err, cmd=""):
    for pat, kind in MISSING:
        m = pat.search(err or "")
        if m:
            return m.group(1), kind
    if cmd and PROBE.search(cmd):
        m = PROBE_MISSING.search(err or "")
        if m:
            return m.group(1), "command"
    return None, None


def remember(slot, value):
    s, _d = api("/rules", "POST", {"slot": slot[:180], "value": value})
    if s != 200:
        return 0
    # Said out loud. A memory written behind somebody's back is one they cannot correct, and the
    # first time they notice it they will assume there are others.
    return out({"systemMessage": f"hebb learned: {slot} -> {value}"})


def forget(slot):
    api("/rules/" + urllib.parse.quote(slot), "DELETE")
    # The cache is dropped, not left to expire. Otherwise a fact we have just disproved keeps being
    # injected into every prompt for up to a minute -- which is the exact harm being fixed.
    try:
        os.remove(state_path("rules.json"))
    except Exception:
        pass
    return out({"systemMessage": f"hebb unlearned: {slot} -- it is available now."})


INSTALLY = re.compile(r"\b(install|upgrade|brew|apt|apt-get|yum|dnf|pacman|pipx?|uv|cargo)\b")
# `npm i -g pnpm` says install with a single letter, so the subcommand position is checked too --
# matching `i` anywhere in a command would fire on any stray flag or variable.
INSTALL_VERBS = {"i", "in", "add", "install", "get", "upgrade", "up"}


def now_available(cmd, text, rules):
    """Which `missing: X` fact does this SUCCESSFUL command disprove?

    WHY THIS MATTERS MORE THAN THE LEARNING. Machine-profile facts are injected on every prompt, so
    the moment you `brew install pnpm` the memory saying pnpm is absent stops being neutral and
    starts actively misleading the model on every single turn. Without this, the layer gets STALER
    the more you use it, which is the opposite of the claim being made for it.

    Biased towards forgetting on purpose, and the asymmetry is the whole argument: a fact dropped
    too eagerly is re-learned the next time something fails, which costs one command. A wrong fact
    kept is paid on every prompt until someone notices. So weaker evidence is enough to forget than
    was needed to learn.
    """
    toks = set(t for t in re.split(r"[^A-Za-z0-9_.+-]+", cmd) if t)
    head = (cmd.split() or [""])[0].split("/")[-1]
    for r in rules:
        slot = str(r.get("slot", ""))
        if not slot.startswith("missing: "):
            continue
        name = slot[len("missing: "):].strip()
        if not name:
            continue
        if head == name:
            return slot                                  # they ran it, and it worked
        if name not in toks:
            continue
        if f"-m {name}" in cmd:
            return slot                                  # python3 -m pytest ... succeeded
        parts = cmd.split()
        verb = parts[1].lstrip("-") if len(parts) > 1 else ""
        if INSTALLY.search(cmd) or verb in INSTALL_VERBS:
            return slot                                  # brew install pnpm / npm i -g pnpm
        # A probe that now prints a path where it used to print "not found".
        if PROBE.search(cmd) and re.search(r"/" + re.escape(name) + r"\b", text or ""):
            return slot
    return None


def failed(event):
    """Learn what the failure itself proves, and keep the command in case a fix follows.

    WHY THE ERROR TEXT IS THE BETTER SIGNAL. The first version of this only learned by pairing a
    failed command with a corrected one, and a real session showed why that was not enough: Claude
    explores with compound commands like `ls -la; which python python3; python --version`, so clean
    pairs are rarer than they look. But that same error said `command not found: python` -- a
    durable fact about the machine, needing nothing else to confirm it. That is learned here and
    now, and upgraded later if a working substitute turns up.
    """
    if event.get("tool_name") != "Bash":
        return 0
    cmd = ((event.get("tool_input") or {}).get("command") or "").strip()
    err = str(event.get("tool_error") or "")
    if not cmd or SECRETISH.search(cmd):
        return 0

    fails = read_json(state_path("fails.json"), [])
    fails = [f for f in fails if f["at"] + PAIR_WINDOW_S > time.time()][-8:]
    name, kind = missing_dep(err, cmd)
    fails.append({"at": time.time(), "cmd": cmd[:300], "err": err[:300],
                  "missing": name, "kind": kind})
    write_json(state_path("fails.json"), fails)

    if not name:
        return 0
    # Slotted by the missing tool's name, so the same discovery revises one memory instead of
    # adding another -- and so the upgrade below lands on this same row.
    what = f"`{name}` is not installed" if kind == "command" else f"the `{name}` module is not installed"
    return remember(f"missing: {name}", f"{what} on {host()}.")


def variant_of(head, name):
    """Is `head` the versioned sibling of a missing `name`? python -> python3, pip -> pip3,
    node -> nodejs. Kept to a short suffix on purpose: a loose rule here would call `npm` a
    variant of `n` and teach nonsense."""
    a, b = head.lower(), name.lower()
    if a == b:
        return False
    long, short = (a, b) if len(a) > len(b) else (b, a)
    if len(short) < 3 or not long.startswith(short):
        return False
    # The difference has to look like a version or a spelling variant -- `3` in python3, `js` in
    # nodejs -- and nothing else. Without this, `n` counts as the root of `npm`.
    suffix = long[len(short):]
    return suffix.isdigit() or (suffix.isalpha() and len(suffix) <= 2)


def same_intent(a, b):
    """Is `b` plausibly a corrected version of `a`?

    Bag-of-words similarity cannot do this job. `npm test` -> `pnpm test` and
    `cat README.md` -> `cat package.json` share exactly one token each and differ in exactly one,
    yet the first is a lesson and the second is just reading two files. What separates them is
    structural: a correction keeps the JOB and changes the TOOL, or keeps both and adds a flag.

    So only two shapes are accepted:

      the program changed, the arguments did not     `npm test`     -> `pnpm test`
      the program stayed, arguments were only added  `docker build` -> `docker buildx build`

    Anything else -- including the same program with a swapped argument -- is refused, because it
    is indistinguishable from doing a different piece of work, and a wrong lesson gets injected
    into every prompt from then on.
    """
    a, b = a.strip(), b.strip()
    if not a or not b or a == b:
        return False                                   # a retry is not a correction
    at, bt = a.split(), b.split()
    aw, bw = set(words(" ".join(at[1:]))), set(words(" ".join(bt[1:])))
    if not aw and not bw:
        return False                                   # two bare programs say nothing about why

    if at[0] != bt[0]:
        # Same job, different tool. Requires the arguments to genuinely correspond.
        return bool(aw) and len(aw & bw) / max(1, len(aw | bw)) >= 0.6
    # Same tool: only an addition counts. A swap is somebody doing something else.
    return bw > aw


def succeeded(event):
    """A success can close out a failure two ways: it names the substitute for a missing tool, or
    it is a recognisable correction of the command that broke."""
    if event.get("tool_name") != "Bash":
        return 0
    cmd = ((event.get("tool_input") or {}).get("command") or "").strip()
    if not cmd or SECRETISH.search(cmd):
        return 0
    resp0 = event.get("tool_response")
    stdout = (resp0.get("stdout", "") if isinstance(resp0, dict) else "") or ""

    # Unlearning comes FIRST. A fact this command disproves must not survive the same call that
    # might otherwise go on to learn something new.
    stale = now_available(cmd, stdout, rules_cached())
    if stale:
        return forget(stale)

    fails = read_json(state_path("fails.json"), [])
    now = time.time()
    live = [f for f in fails if f["at"] + PAIR_WINDOW_S > now]

    # A probe that ran inside an otherwise-successful command is the same evidence, so it is read
    # here too rather than only on the failure path.
    if PROBE.search(cmd):
        name, kind = missing_dep(stdout, cmd)
        if name and kind == "command":
            return remember(f"missing: {name}",
                            f"`{name}` is not installed on {host()}.")
    heads = {t.split("/")[-1] for t in cmd.split()[:1]} | {
        w for w in cmd.split() if "/" not in w and w.isalnum()}

    # 1. The substitute for something that was missing. Revises the row `failed` already wrote,
    #    turning "python is not installed" into the actionable form.
    for f in live:
        if f.get("missing") and f.get("kind") == "command":
            hit = next((h for h in heads if variant_of(h, f["missing"])), None)
            if hit:
                write_json(state_path("fails.json"), [x for x in fails if x is not f])
                return remember(
                    f"missing: {f['missing']}",
                    f"`{f['missing']}` is not installed on {host()} -- use `{hit}` instead.")

    # 2. A plain correction: same job, different command.
    f = next((x for x in reversed(live) if same_intent(x["cmd"], cmd)), None)
    if not f:
        return 0
    shared = [w for w in words(cmd) if w in set(words(f["cmd"]))]
    reason = re.sub(r"\s+", " ", f["err"]).strip()[:120]
    write_json(state_path("fails.json"), [x for x in fails if x is not f])
    return remember(
        f"command: {' '.join(shared[:3]) or 'usage'}",
        f"use `{cmd[:160]}` here, not `{f['cmd'][:160]}`" + (f" ({reason})" if reason else ""))


def observe(event):
    """One entry point for both tool events, routing on the payload rather than on which hook
    fired.

    Deliberate: a failing shell command may reach us as PostToolUseFailure with `tool_error`, or as
    PostToolUse whose output carries the failure, and that is a detail of the client we should not
    be betting on. Wiring this to both events and deciding here means the layer keeps working when
    the client changes its mind.
    """
    # `error` is what Claude Code actually sends on PostToolUseFailure; the others are accepted
    # because this contract is the client's to change and a hook that reads one field name is one
    # release away from silently learning nothing.
    err = event.get("error") or event.get("tool_error")
    resp = event.get("tool_response")
    if not err and isinstance(resp, dict):
        err = resp.get("error") or (resp.get("stderr") if resp.get("is_error") else None)
    if err:
        return failed({**event, "tool_error": err})
    return succeeded(event)




# ---------------------------------------------------------------------- session

DASHBOARD = "https://hebb-site.pages.dev/dashboard.html"


def session(event):
    """SessionStart, in the plugin. With no key every other hook is silent, which from the outside
    looks exactly like a memory that works and has nothing to say. So say it, once per session."""
    if KEY:
        return 0
    return out({
        "systemMessage": ("Hebb is installed but has no key yet, so it can't learn or remember "
                          f"anything. Create one at {DASHBOARD} (Connect to Claude Code), then add "
                          "it to the Hebb plugin with /plugin."),
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": ("Hebb (memory for Claude Code) is installed without a key, so it is "
                                  "off. If the user asks about Hebb or memory, tell them to create a "
                                  f"key at {DASHBOARD} and add it to the Hebb plugin with /plugin."),
        },
    })


# ---------------------------------------------------------------------- refuse

# A memory that only informs is advice, and advice gets ignored -- an instruction written at turn 2
# is competing with eighty turns of context by the time it matters. `PreToolUse` can answer `deny`,
# which turns one class of memory from advice into a rule the model cannot talk itself past.
#
# NEVER AUTO-LEARNED. Every other memory in this file is written by watching what happened; these
# are not, and that asymmetry is deliberate. A block is a standing refusal to do something the user
# has decided is off limits, so it only ever comes from the user saying so:
#
#     remember("never: git push --force", "it wiped the branch on 2026-09-01")
#
# Inferring a refusal from a failure would eventually refuse something important on the strength of
# one bad afternoon.

NEVER = "never: "


# ---------------------------------------------------------------------- fix
#
# CAUGHT BEFORE IT RUNS. A vault, a CLAUDE.md, an MCP tool and a skill all have exactly one move: put
# text in front of the model and hope it complies. `PreToolUse` can refuse, so a command the machine
# is known to fail on never reaches the shell. `python app.py` on a machine that only has `python3`
# is stopped, and the refusal hands the model the corrected command, which it runs next. No failed
# command, no error output to read, and nothing for the model to ignore.
#
# It does not edit the command on its way to the shell. A command that silently differs from the
# one the model chose is exactly what a person reviewing a session cannot see, and the plugin
# directory does not allow it. The model runs the corrected command itself, in the open.
#
# The rules below decide what counts as a known mistake, and they are strict, because a wrong
# refusal stops somebody's work:
#
#   * only a PROGRAM NAME is ever substituted, never an argument, path, flag or filename;
#   * only where a program name can legally appear -- the start of the command, or straight after a
#     shell separator -- so text inside a quoted string is untouchable;
#   * only towards a substitute that was OBSERVED WORKING on this machine, never a guess;
#   * every catch is printed on screen;
#   * and `HEBB_NO_REWRITE=1` turns the whole thing off.

SUBSTITUTE = re.compile(r"use `([^`]+)`")

# A program name and nothing else. Anything with a slash, space or shell metacharacter is refused
# outright rather than escaped -- there is no reason for a legitimate substitute to contain one, and
# refusing is safer than trying to be clever about quoting. It must also START with a letter or
# digit, which rules out `-rf`, `.` and `..`: a leading dash or dot is a flag or a shell builtin,
# never a program somebody meant to name.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")

# Where a program name can legally appear: the start of the command, or straight after a shell
# separator. The trailing lookahead is what protects quoted text -- the name must be followed by
# whitespace or the end of the string, so `echo "pnpm test"` is untouched (nothing separator-like
# precedes `pnpm`) and a token such as `b"` can never match a tool called `b`.
HEAD = r"(^\s*|[;&|]\s*|\n\s*)"


def rewrites(rules):
    """Derive program substitutions from the machine profile.

    Read out of the value text rather than kept in a second row, because the free plan counts
    memories and a parallel namespace would double what every discovery costs. The wording is ours --
    `failed()` and `succeeded()` write it -- and anything that does not parse is skipped rather than
    guessed at.
    """
    found = {}
    for r in rules:
        slot, val = str(r.get("slot", "")), str(r.get("value", ""))
        if not slot.startswith("missing: "):
            continue
        frm = slot[len("missing: "):].strip()
        m = SUBSTITUTE.search(val)
        if not m:
            continue
        to = m.group(1).strip()
        # Both sides must be bare program names, and it must be a real change. This is the check
        # that stops a memory whose value someone has edited by hand from turning into an arbitrary
        # command substitution.
        if frm != to and SAFE_NAME.match(frm) and SAFE_NAME.match(to):
            found[frm] = to
    return found


def fix(cmd, rules):
    """Return the corrected command and what changed, or (cmd, []) if nothing applies.

    ONE PASS over the original, deliberately. Applying rules one after another lets them cascade:
    with `a -> b` and `b -> c` in the profile, `a x` would become `b x` and then `c x`, producing a
    substitution that was never observed working. A single pass can only ever replace what the user's
    own command actually said.
    """
    subs = rewrites(rules)
    if not subs:
        return cmd, []
    # Longest name first so `python` cannot win a position that `python3` should have had.
    names = sorted(subs, key=len, reverse=True)
    pat = re.compile(HEAD + "(" + "|".join(re.escape(n) for n in names) + r")(?=\s|$)")
    done = []

    def swap(m):
        to = subs[m.group(2)]
        note = f"{m.group(2)} -> {to}"
        if note not in done:
            done.append(note)
        return m.group(1) + to

    return pat.sub(swap, cmd), done


def guard(event):
    if event.get("tool_name") != "Bash":
        return 0
    cmd = ((event.get("tool_input") or {}).get("command") or "")
    if not cmd.strip():
        return 0

    # Matching is loose on purpose. For a refusal the errors are not symmetric: a false positive is
    # an annoying prompt the user can override, a false negative is the incident the rule existed to
    # prevent. Whitespace is normalised so a reformatted command cannot slip past.
    low = " ".join(cmd.lower().split())
    rules = rules_cached()
    for r in rules:
        slot = str(r.get("slot", ""))
        if not slot.lower().startswith(NEVER):
            continue
        pat = " ".join(slot[len(NEVER):].lower().split())
        if pat and pat in low:
            return out({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                # The reason is carried back so the model can explain itself and choose another
                # route, rather than reporting an unexplained refusal and stalling.
                "permissionDecisionReason":
                    (f"Blocked by {'your team' if r.get('scope') == 'team' else 'your'} saved rule "
                     f"{slot!r}: {r.get('value')}"
                     + (f" -- shared by {r['shared_by']}" if r.get("shared_by") else "") +
                     ". Do not retry this another way -- tell the user it was blocked and why."),
            }})

    # Refusal is settled before anything is corrected: a banned command is refused for its own
    # reason, never offered back in a corrected form that would get through.
    if os.environ.get("HEBB_NO_REWRITE"):
        return 0
    new, done = fix(cmd, rules)
    if not done or new == cmd:
        return 0
    facts = "; ".join(f"`{a}` is not installed on this machine, `{b}` is"
                      for a, b in (d.split(" -> ", 1) for d in done))
    return out({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            # The corrected command is the last thing in the reason, so the model can run it as is.
            "permissionDecisionReason": (f"Not run: {facts} (Hebb learned this here earlier). "
                                         f"Run this instead: {new.strip()}"),
        },
        "systemMessage": "hebb caught it: " + ", ".join(done),
    })


def main():
    if len(sys.argv) < 2:
        return 0
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0
    action = sys.argv[1]
    if os.environ.get("HEBB_HOOK_DEBUG"):
        # Written, not printed: stdout is the hook's reply channel and stderr is shown to the user.
        try:
            with open(state_path("debug.log"), "a") as f:
                f.write(json.dumps({"action": action, "event": event}) + "\n")
        except Exception:
            pass
    try:
        if action == "inject":
            return inject(event)
        if action == "failed":
            return failed(event)
        if action == "succeeded":
            return succeeded(event)
        if action == "observe":
            return observe(event)
        if action == "guard":
            return guard(event)
        if action == "session":
            return session(event)
    except Exception:
        # A hook must never take the session down. Silence is the right failure here: the person
        # is in the middle of something else.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
