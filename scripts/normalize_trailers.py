#!/usr/bin/env python3
"""Normalize agent attribution in a commit message.

One definition of the transform, shared by every caller: `.githooks/commit-msg`
on each commit, the one-time history backfill, and any later repair run. Drift
gets fixed by running this over a branch, not by a check on every PR.

Two things happen here, and a third deliberately does not.

Dropped — the verbose lines. `Claude-Session:` points at a session only one
person can open and writes a session identifier permanently into history; the
bare session URL is the same identifier undressed; `Generated with …` is tool
advertising.

Translated — an agent's `Co-authored-by:` becomes the repo's `Pair:` trailer:

    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
        -> Pair: voolyvex & Claude Opus 5
    Co-authored-by: aider (gpt-5) <aider@aider.chat>
        -> Pair: voolyvex & aider (gpt-5)

Both tools put the model in the trailer's name field, so one rule preserves
model metadata across tools with no model list to maintain.

Never added — nothing here stamps a commit that carries no agent trailer.
Stripping and translating are truthful; stamping a commit this cannot attribute
would not be. `stamp()` exists for the backfill alone, where the operator
supplies the attribution knowingly.

The translation is gated on an agent-email allowlist. GitHub's privacy address
for humans is `12345+user@users.noreply.github.com`, so "noreply" identifies
nothing on its own. A blanket transform would rewrite a human co-author into the
pair's agent half — destroying a real attribution and asserting something false,
permanently. An unrecognized co-author is therefore left as `Co-authored-by:`,
which is merely untidy.

Usage:
    normalize_trailers.py            < msg > out    # translate and strip
    normalize_trailers.py --stamp    < msg > out    # …then add Pair: if absent
    normalize_trailers.py --stamp='Claude Opus 5' < msg > out
"""

from __future__ import annotations

import re
import sys

# The human half of the pair. This trailer is repo-local, and so is the name.
HUMAN = "voolyvex"

PAIR_KEY = "Pair"

# The agent half when the backfill has no model to record. Unversioned on
# purpose: no model was ever written down for those commits, and inventing one
# would make the stamp untrustworthy everywhere else.
DEFAULT_AGENT = "Claude"

# Addresses that identify a coding agent rather than a person.
AGENT_EMAILS = frozenset(
    {
        "noreply@anthropic.com",  # Claude Code
        "aider@aider.chat",  # aider
    }
)

# GitHub App identities: `12345+dependabot[bot]@users.noreply.github.com`.
BOT_EMAIL = re.compile(r"^[^@]*\[bot\]@")

# The one emailless form this repo's own history produced. A trailer with no
# address is not a co-author GitHub ever rendered, so translating it destroys
# no attribution — but the match stays exact rather than becoming a pattern,
# because Claude is also a person's name.
EMAILLESS_AGENTS = frozenset({"claude"})

DROP_LINES = (
    # The session URL as a trailer, and again as a bare line: same identifier.
    re.compile(r"^\s*Claude-Session:", re.IGNORECASE),
    re.compile(r"^\s*https://claude\.ai/code/session_"),
    # Anchored on a Claude reference so it cannot eat prose that happens to use
    # the phrase. The leading \W* absorbs an emoji and any indent.
    re.compile(r"^\W*Generated with .*[Cc]laude"),
)

COAUTHOR_LINE = re.compile(r"^\s*Co-authored-by:\s*(.*?)\s*$", re.IGNORECASE)
NAME_AND_EMAIL = re.compile(r"^(.*?)\s*<([^>]*)>$")
PAIR_LINE = re.compile(rf"^\s*{PAIR_KEY}:\s", re.IGNORECASE)
# A trailer key: one token, no spaces, then a colon and a value.
TRAILER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:\s")


def is_agent_email(email: str) -> bool:
    email = email.strip().lower()
    return email in AGENT_EMAILS or bool(BOT_EMAIL.match(email))


def _pair_for(value: str) -> str | None:
    """The `Pair:` line for a co-author value, or None to leave it alone."""
    match = NAME_AND_EMAIL.match(value)
    if match:
        name, email = match.group(1).strip(), match.group(2)
        if not name or not is_agent_email(email):
            return None
        return f"{PAIR_KEY}: {HUMAN} & {name}"
    if value.lower() in EMAILLESS_AGENTS:
        return f"{PAIR_KEY}: {HUMAN} & {value}"
    return None


def normalize(message: str) -> str:
    """Drop the verbose lines and translate agent co-authors to `Pair:`."""
    kept: list[str] = []
    seen_pairs: set[str] = set()
    for line in message.splitlines():
        if any(pattern.match(line) for pattern in DROP_LINES):
            continue
        match = COAUTHOR_LINE.match(line)
        if match:
            pair = _pair_for(match.group(1))
            if pair is not None:
                line = pair
        if PAIR_LINE.match(line):
            # Translating can produce a duplicate of a `Pair:` already there.
            key = line.strip()
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
        kept.append(line)
    return _tidy(kept)


def stamp(message: str, agent: str = DEFAULT_AGENT, human: str = HUMAN) -> str:
    """Append `Pair: <human> & <agent>` unless a `Pair:` trailer is present.

    Only the backfill calls this. The hook must never add a stamp it cannot
    justify from what the message already says.
    """
    lines = message.splitlines()
    if any(PAIR_LINE.match(line) for line in lines):
        return _tidy(lines)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and not _ends_in_trailer_block(lines):
        lines.append("")
    lines.append(f"{PAIR_KEY}: {human} & {agent}")
    return _tidy(lines)


def _ends_in_trailer_block(lines: list[str]) -> bool:
    """Whether the message's last paragraph is already a block of trailers."""
    block: list[str] = []
    for line in reversed(lines):
        if not line.strip():
            break
        block.append(line)
    if len(block) == len(lines):
        # No blank line above it, so that paragraph is the subject.
        return False
    return bool(block) and all(TRAILER_LINE.match(line) for line in block)


def _tidy(lines: list[str]) -> str:
    """Close the gaps a removed line leaves behind, and end with a newline.

    Dropping a line from the middle of a message leaves a run of blank lines
    where it was. Collapsing runs to one is git's own `--cleanup=whitespace`
    rule, so this cannot produce a message git would have written differently.
    """
    collapsed: list[str] = []
    for line in lines:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    while collapsed and not collapsed[-1].strip():
        collapsed.pop()
    if not collapsed:
        return ""
    return "\n".join(collapsed) + "\n"


def main(argv: list[str]) -> int:
    agent: str | None = None
    for arg in argv:
        if arg == "--stamp":
            agent = DEFAULT_AGENT
        elif arg.startswith("--stamp="):
            agent = arg.split("=", 1)[1]
        else:
            print(f"{__file__}: unknown argument {arg!r}", file=sys.stderr)
            return 2
    message = sys.stdin.read()
    result = normalize(message)
    if agent is not None:
        result = stamp(result, agent)
    sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
