"""Removing what the model and the tool wrap around the answer.

Two kinds of scaffolding reach us, and neither is the answer:

**The tool's.** Recent `llama-cli` builds are a chat application. They print a
banner, a command list and the prompt itself before generating, and
`--no-display-prompt` is not honoured once the interactive UI is up. All of it
arrives on stdout and would otherwise be scored as the model's output --
observed as "5 sentences not supported by the excerpts" on a run where the model
had produced no claim at all.

**The model's.** Hybrid reasoning models emit a thinking block first. It is not
an answer, it consumes the token budget, and when the budget runs out mid-thought
the run yields nothing usable.

Stripping is deliberately conservative: if it would remove everything, the text
is returned untouched and the caller decides. Silently emptying a response is
how a generation failure becomes an accuracy figure.
"""
from __future__ import annotations

import re

# Opening markers for a reasoning block, and the closers that end one.
_THINK_BLOCKS = (
    (re.compile(r"<think>", re.IGNORECASE), re.compile(r"</think>", re.IGNORECASE)),
    (re.compile(r"\[start thinking\]", re.IGNORECASE),
     re.compile(r"\[end thinking\]|\[done thinking\]", re.IGNORECASE)),
    (re.compile(r"^\s*thinking process\s*:", re.IGNORECASE | re.MULTILINE),
     re.compile(r"\n\s*(?:answer|response)\s*:", re.IGNORECASE)),
)

# Lines a chat UI prints around the exchange.
_BANNER_MARKERS = (
    "available commands:",
    "/exit or ctrl+c",
    "loading model...",
)

# The status line recent builds print when generation ends.
_TRAILING_NOISE = re.compile(
    r"\[\s*prompt:.*?\]|\bexiting\.\.\.|^\s*>\s*$", re.IGNORECASE | re.MULTILINE)


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove reasoning blocks. Returns the text and whether one was found.

    An unterminated block -- the budget ran out mid-thought -- removes
    everything from the opener onwards, because what follows it is a truncated
    thought rather than an answer.
    """
    found = False
    for opener, closer in _THINK_BLOCKS:
        while True:
            start = opener.search(text)
            if not start:
                break
            found = True
            end = closer.search(text, start.end())
            text = (text[:start.start()] + text[end.end():]) if end \
                else text[:start.start()]
    return text.strip(), found


def strip_tool_scaffolding(text: str, prompt: str = "") -> tuple[str, bool]:
    """Remove a chat UI's banner and its echo of the prompt."""
    original = text
    found = False

    lowered = text.lower()
    for marker in _BANNER_MARKERS:
        index = lowered.rfind(marker)
        if index != -1:
            found = True
            cut = text.find("\n", index)
            text = text[cut + 1:] if cut != -1 else ""
            lowered = text.lower()

    # The prompt echoed back: cut at its last distinctive line rather than
    # matching the whole thing, which the UI may have truncated or reflowed.
    if prompt:
        for line in reversed([ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 24]):
            index = text.rfind(line[:60])
            if index != -1:
                found = True
                cut = text.find("\n", index)
                text = text[cut + 1:] if cut != -1 else ""
                break

    cleaned = _TRAILING_NOISE.sub("", text).strip()
    if cleaned != text.strip():
        found = True
    return (cleaned or original.strip()), found


def clean_generation(text: str, prompt: str = "") -> tuple[str, tuple[str, ...]]:
    """Both passes, with a note of what was removed.

    The notes travel with the response: a run whose output needed scaffolding
    removed is a run whose configuration is not right yet, and that should be
    visible rather than tidied away.
    """
    notes: list[str] = []
    text, tool = strip_tool_scaffolding(text, prompt)
    if tool:
        notes.append(
            "removed a chat UI banner or an echoed prompt from the output: this "
            "build of llama-cli is interactive by default. Prefer the "
            "llama-server backend, which has no such wrapper")

    stripped, thinking = strip_thinking(text)
    if thinking:
        notes.append(
            "removed a reasoning block: this model thinks before answering, "
            "which spends the token budget and can leave no answer at all. "
            "Disable thinking, or raise inference.max_output_tokens")
        if not stripped:
            # Everything was a thought. Say so instead of returning nothing.
            return "", tuple(notes + [
                "the model produced only a reasoning block; the token budget ran "
                "out before it answered"])
        text = stripped

    return text.strip(), tuple(notes)
