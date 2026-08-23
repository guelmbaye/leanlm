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

    # The prompt echoed back. Only an echo at the *start* counts: a reasoning
    # model quotes its own instructions while working -- "Answer in the language
    # of the question. Be concise and factual." appears mid-thought -- and
    # searching the whole output for that line cut the answer in half, leaving
    # fragments like "in the language of the question (English)." that then
    # looked truncated and scored zero.
    #
    # A chat UI lists its slash commands after the banner. Cutting at the banner
    # marker leaves them behind, and they then stood between the output and the
    # echoed prompt, so the echo was no longer recognisable as a prefix.
    lines = text.splitlines()
    original_count = len(lines)
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("/")):
        lines.pop(0)
    # The input marker is removed from the line, never the line from the output:
    # the UI writes "> " before the prompt *and* the answer can follow it, so
    # dropping the line deleted the answer.
    if lines and lines[0].lstrip().startswith(">"):
        lines[0] = lines[0].lstrip().lstrip(">").lstrip()
    if lines != text.splitlines()[original_count - len(lines):] or \
            len(lines) != original_count:
        found = True
    text = "\n".join(lines)

    # An echo reproduces the prompt at the head, contiguously. A reasoning model
    # quotes fragments of it anywhere. So: confirm the output opens with the
    # prompt, and only then cut past the prompt's final line.
    if prompt:
        opening = prompt.strip()[:48]
        # A chat UI prefixes the echoed prompt with its input marker.
        head = text.lstrip().lstrip(">").lstrip()
        if opening and head.startswith(opening):
            text = head
            found = True
            last_line = next(
                (ln.strip() for ln in reversed(prompt.splitlines())
                 if len(ln.strip()) > 8), "")
            index = text.rfind(last_line[:60]) if last_line else -1
            if index != -1:
                cut = text.find("\n", index)
                text = text[cut + 1:] if cut != -1 else ""
            else:
                text = text[len(prompt):]

    cleaned = _TRAILING_NOISE.sub("", text).strip()
    if cleaned != text.strip():
        found = True
    return (cleaned or original.strip()), found


# A completed sentence ends somewhere. These are the endings that count.
_TERMINAL = tuple('.!?"\')]}»') + ("\u2019",)

# Markers a reasoning model puts before the thing it actually wants to say.
_FINAL_ANSWER = re.compile(
    r"(?:\*\*)?(?:draft\s+answer|final\s+answer|answer)\s*:?(?:\*\*)?\s*:?\s*",
    re.IGNORECASE)


def looks_truncated(text: str) -> bool:
    """Did the generation stop mid-thought?

    Token accounting was the first signal tried and it is not dependable: the
    count is parsed out of llama.cpp's stderr and falls back to an estimate when
    the format differs, so a generation cut off at the budget was reported as
    complete. Where the text ends is observable regardless.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return not stripped.endswith(_TERMINAL)


def extract_final_answer(text: str) -> tuple[str, bool]:
    """Pull the answer out of a reasoning block that reached one.

    Returns the text unchanged when no marker is found, because guessing where
    reasoning ends is how a correct answer gets truncated by the cleaner instead
    of by the model.
    """
    matches = list(_FINAL_ANSWER.finditer(text))
    if not matches:
        return text, False
    tail = text[matches[-1].end():].strip()
    # A marker at the very end means the model announced an answer and stopped.
    if len(tail) < 8:
        return text, False
    return tail, True


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

    # Bare numbered reasoning: no marker to strip, but often a stated answer
    # somewhere inside it.
    extracted, found_answer = extract_final_answer(text)
    if found_answer:
        notes.append(
            "the model reasoned before answering and the answer was extracted "
            "from the end of that reasoning. Raise inference.max_output_tokens "
            "so it has room to finish, or disable thinking")
        text = extracted

    if looks_truncated(text):
        notes.append(
            "the generation ends mid-sentence: it was cut off before finishing. "
            "Nothing in it should be read as a completed answer")

    return text.strip(), tuple(notes)
