"""Text normalization and token accounting.

Token counting matters here: the context budget is expressed in tokens, and the
budget is the central resource LeanLM manages. Two counters exist:

  * ``HeuristicTokenCounter`` -- dependency free, deterministic, used when no
    model is loaded (CI, unit tests, document ingestion);
  * ``LlamaTokenCounter``     -- the real tokenizer exposed by llama.cpp, used
    as soon as a model is bound to the runtime.

Both implement the same protocol, so a budget computed in CI has the same
semantics as one computed on the target laptop; only the precision changes,
and which counter was used is recorded in the metrics.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Protocol

_WS = re.compile(r"[ \t\x0b\f\r]+")
_MULTI_NL = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_ELISION = re.compile(r"\b([cdjlmnstqu]|qu|jusqu|lorsqu|puisqu)['\u2019]")
_WORD = re.compile(r"[0-9\w][0-9\w'\u2019-]*", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")

# Content-word stop lists (FR + EN). Used by retrieval and grounding checks.
# Interrogatives carry no topic: they say what shape the answer takes, never what
# it is about. Counting them as content made every question look partly unknown
# to the corpus and, once the off-topic gate existed, silently unanswerable.
_INTERROGATIVES = (
    "quel", "quelle", "quels", "quelles", "combien", "comment", "pourquoi",
    "quand", "quoi", "lequel", "laquelle", "ou",
    "what", "which", "how", "why", "when", "who", "whom", "whose", "where",
)

STOPWORDS: frozenset[str] = frozenset(_INTERROGATIVES) | frozenset(
    """
    a ai au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma
    mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son
    sur ta te tes toi ton tu un une vos votre vous y d l n s c j m t est sont etre ete cette
    cet celui celle ceux plus moins tres bien alors donc car si comme sans sous entre chez
    the a an and or but of to in on at for with by from as is are was were be been being
    this that these those it its their his her our your my we you they he she i not no do
    does did done can could should would may might will shall have has had there here what
    which who whom when where why how all any each few more most other some such than too
    very s t don now
    """.split()
)


def normalize_text(raw: str) -> str:
    """DP-04 minimal transformation: fix encoding artefacts, keep the meaning.

    Unicode NFC, control characters removed, CRLF unified, trailing spaces
    trimmed, runs of blank lines collapsed to one. Words are never altered.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = _CONTROL.sub("", text)
    text = "\n".join(_WS.sub(" ", line).rstrip() for line in text.split("\n"))
    return _MULTI_NL.sub("\n\n", text).strip()


def fold_accents(text: str) -> str:
    """Strip combining diacritics for *matching only*.

    Corpora and questions rarely agree on accents: a document typed on one
    keyboard says "delai", a user on another types "délai", and a terminal with
    the wrong code page turns it into "dlai". Left alone these are three
    different tokens and only one of them retrieves anything.

    Folding happens at tokenisation, never on stored or displayed text: the
    answer a user reads keeps its accents, and the artifact checksums are
    computed over the original bytes.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def words(text: str) -> list[str]:
    """Tokenise, splitting French elision.

    ``l'indemnite`` is an article glued to a noun. Kept whole it becomes a token
    no corpus ever contains, which the retrieval gate reads as a sign that the
    question is about something the documents do not cover -- and a legitimate
    question is silently refused.
    """
    return _WORD.findall(fold_accents(_ELISION.sub(" ", text.lower())))


def content_words(text: str) -> list[str]:
    kept = []
    for word in words(text):
        # French inversion glues a pronoun to the verb ("faut-il", "peut-on").
        # Left alone it becomes a token the corpus has never seen, which the
        # off-topic gate then reads as evidence the question is off topic.
        for enclitic in ("-t-il", "-t-elle", "-t-on", "-il", "-elle", "-on",
                         "-ils", "-elles", "-je", "-nous", "-vous"):
            if word.endswith(enclitic) and len(word) > len(enclitic) + 2:
                word = word[: -len(enclitic)]
                break
        if word not in STOPWORDS and len(word) > 1:
            kept.append(word)
    return kept


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def shingles(text: str, size: int = 4) -> frozenset[tuple[str, ...]]:
    """Word n-grams, used for near-duplicate detection and grounding."""
    tokens = content_words(text)
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1))


def jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Dependency-free estimator.

    Calibrated against Llama/Qwen/Gemma BPE vocabularies on mixed FR/EN prose:
    a token averages ~3.6 characters, and punctuation-heavy or numeric text
    tokenizes more finely. The estimator takes the maximum of a character based
    and a word based estimate so it errs on the safe side -- overestimating the
    context cost can only make LeanLM more conservative with memory.
    """

    name = "heuristic-v1"
    CHARS_PER_TOKEN = 3.6
    TOKENS_PER_WORD = 1.32

    def count(self, text: str) -> int:
        if not text:
            return 0
        by_chars = len(text) / self.CHARS_PER_TOKEN
        by_words = len(text.split()) * self.TOKENS_PER_WORD
        return max(1, int(math.ceil(max(by_chars, by_words))))


class LlamaTokenCounter:
    """Exact counter backed by the loaded llama.cpp model."""

    name = "llama.cpp"

    def __init__(self, tokenize) -> None:
        self._tokenize = tokenize

    def count(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._tokenize(text.encode("utf-8"), add_bos=False, special=False))
        except TypeError:
            return len(self._tokenize(text.encode("utf-8")))


DEFAULT_TOKEN_COUNTER: TokenCounter = HeuristicTokenCounter()


def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter | None = None) -> str:
    """Cut on a sentence boundary when possible, never mid-word."""
    counter = counter or DEFAULT_TOKEN_COUNTER
    if max_tokens <= 0:
        return ""
    if counter.count(text) <= max_tokens:
        return text
    kept: list[str] = []
    used = 0
    for sentence in sentences(text):
        cost = counter.count(sentence) + 1
        if used + cost > max_tokens:
            break
        kept.append(sentence)
        used += cost
    if kept:
        return " ".join(kept)
    ratio = max_tokens / max(1, counter.count(text))
    cut = max(1, int(len(text) * ratio))
    return text[:cut].rsplit(" ", 1)[0]
