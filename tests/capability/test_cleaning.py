"""Stripping what the tool and the model wrap around an answer.

Written from a real run on Windows: llama-cli b10590 printed its chat banner,
its command list and the prompt itself to stdout, and the model spent all 128
tokens on a reasoning block. The validator scored the banner as "5 sentences not
supported by the excerpts" -- on a run where the model had made no claim at all.
"""
from __future__ import annotations

import pytest

from leanlm.packages.inference.cleaning import (clean_generation, strip_thinking,
                                                strip_tool_scaffolding)

BANNER = """Loading model...

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response

> You are LeanLM, an offline assistant answering strictly from the supplied excerpts.
Rules:
5. Answer in the language of the question. Be concise and factual.

"""

PROMPT = ("You are LeanLM, an offline assistant answering strictly from the "
          "supplied excerpts.\nRules:\n5. Answer in the language of the question. "
          "Be concise and factual.")

ANSWER = "Approved expense reports are reimbursed within 30 working days. [S1]"


class TestThinking:
    def test_a_closed_block_is_removed(self):
        text, found = strip_thinking(f"<think>weighing options</think>{ANSWER}")
        assert text == ANSWER
        assert found

    def test_an_unterminated_block_takes_everything_after_it(self):
        """The budget ran out mid-thought: what follows is a truncated thought,
        not an answer."""
        text, found = strip_thinking(f"{ANSWER}\n[Start thinking]\n1. Analyse the")
        assert text == ANSWER
        assert found

    def test_a_bare_thinking_header_is_recognised(self):
        text, _ = strip_thinking("Thinking Process:\n1. Read it\nAnswer: " + ANSWER)
        assert ANSWER in text
        assert "1. Read it" not in text

    def test_an_answer_without_thinking_is_untouched(self):
        text, found = strip_thinking(ANSWER)
        assert text == ANSWER
        assert not found


class TestToolScaffolding:
    def test_the_banner_and_command_list_are_removed(self):
        text, found = strip_tool_scaffolding(BANNER + ANSWER, PROMPT)
        assert text == ANSWER
        assert found

    def test_the_echoed_prompt_is_removed(self):
        text, _ = strip_tool_scaffolding(f"> {PROMPT}\n{ANSWER}", PROMPT)
        assert text == ANSWER

    def test_the_trailing_status_line_is_removed(self):
        text, _ = strip_tool_scaffolding(
            f"{ANSWER}\n[ Prompt: 14.3 t/s | Generation: 4.5 t/s ]\n\nExiting...", "")
        assert text == ANSWER

    def test_a_clean_output_is_untouched(self):
        text, found = strip_tool_scaffolding(ANSWER, PROMPT)
        assert text == ANSWER
        assert not found

    def test_stripping_never_empties_a_real_answer(self):
        """Conservative by design: emptying a response is how a generation
        failure becomes an accuracy figure."""
        text, _ = strip_tool_scaffolding("available commands:", "")
        assert text


class TestCombined:
    def test_the_observed_failure_is_recognised_as_one(self):
        """Banner plus reasoning block plus nothing else."""
        raw = BANNER + "[Start thinking]\nThinking Process:\n1. Analyse the req"
        text, notes = clean_generation(raw, PROMPT)
        assert text == ""
        assert any("reasoning block" in note for note in notes)
        assert any("only a reasoning block" in note for note in notes)

    def test_an_answer_survives_both_passes(self):
        raw = BANNER + f"<think>weighing</think>{ANSWER}\n\nExiting..."
        text, notes = clean_generation(raw, PROMPT)
        assert text == ANSWER
        assert notes

    def test_the_notes_name_the_configuration_to_change(self):
        _, notes = clean_generation(f"<think>x</think>{ANSWER}", "")
        joined = " ".join(notes)
        assert "Disable thinking" in joined or "max_output_tokens" in joined

    def test_a_clean_run_produces_no_notes(self):
        text, notes = clean_generation(ANSWER, PROMPT)
        assert text == ANSWER
        assert notes == ()
