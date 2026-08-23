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


class TestTruncationFromText:
    """Where the text ends is observable; the token count is not always.

    The first truncation signal was `generated_tokens >= max_output_tokens`,
    parsed out of llama.cpp's stderr with an estimate as fallback. On a real run
    it reported `truncated: 0.0` for twelve generations that all stopped
    mid-word.
    """

    def test_a_sentence_that_stops_mid_word_is_truncated(self):
        from leanlm.packages.inference.cleaning import looks_truncated
        assert looks_truncated("5. **Review Constrain")
        assert looks_truncated("* State the value: 20 EUR per month.\n * En")

    def test_a_completed_sentence_is_not(self):
        from leanlm.packages.inference.cleaning import looks_truncated
        assert not looks_truncated("The ceiling for lunch is 25 EUR. [S1]")
        assert not looks_truncated("Is that clear?")
        assert not looks_truncated('He said "yes."')

    def test_empty_output_is_not_called_truncated(self):
        """It is a different failure with a different cause."""
        from leanlm.packages.inference.cleaning import looks_truncated
        assert not looks_truncated("   ")


class TestFinalAnswerExtraction:
    """Bare numbered reasoning carries no marker to strip, but often states the
    answer somewhere inside itself."""

    def test_the_answer_after_a_draft_marker_is_taken(self):
        from leanlm.packages.inference.cleaning import extract_final_answer
        text = ("2. **Scan:** keywords\n\n**Draft Answer:** The notice period is "
                "three (3) months. [S1]")
        answer, found = extract_final_answer(text)
        assert found
        assert answer.startswith("The notice period is three (3) months")

    def test_text_without_a_marker_is_returned_whole(self):
        """Guessing where reasoning ends is how the cleaner truncates a correct
        answer that the model had finished."""
        from leanlm.packages.inference.cleaning import extract_final_answer
        answer, found = extract_final_answer("The ceiling is 25 EUR. [S1]")
        assert not found
        assert answer == "The ceiling is 25 EUR. [S1]"

    def test_a_marker_with_nothing_after_it_is_not_an_answer(self):
        from leanlm.packages.inference.cleaning import extract_final_answer
        answer, found = extract_final_answer("3. Reasoning\n\n**Draft Answer:**")
        assert not found

    def test_the_extraction_is_reported(self):
        from leanlm.packages.inference.cleaning import clean_generation
        _, notes = clean_generation(
            "1. think\n\n**Answer:** The ceiling is 25 EUR. [S1]", "")
        assert any("max_output_tokens" in note for note in notes)


class TestEchoStrippingIsNotOverEager:
    """Every case here comes from a real run, including the ones I broke.

    The first echo remover searched the whole output for any prompt line. A
    reasoning model quotes its own instructions while working, so the search hit
    mid-thought and cut the answer in half -- producing fragments like
    "in the language of the question (English)." that then looked truncated and
    scored zero. Ten of twelve probes were corrupted this way by the cleaner,
    not by the model.
    """

    PROMPT = ("You are LeanLM, an offline assistant answering strictly from the "
              "supplied excerpts.\nRules:\n5. Answer in the language of the "
              "question. Be concise and factual.\n\n### Question\nWhat is the "
              "ceiling for a lunch?")
    ANSWER = "The ceiling for lunch is 25 EUR. [S1]"

    def _clean(self, raw: str) -> str:
        return clean_generation(raw, self.PROMPT)[0].strip()

    def test_a_quoted_instruction_mid_reasoning_is_not_a_cut_point(self):
        raw = ("1. **Analyse:** answer in the language of the question. Be "
               f"concise and factual.\n2. **Draft Answer:** {self.ANSWER}")
        assert self._clean(raw) == self.ANSWER

    def test_a_genuine_prefix_echo_is_removed(self):
        assert self._clean(f"{self.PROMPT}\n{self.ANSWER}") == self.ANSWER

    def test_a_bare_answer_is_untouched(self):
        assert self._clean(self.ANSWER) == self.ANSWER

    def test_banner_commands_and_echo_together(self):
        raw = ("Loading model...\n\navailable commands:\n"
               "  /exit or Ctrl+C     stop\n  /regen              redo\n\n"
               f"> {self.PROMPT}\n{self.ANSWER}")
        assert self._clean(raw) == self.ANSWER

    def test_the_input_marker_is_stripped_not_the_line(self):
        """The UI writes "> " before the prompt, and an answer can follow it on
        the same line. Dropping the line deleted the answer."""
        assert self._clean(f"  /exit\n> {self.ANSWER}") == self.ANSWER

    def test_a_truncated_draft_stays_truncated(self):
        """Cleaning must not rescue what the model never finished."""
        raw = "3. **Locate:** 25 EUR\n4. **Draft Answer:** The ceiling for lunch is 25"
        cleaned, notes = clean_generation(raw, self.PROMPT)
        assert cleaned == "The ceiling for lunch is 25"
        assert any("mid-sentence" in note for note in notes)
