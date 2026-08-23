"""Foundations: identity, determinism, text, config, canonical objects."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from leanlm.shared.com import CanonicalObject, LifecycleState
from leanlm.shared.config import deep_merge, get_path, parse_yaml
from leanlm.shared.hashing import sha256_text, short
from leanlm.shared.ids import artifact_id, request_id, session_id
from leanlm.shared.serialization import canonical_json
from leanlm.shared.text import (HeuristicTokenCounter, content_words, jaccard,
                                normalize_text, sentences, shingles, truncate_to_tokens)


class TestIdentity:
    def test_artifact_id_is_content_derived(self):
        assert artifact_id("Unit", "a", "b") == artifact_id("Unit", "a", "b")
        assert artifact_id("Unit", "a", "b") != artifact_id("Unit", "a", "c")

    def test_session_and_request_ids_are_unique(self):
        assert session_id() != session_id()
        assert request_id("s", 1) != request_id("s", 2)

    def test_short_hash_is_a_prefix(self):
        digest = sha256_text("leanlm")
        assert digest.startswith(short(digest))


class TestCanonicalJson:
    def test_key_order_does_not_change_the_bytes(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_unicode_survives_round_trip(self):
        assert "é" in canonical_json({"k": "é"})

    def test_floats_are_stable(self):
        assert canonical_json({"x": 1.0}) == canonical_json({"x": 1.0})


class TestNormalization:
    def test_crlf_becomes_lf(self):
        assert "\r" not in normalize_text("a\r\nb")

    def test_control_characters_are_removed(self):
        assert "\x00" not in normalize_text("a\x00b")

    def test_normalization_is_idempotent(self):
        once = normalize_text("Café\r\n\u0000ok")
        assert normalize_text(once) == once


class TestTextTools:
    def test_content_words_drop_stopwords_in_both_languages(self):
        assert "the" not in content_words("the delay of the payment")
        assert "les" not in content_words("les delais de paiement")

    def test_sentences_split_on_terminators(self):
        assert len(sentences("Un. Deux. Trois.")) == 3

    def test_jaccard_bounds(self):
        # Two empty sets are identical, not disjoint: the near-duplicate filter
        # relies on this so that two empty passages compare as the same.
        assert jaccard(set(), set()) == 1.0
        assert jaccard({"a"}, {"a"}) == 1.0
        assert jaccard({"a"}, set()) == 0.0

    def test_shingles_capture_word_order(self):
        assert (shingles("remboursement notes frais", 2)
                != shingles("frais notes remboursement", 2))

    def test_token_counter_never_underestimates_badly(self):
        counter = HeuristicTokenCounter()
        text = "remboursement des notes de frais sous trente jours ouvres"
        assert counter.count(text) >= len(text.split())

    def test_truncate_respects_the_ceiling(self):
        counter = HeuristicTokenCounter()
        text = " ".join(["mot"] * 400)
        assert counter.count(truncate_to_tokens(text, 50, counter)) <= 50


class TestMiniYaml:
    def test_nested_mappings(self):
        parsed = parse_yaml("a:\n  b: 1\n  c: two\n")
        assert parsed["a"]["b"] == 1 and parsed["a"]["c"] == "two"

    def test_list_after_key(self):
        parsed = parse_yaml("items:\n  - one\n  - two\n")
        assert parsed["items"] == ["one", "two"]

    def test_quoted_scalar_containing_a_colon(self):
        parsed = parse_yaml('url: "http://example.com:8080"\n')
        assert parsed["url"] == "http://example.com:8080"

    def test_booleans_and_null(self):
        parsed = parse_yaml("a: true\nb: false\nc: null\n")
        assert parsed["a"] is True and parsed["b"] is False and parsed["c"] is None

    def test_comments_are_ignored(self):
        assert parse_yaml("# lead\na: 1  # trailing\n")["a"] == 1


class TestConfigHelpers:
    def test_deep_merge_keeps_untouched_branches(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_get_path_returns_the_default(self):
        assert get_path({"a": {"b": 1}}, "a.c", "fallback") == "fallback"


@dataclass(frozen=True)
class _Sample(CanonicalObject):
    SCHEMA_VERSION = "1.0.0"
    value: str = ""

    def identity(self):
        return (self.value,)


class TestCanonicalObject:
    def test_sealing_sets_a_checksum_and_verifies(self):
        obj = _Sample(meta=_Sample.build_meta(identity=("x",)), value="x").sealed()
        assert obj.meta.checksum and obj.verify_checksum()

    def test_tampering_breaks_verification(self):
        import dataclasses
        obj = _Sample(meta=_Sample.build_meta(identity=("x",)), value="x").sealed()
        tampered = dataclasses.replace(obj, value="y")
        assert not tampered.verify_checksum()

    def test_objects_are_frozen(self):
        obj = _Sample(meta=_Sample.build_meta(identity=("x",)), value="x")
        with pytest.raises(Exception):
            obj.value = "mutated"

    def test_lifecycle_state_defaults_to_created(self):
        obj = _Sample(meta=_Sample.build_meta(identity=("x",)), value="x")
        assert obj.meta.lifecycle_state == LifecycleState.CREATED


class TestAccentAndTerminalRobustness:
    """Three spellings of the same French word must retrieve the same passage.

    A document typed on one keyboard says "delai"; a user on another types
    "délai"; a Windows terminal with the wrong code page sends "dlai". Only the
    first two can be made to agree, and they must be.
    """

    def test_accents_fold_at_tokenisation(self):
        from leanlm.shared.text import content_words
        assert content_words("télétravail") == content_words("teletravail")
        assert content_words("délai de dépôt") == content_words("delai de depot")

    def test_folding_does_not_alter_stored_text(self):
        """Only matching folds. What the user reads keeps its accents."""
        from leanlm.shared.text import fold_accents, normalize_text
        assert normalize_text("délai") == "délai"
        assert fold_accents("délai") == "delai"

    def test_folding_is_idempotent(self):
        from leanlm.shared.text import fold_accents
        assert fold_accents(fold_accents("créé")) == fold_accents("créé")

    def test_non_latin_text_survives(self):
        from leanlm.shared.text import fold_accents
        assert fold_accents("عربية") == "عربية"

    def test_case_and_accent_both_normalise(self):
        from leanlm.shared.text import content_words
        assert content_words("DÉLAI") == content_words("delai")
