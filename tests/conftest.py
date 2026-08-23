"""Shared fixtures.

Every test runs against a private corpus in a temporary directory: the suite
must never touch a developer's real workspace, and two tests must never see
each other's documents.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "datasets" / "enterprise_en"
DATASET_FR = REPO_ROOT / "datasets" / "enterprise"

SAMPLE_MARKDOWN = """# Politique interne

## 1. Delais
Toute note de frais doit etre soumise dans les 15 jours calendaires suivant la
depense. Le remboursement intervient sous 30 jours ouvres apres validation.

## 2. Plafonds
Le plafond pour un repas du midi est de 25 EUR par personne. Le plafond pour un
repas du soir est de 35 EUR par personne.

## 3. Exclusions
Les amendes routieres ne sont jamais remboursees.
"""


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    # Only the *working* directory is redirected. LEANLM_HOME must keep pointing
    # at the repository, otherwise the suite would test against absent configs
    # instead of the ones that ship.
    monkeypatch.setenv("LEANLM_WORKDIR", str(tmp_path / "workdir"))
    monkeypatch.delenv("LEANLM_HOME", raising=False)
    monkeypatch.delenv("LEANLM_CONFIG_DIR", raising=False)
    yield tmp_path


@pytest.fixture
def sample_file(tmp_path) -> Path:
    path = tmp_path / "politique.md"
    path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    from leanlm.runtime.corpus import CorpusStore
    store = CorpusStore(tmp_path / "corpus.sqlite3")
    yield store
    store.close()


@pytest.fixture
def structured(sample_file):
    """A StructuredDocument produced by the real CAP-001 -> CAP-002 pipeline."""
    from leanlm.contracts.ddp import DocumentPipelineState
    from leanlm.packages.ingestion import DocumentIngestionCapability
    from leanlm.packages.understanding import DocumentUnderstandingCapability

    state = DocumentPipelineState(source_path=str(sample_file))
    state = DocumentIngestionCapability().process(state)
    state = DocumentUnderstandingCapability().process(state)
    return state.structured


@pytest.fixture
def runtime(tmp_path):
    """A runtime on the reference corpus, simulated backend, offline enforced."""
    from leanlm.runtime.corpus import CorpusStore
    from leanlm.runtime.runtime import LeanLMRuntime

    store = CorpusStore(tmp_path / "runtime.sqlite3")
    engine = LeanLMRuntime("development", corpus=store, backend="simulated",
                           verify_model=False)
    engine.ingest([DATASET])
    yield engine
    engine.close()
