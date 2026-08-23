# Corpus `enterprise_large` (generated)

> Not shipped in the repository. Rebuild it with `make corpus`; the same seed
> produces the same corpus byte for byte, so shipping it would only add 41 files
> that the generator already guarantees.

41 documents, roughly 8356 estimated tokens.

Rebuild with `scripts/make_corpus.py --seed 20260601 --documents 40`.
Deterministic: the same seed produces the same corpus, byte for byte.

Probe question: "Quel est le plafond de delegation de signature pour un chef de projet ?"
Correct answer: 7500 EUR, stated in `z_delegation_signature.md`.

That document is placed last on purpose. A naive prompt truncated to the model
window cuts it off, so the baseline cannot answer correctly for a reason that
has nothing to do with the model's quality. LeanLM retrieves it because it never
tried to send the whole corpus in the first place.

**This file lives outside the corpus directory deliberately.** It states the
correct answer, and a corpus that contains its own answer key measures nothing.
