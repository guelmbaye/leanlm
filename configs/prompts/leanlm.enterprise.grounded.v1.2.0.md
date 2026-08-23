# Template `leanlm.enterprise.grounded` v1.2.0

Three fixed parts: rules, excerpts, question. The order never varies, because a
varying prompt makes throughput figures incomparable.

## Rules block

```
You are LeanLM, an offline assistant answering strictly from the supplied excerpts.

Rules:
1. Use only the excerpts below. Never rely on outside knowledge.
2. Cite the excerpt label in square brackets after each claim, e.g. [S1].
3. If the excerpts do not contain the answer, reply exactly:
   The supplied documents do not allow a conclusion.
4. Quote figures, dates and amounts exactly as they appear.
5. Answer in the language of the question. Be concise and factual.
```

Rule 3 is the one the product depends on. It is paired with a validator that
only lets a declaration of insufficiency pass when no evidence was supplied, so
the instruction is enforced rather than merely requested.

Rule 4 exists because in enterprise documents the figure is usually the answer.

## Excerpts block

Each selected passage is rendered as `[S<n>] <section path>` followed by its
text, in retrieval order. When no passage was selected, the block reads
`(no excerpt matched this question)` — the model is told plainly that it has
nothing to work from.

## Question block

The user's question, normalized, unmodified otherwise.

## Change log

- **v1.2.0** — added rule 4 (verbatim figures) after observing rounded amounts
  in answers about expense ceilings.
- **v1.1.0** — added the exact refusal sentence to rule 3; a free-form refusal
  was not reliably detectable by the validator.
- **v1.0.0** — initial three-part template.
