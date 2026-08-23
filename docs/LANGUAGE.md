# Language policy

**English is the default. French is a demonstrated capability, not the medium.**

Set after reading the published rules: the challenge, the submission template,
the report and the video are English, documentation quality is scored inside the
50% accuracy weight, and the accuracy benchmark is multiple-choice with no
indication it is anything but English.

## What goes in which language

| Artefact | Language | Why |
|---|---|---|
| Code, identifiers, comments | English | already the case |
| ADRs, EDB, traceability, README | English | already the case, and scored |
| `REPORT.md` (submission) | English | read by judges |
| Two-minute video | English | read by judges |
| **Reference corpus** | **English** | the accuracy benchmark is the thing being scored |
| **`evaluation.json`** | **English**, plus a French section | the probes must mirror the benchmark |
| Workspace UI | English default, French available | the demo is watched in English |
| Demonstration script | English | Act 6 of the pitch |

## What we do *not* do

**We do not delete the French corpus.** It is the evidence for two claims that
are hard to make otherwise:

- *multilingual capability* — the same pipeline, unmodified, answering correctly
  in a second language, is a stronger statement than a claim of multilingual
  support;
- *the African Use Case Bonus* — French is a working language across many of the
  eligible countries, Morocco included.

So: `datasets/enterprise_en/` becomes the default corpus and the accuracy set;
`datasets/enterprise/` stays as the French set, with its own probes, and gets one
slide in the demonstration.

## What this costs, honestly

The tokenisation work that accuracy currently depends on — elision splitting,
enclitic stripping, accent folding, the off-topic gate threshold — was tuned on
French. None of it *breaks* on English, but none of it was measured on English
either. Translating the corpus without re-running `leanlm accuracy` on the
English set would be replacing a measured 100% with an assumed one.

The order matters: translate, then measure, then report. Not the reverse.
