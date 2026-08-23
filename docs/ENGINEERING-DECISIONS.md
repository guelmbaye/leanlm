# Engineering Decision Book

Decisions that shaped LeanLM's behaviour, with the evidence behind them and the
conditions under which each should be revisited. A decision without a stated
reversal condition is a preference, not an engineering decision.

---

## EDB-004 — BM25 rather than embeddings for evidence retrieval

**Decision.** Retrieval uses Okapi BM25 over the segmented corpus, with
structural boosts (section title match, heading proximity, exact phrase,
numeric question) and a greedy MMR pass for diversity.

**Why.** An embedding model is a *second* model resident in RAM. On an 8 GB
laptop already holding a 2.6 GB quantized LLM, a 400 MB encoder plus its index
competes directly with the thing the user came for. BM25 needs no model, no
index build ahead of time, and no GPU.

**Measured.** On the reference corpus, retrieval completes in 3.7 ms with
precision 1.0 on the enterprise question set. The passages a semantic search
would add are, in this domain, mostly paraphrases of passages BM25 already
found — enterprise policy documents repeat their own vocabulary.

**Reverse this if.** The corpus becomes conversational rather than
documentary, or questions start being asked with vocabulary that does not
appear in the documents. A hybrid (BM25 candidates re-ranked by a small
encoder) is the first step, not a replacement.

---

## EDB-006 — Grounding by n-gram overlap, not by a second model call

**Decision.** `ResponseValidation` scores each answer sentence by 3-gram
overlap against the evidence actually sent to the model, checks citation labels
against the real label set, and derives a qualitative confidence level.

**Why.** Asking a model to grade its own output costs a second inference pass —
on this hardware, roughly doubling the latency of every request — and produces
a number that is itself unverifiable. N-gram overlap is deterministic, costs
0.4 ms, and a reader can reproduce it by hand.

**What it catches.** Invented figures, fabricated citation labels, and answers
drawn from outside the supplied passages. Verified by
`tests/capability/test_validation.py`.

**What it does not catch.** A fluent answer that recombines real passages into
a false claim. This is stated in the submission report's "Honest limits"
section rather than hidden.

---

## EDB-009 — A figure makes a passage unique

**Decision.** Near-duplicate elimination compares numeric signatures *before*
Jaccard similarity. Two passages that differ only by a number are never treated
as duplicates.

**Why.** In enterprise documents, "reimbursed within 30 days" and "reimbursed
within 45 days" are 96% lexically identical and mean different things. The
figure is usually the answer the user came for. A deduplicator that collapses
them silently deletes the correct answer and keeps a plausible wrong one.

**Cost.** A handful of near-duplicates survive into the context. Measured
compression fell by about one percentage point. Accepted without hesitation.

---

## EDB-011 — No evidence means no answer

**Decision.** When no passage scores above the absolute floor, retrieval returns
*nothing* rather than the best of a bad set, and the validator only lets a
declaration of insufficiency pass.

**Why this was not the original behaviour.** The first implementation fell back
to the top three passages whenever the floor filtered everything out. Asked for
Tesla's share price against a corpus of HR policies, the system answered with a
verbatim quote about annual leave — scoring a perfect grounding rate and *high*
confidence, because the answer was faithful to a passage that had nothing to do
with the question.

**Two fixes, because one was not enough.**
1. Retrieval returns an empty set when nothing clears the floor.
2. High confidence now additionally requires that the answer share at least one
   term with the question. Faithfulness to the wrong passage is not confidence.

**Evidence.** Scenario S6 asserts this on every benchmark run; the campaign
fails if the system answers.

---

## EDB-013 — The simulator is never a fallback

**Decision.** When no GGUF model is available, LeanLM runs a clearly labelled
extractive simulator so that development and testing can proceed. It is
labelled `is_simulated=true` on every artifact, warned about in the IEC, shown
as a banner in the workspace, and **refused by the submission gate** (SUB-006).

**Why.** A simulator that quietly stands in for a model is how a project ends up
reporting throughput figures no model ever produced. The failure is not
technical, it is a failure of honesty, and it is best prevented mechanically.

**Evidence.** `tests/submission/test_delivery.py::test_a_simulated_campaign_cannot_be_submitted`.

---

## EDB-015 — Peak, not final, resource usage

**Decision.** Memory is reported as the peak RSS observed during a request, with
the minimum available RAM alongside it.

**Why.** A run that briefly touched 7.6 GB on an 8 GB laptop nearly failed, even
if it ended at 2 GB. On the target hardware the peak is the number that decides
whether the machine swaps, and swapping is the difference between 11 tok/s and
an unusable tool.

---

## EDB-017 — "No sensor" is not "cool"

**Decision.** Every resource observation carries a `thermal_source`. When no
sensor is exposed, the temperature field is `null` and the source says
`unavailable` — never a default value.

**Why.** A thermal claim is one of the competition's evaluation criteria. A
report that prints `41 °C` because that is what a missing sensor defaults to is
worse than a report that admits it could not measure.

---

## EDB-019 — Caching the corpus snapshot, because the profiler said so

**How this was found.** `leanlm profile` on the reference corpus reported
`document_discovery` consuming **37.9%** of a request — more than context
optimization and retrieval combined. The cause was visible in the hot-spot list:
`com.py::sealed` at 41 ms cumulative over 123 calls. Every question re-read the
whole corpus from SQLite and re-computed a content checksum for every semantic
unit, to rebuild a snapshot that had not changed since the previous question.

**Decision.** `CorpusStore` caches the `ContextPackage` per document selection,
and the corpus checksum alongside it. Every write path — `upsert_document`,
`remove_document`, `clear` — invalidates both.

**Why this is safe here specifically.** The snapshot is an immutable canonical
object whose id is derived from its content. A cached package and a freshly built
one are byte-identical, so caching cannot change what a run does — only how long
it takes. That property is what makes this a cache rather than a liability.

**Measured, same machine, same corpus, same profile:**

| | before | after |
|---|---|---|
| total request | 17.57 ms | **10.66 ms** |
| `document_discovery` share | 37.9% | out of the top four |

A 39% reduction in layer latency, which is close to what the profile predicted —
the useful confirmation being that the prediction and the outcome agreed.

**The risk this introduces.** A stale read after a write. That risk is where the
tests went: `tests/integration/test_runtime.py::TestCorpusCaching` asserts
invalidation on adding, removing and clearing, and that a document subset is
cached separately from the whole corpus. A cache that can go stale is a
correctness bug wearing a performance costume.

**Reverse this if.** The corpus becomes large enough that holding a snapshot in
memory competes with the model for RAM. The measurement to watch is peak RSS on
scenario S3; at that point the cache should become a bounded LRU rather than
disappear.

---

## EDB-021 — Measuring accuracy, and what it immediately caught

**Why this came late, and why that was a mistake.** The rubric weights accuracy
at 50%. Until this module existed LeanLM measured grounding, compression,
latency and memory — every proxy for "the answer is good", and no check that the
answer was *correct*. Twelve ground-truth probes now sit next to the corpus, ten
answerable and two not, each stating the figure the corpus contains.

**What the first run found.** The system answered "quelle est la politique de
remboursement des frais de scolarite des enfants ?" — a topic the corpus does
not cover — with content from the expense policy. The overlap is ordinary
vocabulary (`remboursement`, `frais`); the words that carry the question
(`scolarite`, `enfants`) appear nowhere. Grounding was perfect, the citation was
real, and the answer was about something else entirely. The Tesla probe does not
catch this because it shares no vocabulary at all.

**The gate.** Retrieval now rejects a question before scoring it when 40% or more
of its content words are absent from the corpus vocabulary.

**Then it broke five correct answers, which is the useful part.** Accuracy fell
from 100% to 50%. Four defects in tokenisation, each real and each invisible
until something depended on it:

| Defect | Effect |
|---|---|
| interrogatives (`quel`, `combien`) counted as content | every question looked partly unknown |
| section titles never indexed | the corpus had "never heard of" its own topics |
| French elision (`l'indemnite`) left glued | a token no corpus can contain |
| inverted verbs (`faut-il`) left glued | same |

Fixing those took accuracy to **100% answer, 100% source, 100% refusal, 0%
hallucination**.

**The honest caveat.** Twelve probes is a small set, and the 0.4 threshold was
chosen while looking at it. That is overfitting risk, plainly. The parts that
generalise are the tokenisation fixes, which are correct independently of any
probe set; the threshold is the tunable, and it should be re-checked the first
time the corpus changes. A larger probe set is the obvious next investment.

---

## EDB-022 — Computing our own score, and labelling it provisional

**Decision.** `leanlm score` applies the rubric — 50% accuracy, 30% throughput,
20% efficiency, thermal penalty, disqualification on out-of-memory or crash — to
our own campaign, from constants in `configs/scoring.yaml`.

**Why a score and not just metrics.** Fifteen metrics tell you the system works.
They do not tell you what it would score. Only the second question decides
anything, and the difference showed immediately: the provisional score is 70%,
not because anything is broken but because throughput is unmeasurable without a
real model. That is a planning fact worth knowing early.

**Two deliberate refusals.**

*Disqualification is a cliff, not a penalty.* An out-of-memory kill or a crash
returns zero and short-circuits the rest. On an 8 GB target it is the most
expensive failure available, and modelling it as a deduction would let a strong
accuracy figure disguise it.

*A missing measurement scores zero, never a guess.* With no accuracy file the
accuracy component is zero and says `NO GROUND TRUTH` in its basis field.
Assuming it would be fine is how a planning number quietly becomes a claim.

**The label.** Every score carries `rubric_source: assumed`, and while it does,
the CLI, the JSON and the report all read *provisional*. The weights come from
our strategy synthesis, not from a rulebook in hand. When the official rubric
arrives, edit the YAML and set the field to `official` — no code changes.

---

## EDB-023 — What a Windows laptop found in ten minutes

Three defects, reported from a first run on Windows 11 in a virtualenv. None of
them was visible on Linux, and the first two were only visible because someone
ran the product on the machine it was designed for.

**1. Every memory reading was 0 MB.** The resource sampler's fallbacks read
`/proc/meminfo` and `/proc/self/status`. Windows has neither, psutil was not
installed, and the sampler returned zeros without complaint. Fixed with a Win32
path through `ctypes` — `GlobalMemoryStatusEx` for system memory,
`GetProcessMemoryInfo` for the working set — so the zero-dependency promise holds
on Windows as it does on Linux.

**2. The zeros disqualified a healthy run.** This is the one that mattered. The
scoring module read `available_ram_mb: 0.0` as an out-of-memory kill and returned
**DISQUALIFIED, score 0**. A silent sensor had been turned into a catastrophic
result.

The principle was already written down for thermal readings — *"no sensor" is not
"cool"* — and simply had not been applied to memory. It is now: an unmeasured
value never disqualifies. It still scores zero for efficiency, because not
disqualifying is not the same as assuming things are fine, and the warning says
which sampler was active so the gap is attributable.

**3. Correctly accented French did not retrieve.** The terminal mangled `délai`
into `dlai`, which is unfixable from our side — but investigating it exposed
something that is: `télétravail` typed properly matched *nothing*, because the
corpus spells it `teletravail`. A user typing correct French was being penalised
for it.

Tokenisation now folds combining diacritics. Matching only: stored text,
displayed answers and artifact checksums keep their accents. Accuracy stayed at
100% across the probe set, and `télétravail` now finds the teleworking policy.

**The lesson worth keeping.** Two of these were invisible to a test suite that
passes on Linux, and the second was a measurement failure wearing the costume of
a product failure. Cross-platform coverage is not a nice-to-have for something
whose entire premise is *the laptop you already have*.

---

## EDB-025 — Translating the corpus, and what English broke

English became the default after reading the published rules: the challenge, the
report and the multiple-choice accuracy benchmark are English, and documentation
quality is graded *inside* the 50% accuracy weight.

The translation was done figure-for-figure so the two probe sets mirror each
other and the languages stay comparable. Then it was measured, which was the
point: **English scored 83%, not 100%.** Two failures, two distinct defects,
neither visible in French.

**1. Structural boosts were additive, so they faded as the corpus grew.**
"What notice period is required to terminate the service contract?" retrieved
the *remote work* termination clause. Both documents say "either party may
terminate ... notice"; only one is a service contract, and its heading says so.
The section-match boost fired correctly — and added 0.25 to a BM25 score of 5.
Five percent. On a larger corpus it would have been less.

A boost expressed as a fraction should scale with what it modifies:
`total = lexical × (1 + boost)`. A 0.22 section match now means +22% at any
corpus size. This is a defect the French set could not have exposed, because
`prestation` was distinctive enough to win on lexical score alone.

**2. The off-topic gate was tuned on French and refused correct English.**
"How long does the confidentiality obligation last after the contract ends?" was
refused: `long`, `last`, `ends` and `obligation` are absent from a corpus that
plainly discusses confidentiality, giving an unsupported ratio of 0.57 against a
0.4 threshold.

Raising the threshold was tried first and **does not work**. There is no value
that accepts that question (0.57) while rejecting "reimbursement of children's
school fees" (0.40). At 0.6 English reached 100% and French fell to 92%. The
ratio alone cannot separate them.

What separates them is *which* words are missing. In the school-fees question the
unsupported words are the subject; in the confidentiality question they describe
the shape of the answer — duration, occurrence, extent. So the gate now excludes
a small list of generic framing words from both sides of the ratio, and the
threshold went back to 0.4. Both languages: **100% answer, 100% source, 100%
refusal, 0% hallucination.**

The generic list applies to the gate only. Those words stay in the scoring
vocabulary, because "3 days allowed" is a phrase worth matching on.

**3. The simulator refused in French regardless of the question's language.**
Cosmetic-looking, and not: the validator detects insufficiency by matching the
exact sentence the prompt asked for, so a French refusal to an English question
turns a *correct* refusal into a scored failure.

**What this cost, and what it bought.** Two hours, and the discovery that a
number I reported as 100% was measuring a system tuned to one language. The rule
holds: translate, then measure, then report — and never the reverse. The French
corpus stays, with its own probes and its own assertion in the suite, because a
bilingual claim is only evidence while it is still measured.

---

## EDB-027 — Designing a submission format instead of reading the published one

For most of this project the packager emitted nine files: `metadata.json`,
`REPORT.md`, `download_model.sh`, `model/model_manifest.json`,
`runtime/runtime_profile.yaml`, `README.md`, an `evidence/` directory and a
`MANIFEST.sha256`. Eleven compliance checks verified them. All of it was
inferred from a blueprint.

The official template asks for **four files and a directory**:

```
metadata.json  download_model.sh  REPORT.md  .gitignore  model/
```

`model_manifest.json`, `runtime_profile.yaml`, `MANIFEST.sha256` and `evidence/`
are ours and an evaluator reads none of them. `metadata.json` is a strict schema
that shares no field with the one we were writing — `team_id`,
`language_scope`, `african_alpha_claim`, `budget_laptop_claim`,
`cross_disciplinary_pairing`, exactly two `test_prompts`, `_runtime.model_path`.
`domain` is an enum: `corporate_enterprise`, where we shipped
`"Corporate / Enterprise"`.

**What was actually wrong.** Not the engineering — the packager worked, the gate
blocked, the tests passed. What was wrong is that a green build meant nothing,
because it verified conformance to a format nobody would evaluate. A test suite
is only as truthful as the specification it encodes.

**What we kept.** Two things the template does not require and that survive on
merit: the download script verifies GGUF magic bytes and the SHA-256 before
keeping a file, and the compliance gate refuses to package results produced by
the simulated backend. Neither is asked for. Both prevent a specific,
foreseeable failure.

**The check we can no longer pass ourselves.** `SUB-023` asks whether the
official profiler produced a report, and it is advisory rather than blocking
because we cannot run `adtc-profiler` without `llama-bench` and a real model.
It says plainly that our cProfile output is not a substitute — an advisory check
that names what is missing is more useful than a blocking one that cannot be
satisfied.

**The rule this leaves behind.** When an organiser publishes a format, read it
before building against it. Where no published format exists, say `assumed` in
the artifact — as `configs/scoring.yaml` did for six weeks, which is exactly why
correcting it took an afternoon rather than a rewrite.

---

## EDB-029 — Asking the binary instead of assuming the flag

Three abandoned runs on a Windows laptop, each interrupted after minutes of
silence, traced to one line: `-no-cnv` is not a valid argument in llama.cpp
build b10590. The flag that disables conversation mode has been spelled
differently across releases, and this build rejects that spelling outright.

**Why it presented as a hang rather than an error.** The backend caught the
rejection and retried with the flag removed -- which is precisely what must not
happen. Without it llama-cli enters conversation mode, waits for input, and the
run ends at the timeout. A clear, immediate error had been converted into an
unexplained five-minute wait. The retry was written to be resilient and its
effect was to hide the diagnosis.

**Decision.** The backend runs `llama-cli --help` once and builds its command
from what the binary says it accepts:

- the conversation flag is chosen from `--no-conversation`, `-no-cnv`,
  `--single-turn`, `-st`, newest spelling first;
- optional niceties (`--no-display-prompt`, `--simple-io`) are included only
  when advertised, because they are worth having and not worth failing over;
- sampling parameters are never conditional: they decide what the model
  produces;
- a rejected argument now raises **INF-125** naming the argument, instead of
  being stripped.

`leanlm doctor` prints which flags were detected, so the answer is visible
before a request is ever made.

**What this cost, and what made it slow to find.** The diagnostic that should
have exposed it built its own, shorter command list: `--seed`, `--top-k`,
`--top-p`, `--repeat-penalty` and `--simple-io` were omitted from what
`--dry-run` printed. The offending flag *was* shown, but the discrepancy meant
the printed command could not be trusted as the command that ran. Both are now
built by one function, and a test compares them.

**The rule.** When an external tool's interface can vary, ask it. `--help`
costs one subprocess call at load time and removes a whole class of
version-guessing from the code.

---

## EDB-030 — Two kinds of scaffolding, and why the validator was scoring them

A run on Windows produced `grounding 0.0` and "5 sentences not supported by the
excerpts" on a request where the model had made **no claim at all**. What the
validator scored was scaffolding.

**The tool's.** `llama-cli` b10590 is a chat application. It prints a banner, a
command list and the prompt itself to stdout before generating, and
`--no-display-prompt` is not honoured once that UI is up. `--single-turn` limits
the exchange to one turn; it does not remove the wrapper.

**The model's.** Qwen3.5 is a hybrid reasoning model. It opened a thinking block,
spent the whole 128-token budget on it, and was cut off mid-thought. Nothing that
could be called an answer was ever emitted.

**Decision.** `packages/inference/cleaning.py` strips both, conservatively: if
stripping would empty the text, the original is returned and the caller decides.
Silently emptying a response is how a generation failure becomes an accuracy
figure -- which is precisely what had happened, twelve probes at 0%.

When nothing survives, the backend raises **INF-126** naming both causes and the
configuration to change.

**What the notes are for.** A run that needed scaffolding removed is a run whose
configuration is not right yet, so the removal is reported rather than tidied
away. The right answer here is not better stripping: it is the `llama-server`
backend, which has no interactive wrapper, and disabling the reasoning mode.

**The measurement this invalidates.** Every timing from that run. "first token
36343 ms" was the banner appearing, not a token; "19 tok/s" counted banner text.
A number measured through scaffolding measures the scaffolding.

---

## EDB-031 — UTF-8, and a parser held hostage by one character

`leanlm speed` reported "no readable results" on a `llama-bench` table that was
entirely correct. Two faults, one cause.

**The cause.** llama.cpp writes UTF-8. Python decodes a subprocess with the
platform default, which on Windows is the ANSI code page. `±` arrived as `Â±`.
The clue had been visible for several runs already: llama-cli's banner rendered
as `â–„â–„` in the captured output and nobody read it as an encoding signal.

**Why it mattered beyond a parser.** The same stream carries the model's answer.
Any accented character in a French reply was being corrupted the same way --
silently, since a mangled word is still a word. Decoding is now pinned to UTF-8
with `errors="replace"` wherever a subprocess is read: an answer must never be
lost to a byte, and a byte must never be lost in silence.

**The second fault, which the first exposed.** The row parser required the value
to be followed by `±`. A parser that depends on one character surviving a decode
is brittle by construction. The final cell is now read as "whatever is not a
pipe", with the value and the deviation extracted from it separately.

**And a defect found while testing that fix.** The first replacement matched
`32.96` as a value of `32` with a deviation of `96` — inventing a dispersion the
benchmark never reported. Reporting no dispersion is a gap; reporting a
fabricated one is a lie about the measurement's precision. The pattern now
requires a genuine separator between the two numbers, and a test covers the
no-deviation case explicitly.

**The rule.** When reading another program's output, pin the encoding and parse
the shape, not the punctuation.

---

## EDB-032 — Saturation breaks the claim that rankings transfer

Earlier in this project I told the user that relative rankings between model
candidates survive a change of machine, even when absolute figures do not. That
is true of a monotone scoring function. It is false of this one.

`S_perf = min(TPS / 15, 1)` is **flat above the reference**. The consequence:

| | 2B | 4B |
|---|---|---|
| on a 2-core laptop (7.0 / 3.5 tok/s) | **66.7** | 59.9 |
| where both saturate | 82.6 | **82.9** |

On slow hardware neither candidate saturates, the throughput term is
proportional to speed, and the smaller model wins on it. On the evaluation
machine both may clear 15 tok/s, at which point the term is identical for both,
cancels, and the decision passes to accuracy -- which favours the larger model.

The ranking does not merely shift. It inverts.

`leanlm candidates` now detects this: when any candidate is below the reference,
it re-scores with everything saturated and says plainly whether the winner would
change. If it would, the output states that the ranking is hardware-dependent
and that the choice must be made on hardware close to the profile.

**The general form.** A ranking transfers across machines only where the scoring
function is monotone in the thing the machine changes. A cap, a floor or a step
anywhere in the rubric breaks that, and the break is invisible in the numbers
themselves -- both models simply look slow.

---

## EDB-033 — Scoring a scratchpad, and reporting 83%

The first real accuracy run on conforming hardware reported **83% overall, 100%
source accuracy, 0% hallucination**. Every one of those numbers was wrong,
because the model had not answered a single question.

Qwen3.5 reasons before answering. With a 128-token budget it produced twelve
unfinished drafts:

```
3. **Locate Information:**
 * Row 1 under "Category": "Lunch".
 * Column "Ceiling" corresponding to "Lunch": "25 EUR".
4. **Draft Answer:**
 * Based on Excerpt [S1], the ceiling for lunch is 25 EUR.
5. **Review Constrain
```

Cut off mid-word. The scorer looked for `25` in the text, found it inside the
draft, and marked the probe correct. Eight of ten answerable probes were scored
on a scratchpad.

**Why the existing guard missed it.** `cleaning.py` strips reasoning marked with
`<think>` or `[Start thinking]`. This model emits bare numbered steps with bold
headers -- no marker to match. Pattern-matching prose was never going to be
reliable here.

**The signal that is reliable.** `generated_tokens >= max_output_tokens` means
the generation was cut off by the budget, whatever it looks like. A truncated
response is now flagged at the backend and can never be scored correct, in
either direction: a truncated refusal is not a successful refusal either.

**The second fix, which addresses the cause.** `prompt.thinking_suffix` appends
an instruction to answer directly (`/no_think` for this family), configured in
all four shipped profiles now that the model is chosen. Leaving it to be
discovered costs an entire measurement run, which on this timeline is most of a
day.

**What this says about the accuracy number generally.** A scorer that searches
for a substring will find it wherever it appears, including in text the model
never meant as an answer. The check is now: was the generation complete, then
does it contain the fact. Order matters.

---

## EDB-034 — Two truncation signals, and only one that works

The fix in EDB-033 flagged a generation as truncated when
`generated_tokens >= max_output_tokens`. The next run reported
**`truncated: 0.0`** for twelve generations that every one of them ended
mid-word: `5. **Review Constrain`, `* En`, `* Calc`.

The count is parsed out of llama.cpp's stderr timing block and falls back to a
heuristic estimate when the format does not match. A signal that depends on
another program's log format is a signal that can quietly stop working.

**Where the text ends is observable.** A completed sentence ends with terminal
punctuation. `looks_truncated` checks that, and either signal now marks a
generation truncated.

**And the cause, which `/no_think` did not address.** The suffix was added to
the prompt and the model reasoned anyway: Qwen honours it through the chat
template, and LeanLM sends a raw completion prompt with no template at all. So
the budget was raised to 768 tokens in all four profiles, giving the model room
to think *and* answer, with `extract_final_answer` pulling the answer out of the
reasoning that precedes it.

**Why raising it is free.** The throughput component comes from `llama-bench
-p 512 -n 128`, which never reads our configuration. A larger budget costs
wall-clock on our own accuracy runs and nothing at all on the score. That was
worth checking before trading accuracy for speed that was never at stake.

---

## EDB-035 — The cleaner was corrupting the answers it was meant to rescue

Accuracy fell to 8% and the failures said "cut off by the token budget" for
eleven of twelve probes. The token budget was not the whole story: several
answers *began mid-word*.

```
"is direct.\n\n Draft:"
"s the question.\n\n4. **Formulate the Output:**"
", reply exactly: \"The supplied documents do not allow a conclusion.\""
```

Those cut points are not the model's. They are mine. The echo remover searched
the whole output for any line of the prompt and cut everything before it. A
reasoning model quotes its own instructions while working -- *"Answer in the
language of the question. Be concise and factual."* appears mid-thought -- so
the search hit inside the reasoning and removed the first half of the response,
including, in several cases, the answer.

**The distinguishing property.** An echo reproduces the prompt at the head,
contiguously. A quotation appears anywhere. So the remover now confirms the
output *opens* with the prompt before cutting anything, and cuts only as far as
the prompt's last line.

Two smaller faults surfaced while fixing it, both from the same instinct to
delete rather than trim:

- the chat UI's slash-command list survived the banner pass and stood between
  the output and the echoed prompt, so the echo was no longer a prefix;
- dropping lines that begin with the UI's `> ` input marker deleted answers,
  because an answer can follow that marker on the same line. The marker is now
  stripped from the line; the line is kept.

**What this cost.** One measurement run reported 83% on a scratchpad; the next
reported 8% on text the cleaner had mangled. Neither number described the model.
Six shapes taken verbatim from those runs are now regression tests.

**The rule.** A cleaner that removes more than it can justify is indistinguishable
from a broken model, and it fails in the direction that looks like someone
else's fault.
