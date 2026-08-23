# Positioning — what to say, and in what order

Derived from the strategy synthesis. It exists because the repository is full of
internal machinery that is excellent evidence and terrible storytelling.

## The line

> Most AI assumes the cloud. LeanLM assumes the laptop.

Then: *a useful, fully offline language model engineered to run fast and
efficiently on the 8 GB laptops Africa already has.*

## What a jury must understand in ten seconds

```
local LLM → no cloud → 8 GB laptop → fast → low memory → real work
```

Then, immediately: unplug the network, and ask a real question.

## Order of the demonstration

1. **The problem** — modern AI assumes the cloud.
2. **The constraint** — show the machine: 8 GB, integrated graphics, no GPU.
3. **The gesture** — unplug the network.
4. **The intelligence** — a real question from the chosen domain, answered locally.
5. **The proof** — the panel: offline, model, quantization, tokens/s, memory,
   temperature, latency. It is the "La preuve" section of the workspace, visible
   in demonstration mode without touching anything.
6. **The message** — this is not AI running in the cloud. This is useful
   intelligence running on the laptop Africa already has.

## What NOT to lead with

COA, DIC, IEC, CCM, PEM, RPCB, TILAB, Pipeline-as-Code, the ten phases, the
capability registry. Every one of them is real and every one of them is evidence
— *when a technical judge digs*. None of them is the story.

The workspace enforces this: demonstration mode shows the answer, its sources and
the proof panel. Engineering mode reveals the trace rail, the budget decisions
and the full execution context. The order is deliberate.

## Where the acronyms belong

| Question a judge asks | What to open |
|---|---|
| "How do you know it's actually offline?" | the offline guard, and its test |
| "How do you know the answer is right?" | `leanlm accuracy` — ground truth, 12 probes |
| "What would this score?" | `leanlm score` — the rubric applied to our own run |
| "Is this reproducible?" | profile fingerprint, corpus checksum, temperature 0 |
| "Where does the time go?" | `leanlm profile` |
| "Is the architecture real or a diagram?" | `leanlm ccm` |

## The risk named in the synthesis

**Overengineering.** The documentation is sufficient; no further concepts should
be added to appear innovative. What remains is measurement:

```
Build → Measure → Optimize → Validate → Demonstrate
```

Everything added from here should move a number, or it should not be added.
