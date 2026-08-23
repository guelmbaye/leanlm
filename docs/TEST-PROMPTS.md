# Choosing the two test prompts

`metadata.json` carries exactly two, and the organisers add two hidden prompts in
the same domain to test for overfitting. All four are judged on the model's
responses.

## The constraint that decides everything

**The profiler loads the GGUF and runs it directly.** LeanLM's retrieval is not
in the path. So a prompt that assumes retrieved excerpts —

> What is the reimbursement deadline for expense reports?

— is handed to a bare 2B model with no document in front of it. It will invent a
number. That is the worst possible answer to show a judge, and it would be *our*
fault, not the model's.

**Every prompt must carry its own context.** Paste the document into the prompt.
This is not a workaround: it is exactly what LeanLM does at scale, one selected
passage at a time, and it lets the same prompt be answered well by a model
holding nothing in memory.

## What a 2B model does well, and badly

| Plays to its strengths | Exposes its limits |
|---|---|
| extracting a figure from supplied text | recalling facts from training |
| summarising into a fixed structure | multi-step arithmetic |
| rewriting tone or format | long chains of inference |
| saying a document does not cover something | open-ended advice |

Choose from the left column. A prompt that asks a small model to know things is
a prompt designed to fail.

## Do not tune them

Two hidden prompts arrive in the same domain. A prompt selected because *this*
model happens to answer it well buys nothing on the hidden pair and signals that
the visible pair was chosen for effect. Pick two that are representative of the
domain, and accept the score they honestly produce.

## The two we submit

### tp_001 — grounded extraction with calibrated refusal

```
You are answering questions about an internal company policy. Use only the
policy text provided. If the policy does not state something, say so
explicitly rather than guessing.

POLICY EXTRACT
Expense reports must be submitted within 15 calendar days of the expense.
Approved reports are reimbursed within 30 working days from the date of
approval by the line manager. Daily ceilings apply per person: lunch 25 EUR,
dinner 35 EUR, accommodation 120 EUR. Traffic fines are never reimbursed.

QUESTIONS
1. How long after approval is an expense report reimbursed?
2. What is the ceiling for dinner?
3. Are parking fees reimbursed?

Answer each question in one sentence, quoting figures exactly as written.
```

Question 3 is not covered by the extract. A model that answers it invents; a
model that says the policy does not address parking fees is doing the harder and
more useful thing. This is the single most important behaviour for enterprise
document work, and it is visible in one response.

Questions 1 and 2 test a trap worth having: the extract contains **15 days** and
**30 working days**, and **25** and **35 EUR**. Picking the wrong one of a
neighbouring pair is the realistic failure, not an exotic one.

### tp_002 — structured transformation

```
Summarise the following contract clauses for a project manager who has five
minutes. Produce exactly three bullet points: one on duration, one on payment,
one on how to end the contract. Include every date and amount. Do not add
anything the clauses do not say.

ARTICLE 3 - TERM
This contract is entered into for a term of twelve (12) months from the date of
signature. It renews by tacit agreement for successive periods of six (6)
months.

ARTICLE 5 - PAYMENT TERMS
Invoices are payable within 45 days end of month. Late payment incurs a penalty
at three times the statutory interest rate, plus a flat recovery indemnity of
40 EUR.

ARTICLE 11 - TERMINATION
Either party may terminate by giving three (3) months' written notice sent by
registered letter.
```

Everyday enterprise work: someone has a contract and five minutes. It tests
instruction-following (*exactly three bullets*, one topic each), figure
retention across a longer input, and restraint — the clauses say nothing about
penalties for early termination, and a model that adds some has failed.

## Test them before submitting

Run each against the bare model, the way a judge would:

```bash
leanlm ask --dry-run "anything"          # shows the invocation shape
llama-cli -m models/<model>.gguf -f prompts/tp_001.txt -n 256 -c 4096 -t 4 \
  --temp 0.2 --no-display-prompt --single-turn
```

Read the answer as a sceptic. Is every figure right? Did it refuse what it
should have refused? Did it add anything? If the answer disappoints, the fix is
the model or the quantization — not a friendlier prompt.
