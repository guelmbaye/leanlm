# Choosing the model

LeanLM ships no weights. The submission gate refuses any package containing a
`.gguf` (SUB-002), and the choice is yours to make because it is yours to
defend: the model manifest declares an identity, a source URL, a checksum and a
**licence**, and a jury may ask about all four.

## What has to fit

On an 8 GB laptop, the arithmetic that matters:

```
8 GB total
 −  ~2.0 GB  operating system and whatever else is open
 −  ~0.3 GB  LeanLM itself, corpus included
 =  ~5.7 GB  left for the model, its KV cache and headroom
```

A model whose file is 2.2–2.8 GB leaves room for the context window and for the
machine not to swap. Swapping is the difference between a usable tool and an
unusable one, and it is invisible in an average — which is why LeanLM reports
**peak** RSS rather than final (EDB-015).

That points at **3–4B parameters, quantized Q4_K_M**. Below 3B the answers get
noticeably weaker on French policy text; above 4B the file grows past 3 GB and
the headroom disappears.

## What to check on the model card

Verify each of these yourself rather than taking a recommendation on trust —
model cards change, and a licence you misreported is worse than a model you
chose imperfectly.

| Check | Why it matters here |
|---|---|
| **Licence** | it goes in `metadata.json` and the jury can read it. Apache-2.0 and MIT are unambiguous; some community licences carry conditions worth reading before the deadline |
| **Instruction-tuned** | LeanLM sends a rules-and-excerpts prompt. A base completion model will not follow "reply exactly: the supplied documents do not allow a conclusion" |
| **French** | the reference corpus is French. Multilingual training matters more than raw parameter count for this corpus |
| **GGUF available** | a conversion step you have to perform yourself is a step that can go wrong the night before |
| **Q4_K_M variant** | the usual quality/size compromise. Q5_K_M is better and larger; Q3 is noticeably worse |

## Recommendation (checked August 2026)

**Qwen3.5-4B, Q4_K_M.** Apache-2.0, strongly multilingual, roughly 2.5 GB, and
the current consensus first pick for CPU-only machines in this size class.

The deciding factor for this project is not speed — it is French. Phi-4-mini is
smaller and faster and would win a benchmark on an English corpus, but it was
trained primarily on English and underperforms elsewhere. Our corpus is French
enterprise policy text, so a model that is 20% faster and wrong about
`indemnité forfaitaire` costs us the 50% accuracy weight to buy back part of the
30% throughput weight. That trade is not worth making.

| Model | Size (Q4_K_M) | Licence | Why / why not |
|---|---|---|---|
| **Qwen3.5-4B** | ~2.5 GB | Apache-2.0 | recommended: multilingual, permissive, fits with headroom |
| Llama 3.2 3B | ~2.0 GB | Llama Community | good fallback if 4B is too slow; licence has attribution conditions |
| Phi-4-mini | ~2.5 GB | MIT | fast and permissive, but English-first — wrong tool for a French corpus |
| Mistral 7B / Qwen3 8B | ~4.4 GB+ | Apache-2.0 | too cramped as a daily model on 8 GB |

Verify the licence on the exact repository you download. Official weights and
community GGUF re-quantizations live in different repos, and lookalike
repositories exist — check the publisher, not the name.

### Two settings that matter more than the model choice

**Cap the context window.** Qwen3.5 advertises 262 144 tokens natively. Loading
it that way on an 8 GB laptop is how you get an out-of-memory kill, because the
KV cache scales with context, not with the weights. Leave
`model.context_tokens: 4096` in the profile. A large advertised window is a
model-card feature; on CPU it is a memory bill.

**Check for thinking tokens.** Recent Qwen models can emit reasoning before the
answer. LeanLM's prompt asks for an exact refusal sentence and its validator
matches on it, so a model that thinks out loud first can wreck the refusal
accuracy — which is scored. Run `leanlm accuracy` immediately after installing:
if refusal accuracy drops below 100%, disable thinking mode or pick the
non-thinking instruct variant. That single command turns a silent failure into a
number.

## Fetching it

```powershell
# Windows
.\scripts\download_model.ps1 -Url "<direct link to a .gguf>"
```

```bash
# Linux / macOS
MODEL_URL="<direct link to a .gguf>" ./scripts/download_model.sh
```

Both derive the filename from the URL, so candidates land side by side:

```
models/Qwen3.5-2B-Q4_K_M.gguf
models/Qwen3.5-4B-Q4_K_M.gguf
```

That is not tidiness. A fixed `model.gguf` means the second download silently
replaces the first, and every measurement already attributed to "the model"
becomes wrong with nothing to signal it. Override with `-Destination` or
`MODEL_FILE` when you do want a specific name.

Both refuse a file that does not begin with the GGUF magic bytes, which is what
an HTML error page saved under a `.gguf` name looks like. Both print the
SHA-256 when you have not declared one.

## Wiring it in

The default path is `models/model.gguf`, relative to where you run `leanlm`.
Point at something else either way:

```powershell
leanlm --model models\qwen2.5-3b-instruct-q4_k_m.gguf doctor
```

or permanently, in `configs/runtime/competition.yaml`:

```yaml
model:
  id: qwen2.5-3b-instruct
  path: models/qwen2.5-3b-instruct-q4_k_m.gguf
  quantization: Q4_K_M
  parameters: "3.1B"
  context_tokens: 4096
  sha256: "<the checksum the download script printed>"
  source_url: "<the exact URL you fetched>"
  license: "<as written on the model card>"
```

Then:

```powershell
leanlm doctor          # model: verified
leanlm accuracy        # the heaviest criterion, now with a real model
leanlm bench --with-profile --save-baseline
leanlm score           # throughput stops being zero
```

`sha256`, `source_url` and `license` are not decoration: SUB-004 and SUB-005
block the submission until they are filled in, because an evaluator who cannot
fetch the exact file you measured cannot reproduce anything you claim.

## If the in-process backend will not install

`llama-cpp-python` compiles, and on Windows its source archive exceeds the
260-character path limit unless long paths are enabled. You do not need it.
Download an official llama.cpp release and use either:

```powershell
.\llama-server.exe -m models\model.gguf --host 127.0.0.1 --port 8080
leanlm --backend llama-server ask "..."

leanlm --backend llama-cli --model models\model.gguf ask "..."
```

The server backend refuses any address that is not loopback, so it cannot
quietly become a cloud dependency.
