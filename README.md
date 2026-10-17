# feast AI

A terminal app for chatting with edge-optimised language models on a **CPU
inference engine**, while a live telemetry panel shows exactly what the model
is costing you — and a post-session report turns that into numbers you can put
in a thesis.

The point is not to run an LLM. The point is to answer *"could this model
actually live on a phone?"* — so feast AI constrains the CPU to a phone-class
budget, refuses to cheat with a GPU, and measures the result honestly.

```
┌ feast AI ────────────────────────────────────────────┬─ LIVE TELEMETRY ─────┐
│ you                                                  │ CPU    ███████████░░  │
│ Merhaba! Kendini kısaca tanıtır mısın?               │ RAM    ████░░░░░░░░░  │
│                                                      │ cpu    ▃▅▇█▇▆▅▃▂▁▁▂▄  │
│ Llama 3.2 1B Instruct · Turkish                      │ t/s    ▅▆▇█▇▇▆▆▇█▇▆▅  │
│ Merhaba! Ben feast AI içinde çalışan, cihazınızda    │                       │
│ yerel olarak koşan bir asistanım…                    │  33.0      0.28       │
│ 79 tok · 32.3 t/s · TTFT 0.64s · CPU 90% · RAM 935MB │  tokens/s  TTFT       │
└──────────────────────────────────────────────────────┴───────────────────────┘
```

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m feast_ai runtime
```

```bash
python -m feast_ai pull llama32-1b-tr
```

```bash
python -m feast_ai
```

On first launch feast AI opens Settings so you can pick a device profile and a
system prompt before anything runs. Both stay editable at any time — Settings
from the welcome screen, or `Ctrl+P` mid-conversation for just the prompt.

## Run it on your own machine

**You need** Python 3.10 or newer, a 64-bit CPU (no GPU is used), about 4 GB
of free disk per model you pull, and 8 GB of RAM or more for the 2B-class
models. Windows 11 is the main development machine. Linux works with every
feature. macOS runs, but without the RAM ceiling, because macOS has no
working-set cap to apply. Run every command from the repository root.

**1. Get the code and install.** A virtual environment keeps the
dependencies out of your system Python.

```bash
git clone <this repository> feast-ai && cd feast-ai
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`run.ps1` (Windows) and `run.sh` (Linux/macOS) do the install on first use
and then start the app. `onnxruntime`, `tokenizers` and `numpy` are only for
the intent router, and `onnxruntime-genai` only for the ONNX models. Without
them the app still runs, and those features say what is missing.

**2. Download the inference engine and check your setup.**

```bash
python -m feast_ai runtime      # prebuilt llama.cpp llama-server for your OS
python -m feast_ai doctor       # engines, cores, RAM-cap support, models on disk
```

**3. Pull a model and chat.** `gemma-3-270m` is the quickest to test
(~200 MB). `gemma-4-e2b` is the model the agent results below were measured
on (~3 GB). `python -m feast_ai models` lists the rest.

```bash
python -m feast_ai pull gemma-4-e2b
python -m feast_ai
```

**4. Try the agent.** Pull the router, then switch agent modes in a chat with
`Ctrl+R`, or in Settings → *Agent / intent routing*.

```bash
python -m feast_ai pull modernbert-tr-embed               # 150 MB intent router
python -m feast_ai route "Kaydedilenlerimi gösterir misin" # how one message is routed
```

**5. Reproduce the comparison.** This is the run behind the prompt-cycle
table below: four pipelines on one simulated phone (3 cores, 4 GB). Each
suite takes about 10 minutes on a desktop CPU. Every run writes a Markdown
report, a JSON file and a side-by-side HTML page to `~/.feast-ai/sessions/`.

```bash
python -m feast_ai bench gemma-4-e2b --modes direct,active,classes --suite kanka --cores 3 --ram-gb 4
python -m feast_ai bench gemma-4-e2b --modes direct,active,classes --suite standard --cores 3 --ram-gb 4
```

**Optional: the Decision 2.0 router.** It needs a llama.cpp fork that stock
builds lack, so it is the only part you have to build yourself. You need git,
CMake and a C++ compiler (Visual Studio Build Tools on Windows,
`build-essential` on Linux).

```bash
git clone --depth 1 --branch decision2-gguf https://github.com/Xunzhuo/llama.cpp llama.cpp-decision2
cmake -S llama.cpp-decision2 -B llama.cpp-decision2/build -DLLAMA_CURL=OFF
cmake --build llama.cpp-decision2/build --config Release --target llama-server
```

Copy the built `llama-server` and its shared libraries (on Windows,
`build/bin/Release/*.exe` and `*.dll`) into
`~/.feast-ai/bin/llama.cpp-decision2/`. Then download
[`Decision-2.0-Kai-0.6B-Q4_K_M.gguf`](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B-GGUF)
into `~/.feast-ai/encoders/decision-2.0-kai-0.6b/`. Its SHA-256 is listed in
`feast_ai/engine/decision_server.py`. Add `classes@decision-2.0-kai-0.6b` to
`--modes`, or pass `--router decision-2.0-kai-0.6b`.

**Where things are stored.** Models, runtimes, settings, reports and logs live
in `~/.feast-ai/` (`%USERPROFILE%\.feast-ai` on Windows), never in the
repository. Set `FEAST_AI_HOME` to put them somewhere else, for example on a
bigger drive.

---

## How the edge simulation works

feast AI constrains **CPU only**, and it does so twice:

| Control | Mechanism | Why |
| --- | --- | --- |
| `--threads N` | llama.cpp worker-thread count | Limits how much parallel work the model *creates* |
| CPU affinity | `psutil.cpu_affinity([...])` | Limits which cores the OS may *schedule it on* |
| Priority | process scheduling class | Optional: simulates contention with foreground apps |

Threads alone are not enough. llama.cpp spawns helper threads for loading and
batching, and the scheduler is free to migrate work across every core on the
machine. Affinity pinning turns the budget into a hard ceiling — a "2-core
phone" genuinely cannot exceed two cores' worth of CPU. You can watch this in
the summary: `CPU peak (host)` lands almost exactly on `threads / host_cores`.

**RAM is capped as a working set, not as a commit limit.** This distinction is
the whole design. A commit limit (Windows Job Objects, cgroup `memory.max`)
kills the process the instant it asks for one byte too many, turning every
oversized model into a crash instead of a measurement — which is why feast AI
originally shipped with no RAM cap at all. A *working-set* ceiling bounds only
how much stays resident: overshooting causes the OS to evict pages and fault
them back in. The model gets slow, not dead. That is exactly what a phone does
when the weights do not fit, so it is the honest simulation.

The ceiling is off by default. Turn it on in Settings and every profile's RAM
figure becomes enforceable.

### What streaming actually costs

Measured here on Qwen3 1.7B Q4_K_M (1.0 GB) at 4 cores, ctx 2048:

| Ceiling | Peak RSS | Decode t/s | Page faults | Weights re-read |
| --- | --- | --- | --- | --- |
| none | 1381 MB | **23.6** | 6.8 K | — |
| 512 MB | 520 MB | 2.71 | 11.1 M | 41× |
| 256 MB | 256 MB | 2.64 | 17.0 M | — |

The generated text is byte-identical in every case; only the speed changes.
11.1 M faults × 4 KB is 43 GB paged for 40 tokens — 1.1 GB per token, i.e. the
entire weight set really is re-read for every single token. The ceiling holds
to within ~1.5% (Windows cannot trim locked pages, so a 512 MB cap settles at
~520 MB).

When a ceiling is active, `--no-mmap` is suppressed automatically. Without mmap
the weights sit in anonymous memory, so eviction writes them to the pagefile
and reads them back — measured at ~2× the disk traffic for identical throughput,
plus the SSD wear. With mmap the evicted pages are clean and file-backed, so
they are simply dropped and re-read from the GGUF, which is what a phone
streaming from flash actually does.

### Fidelity, and the honest caveat

The paging *behaviour* is faithful; the *storage latency* is optimistic. On a
host with spare RAM the evicted pages sit on the system standby list and get
re-faulted from memory rather than disk, so the run is faster than the device
it stands in for. feast AI reports this rather than hiding it, as
**hard-fault ratio** — the share of faulted pages that genuinely hit storage.

But the page-in *volume* is measured exactly even when the host serves it from
RAM, so dividing it by a real device's sequential read bandwidth bounds what
that device could sustain. From the single 512 MB run above:

| Device | Storage | Ceiling | Projected |
| --- | --- | --- | --- |
| Flagship phone | UFS 4.0, 3500 MB/s | 3.23 t/s | 2.71 t/s |
| Mid-range phone | UFS 3.1, 1700 MB/s | 1.57 t/s | **1.57 t/s** (I/O bound) |
| Budget phone | UFS 2.2, 800 MB/s | 0.74 t/s | **0.74 t/s** (I/O bound) |
| Entry / SBC | eMMC 5.1, 250 MB/s | 0.23 t/s | **0.23 t/s** (I/O bound) |

Read that as: a 1.7B model streamed through 512 MB is flash-bound on anything
below a flagship. The session report prints this table for whichever profile
you ran.

### Device profiles

| Profile | Cores | Device RAM | Storage | Represents |
| --- | --- | --- | --- | --- |
| Flagship phone | 8 | 12 GB | UFS 4.0, 3500 MB/s | Snapdragon 8 Gen 3 / A17 Pro |
| Mid-range phone | 6 | 8 GB | UFS 3.1, 1700 MB/s | Snapdragon 7 Gen 3 / Dimensity 7200 |
| Budget phone | 4 | 4 GB | UFS 2.2, 800 MB/s | Snapdragon 680 / Helio G85 |
| Entry / SBC | 2 | 2 GB | eMMC 5.1, 250 MB/s | Raspberry Pi 5, Android Go |
| Wearable / MCU | 1 | 1 GB | eMMC, 120 MB/s | Smartwatch, industrial gateway |
| Host (no cap) | all | — | NVMe, 3000 MB/s | Reference ceiling |
| Custom | you decide | you decide | 1000 MB/s | — |

With the RAM ceiling enabled, the model gets `ram_budget_pct` of the device RAM
(70% by default) — a phone never hands one app the whole of memory. The storage
figure is never enforced; it feeds the projection table above.

Profiles are presets, not walls: change any value and the profile silently
becomes `Custom` so the report never misrepresents what you actually ran.

---

## The model catalog

Eighteen models, every repo and filename verified live against the Hugging Face
API. Selection criteria: designed for on-device deployment, runnable on CPU
without a GPU, and — per the project brief — weighted heavily toward Turkish.
Fifteen are GGUF for llama.cpp; three are ONNX builds for the second engine
(see below).

### Ten primary edge picks

| # | Model | Params | Q4 size | TR | Why it's here |
| --- | --- | --- | --- | --- | --- |
| 1 | **Gemma 4 E2B** | ~2B eff. | 2.8 GB | ★★★★☆ | Google's edge flagship. Per-layer embeddings give larger-model quality at a ~2B footprint. 140+ languages, 128K context, QAT weights tuned for 4-bit. |
| 2 | **Qwen3 1.7B** | 1.7B | 1.1 GB | ★★★★☆ | Strongest genuinely multilingual model in the 1–2B band. 100+ languages, Apache-2.0, switchable thinking mode. |
| 3 | **LFM2.5 350M** | 350M | 240 MB | ★★☆☆☆ | Purpose-built for on-device. Hybrid conv+gated architecture decodes ~2× faster than a same-size transformer on CPU. |
| 4 | **Gemma 3 1B** | 1B | 810 MB | ★★★☆☆ | The reference 1B most Android on-device demos are built on. Good baseline. |
| 5 | **LFM2 1.2B** | 1.2B | 760 MB | ★★★☆☆ | Matches Qwen3-1.7B on many tasks with 47% fewer parameters. |
| 6 | **Qwen3 0.6B** | 0.6B | 400 MB | ★★★☆☆ | Smallest model here that still answers Turkish coherently. |
| 7 | **Phi-4 Mini** | 3.8B | 2.4 GB | ★★★☆☆ | Class leader under 4B on reasoning (~67% MMLU, ~87% GSM8K). The accuracy ceiling on a flagship profile. |
| 8 | **Granite 4.0 1B** | 1B | 700 MB | ★★☆☆☆ | IBM's edge line. Tool calling built in, documented data provenance, Apache-2.0. |
| 9 | **SmolLM3 3B** | 3B | 1.9 GB | ★★☆☆☆ | Fully open pipeline — data, recipe and weights all public. The reproducible choice. |
| 10 | **Gemma 3 270M** | 270M | 200 MB | ★☆☆☆☆ | The floor of the field. Not a general chatbot — it shows what one core actually buys. |
| 11 | Granite 4.0 350M | 350M | 260 MB | ★☆☆☆☆ | Bonus second data point at the bottom of the size curve. |

### Turkish specialists

| # | Model | Params | Q4 size | Notes |
| --- | --- | --- | --- | --- |
| 12 | **Llama 3.2 1B · Turkish** | 1B | 810 MB | **Start here for Turkish on edge.** The only model that is both Turkish-native *and* genuinely phone-sized. |
| 13 | **Cosmos Turkish-Llama 8B** | 8B | 4.8 GB | YTÜ COSMOS group's Llama-3 continued-pretrained on ~30 GB of Turkish. The academic reference. |
| 14 | **Cosmos Turkish-Gemma 9B T1** | 9B | 5.7 GB | Reasoning-tuned. 77.4 on Turkish GSM8K, 68.7% win rate over its base. |
| 15 | **Trendyol LLM 8B Chat v2.0** | 8B | 4.8 GB | Industry counterpart — Turkish + English with a conversational/commerce slant. |

Models 13–15 will not fit a phone. That is the point: run one on the *Host*
profile to establish a quality ceiling, then measure how close model 12 gets on
a Budget profile. The catalog marks them **Fits: no** rather than hiding them.

`Q4_K_M` is the default quantisation. Change **Preferred quantisation** in
Settings and feast AI resolves the best available match in the repo at download
time, remembering the choice.

### Adding your own models

Press **`A`** in the model list. Paste any of:

| Input | Behaviour |
| --- | --- |
| `unsloth/gemma-3-1b-it-GGUF` | Lists every GGUF in the repo with real sizes |
| `https://huggingface.co/owner/name` | Same, URL form |
| `https://huggingface.co/…/resolve/main/x.gguf` | Takes that exact file |
| `https://any-host.example/model.gguf` | Direct link, any platform |

Pick a quantisation from the dropdown (it pre-selects your preferred quant when
the repo has it), optionally set a display name, parameter count, context and
your own Turkish rating, and it joins the catalog permanently — stored in
`~/.feast-ai/models/custom.json`. Custom entries are shown in blue and skip
quant re-resolution, so the file you chose is the file you get. Deleting one
removes both the weights and the catalog entry.

Same thing from the shell:

```bash
python -m feast_ai add unsloth/Qwen3-1.7B-GGUF --name "Qwen3 1.7B" --quant Q5_K_M
```

Only public weights work — gated or private repos will report that they cannot
be reached.

---

## System prompts

Set in Settings, or with **`Ctrl+P`** mid-conversation — the change applies from
your next message with no reload.

Prompts can be **saved by name**: type a name in the *Save as…* box and press
Save. Saved prompts appear in the picker marked **★**, built-in presets marked
**·**, and both are available from Settings and the chat modal — the library is
shared, so a prompt saved mid-chat shows up in Settings immediately. Built-ins
cannot be overwritten or deleted; your own can be updated by saving over the
same name. Stored in `~/.feast-ai/prompts.json`.

The active system prompt is recorded in every session report, so a benchmark
run always says what the model was told.

---

## Thinking / reasoning control

Reasoning models spend real tokens thinking before they answer, and on a CPU
budget that is the single biggest factor in perceived latency. **Thinking mode**
in Settings — or `Ctrl+T` mid-conversation — cycles `auto → off → on`:

| Mode | Behaviour |
| --- | --- |
| `auto` | Each model's own default |
| `off` | Reasoning suppressed — fastest replies |
| `on` | Reasoning forced where the model supports it |

Measured here on Qwen3 0.6B, 4 cores, same Turkish arithmetic prompt:

| Mode | Output tokens | Of which thinking | Decode time |
| --- | --- | --- | --- |
| `on` | 217 | 207 (95%) | 3.58 s |
| `off` | 6 | 0 | **0.08 s** |

That is a 45× latency difference from one toggle, which is exactly the kind of
thing an edge deployment decision hinges on.

It is applied per request via llama.cpp's `enable_thinking` chat-template
kwarg, verified against build `b10144` — so it takes effect on your **next
message with no model reload**. Templates that don't define the variable (Gemma
3, Llama 3.2) ignore it rather than erroring, so it is safe to leave set for any
model. Per-request `reasoning_budget` was tested and is *not* honoured by the
server, which is why feast AI doesn't use it.

Reasoning tokens are counted separately from answer tokens and reported per
turn and per session. **Show reasoning in transcript** is a separate,
display-only switch — thinking still counts toward throughput either way.

Because the mode can change mid-session, the report records it **per turn** and
labels a session `mixed (auto, off, on)` rather than pretending it was uniform.

---

## Agent pipeline: direct vs intent-hierarchical

feast AI began as a model benchmark. The feast mobile app needs more than
chat, though. It needs an on-device agent that turns "Kaydedilenlerimi
gösterir misin" into an action. feast AI compares two ways to build one, on
the same model and the same device budget:

| Agent mode | What the SLM sees | When the SLM runs |
| --- | --- | --- |
| `off` | your system prompt | every message (plain chat, as before) |
| `direct` | one prompt with all 15 tools (~1,000 tokens) | every message |
| `hierarchical` | only the routed intent's tools (165–490 tokens) | only when the router can't answer alone |

Switch modes with `Ctrl+R` mid-chat, or in Settings → *Agent / intent
routing*. The router loads with hierarchical mode and unloads when you leave
it, so a non-hierarchical turn never pays for an idle encoder. Every turn
records its route and whether the tool call that came back was valid, and the
session report gains an *Agent pipeline* section. To compare the modes on a
fixed test set, run `python -m feast_ai bench <model>`.

### The tools

The 11 original tools are joined by four read-only ones, each based on
something feast mobile already has: `get_menu_item_details` (follow-up on a
dish), `get_directions` (opens the maps app), `open_screen` (navigation: the
cheapest action there is) and `get_suggested_users` (the friends-of-friends
list). `send_message_to_friends` is still the only tool that changes
anything. Its confirm-first rule is part of the benchmark: the suite fails
any turn that sends before the user has confirmed.

Each tool is defined once as data. That single definition writes the compact
prompt line (`query*(str), limit(int=10)`) and validates whatever the model
emits. This is how hallucination is *counted* rather than eyeballed:

| Class | Meaning |
| --- | --- |
| `parse_error` | tried to call a tool, payload unparseable (incl. a bare tool name) |
| `unknown_tool` | the tool doesn't exist |
| `unoffered_tool` | it exists, but this prompt didn't list it |
| `invalid_args` | invented argument, value outside its enum, missing required, both/neither of an either-or pair |

Format slips are accepted but recorded: a ```json fence, a bare object,
`name(arg=…)`. Wrong types that an app could coerce ("10" for 10) count as
warnings, not hallucinations. Answering in plain text when a tool was due is
counted separately, because that is where small models make things up.

### The router

`modernbert-tr-embed` embeds each message and finds the nearest of 212
example utterances spread over 19 intents. It needs no training: a new intent
is just a list of sentences. It runs as a third supervised subprocess:

- pinned to the same cores as the model;
- sampled together with the model;
- under a RAM ceiling, given its own slice (`router_ram_mb`, default 320 MB)
  carved out of the model's share, so the pair never exceeds one device.

ONNX Runtime's spin-waiting is disabled, so the router doesn't steal the
SLM's cores right after it routes.

The policy, in order:

1. **Unsure → widen, don't guess.** If an intent the top group can't serve
   scores within 0.04 of it (a near-tie), the SLM gets the union of the tied
   groups: "Bir latte kaç lira" ties out_of_scope with menu_search and gets
   the 3 menu tools. Only a weak top score (< 0.60) or a union over 8 tools
   falls back to the full prompt, which is never worse than `direct`.
2. **Follow-up → inherit.** "evet" and "daha ucuzu var mı" reuse the previous
   turn's tool group. Without this, confirming a message would be routed as
   small talk.
3. **Confident and trivial → no SLM.** Opening a screen, going back to GPS,
   listing friends or suggestions, the time and refusals are answered
   deterministically above 0.75.
4. **Otherwise → SLM with the intent's tool group.** Intents the router
   confuses share tools, so a near-miss still lands on a usable prompt.

Measured on 60 held-out turns, Budget profile (4 pinned cores):

| Metric | Value |
| --- | --- |
| Top-1 intent accuracy | 83% |
| Accuracy on turns it doesn't fall back on (margin ≥ 0.04) | 94% (20% fall back) |
| No-SLM answers that were exactly right | 14 of 14 |
| Router latency | 24 ms (23 ms of it inference) |
| Router RSS | ~250 MB |

The raw similarity score barely separates right routes from wrong ones (wrong
routes scored 0.62–0.82). The *margin* does separate them, because a wrong
route is usually a near-tie, which is why near-ties are widened rather than
trusted. A tie only counts if the rival needs tools the top group lacks:
"Kaydedilenlerimi göster" ties saved_open with navigate, but both mean
`open_screen(saved)`, so it is still answered with no SLM. The margin, the
merge rule and three tool-group overlaps were set after the first router run,
so the suite is no longer blind for them. Tune exemplars on new queries,
never on suite sentences.

### Alternative router: Decision 2.0

`bench --router decision-2.0-kai-0.6b` swaps the encoder for
[Decision 2.0](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B), the vLLM
Semantic Router team's decision models. Kai-0.6B is a Qwen3-0.6B finetune with
a decision head. The router asks it one `choice` question whose options are
the 19 intents, and it answers with a probability per intent from
`/v1/systemone`. The routing policy is unchanged, on its own thresholds
(probabilities are not cosines).

Its GGUF needs the `decision2-gguf` llama.cpp branch
(`github.com/Xunzhuo/llama.cpp`). Build `llama-server` from it and copy the
binaries to `~/.feast-ai/bin/llama.cpp-decision2/`. Put the Q4_K_M GGUF in
`~/.feast-ai/encoders/decision-2.0-kai-0.6b/`. The build reproduced the
published Q4_K_M validation outputs to within 0.053 probability (most answers
within 0.01).

Measured router-only on the standard suite, 3 cores:

| Router | Top-1 accuracy | Per message | RSS |
| --- | --- | --- | --- |
| ModernBERT-TR embed (kNN, 212 exemplars) | 83% | ~35 ms | ~250 MB |
| Decision 2.0 Kai-0.6B, intent summary + 2 examples | 70% | ~2.6 s | ~640 MB |
| Decision 2.0 Kai-0.6B, intent summary only | 50% | ~1.1 s | ~640 MB |

Each routing call processes the whole question, about 990 tokens. The message
comes first in the prompt, so llama.cpp can't reuse the question across calls.
Decision 2.0's probabilities are well calibrated, though: a top probability
of 0.50 or more was right on every turn, but only 38% of turns reached it.

### The Cosmos ModernBERT-TR family

YTÜ COSMOS's Turkish encoders: 149M parameters, 22 layers, 8,192-token
context, cased WordPiece vocabulary of 50,008. All Apache-2.0 and all ship an
int8 ONNX build of about 150 MB. They are not chat models: they read a
sentence and return a vector, a label or per-token tags in one forward pass,
which is why they are cheap enough to run before the SLM on every message.
`python -m feast_ai models` lists them, and `pull` fetches any of them.

| Encoder | Task | Reported | In feast AI |
| --- | --- | --- | --- |
| `modernbert-tr-embed` | 768-d sentence embeddings | MTEB(Turkish) mean 68.14; retrieval 59.42, STS 77.62 | **The router.** kNN over feast exemplars |
| `modernbert-tr-massive-intent` | 60 MASSIVE intents | accuracy 88.21, macro-F1 85.43 | Measured, not used (see below) |
| `modernbert-tr-massive-slot` | 55 slot types (place_name, person, food_type, time…) | entity-F1 75.27 | Next step: arguments for more direct calls |
| `modernbert-tr-guardrail` | prompt safety, 10 hazard heads | binary F1 93.0 | Candidate gate before `send_message_to_friends` |
| `modernbert-tr-reranker` | cross-encoder reranking | +8.38 NDCG@10 | Candidate for on-device post/memory search |
| `modernbert-tr-absa-polarity` | aspect sentiment | macro-F1 72.74 | Candidate: "is the service good here?" from posts |
| `modernbert-tr-absa-extraction` | aspect terms | span F1 64.09 | Pairs with polarity |
| `modernbert-tr-pii-ner` | 25 KVKK PII types | F2 80.85 | Candidate: scrub memory and logs (KVKK) |
| `modernbert-tr-base` | masked-LM base | above BERTurk on TabiBench | Base for a feast-specific head, once real queries exist |

**Why the intent classifier doesn't route.** `massive-intent` is fast (~7 ms),
but its taxonomy is a voice assistant's: alarms, lights, music. It has no
saved items, posts or active location, and on feast queries it is
*confidently* wrong. "Ayarları aç" → `music_settings` (0.65), "Üsküdar'a taşı
konumu" → `transport_query` (0.70), and restaurant, dish and saved-item
searches all collapse into `recommendation_locations`. The embedding model
needs no retraining. On a 16-query probe with 3–4 exemplars per intent, its
instruction prefix lifted accuracy from 10/16 to 15/16. Applying the prefix
to queries *and* exemplars beat the query-only (asymmetric) variant, 15/16 vs
13/16.

### Results

`python -m feast_ai bench <model> --profile <p>`: 57 cases / 60 turns,
greedy decoding, thinking off, shuffled with a fixed seed, latency
end-to-end with routing included.

#### Standard suite (60 labelled turns)

Each cell reads direct → hierarchical.

| Model | Setup | Exact | Hallucinated | Unsafe sends | No-SLM turns | Latency avg | TTFT | Prefilled tokens | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 0.6B | Budget (4 cores), warm | 27% → 42% | 25% → 15% | 0 → 0 | 14/60 | 0.87 s → 1.02 s | 0.30 s → 0.90 s | 26 → 354 | 1,119 → 1,378 MB |
| Qwen3 0.6B | Budget, warm + 1 GB saved prompts | 27% → 43% | 23% → 15% | 0 → 0 | 14/60 | 0.81 s → 0.85 s | 0.29 s → 0.62 s | 26 → 183 | 1,120 → 1,967 MB |
| Qwen3 0.6B | Budget, cold | 28% → 43% | 23% → 18% | 0 → 0 | 14/60 | 2.88 s → 1.21 s | 2.40 s → 1.11 s | 1,064 → 487 | 1,120 → 1,378 MB |
| Llama 3.2 1B TR | Budget, warm | 15% → 37% | 32% → 30% | 1 → 0 | 14/60 | 3.27 s → 2.68 s | 0.29 s → 1.33 s | 21 → 330 | 1,460 → 1,719 MB |
| Qwen3 1.7B | Mid-range (6 cores), warm | 52% → 68% | 7% → 8% | 0 → 0 | 14/60 | 2.20 s → 2.47 s | 0.33 s → 1.48 s | 27 → 357 | 2,254 → 2,513 MB |

"Warm" is llama.cpp's prompt cache as an app would run it, with the
cross-prompt cache off (`prompt_cache_ram_mb` 0, see below). The peak RSS
difference is mostly the router, about 250 MB.

#### End-user simulation (`--suite kanka`, Gemma 4 E2B)

These are 31 casual turns in 20 chats, written the way people text a friend:
"Kanka çok acıktım yaa…", no diacritics, emoji, a bit of English, and
follow-ups like "150 lirayı geçmesin ama 😅". The same model ran on two
phones. Each cell reads direct → hierarchical. `bench` writes a side-by-side
chat page next to each report.

| Phone | Exact | Hallucinated | Unsafe sends | No-SLM turns | Latency avg | TTFT | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mid-range (6 cores, 8 GB) | 61% → 65% | 13% → 16% | 1 → 1 | 4/31 | 2.30 s → 3.69 s | 0.47 s → 2.48 s | 2.7 → 2.9 GB |
| Low-budget (2 cores, 3 GB, 2.1 GB ceiling) | 74% → 65% | 6% → 16% | 0 → 1 | 4/31 | 3.29 s → 7.43 s | 0.76 s → 6.29 s | 2.1 → 2.0 GB |

- **No accuracy winner for Gemma 4 E2B on casual chat.** Direct scored 61%
  and 74% on identical prompts; only the thread count and the clock minute
  changed. A swing of four messages is the noise floor of a 31-turn suite.
- **On a budget phone, latency clearly favours direct.** Two cores prefill at
  45–64 tokens/s, so re-prefilling the ~410 tokens of a routed prompt costs
  about 6 s per message. Direct's cached prefix costs about 26 tokens. A 2.1 GB
  ceiling leaves no room for llama.cpp's RAM prompt cache. The fix is
  storage: compute each intent prompt's KV once and restore it from flash per
  message (llama.cpp slot save/restore).
- **Gemma 4 E2B fits the 3 GB phone.** Under the 2.1 GB ceiling it held
  2.1 GB RSS and decoded at 14 tokens/s, with about 560 MB paged in over the
  whole run. That is less than the 2.9 GB file, so most of the weights it
  couldn't hold were never read. This is consistent with its sparsely read
  per-layer embeddings.
- **Gemma writes calls in its own syntax.** Of 21 calls, 9 used the native
  `name{key:<|"|>value<|"|>}` form and 8 left out the `arguments` wrapper. The
  judge accepts both as format slips.

What the numbers say:

- **Routing helps the small models most.** On the standard suite,
  exact answers rose by 15–22 points: Qwen3 0.6B 27% → 42%, Llama 3.2 1B
  Turkish 15% → 37%, Qwen3 1.7B 52% → 68%. Hallucinated calls fell on the
  sub-1B models and stayed level on Qwen3 1.7B. The only unsafe send in the
  standard suite came from direct mode. For Gemma 4 E2B on casual chat (below),
  there was no clear difference.
- **About a quarter of turns don't need the SLM at all** (14 of 60, ~25 ms
  each). That is the biggest latency win, and those answers were 14/14 exact.
- **Latency depends on the prompt cache.** `direct` sends the same
  ~1,000-token prompt every time, so llama.cpp reuses it and prefills about 25
  new tokens per turn. A hierarchical prompt changes with the intent, so each
  switch re-prefills about 350 tokens. With a warm cache, `direct` reaches the
  first token sooner (0.30 s vs 0.90 s on Qwen3 0.6B). Cold (`--no-cache`),
  hierarchical is 2.4× faster end to end (1.21 s vs 2.88 s).
- **llama-server keeps up to 8 GB of old prompts by default.** Its
  `--cache-ram` (default 8,192 MiB) stores the KV state of prompts it has
  seen, so returning to an intent restores instead of re-prefilling. That cut
  hierarchical TTFT from 0.90 to 0.62 s, but it cost **+589 MB** of
  llama-server RSS (1,378 → 1,967 MB) on a 4 GB phone profile. feast AI now
  passes `prompt_cache_ram_mb` (default 0), records it in every report, and
  `bench --cache-ram N` measures that trade-off on purpose.
- **Absolute accuracy at this size is low**, and the failure mode is the
  dangerous one: small models often answer in text instead of calling the
  tool, echo the question or invent a price. Gemma 4 E2B mostly falls back to
  its own trained call syntax (`name{key:<|"|>value<|"|>}`). The judge accepts
  that as a format slip, but it points at the next experiment: each model's
  native chat-template tool format against this compact one.

#### Prompt cycle: tool manuals, active prompting and class prompts

The routed pipelines above only *shortened* the tool list; the words around
it stayed the same. This run gives a correct route something to buy. Every
tool now has a manual (`agent/manuals.py`): a hint per argument, one complete
example call and, for `send_message_to_friends`, the two-step confirm
protocol. Two pipelines use the same manuals, so a difference between them
is routing, not wording:

| Pipeline | What happens |
| --- | --- |
| `direct` | unchanged: all 15 compact tool lines, one prompt |
| `active` | one fixed prompt lists every tool by name and purpose (494 tokens); the model answers `<use>tool</use>` (generation stops there) or in text; for a tool, its manual (150–350 tokens) is appended as a system turn, the model writes the call, and the manual is dropped from the history |
| `classes` | the router picks one of 13 action classes (`agent/classes.py`: the 19 intents merged by what the SLM must do, e.g. venue + dish search = `food`); the class prompt carries its tools' manuals and its own rules (250–960 tokens), restored from a KV snapshot only when the class changes |

`bench gemma-4-e2b --modes direct,active,classes,classes@decision-2.0-kai-0.6b
--cores 3 --ram-gb 4`, kanka (31 turns) and standard (60 turns) suites, 91
turns in all:

| | Direct | Active prompting | Classes, ModernBERT-TR | Classes, Decision 2.0 |
| --- | --- | --- | --- | --- |
| Exact | 73% | 74% | **87%** | 85% |
| Hallucinated calls | 6 | 1 | **0** | **0** |
| Unsafe sends | 0 | 0 | 0 | 0 |
| Router picked the right class | – | – | 86% | 70% |
| Fell back to the all-tools prompt | – | – | 4 | 35 |
| TTFT median | **0.44 s** | 4.04 s | 1.84 s | 3.46 s |
| Reply time median / average | **2.35 / 2.46 s** | 5.76 / 5.24 s | 3.36 / 4.50 s | 5.07 / 5.50 s |
| Decode | 16.2 tok/s | 16.8 tok/s | 16.5 tok/s | 16.3 tok/s |
| Routing per turn | – | – | 34 ms | 1,706 ms |
| Page-ins over the run | 282 MB | 494 MB | 371 MB | 6.4 GB |

- **A right class with its own prompt is where hallucination goes.** On the
  74 turns where ModernBERT-TR picked the right class and its class prompt
  was sent, Gemma was exact on 93% with no hallucinated call; direct, on the
  same turns, scored 73% with 5. A wrong class costs: on the 13 misrouted
  turns the class prompt scored 46% against direct's 62%.
- **Active prompting cuts hallucination, not misses.** 6 → 1 hallucinated
  calls, but the same 74% exact: picking the tool from names alone is the
  weak step (shared items read as chat memory, the QR scanner as a menu
  search), and 6 times the model repeated its `<use>` tag instead of calling.
  Each tool turn pays two passes: ~4 s to the first token of the call.
- **Decision 2.0 classifies worse but its probabilities are honest.** 70%
  right on 13 classes, 1.7 s per message (down from 2.6 s on 19 intents), and
  its 640 MB next to Gemma pushed the 4 GB phone into 6.4 GB of page-ins.
  Its confident turns (top >= 0.25, 56 of 91) scored 88%, and 96% on the 48
  of them it routed right.
- **The ModernBERT average hides a cold cost.** Nine near-tie class pairs
  were first seen mid-run and their snapshots built on the spot (~12 s each).
  The median, 3.36 s, is the steady state; merged prompts are now ordered
  canonically so a pair is built once.

Five router exemplars that echoed suite turns were reworded before this run
(the 13 classes reuse the 212 exemplars, relabelled). Manual examples follow
the same rule: none paraphrases a suite sentence.

Limits of the test: tools are not executed, so only the first hop is judged.
Follow-up cases start from a history that already lists results with ids.

---

## What gets measured

| Metric | Definition |
| --- | --- |
| **TTFT** | Time to first token. On CPU this is dominated by prompt processing. |
| **Prefill t/s** | Prompt tokens ÷ prompt-eval time — how fast the device *reads*. |
| **Decode t/s** | Generated tokens ÷ generation time. What users perceive as speed. |
| **End-to-end t/s** | Generated tokens ÷ total wall clock *including* TTFT. The honest figure for short answers. |
| **CPU % (device)** | Share of the *simulated* budget. Saturates at 100%. |
| **CPU % (host)** | Share of the whole machine — proves the cap is real. |
| **RAM RSS / USS** | Resident set, and memory unique to the process. |
| **Reasoning tokens** | Tokens spent thinking rather than answering, and their share of output. |

llama.cpp's own `timings` are used where available and treated as
authoritative; feast AI's wall-clock measurements are the fallback.

`Disable mmap` is on by default so weights load into anonymous memory and RSS
reflects the model's true cost. Turn it off for faster loads and fuzzier
numbers.

Sessions auto-save to `~/.feast-ai/sessions/` as both Markdown (for writing up)
and JSON (for plotting).

---

## Keys

**Welcome** — `Enter` start · `M` models · `S` settings · `R` reinstall runtime · `Q` quit

**Models** — `↑↓` browse · `Enter` load & chat · `D` download · `A` add a model · `X` delete · `R` refresh · `Ctrl+C` cancel download · `Esc` back

**Chat** — `Enter` send · `Ctrl+E` end & report · `Ctrl+P` system prompt · `Ctrl+T` thinking mode · `Ctrl+R` agent mode (off → direct → hierarchical) · `Ctrl+X` stop · `Ctrl+L` clear context · `F2` settings · `Esc` leave

**Summary** — `S` save report · `N` another model · `Esc` back to chat · `Q` quit

---

## CLI

```bash
python -m feast_ai doctor
```

```bash
python -m feast_ai models
```

```bash
python -m feast_ai pull gemma-3-270m
```

```bash
python -m feast_ai add unsloth/gemma-3-1b-it-GGUF
```

```bash
pip install onnxruntime-genai && python -m feast_ai pull qwen3-0.6b-onnx
```

```bash
python -m feast_ai pull modernbert-tr-embed
```

```bash
python -m feast_ai route "Kaydedilenlerimi gösterir misin" "Etrafta 150 TL altı döner var mı"
```

```bash
python -m feast_ai bench --router-only --profile budget
```

```bash
python -m feast_ai bench qwen3-0.6b --profile budget
```

```bash
python -m feast_ai bench gemma-4-e2b --profile midrange --suite kanka
```

`pull` also fetches encoders. `route` prints how the intent router would serve
each sentence. `bench` runs a labelled agent suite through both pipelines on
one model and one device budget, then writes Markdown, JSON and a side-by-side
chat page (`.html`) to `sessions/`. Options: `--suite kanka` for casual
end-user chats, `--no-cache` for cold prefill, `--cache-ram N`, `--modes`,
`--limit`, `--thinking`, `--temperature`, and `--router-only` to skip the SLM.
`compare a.json b.json` merges several runs into one page with a model
switcher.

`runtime` downloads the llama.cpp CPU build; `run` (the default) opens the TUI.
`doctor` reports both engines and probes whether this OS can actually enforce
the RAM ceiling, rather than assuming it from the platform name.

---

## Architecture

```
feast_ai/
  cli.py            argparse entry point: run · models · pull · add · runtime · doctor · bench · route
  config.py         Settings, device profiles, thinking and agent modes, JSON persistence
  catalog.py        the 18 models, user-added models, HF lookup, fit heuristics
  encoders.py       the Cosmos ModernBERT-TR family: what each is for, which one routes
  prompts.py        saved system prompts (built-in presets + your own)
  download.py       resumable HTTP with progress; multi-file for ONNX dirs
  paths.py          ~/.feast-ai layout
  engine/
    binaries.py     fetches the prebuilt llama.cpp CPU build from GitHub releases
    server.py       BaseServer (affinity · RAM ceiling · health) + LlamaServer
    memcap.py       working-set ceiling: Windows hard WS, Linux cgroup memory.high
    onnx_server.py  OnnxServer - same supervision, different runtime
    onnx_worker.py  the ONNX subprocess: OpenAI-compatible HTTP over ORT GenAI
    encoder_server.py  the intent router's supervisor - same cores, its RAM share
    encoder_worker.py  the router subprocess: ONNX Runtime + tokenizers, kNN over exemplars
    client.py       OpenAI-compatible SSE streaming, <think> filtering
  agent/
    tools.py        the 15 feast mobile tools as data: rendered into prompts, used to validate
    calls.py        find the tool call in a reply and classify what is wrong with it
    context.py      the CONTEXT block (clock, GPS, active location) - rendered last
    intents.py      19 intents: exemplars, route (slm/direct/inherit), tool group
    prompts.py      the direct (all tools) and routed (one group) system prompts
    router.py       the routing policy: fall back, inherit, answer directly, or narrow
    direct.py       no-SLM answers: open a screen, back to GPS, time, refusals
    suite.py        the standard (60 turns) and kanka (31 casual turns) suites, scoring
    bench.py        direct vs hierarchical on one budget -> sessions/bench-*.md/.json/.html
    page.py         the side-by-side chat comparison page
  metrics/
    sampler.py      psutil ring buffers, dual CPU normalisation, RSS/USS, paging
    session.py      per-turn stats, aggregates, percentiles, MD/JSON export
  ui/
    app.py          Textual app shell and shared state
    widgets.py      meters, stat tiles, sparklines, banner
    screens/        welcome · settings · models · addmodel · chat · summary · modals
```

`ChatScreen` is the one screen pushed as a **fresh instance** rather than a
cached named screen: it owns a live server connection and per-session
statistics, so reusing it across models would leave the client pointed at a
dead port and mix two models' turns into one report.

Adding a third engine means subclassing `BaseServer` and supplying a command
line — affinity, the RAM ceiling, priority, health polling and telemetry come
with it, so a new backend cannot accidentally opt out of the constraints that
make runs comparable.

Nothing is compiled. `python -m feast_ai runtime` downloads the official
prebuilt `llama-server` for your platform (~17 MB on Windows x64) and caches it
under `~/.feast-ai/bin/`. Point `FEAST_AI_LLAMA_SERVER` at your own build to
override, or `FEAST_AI_HOME` to relocate all state.

Requires Python 3.10+. Tested on Windows 11 with Python 3.13, llama.cpp
`b10144` and `onnxruntime-genai` 0.14.1.

---

## Two inference engines

feast AI runs two CPU backends behind one interface. Each model declares which
one it needs, so a GGUF can never be handed to ONNX Runtime or vice versa.

| Engine | Weights | Install | Notes |
| --- | --- | --- | --- |
| **llama.cpp** | GGUF, one file | bundled prebuilt binary | The default. |
| **ONNX Runtime GenAI** | ONNX, a directory | `pip install onnxruntime-genai` | Optional; catalog entries report it as missing if absent. |

Both run as a **subprocess** exposing the same OpenAI-compatible endpoint.
That is deliberate: `onnxruntime-genai` is a Python library, so calling it
in-process would pin *the TUI* to the device's core budget, cap the TUI's
working set instead of the model's, and leave no separate process tree to
sample. A subprocess keeps affinity, the RAM ceiling and telemetry honest and
identical across engines.

Two of the three ONNX entries deliberately mirror GGUF entries (Qwen3 0.6B,
Llama 3.2 1B) so the runtimes can be compared on the same weights.

## Verified results

Measured on this repo, Intel Core i5-14600KF, affinity pinned.

**CPU budget is enforced** (`Q4_K_M`, llama.cpp):

| Model | Cores | Decode t/s | TTFT | Peak RSS | Host CPU |
| --- | --- | --- | --- | --- | --- |
| Llama 3.2 1B Turkish | 4 (Budget) | 33.0 | 0.28 s | 935 MB | 20.6% |
| Gemma 3 270M | 1 (Wearable) | 70.8 | 0.26 s | 338 MB | 4.0% |

Host CPU of 20.6% on 4 of 20 cores, and 4.0% on 1 of 20, confirms the affinity
cap is doing real work rather than being advisory.

**Engine comparison** — Qwen3 0.6B, same prompt, 4 cores, thinking off. Both
render 134 prompt tokens, so the templates agree and the comparison is fair:

| Run | Decode t/s | Prefill t/s | TTFT | Peak RSS |
| --- | --- | --- | --- | --- |
| llama.cpp (Q4_K_M) | **64.2** | **556** | 0.44 s | 688 MB |
| ONNX Runtime (int8) | 7.6 | 43 | 3.31 s | 945 MB |
| llama.cpp @ 300 MB ceiling | 7.1 | 345 | 0.64 s | 308 MB |
| ONNX Runtime @ 300 MB ceiling | 1.2 | 31 | 4.55 s | 308 MB |

llama.cpp is ~8x faster at decode and ~13x at prefill here. Two things are
being measured at once and it is worth separating them: this ONNX build is
int8 against llama.cpp's Q4_K_M, so it moves roughly twice the bytes per
token, and llama.cpp's CPU kernels are hand-tuned for exactly this workload
while ONNX Runtime's CPU EP is general-purpose. The RAM ceiling binds both
backends to the same 308 MB, which is the point — the constraint is a property
of the harness, not of either engine.
