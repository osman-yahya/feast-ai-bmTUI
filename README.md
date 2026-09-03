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

**Chat** — `Enter` send · `Ctrl+E` end & report · `Ctrl+P` system prompt · `Ctrl+T` thinking mode · `Ctrl+X` stop · `Ctrl+L` clear context · `F2` settings · `Esc` leave

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

`runtime` downloads the llama.cpp CPU build; `run` (the default) opens the TUI.
`doctor` reports both engines and probes whether this OS can actually enforce
the RAM ceiling, rather than assuming it from the platform name.

---

## Architecture

```
feast_ai/
  cli.py            argparse entry point: run · models · pull · add · runtime · doctor
  config.py         Settings, device profiles, thinking modes, JSON persistence
  catalog.py        the 18 models, user-added models, HF lookup, fit heuristics
  prompts.py        saved system prompts (built-in presets + your own)
  download.py       resumable HTTP with progress; multi-file for ONNX dirs
  paths.py          ~/.feast-ai layout
  engine/
    binaries.py     fetches the prebuilt llama.cpp CPU build from GitHub releases
    server.py       BaseServer (affinity · RAM ceiling · health) + LlamaServer
    memcap.py       working-set ceiling: Windows hard WS, Linux cgroup memory.high
    onnx_server.py  OnnxServer - same supervision, different runtime
    onnx_worker.py  the ONNX subprocess: OpenAI-compatible HTTP over ORT GenAI
    client.py       OpenAI-compatible SSE streaming, <think> filtering
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
