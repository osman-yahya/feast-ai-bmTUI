# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**feast AI**: a graduation project (*bitirme*) that asks whether a small LLM could actually run as an AI agent on a phone. It's a Textual TUI plus a CLI. It runs edge-optimized LLMs on a **CPU-only** inference engine, clamps CPU and RAM to a phone-class budget, streams live telemetry, and writes a per-session report (Markdown + JSON) that feeds the thesis.

It is also the R&D step for the **on-device AI agent that will live inside the feast mobile app**, part of the feast restaurant platform (feast.tr). The model, quantization and device-profile results measured here decide what ships in that app. That's why the catalog weights Turkish so heavily and why the mid-range and budget phone profiles matter most. For platform context (the mobile app, backends, product decisions), use the `feast` skill.

The core value is **honest measurement**. Before changing anything, ask whether the change could let a run use more than its simulated budget, or whether it would report numbers that don't reflect what actually ran. Either one is a bug, even if the app still works.

`README.md` is the user-facing source of truth (design rationale, measured results, catalog, keys). Read it before larger changes, and update it when behaviour, keys, the catalog or measured figures change.

## Layout

```
Bitirme/                 <- git root
  CLAUDE.md  README.md  pyproject.toml  requirements.txt  run.ps1  run.sh
  feast_ai/              <- the Python package (import name feast_ai)
    cli.py  config.py  catalog.py  encoders.py  prompts.py  download.py  paths.py
    agent/   tools.py · manuals.py · calls.py · context.py · intents.py · classes.py
             prompts.py · direct.py · router.py · decision.py · suite.py · bench.py
             page.py · text.py · grammar.py · native.py · replies.py · mock.py
    engine/  server.py (BaseServer, LlamaServer) · onnx_server.py · onnx_worker.py
             litert_server.py · litert_worker.py
             encoder_server.py · encoder_worker.py · memcap.py · binaries.py · client.py
    metrics/ sampler.py · session.py
    ui/      app.py · widgets.py · feast.tcss · screens/
  mustachio/             <- the model workspace (import name mustachio): text-only
                            Gemma 4 E2B, training data, LoRA fine-tune, eval, export.
                            Its own README is the source of truth; state lives in
                            ~/.feast-ai/mustachio/. It imports feast_ai's tools and suites.
```

Run every command from the **git root** (`Bitirme/`), not from inside `feast_ai/`. Otherwise `python -m feast_ai` won't resolve.

## Commands

```powershell
pip install -r requirements.txt          # textual, psutil, httpx, rich, (optional) onnxruntime-genai
python -m feast_ai                       # TUI (same as `run`)
python -m feast_ai doctor                # env check: engines, cores, RAM-cap support, models on disk
python -m feast_ai models                # list catalog
python -m feast_ai runtime               # download prebuilt llama.cpp CPU llama-server
python -m feast_ai pull <model-id>       # e.g. gemma-3-270m (smallest), llama32-1b-tr, qwen3-0.6b
python -m feast_ai add <hf-repo|url> [--name N] [--quant Q4_K_M]
python -m feast_ai pull modernbert-tr-embed          # the intent router (int8 ONNX, 150 MB)
python -m feast_ai route "Kaydedilenlerimi göster"    # how the router serves a sentence
python -m feast_ai bench --router-only --profile budget   # router accuracy/latency, no SLM
python -m feast_ai bench qwen3-0.6b --profile budget      # direct vs hierarchical, MD+JSON report
python -m mustachio install --activate   # register the exported .litertlm, set agent mode native
python -m feast_ai bench mustachio-e2b --modes native,native-raw --suite kanka
python -m pyflakes feast_ai mustachio/*.py   # the only static check in use (not mustachio/: it holds a .venv)
```

There is **no test suite**. Verify a change in these steps:
1. Run `pyflakes`.
2. Exercise the change with a throwaway script in the scratchpad. Earlier sessions used e2e scripts that start a server, stream a chat and print the stats, plus Textual `run_test()` pilot scripts for the TUI.
3. Run `doctor` or a real `pull` + chat with a tiny model (`gemma-3-270m`).

Primary dev machine: Windows 11, Python 3.13, Intel i5-14600KF (20 logical cores), llama.cpp `b10144`, onnxruntime-genai 0.14.1.

## Runtime state (outside the repo)

Everything is stored under `~/.feast-ai/`, which `FEAST_AI_HOME` can override (see `paths.py`):
`config.json`, `prompts.json`, `agent_context.json` (the simulated phone state behind the agent prompt's CONTEXT block), `models/<id>/…` (plus `models/resolved.json` quant overrides and `models/custom.json` user-added models), `encoders/<id>/…` (ModernBERT-TR ONNX + cached exemplar-bank vectors), `bin/llama.cpp/` (runtime + `VERSION.json`), `sessions/` (reports, including `bench-*`), `logs/` (server logs).
`FEAST_AI_LLAMA_SERVER` points the app at a custom `llama-server` binary.

Never write weights, sessions or logs into the repo.

## Architecture essentials

- **Three engines, one interface.** `engine.make_server(spec.engine, …)` picks `LlamaServer` (GGUF, single file), `OnnxServer` (ONNX directory) or `LiteRtServer` (`.litertlm` bundle, the runtime the mobile app ships; needs `pip install litert-lm`). The *model's* `ModelSpec.engine` decides which, and there is no global switch. Both run as a **subprocess** that exposes an OpenAI-compatible `/v1/chat/completions` (SSE), so `ChatClient` and the telemetry treat them the same way.
  - ONNX runs as a subprocess on purpose. Running it in-process would pin the TUI's own process to the core budget and cap the TUI's memory instead of the model's.
  - A new engine should subclass `BaseServer` and implement `_commands()`, which returns candidate command lines in order, later ones being fallbacks. Affinity, RAM cap, priority, health polling and process-tree telemetry then come from the base class. Don't bypass them.
  - Workers are started with `server.worker_python()`, not `sys.executable`: in a Windows venv `python.exe` is a launcher stub, and the affinity and RAM cap would land on the stub (measured: RSS 4 MB "model").
  - The LiteRT worker keeps the conversation (`live` / `clone` / `replay`, reported as `timings.path`). A new conversation from a history re-reads all of it (14 s for 2,650 tokens at 4 threads), so: the warm-up must use the same `max_tokens`, tools and sampler as real turns (they are part of the reuse key), and `_fingerprint` deliberately ignores a call's arguments and the sentence after a tool result, the two things the app-side checks may have edited.
- **CPU budget is applied twice:** `--threads N` and `psutil.cpu_affinity` pinning (`select_cores` prefers even indices, i.e. distinct physical cores on SMT). Threads alone are not a cap.
- **RAM ceiling = working set, never a commit limit** (`engine/memcap.py`). On Windows it uses `SetProcessWorkingSetSizeEx` with a hard max. On Linux it uses cgroup v2 `memory.high`. Never switch to Job Object `ProcessMemoryLimit` or `memory.max`: those kill the process instead of slowing it down, and slowing down is what a phone actually does. The cap is off by default. When it's on, `Settings.effective_no_mmap` turns `--no-mmap` off on purpose.
- **Settings** (`config.py`) is a dataclass persisted as JSON. `from_dict` drops unknown keys and validates enums, so you can add fields freely as long as they have defaults. Device profiles are presets. If the user changes any value, `mark_custom_if_needed` relabels the profile as `custom` so reports stay truthful.
- **Thinking control** is sent per request as `chat_template_kwargs.enable_thinking` (verified on b10144, no reload needed). Inline `<think>` spans are split by `ThinkFilter`. Reasoning tokens still count toward throughput. `reasoning_budget` is *not* honoured by llama-server, so don't use it.
- **Metrics:** `metrics/sampler.py` keeps psutil ring buffers with two CPU normalizations, "device" (share of the budget) and "host" (share of the whole machine), plus RSS/USS and page faults. `metrics/session.py` holds per-turn stats, aggregates, the storage-bandwidth projection and the MD/JSON export. Where llama.cpp's own `timings` exist they're authoritative, and wall-clock timing is the fallback.
- **UI** (`ui/app.py`, `FeastApp`) holds process-wide state: settings, the running server, the sampler and the active model. Named screens (welcome/models/settings/summary) are cached instances and refresh in `on_screen_resume`. **`ChatScreen` is always pushed as a fresh instance** because it owns a live server and per-session stats. Keep it that way. Styles live in `ui/feast.tcss`.
- **Catalog** (`catalog.py`): 18 curated `ModelSpec`s. Every repo and filename was verified against the live Hugging Face API, so re-verify any entry you add or change. Ranking weights Turkish capability heavily (`tr` 1–5, per the project brief). User-added models live in `custom.json` and skip quant re-resolution.
- **Downloads** (`download.py`): httpx with resume via `.part` files. ONNX models are multi-file (`download_many`). No `huggingface_hub`, so progress can render in the TUI.
- **Agent pipeline** (`agent/`, `Settings.agent_mode`): `off` = plain chat; `direct` = one prompt with all 15 tools; `hierarchical` = the ModernBERT-TR router picks an intent first, then the SLM sees only that intent's tool group, or a deterministic handler answers with no SLM (`direct.py`). Tools are data (`tools.py`): the same `ToolSpec` renders the compact prompt line and validates calls, which is how hallucination is counted (`calls.py`: parse_error / unknown_tool / unoffered_tool / invalid_args). Both pipelines share every prompt word except the TOOLS block; keep it that way or the comparison stops isolating one variable. CONTEXT goes last so the static tool block stays in llama.cpp's prompt cache.
  - The router is a third `BaseServer` subprocess (`EncoderServer`), pinned to the **same cores** as the SLM, sampled with it (`FeastApp.attach_telemetry`), and under a RAM ceiling it gets `router_ram_mb` carved out of the model's share (`agent.start_router` → `set_ram_limit`). Never run it in-process, and never let the pair exceed one device budget. Each server has its own Linux cgroup (`cap_group`), or the router's slice would overwrite the model's `memory.high`.
  - ORT intra-op spinning is disabled in the worker; otherwise its threads steal the SLM's pinned cores right after routing.
  - A second router backend exists: Decision 2.0 Kai-0.6B (`engine/decision_server.py`, `agent/decision.py`, `--router decision-2.0-kai-0.6b`). It needs the `decision2-gguf` llama.cpp fork built into `~/.feast-ai/bin/llama.cpp-decision2/`; the SLM stays on the stock runtime. It returns probabilities, so `DecisionRouter.thresholds()` overrides the cosine thresholds. Don't run its `trust_remote_code` Python package; the GGUF path needs no downloaded code.
  - The `routed` and `contexts` bench modes make the router a system-prompt picker only (SLM always answers). `contexts` restores a precomputed per-intent KV snapshot (`--slot-save-path`, `ContextStore`), which needs `--swa-full` on Gemma (`Settings.swa_full`) or llama.cpp discards the restored state.
  - Two more bench modes share the tool manuals in `manuals.py` (per-argument hints, one complete example call per tool, the send protocol), so a difference between them is routing, not wording. `active`: one fixed prompt names every tool (`<use>tool</use>`, generation stops at the tag), then that tool's manual is appended as a *system* turn (Gemma 4 renders any role as `<|turn>role`) and dropped from history afterwards; both passes reuse the cached conversation. `classes`: the router picks one of 13 action classes (`classes.py`, the 19 intents merged by what the SLM must do; exemplars are the old ones relabelled, `gold()` maps suite labels) and the class prompt (`prompts.class_static`) carries manuals + class rules; its KV snapshot is restored only when the class changes. A router mode can name its router: `classes@decision-2.0-kai-0.6b`. Decision 2.0 uses `CLASS_THRESHOLDS` and sees the previous user message on that taxonomy. Manual examples follow the exemplar rule: never a paraphrase of a suite sentence.
  - The `active2` bench mode keeps `active`'s idea: the SLM picks a tool by name, and that tool's manual is injected for one pass and then dropped. It adds constraints. `agent/grammar.py` builds GBNF from the same `ToolSpec`s and sends it as `grammar` on `/v1/chat/completions`. Pass A may only produce `<use>`+an exact tool name or plain text. In pass B the call's opening is prefilled as a trailing assistant message, which llama.cpp continues, and the grammar admits only that tool's arguments. `send_message_to_friends` is callable only right after a text turn. On b10144, `json_schema` with a prefill returns HTTP 400, so use `grammar`. Never end a prefill with a space (token boundary). Measured on Gemma 4 E2B over 91 turns: 85% exact (`active` 73%, `classes` 84%), 0 hallucinated calls, reply median 3.43 s. The arguments grammar inflates the bench's page-fault ("page-ins") figure with allocator faults, not weight re-reads.
  - `native` (`Settings.agent_mode`, bench modes `native` / `native-raw`) is the pipeline for a model trained on the tools (the mustachio bundle): one greedy pass in Gemma 4's own tool format. The declarations travel as `tools` and are rendered by the bundle's template; the volatile context is a `[ctx]` line on the user turn. `agent/native.py` holds the prompt (`SYSTEM`, the Mustachio persona, `declarations`) and the guard: schema coercion, ids must be ones the app listed, searched words/places/people must be the user's (near misses repaired from the user's wording), `send_message_to_friends` only as the answer to its own proposal. Then the chat screen runs the tool (`agent/mock.py`, invented local data), sends the result back as a `tool` message and checks the sentence about it (`agent/replies.py`: a number or name not in the result replaces the sentence with one written from the result). **The prompt, the declarations, the result shapes (`mock.*_item`) and `replies.say` are also the training format and the training labels** (`mustachio/generate.py` imports them): change one and the model has to be retrained. Every intervention is reported (`Turn.guard_note`, `Turn.answer_note`, the bench's `native-raw` row), so the guard cannot hide what the model did.
  - The routing policy is in `router.py` (`decide()` is pure). The margin, not the raw score, is the confidence signal that works (see its docstring for the calibration numbers). The margin is measured against rivals the top group's tools can't serve. A near-tie gets the union of the tied groups (`kind="merged"`, capped at `MAX_MERGED`), not a guess. A direct (no-SLM) answer on a near-tie needs the rival to produce the identical action. A no-tool intent that isn't answered directly falls back to the full prompt.
  - `suite.py` is the held-out test set. **Never add exemplars that paraphrase suite sentences**, since that measures memory, not routing. Tune exemplars against new real queries instead.

## mustachio essentials

`python -m mustachio {inspect,strip,bundle,data,train,eval,teacher,export,install}` from the git root. Read `mustachio/README.md` before changing it; the measured numbers and open items are there.

- **Environment:** `mustachio/.venv` (created with `--system-site-packages`, so it reuses the system torch; it adds peft, accelerate, litert-lm). `litert-torch` does not install on Windows.
- **Four held-out suites, held out twice.** `mustachio/hybrid.py` is a suite like `suite.py`; `mustachio/blind.py` is generated from `blind_cards.py` by a teacher model that never saw the templates, and is scored once per final model, never tuned against. No training sentence may repeat or paraphrase a suite sentence, and no training slot value may be an entity a suite tests (`lexicon.HELD_OUT`). `python -m mustachio data` enforces both and refuses to build otherwise. Don't loosen the guards to get more data.
- **One prompt format.** Gemma 4's native tool syntax, rendered by `mustachio/assets/chat_template.jinja`, which is byte-identical to the template inside the LiteRT-LM bundle. Don't replace it with the newer template on the Hugging Face repo: LiteRT-LM's on-device Jinja has no `dict.get`. Training, eval and the phone must render the same string (`template.render` was checked against LiteRT-LM's own renderer).
- **Volatile context goes in the user turn** (`[ctx]` line), never in the system text: the native template puts the tool declarations after the system text, so anything volatile there re-prefills all of them.
- **The per-layer embedding table stays in system RAM via two hooks** (`model.place`). Don't use an accelerate `device_map` for it: it copies the 4.7 GB table to the GPU on every forward.
- **Two kinds of training conversation.** `bare` (a call ends the turn, as the suites and the bench run it) and `executed` (call, tool result, one sentence; `EXECUTED_SHARE`). The sentence is written from the result by `feast_ai.agent.replies.say`, never by a model, so its label states nothing the result does not.
- **Quantisation is chosen on the dev split, never on the suites** (`eval --suites dev`). The stock `dynamic_wi4_afp32` and channel-wise int4 break this model (65% on dev, calls the runtime cannot parse); `gemma4_mixed48_b32` (`assets/recipes/`, block-wise int4) keeps it (95%). Conversion needs Linux (`litert-torch`, `litert-lm-builder==0.18.1`, `tensorflow`).
- **Logits only where needed.** The output layer is 262,144 wide; compute it at supervised positions (training) or the last position (decoding), never over the whole 2,087-token prompt.

## Conventions

- Python ≥ 3.10, `from __future__ import annotations`, dataclasses, type hints. Module docstrings explain *why*, often including measured numbers. Keep that style, and when a design decision rests on a measurement, write the measurement down.
- Core deps only: textual, psutil, httpx, rich. `onnxruntime-genai` is optional, so guard any import of it (`onnx_server.available()`). The router's `onnxruntime`/`tokenizers`/`numpy` are optional too (`encoder_server.available()`), and are imported only inside the worker subprocess.
- Keep failures non-fatal and reported. A corrupt config falls back to defaults, the RAM cap returns a `CapResult` and never raises, and HF lookups raise `LookupError_` with a user-readable message.
- The default system prompt and some presets are in Turkish, written in ASCII on purpose (`cli._use_utf8` exists because cp1252 consoles mangle text). The UI is in English. Exception: router exemplars, the bench suite and direct-answer templates use real Turkish characters, because the encoder's cased WordPiece vocabulary is Turkish ("göster" is one token, "goster" is three). Compare Turkish text through `agent.text.fold`, never `str.lower()`.
- The CLI imports its subcommands lazily inside each `_cmd_*` so `--help` stays fast.
