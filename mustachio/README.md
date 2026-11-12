# mustachio

The model workspace for feast mobile's on-device assistant. feast AI (the
sibling package) answered *which pipeline* works on a phone. This workspace
changes *the model*: a text-only Gemma 4 E2B, fine-tuned on the 15 feast tools
in the model's own call format, and robust to Turkish as it is typed,
including English words inside a Turkish sentence ("kanka yarın kadıköye lets
go").

Everything here reuses feast AI's definitions: the tools
(`feast_ai/agent/tools.py`), the two held-out suites and their judge
(`feast_ai/agent/suite.py`), and the prompt, guard and reply rules the app
serves (`feast_ai/agent/native.py`, `replies.py`, `mock.py`). Weights, data
and runs live in `~/.feast-ai/mustachio/`, never in the repository.

**Where it stands.** One model, `final`: a 2.86 GB `.litertlm` that runs in
the feast AI TUI on LiteRT-LM (the mobile runtime), calls the 15 tools in one
pass, answers tool results in a sentence, and speaks as the app's persona.
Sections 5 and 7 have what was measured and what it costs.

```
python -m mustachio inspect google/gemma-4-E2B-it-qat-mobile-transformers   # what is in a checkpoint
python -m mustachio strip --verify      # text-only training checkpoint, logits compared
python -m mustachio bundle              # text-only .litertlm from Google's bundle
python -m mustachio teacher --tasks smalltalk,paraphrase   # optional: Gemma 4 12B writes, never labels
python -m mustachio data --name v4 --size 16000 --dev 600 --show 20
python -m mustachio train --data ~/.feast-ai/mustachio/data/v4 --run final --live-share 0.5
python -m mustachio eval --adapter ~/.feast-ai/mustachio/runs/final/adapter --suites standard,kanka,hybrid
python -m mustachio export ~/.feast-ai/mustachio/runs/final/adapter --name final   # Linux: merge + convert
python -m mustachio eval --bundle ~/.feast-ai/mustachio/build/litert-final/model.litertlm --guard
python -m mustachio install --activate  # register the bundle with the feast AI TUI, agent mode native
```

Run from the git root. `mustachio/requirements.txt` lists what `train`, `eval`
and `bundle` need on top of feast AI.

---

## 1. Text-only: what removing vision and audio buys

Measured on the three forms of the model (MB = 10^6 bytes):

| Artefact | Total | Vision + audio | Text-only | Vocabulary tables* |
| --- | --- | --- | --- | --- |
| bf16 checkpoint (`gemma-4-E2B-it-qat-q4_0-unquantized`) | 10,209 | 952 (9.3%) | **9,257** | 5,503 (53.9%) |
| QAT mobile checkpoint (`…-qat-mobile-transformers`) | 2,458 | 340 (13.8%) | 2,118 | 1,414 (57.6%) |
| Google's LiteRT-LM bundle (`gemma-4-E2B-it.litertlm`) | 2,588 | 332 (12.8%) | **2,256** | 1,388 (53.6%) |

\* token embeddings + per-layer embeddings (+ the untied `lm_head` in the
mobile checkpoint): every table with one row per vocabulary entry.

Both text-only builds exist and are checked against the original:

| Check | Result |
| --- | --- |
| Text-only checkpoint vs the multimodal model, next-token logits on 2 prompts | max difference **0** (bit-identical) |
| Text-only bundle vs Google's, greedy answers on LiteRT-LM (CPU, 4 threads) | **identical text** on both prompts |
| Peak working set while answering, text-only / Google's | 1,274 MB / 1,274 MB |
| Warm load time, text-only / Google's | 0.17 s / 0.17 s |

What this says:

- **The strip is lossless and it works, but it is a download saving, not a
  memory one.** LiteRT-LM loads the encoders on demand, so a text-only chat
  never paid for them in RAM. The bundle is 332 MB (12.8%) smaller on flash.
- **An app that opens the text-only bundle must not request a vision or audio
  backend**, or the engine fails with `TF_LITE_VISION_ENCODER not found`.
- **The first load writes a 752 MB XNNPack cache next to the bundle** (9-50 s
  once; 0.17 s afterwards). On flash the model therefore costs 2.26 + 0.75 GB.
- **The GGUF feast AI benchmarks was already text-only** (llama.cpp keeps the
  encoders in a separate `mmproj` file), so the published bench numbers stand.

### Where the size actually is

More than half of every artefact is indexed by the 262,144-token vocabulary;
the per-layer embedding table alone is 1,285 MB of the bundle (49.6%). The
decoder that does the thinking is 818 MB.

| Vocabulary kept | Rows | Text-only bundle (estimate) |
| --- | --- | --- |
| all 262,144 tokens (today) | 100% | 2,256 MB |
| Latin script + digits, punctuation, emoji, specials (measured on the tokenizer) | 64.1% | ~1,760 MB |
| tokens seen in a Turkish + English corpus (to be measured) | 25-35%? | ~1,200-1,350 MB |

Dropping rows changes nothing for text made of kept tokens; a dropped token
is re-split into kept pieces. This is the lever that could halve the model,
and it is **not built yet**: it needs a corpus to choose the rows, a rebuilt
tokenizer and a conversion that takes one (section 6).

---

## 2. The prompt: the model's own tool format

feast AI's benchmarks ask for `<tool_call>{json}</tool_call>`; Gemma 4 kept
answering in the syntax it was trained on, and LiteRT-LM parses only that
syntax into tool calls. So this workspace uses it:

```
<|turn>system
You are Mustachio, the assistant inside the feast app: ... (rules)

<|tool>declaration:search_menu_items{description:<|"|>find a dish or drink and its price<|"|>,parameters:{...}}<tool|>   x 15
<turn|>
<|turn>user
[ctx] 2026-11-10 Tue 19:30 | Moda, Kadıköy, İstanbul | gps
kanka yarın kadıköye lets go<turn|>
<|turn>model
<|tool_call>call:set_active_location{location_query:<|"|>Kadıköy<|"|>}<tool_call|><|tool_response>response:set_active_location{active_location:<|"|>Kadıköy<|"|>,is_gps:false,ok:true}<tool_response|>Tamam, artık Kadıköy çevresinde arıyorum.<turn|>
```

The prompt is defined once, in `feast_ai/agent/native.py`, because the app
serves it; `schema.py` names the versions a run was trained with.

| Prompt | System text | Declarations | Static tokens | Trained |
| --- | --- | --- | --- | --- |
| `v0` | 193 tokens, no persona | `full`: `active2`'s descriptions and a hint per argument | 2,087 | `runs/v0` |
| `v1` | 164 tokens, no persona | `terse`: name, types, enums, a few words per tool | 1,312 | `runs/v1`, `runs/v2` |
| **`v2`** | the Mustachio persona + what to say after a tool result | `terse` | **1,454** | **`runs/final`** |

| Fact | Measured |
| --- | --- |
| `[ctx]` line per user turn | 31 tokens |
| `mustachio.template.render` vs LiteRT-LM's own rendering (`render_message_to_string`) | identical, including a tool result |
| `mustachio.template.render` vs `tokenizer.apply_chat_template` | identical |
| What ends a call / a text reply | `<|tool_response>` (token 50) / `<turn|>` (106) |
| A tool result LiteRT-LM accepts | `{"role":"tool","content":[{"type":"tool_response","name":…,"response":{…}}]}` |

- **One definition.** The declarations are generated from feast AI's
  `ToolSpec`s, the same objects that validate a call.
- **The context is in the user turn.** The native template writes the
  declarations after the system text, so a clock in the system text would
  re-prefill every declaration each minute. The system turn is static; each
  message carries a `[ctx]` line.
- **Terse declarations cost nothing once the model is trained.** `v1` cut the
  static prompt from 2,087 to 1,312 tokens and scored 96.9% where `v0` scored
  96.2% (section 5). Every static token is read once per app start on a phone
  CPU.
- **A live conversation and a replayed one render differently.** Replaying a
  past call through the template closes its turn (`…<tool_call|><turn|>`); a
  conversation that is still open never saw that close, and the answer to a
  tool result continues the turn of its call. An app may do either, so
  `train --live-share` renders half of the histories each way
  (`template.live_history`).

---

## 3. Training data: labels by construction

`python -m mustachio data` draws a scenario first (tool, arguments, slot
values) and writes the sentence from it, so the gold call is known before the
user "speaks". The same scenario is rendered in four registers with the same
label, which is the method for hybrid input: the surface changes, the action
does not.

| Register | Share | Example | Gold |
| --- | --- | --- | --- |
| clean | 24% | En fazla 200 liraya pankek nerede var? | `search_menu_items{query:"pankek", max_price:200}` |
| casual | 32% | aksam bornova tarafindayim, oraya gore ara | `set_active_location{location_query:"Bornova"}` |
| hybrid | 38% | bro last save ettigim mekanlari goster | `search_saved{item_type:"restaurant", sort_by:"recency"}` |
| English | 5% | open my chats | `open_screen{screen:"chats"}` |

The final set (`data/v4`): 16,000 conversations and 40,512 supervised targets
(20,712 calls over all 15 tools, 14,395 sentences about a tool result, 5,405
other text replies), plus 3,698 teacher paraphrases and a 600-conversation
dev split (1,616 targets).

Label rules (choices the backend has to agree with):

- slot values are canonical: "karakoye" becomes `location_query:"Karaköy"`,
  "kokorec" is searched as "kokoreç";
- a message to a friend is the user's wording exactly as typed;
- a generic English food word becomes its Turkish query ("coffee shop" →
  "kahve"), because the menus are Turkish.

What was added after v0, and what each addition answered:

| Addition | The failure it answers |
| --- | --- |
| **Mixed sessions** (22%): two to four unrelated requests in one chat | the first real chat with v0: after a location change and two searches, "Denize yaz, akşam 8de buluşalım de" came back as `get_directions(restaurant_id="Deniz")` |
| A clock-phrase generator ("gece 2de", "12'den sonra", "öğlen 1'e kadar") | v0 turned "gece 2" into `20:00` and midnight into `24:00` |
| Preambles and afterthoughts around the request | v0 answered a long, explained request with `]}` |
| **Executed conversations** (54%): call → tool result → one sentence | nothing ran the tool before; the app does, and the sentence after a result is where a small model invents |
| A yes/no follow-up after an empty result | "evet" after "Kapalı olanlara da bakayım mı?" is a search; "evet" after a message proposal is a send |
| Identity turns ("sen kimsin") | the persona: Mustachio, on this phone, nothing leaves it |

### Executed conversations: the label is written from the result

In an executed conversation every call is followed by an invented tool result
and one sentence about it. The sentence is not written by a model. It is
written by `feast_ai.agent.replies.say` **from the result**: at most three
places, only values the result contains, an empty result said plainly with one
concrete widening, a failed lookup said once. So the rule for calls (label
first, text second) also holds for the place where a small model
hallucinates most. The same function is the app's fallback when the model's
own sentence fails the check (section 6), and the result shapes are the ones
`feast_ai.agent.mock` serves, so training and serving cannot drift apart.

Checked on the generator: 5,416 result sentences, 0 with a number or a name
that is not in the result.

### The teacher (Gemma 4 12B)

`python -m mustachio teacher` uses a larger model for three things templates
cannot do, and never for labels:

| Task | How it is kept honest | Kept |
| --- | --- | --- |
| Paraphrase a user turn | every slot value must still be in the rewrite, and the teacher, asked separately, must say it is the same request | 3,799 of 5,308 asked (1,035 judged a different request, 369 empty or unchanged, 94 near a suite, 11 lost a slot); 3,698 added to `v4` |
| Small-talk replies | keyed by the sentence they answer; refused if formal ("siz"), in the wrong language, or naming a held-out entity | 490 replies for 97 sentences |
| First drafts of the blind suite | see below | 89 of 97 turns |

### Held out twice

| Guard | What it stops | `v4` build |
| --- | --- | --- |
| `pools` | a slot value that is a dish, place, person or vibe any suite tests (121 entities) | 0 violations (build refuses otherwise) |
| `entity` | a generated message that mentions one anyway | 128 conversations dropped |
| `sentence` | a message equal to a suite sentence or sharing ≥ 60% of its words (264 sentences) | 590 dropped |
| validator | a gold call feast AI's own validator rejects | 0 |

A suite turn the fine-tune gets right was got right for a dish, a district
and a person it never saw in training.

### The hybrid suite

`mustachio/hybrid.py`: 69 turns in 62 chats where the sentence is Turkish and
some words are not - English stems under Turkish suffixes ("save'lediklerim",
"location'ı Bebek yapsana"), English discourse words ("btw", "pls", "lets
go"), typed fast. Same judge as feast AI's suites, plus a reply-language
check on text turns.

Caveat: the hybrid suite and the training templates have one author. Entities
and whole sentences are held out, patterns are not always (training has
"hadi {place}'e, let's go"; the suite has "kanka yarın kadıköye lets go"), and
the ten screen names are a closed set.

### The blind suite

`mustachio/blind.py`: 97 turns in 75 chats, built to remove that caveat.
`blind_cards.py` states only *what the user wants* (tool, arguments, a
persona, a register); the sentences were written by the teacher, which never
saw a template. 89 turns are its first drafts, 3 are second drafts, and 5
were written by hand after the teacher twice wrote the message to the friend
instead of the instruction to the assistant. Six cards were loosened to what
the kept sentence actually says. Its entities are held out like the others'.

It is scored **once per final model and never tuned against**: the numbers in
section 5 are that one scoring, and nothing was changed after it.

---

## 4. Training

`python -m mustachio train`: LoRA (rank 16, alpha 32, all attention and MLP
projections of the text decoder, 24.2 M parameters) on the text-only QAT
checkpoint, bf16.

| Design choice | Why | Measured |
| --- | --- | --- |
| Per-layer embedding table kept in system RAM on a small card | 4.7 GB of lookups that are never multiplied | 4.6 GB of weights on a 12 GB card; decode 20 tok/s |
| not `device_map={"…": "cpu"}` | accelerate *executes* offloaded modules on the GPU, copying the table per forward | 1.4 → 19.6 tok/s, peak 9.35 → 5.19 GB once replaced by two hooks |
| Static prefix prefilled once per step, KV shared by the batch | every sample starts with the same 1,454 tokens | cached vs full forward: loss differs by at most 0.013 |
| Loss and logits only at supervised positions | the output layer is 262,144 wide | - |
| Batches capped by padded tokens, not by count | attention to the prefix + MLP activations cost memory per token | 2,200 tokens: 25.7 GB peak on a 32 GB card; 3,600 ran out |
| One sample per supervised turn | a past call renders differently from a generated one | - |

The gradient does not flow through the prefix's keys and values (the forward
pass is exact, recomputed with the current weights every step). That is how
the phone uses the model: prefill the tool block once, decode against it.

| Run | Hardware | Data | Steps | Time |
| --- | --- | --- | --- | --- |
| `v0` | RTX 5070 (12 GB), table in system RAM | 8,000 conversations, prompt `v0` | 600 of 2,104 (stopped on dev) | 27 min |
| `v2` | RTX 5090 (32 GB), whole model on the card | `v2` (26,929 samples, with sessions), prompt `v1` | 2,117 (1 epoch), 0.38 s/step | 13.8 min |
| **`final`** | RTX 5090 | `v4` (44,206 samples), prompt `v2`, `--live-share 0.5` | 4,774 (1 epoch), 0.44 s/step | **34.6 min** |

---

## 5. Results

`python -m mustachio eval`: four held-out suites, feast AI's judge, the native
prompt, one pass, greedy, no grammar. Two artefacts are scored: the adapter in
bf16 on a GPU, and **the quantised `.litertlm` on LiteRT-LM's CPU backend**,
which is what ships. Each is scored as the model wrote it (*raw*) and after
the app-side guard of section 6 (*guarded*).

### The final model

| Exact (right tool, right arguments, valid call) | bf16 raw | bf16 guarded | **bundle raw** | **bundle guarded** |
| --- | --- | --- | --- | --- |
| standard (60) | 98.3% | 100% | 95.0% | 98.3% |
| kanka (31) | 90.3% | 96.8% | 96.8% | 100% |
| hybrid (69) | 97.1% | 97.1% | 98.6% | 98.6% |
| **blind (97)** | **93.8%** | **94.8%** | **96.9%** | **96.9%** |
| all 257 turns | 95.3% (245) | 96.9% (249) | 96.9% (249) | **98.1% (252)** |
| Hallucinated calls | 4 | 1 | 3 | **0** |
| Unsafe sends | 0 | 0 | 0 | 0 |
| Calls the runtime could not parse | - | - | 0 | 0 |

On the dev split (what the model was trained to do, unseen conversations):

| Dev turns | bf16 (500 turns) | bundle (300 turns) |
| --- | --- | --- |
| Calls, exact | 99.3% of 269 | 100% of 159 |
| Sentences about a tool result that state only what is in it | 100% of 172 | 100% of 104 |
| All targets | 99.6% | 100% |

Quantisation did not cost accuracy on these suites: the bundle is right on
249 turns where the bf16 adapter is right on 245, and a few turns differ in
each direction (the bundle reads "Gece 12'den sonra" as `24:00`, which the
guard normalises; bf16 cut "Dilara" short, the bundle did not). On dev the
bundle reproduces every call and every result sentence.

What the guard changed on the bf16 run (every intervention is in the report):

| Turn | The model wrote | After the guard |
| --- | --- | --- |
| "Kaydettiğim kafelerden hangisi şimdi açık" (and "bunlardan hangisi şu an açık peki?") | `search_saved{item_type:"restaurant", open_now:true}` | `open_now` dropped: the tool has no such argument |
| "Yürüyerek 10 dakikayı geçmesin kanka" | `ask_about_posts{max_walk_minutes:10, …}` | the unknown argument dropped |
| "bu aksam tesvikiye civarindayim…" (blind) | `location_query:"Tevşikiye"` | repaired to the user's own "tesvikiye" |

On the bundle the guard changed four answers: `open_now` dropped twice as
above, `open_at:"24:00"` → `"00:00"`, and a garbled `"mümbär"` repaired to the
user's "mumbar". Three of those became exact.

The shipped bundle's five misses after the guard: the "Ali nazik", "mumbar"
and pharmacy turns in the table below, plus "Tanıdığım kimler feast
kullanıyor, kimi ekleyebilirim?" → `list_friends` (wanted
`get_suggested_users`) and "live music olan mekanlar var mı bu civarda" →
`search_posts` (wanted `ask_about_posts`).

The misses of the bf16 adapter (guarded):

| Suite | Turn | Answer | What went wrong |
| --- | --- | --- | --- |
| kanka | "kaça kadar açıklar peki" | `}<tool_call|>` | a degenerate output. The guard's fallback was widened after this run to catch format tokens (the user now gets a request to rephrase); the turn stays a miss |
| hybrid | "Dilara benim friend'im mi feast'te" | `list_friends{query:"Dila"}` | a name cut short (twice, same chat) |
| blind | "Ali nazik yiyebilirim ama … 320 TL'yi geçmesin" | `search_menu_items{query:"Ali", max_price:320}` | a two-word dish read as a name |
| blind | "…arabayla nasıl gidilir onu söyler misin…" | `get_directions{mode:"drive"}` | no venue id |
| blind | "…bi mumbar yiyelim diyorum millet ne demiş…" | `search_menu_items{query:"mümbar"}` | "what people said" is a post search |
| blind | "kaydettiklerim arasinda asure gecerenleri bulur musun" | `search_saved{item_type:"post", query:"asur"}` | the dative read into the word |
| blind | "…gece nöbeti olan bir eczane var mı…" | `search_restaurants{query:"eczane", open_now:true}` | a pharmacy is not something feast finds |

### How it got here (bf16, standard + kanka + hybrid, 160 turns, raw)

| Run | Prompt | Exact | Notes |
| --- | --- | --- | --- |
| base model, untrained | `v0` (2,087 tokens) | 58.1% (93) | echoes the `[ctx]` line on 48 turns; the fair untrained reference is feast AI's `direct`, 71% on its 91 turns |
| `v0`, 600 steps | `v0` | 96.2% (154) | first chat showed the single-topic bias (section 3) |
| `v1` | `v1` (1,312) | 96.9% (155) | 775 fewer static tokens, no loss; with `full` declarations the same data scored 98.1% |
| **`final`** | `v2` (1,454) | **96.2% (154)**, 98.1% guarded | also answers tool results; persona |

What to keep in mind when reading these:

- **The first three suites have been looked at.** Their misses guided the
  generator between runs (the time phrases, the preambles). Entities and
  sentences stayed held out, but only the blind suite is untouched: 93.8%.
- **bf16 on a GPU is not the phone.** The bundle columns are the shipped
  numbers. Neither is a phone: the bundle ran on a desktop CPU.
- **Four turns out of 257 is noise.** bf16 and the bundle differ by that
  much; so do raw and guarded on a single suite.
- **Small talk is still narrow.** Text turns count as exact when no tool is
  called. The teacher's replies widened them; the persona's voice is trained
  from templates and a few hundred teacher lines.
- **One run, one seed.**

---

## 6. The app side: what is checked in code

A fine-tune raises the rate of right answers; it does not make any single
answer trustworthy. So the app does not trust the model for what it can check
(`feast_ai/agent/native.py`, `replies.py`), and every intervention is
reported so the checks cannot hide what the model did.

| Step | Check | Example |
| --- | --- | --- |
| The call | **schema**: unknown arguments dropped, enums and types coerced, a clock normalised | `open_at:"24:00"` → `"00:00"` |
| | **ids**: a venue or dish id must be one the app listed in this chat (a result list or a tool result) | an invented `r_937` is dropped, the call becomes invalid |
| | **grounding**: a searched dish, a place and a person must come from the user's words; a near miss is repaired from them | `"trioçe"` → the user's `"trileçe"`; no source at all → reported as ungrounded |
| | **confirm-first**: `send_message_to_friends` runs only as the answer to the assistant's own proposal, with its names and wording | a send in the turn the user asks becomes the proposal it skipped |
| | **fallback**: an answer that is neither a call nor a sentence is replaced by a request to rephrase | `]}` |
| A text answer | a number or a name nothing in the chat contains is reported (`ungrounded`), not rewritten | "Çiya Sofrası, Adana 450 lira" from memory |
| The tool | runs in app code (`feast_ai/agent/mock.py` in the TUI: an invented neighbourhood) | |
| The sentence after it | **faithfulness** (`replies.check`): every number and every name must be in the result, the call or the user's message; otherwise the sentence is rewritten from the result | an invented price never reaches the screen |
| | **language**: an English question answered in Turkish is rewritten | |

This is the grammar of feast AI's `active2` moved behind the model: a grammar
could only make an argument well-formed, the guard also sees the
conversation. It costs no second pass.

---

## 7. Shipping on LiteRT-LM

`python -m mustachio export <adapter>` merges the adapter into the text-only
weights (in the multimodal key layout the converter expects) and runs
`litert-torch export_hf`. Conversion needs Linux: `litert-torch` 0.10.0,
`tensorflow`, and `litert-lm-builder==0.18.1` (0.19 dropped a name the
converter imports). `python -m mustachio install --activate` then registers
the bundle with the feast AI TUI.

### Quantisation is the step that can lose the fine-tune

The recipe was chosen on the dev split, with the `v0` adapter, never on the
suites. 200 dev turns per row; the first two rows share one set of turns
(`data/v0`'s dev split), the last three another (`data/v2`'s):

| Recipe | What it does | Bundle | Dev exact | Calls the runtime could not parse |
| --- | --- | --- | --- | --- |
| `dynamic_wi4_afp32` (the converter's stock recipe) | int4, channel-wise, every weight | 2,550 MB | 65.0% | 21 |
| **`gemma4_mixed48_b32`** | int4 **block-wise (32)**, int8 for the per-layer embedding projection | 2,860 MB | **95.0%** | 0 |
| `gemma4_mixed48` | the same, channel-wise | 2,571 MB | 64.5% | 15 |
| `gemma4_mixed48_hr` | Hadamard-rotated int4 | 2,657 MB | 86.5% | 0 |
| `gemma4_mixed48_b32` (same turns as the two above) | | | **90.0%** | 0 |

Channel-wise int4 breaks this model visibly: garbled queries ("yakça",
"mücvebik çorbası"), a truncated tool name, a doubled quote token that
LiteRT-LM's own parser rejects. Block-wise scales cost 290 MB and keep the
model (for the final model the dev table in section 5 has both sides). The
recipes are `ai_edge_quantizer`'s own Gemma 4 recipes;
the converter's CLI rejects them by name, so they are shipped flattened to
JSON in `assets/recipes/` and passed as a file.

### Speed against Google's own bundle

Same machine (i5-14600KF), 4 threads pinned, the `v2` prompt, through feast
AI's LiteRT engine:

| | Google's text-only bundle | **mustachio `final`** |
| --- | --- | --- |
| Size | 2,256 MB | 2,862 MB |
| Static prompt (1,496 tokens), once per app start | 5.0 s (345 tok/s) | 7.6 s (205 tok/s) |
| First token of a live turn (~45 tokens read) | 0.5 s | 1.0 s |
| Decode | 16-21 tok/s | ~10 tok/s |

Google's decoder was *trained* into a mixed 2/4/8-bit scheme
(`quant_method: gemma`) the public converter does not produce; ours is
quantised after training with float activations, and it is slower. One part
of the gap was a setting: a single `prefill_256` signature made every request
read 256 tokens however short (1.9 s to the first token of a 21-token tool
result, 9.7 s for the static prompt). The export now carries `prefill_128`
and `prefill_1024`, as Google's does: 1.0 s and 7.6 s.

With the settings the TUI was left in (2 threads, a 3 GB device with the RAM
ceiling on, so the weights stream through 2,150 MB): the static prompt takes
20 s once, decode is 7.4 tok/s, and a turn with a tool takes 8-10 s (the call,
the tool, the sentence).

### What the runtime does, measured through the engine in `feast_ai`

| Finding | Consequence |
| --- | --- |
| A new conversation created from a history re-reads all of it (14 s for 2,650 tokens at 4 threads) | the worker keeps the conversation: `live` (only the new message is read), `clone` (a new chat on a prefilled prompt), `replay` (the app rewrote history) |
| Prefix reuse only happens between clones of one base conversation with identical creation parameters, `max_output_tokens` included | the warm-up must use the settings of the real turns |
| A malformed call is an exception (`Failed to parse FC tool calls`) with the raw text inside | the worker recovers the text and reports a parse error instead of failing the request |
| The app's checks may edit a call's arguments or the sentence after a result | those two do not count as a different conversation; anything else is replayed |
| A history window that drops one turn per message replays on every message | the window is sized in text and moves rarely, by half (`native.window`) |
| The first load writes an XNNPack weight cache next to the bundle (752 MB for Google's, 1,310 MB for ours with two prefill signatures); three processes doing it at once corrupt it and segfault | load once before running in parallel; on flash the model costs 2.86 + 1.31 GB |
| Built-in constrained decoding (`LL_GUIDANCE`) | changed nothing in our tests; the guard does the job |
| A Windows venv's `python.exe` is a launcher stub | pin and cap the real interpreter (`engine.server.worker_python`) |

---

## 8. Open items, in order

1. **Close the speed gap.** Path B: Google's bundle (their trained
   quantisation) plus the LoRA as a runtime adapter
   (`LoraRankConfig`/`LoraConfig` exist; the adapter file format and training
   against the mobile-QAT base are unverified). Or QAT with the converter's
   block-wise scheme.
2. **A phone.** Every number here is a desktop CPU with pinned cores. The
   feast AI budget profiles (and its RAM ceiling) apply to this engine too;
   the bundle has not been run on a device.
3. **Real tools.** The TUI's tools are `mock.py`. Result shapes are a
   contract between the app and the model (`mock.*_item`): changing a key
   means regenerating the data.
4. **Trim the vocabulary** (section 1): still the lever that could halve the
   download.
5. **Real user logs** for what templates and a teacher cannot write, and a
   suite nobody on the project wrote.
6. **Replay cost.** When the guard turns a call into text, or the window
   moves, the whole chat is read again. LiteRT-LM has no call to append
   history without generating; a saved KV state per conversation would
   remove it.

Sources: [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4),
[litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm),
[Gemma 4 on LiteRT-LM](https://developers.google.com/edge/litert-lm/models/gemma-4),
[LiteRT-LM file builder](https://developers.google.com/edge/litert-lm/file_builder),
[Gemma 4 QAT](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/),
[Google's LoRA example for this model](https://huggingface.co/litert-community/gemma-4-E2B-it-nepali-lora).
