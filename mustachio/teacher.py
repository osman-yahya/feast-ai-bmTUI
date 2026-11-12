"""A larger model as a second author: how users phrase things, not what they mean.

The generator writes sentences from templates, so every sentence in the
training set - and in the ``hybrid`` suite - has one author's habits.  A
teacher model (Gemma 4 12B here; any chat model would do) breaks that in two
places, and in both the *label* still comes from the scenario, never from
the teacher:

``blind``       the teacher writes the user turns of the blind suite from an
                English description of what the user wants
                (:mod:`mustachio.blind_cards`).  It never sees a template or
                a tool name, so its wording is its own; the expected call was
                fixed before it wrote anything.  Every sentence is reviewed
                by hand before it enters :mod:`mustachio.blind`.
``paraphrase``  the teacher rewrites training requests the way people type.
                A rewrite is kept only if every slot value of the gold call
                is still in it (:func:`slots_kept`) and the teacher, asked
                separately, says the two messages ask for the same thing.
                That is distillation of *phrasing*: the student learns what
                the teacher knows about Turkish as it is typed, on labels the
                teacher cannot get wrong.
``smalltalk``   the teacher answers greetings, thanks and off-topic requests
                in the assistant's voice, so those replies stop being the
                handful of sentences the templates had.

Runs on a GPU with the teacher in bf16 (24 GB for the 12B).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from feast_ai.agent import native
from feast_ai.agent.text import fold

from . import dataset, lexicon, paths

TEACHER = "google/gemma-4-12B-it"

STYLES = {
    "casual": "the way a young Turkish person texts a friend: lower case, little punctuation, "
              "spoken forms (bi, nerde, yapcam), maybe a word like kanka, abi or ya",
    "hybrid": "Turkish with a few English words mixed in, the way white-collar Turks talk "
              "(check etmek, mood, location, nearby, budget) - the sentence stays Turkish",
    "ascii": "Turkish typed without Turkish letters (c g i o s u instead of ç ğ ı ö ş ü), "
             "lower case, in a hurry",
    "clean": "correct, polite written Turkish",
    "rambling": "a longer message that first says something about the user's situation or "
                "mood and then makes the request",
    "terse": "as few words as possible, like a search query",
}

WRITE = (
    "You write messages that users type to the assistant of feast, a Turkish restaurant "
    "discovery app. Write ONE message for this situation.\n\n"
    "What the user wants: {want}\n"
    "{keep}"
    "How they type: {style}\n\n"
    "Rules: Turkish (unless the style says otherwise). Do not mention apps, tools or "
    "functions. Do not add requests that are not in the situation. Output only the "
    "message, on one line, without quotes."
)
REWRITE = (
    "Here is a message a user typed to the assistant of a Turkish restaurant app:\n\n"
    "{text}\n\n"
    "Rewrite it as a different person would type the same request.\n"
    "How they type: {style}\n"
    "{keep}"
    "Rules: the request must stay exactly the same - add nothing, drop nothing. Output "
    "only the new message, on one line, without quotes."
)
SAME = (
    "Two messages typed to a restaurant app assistant:\n\nA: {a}\nB: {b}\n\n"
    "Do A and B ask the assistant for exactly the same thing (same item or place, same "
    "limits, same order of preference)? Answer with one word: yes or no."
)
REPLY = (
    "You are feast AI, the assistant inside the feast mobile app: you help with nearby "
    "restaurants and dishes, saved items, friends and messages. You are warm, brief and you "
    "never invent venues, dishes or prices.\n\n"
    "A user wrote: {text}\n\n"
    "{task}\n"
    "Reply in the user's language, in one or two short sentences. Output only the reply."
)


class Teacher:
    def __init__(self, model_id: str = TEACHER) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}).eval()

    def ask(self, prompts: list[str], max_new_tokens: int = 96, temperature: float = 0.8,
            batch: int = 32, seed: int = 7) -> list[str]:
        """One reply per prompt, batched."""
        torch = self.torch
        torch.manual_seed(seed)
        out: list[str] = []
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        replies: dict[int, str] = {}
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            texts = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i]}], tokenize=False,
                add_generation_prompt=True) for i in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None, top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id)
            for index, row in zip(chunk, gen):
                text = self.tokenizer.decode(row[enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                replies[index] = _one_line(text)
            print(f"  teacher: {min(start + batch, len(order))}/{len(order)}", flush=True)
        out = [replies[i] for i in range(len(prompts))]
        return out


def _one_line(text: str) -> str:
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S).strip()
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line.strip("\"'“”‘’ ").strip()


# --------------------------------------------------------------------------
# Label preservation
# --------------------------------------------------------------------------
_NUMERIC = ("max_price", "min_price", "max_walk_minutes", "max_distance_meters")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


def slots_kept(args: dict, text: str) -> bool:
    """Is every slot value of the gold call still stated in ``text``?

    Strings are grounded the way the app's guard grounds them; a number must
    appear as digits (1000 m may be written "1 km").  Flags and enum values
    cannot be checked this way - that is what the teacher's "same request?"
    answer is for.
    """
    numbers = {float(n.replace(",", ".")) for n in _DIGITS.findall(text)}
    for key, value in args.items():
        if key in _NUMERIC:
            wanted = {float(value)}
            if key == "max_distance_meters":
                wanted.add(float(value) / 1000.0)
            if not wanted & numbers:
                return False
        elif key == "open_at":
            hour, minute = (int(part) for part in str(value).split(":"))
            if not ({float(hour), float(hour % 12 or 12)} & numbers
                    or (hour == 0 and "gece yar" in fold(text))):
                return False
        elif key in ("query", "location_query", "friend_name"):
            if native.ground(str(value), [text])[0] != "ok":
                return False
        elif key == "friend_names":
            if any(native.ground(str(name), [text])[0] != "ok" for name in value):
                return False
    return True


def _keep_line(args: dict) -> str:
    words = [str(v) for k, v in args.items() if k in ("query", "location_query", "friend_name")]
    words += [str(n) for n in args.get("friend_names", [])]
    words += [str(v) for k, v in args.items() if k in _NUMERIC]
    if not words:
        return ""
    return "These must appear in it as they are: " + ", ".join(words) + "\n"


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def paraphrase(teacher: Teacher, examples: list[dict], per_example: int, seed: int) -> dict:
    """Rewrites of single-request training conversations, with their labels."""
    rng = random.Random(seed)
    suite = dataset.suite_sentences()
    jobs = []
    for example in examples:
        messages = example["messages"]
        if len(messages) != 2 or not messages[1].get("tool_calls"):
            continue  # one request, one call: the only shape a rewrite cannot break
        call = messages[1]["tool_calls"][0]["function"]
        if call["name"] in ("send_message_to_friends",):
            continue
        for style in rng.sample(sorted(STYLES), per_example):
            jobs.append((example, call, style))
    prompts = [REWRITE.format(text=e["messages"][0]["content"], style=STYLES[s],
                              keep=_keep_line(c["arguments"])) for e, c, s in jobs]
    rewrites = teacher.ask(prompts, max_new_tokens=80, temperature=0.9, seed=seed)

    kept, stats = [], {"asked": len(jobs), "empty_or_same": 0, "slot_lost": 0,
                       "held_out": 0, "near_suite": 0, "not_same": 0}
    candidates = []
    for (example, call, style), text in zip(jobs, rewrites):
        original = example["messages"][0]["content"]
        if len(text) < 4 or fold(text) == fold(original) or len(text) > 240:
            stats["empty_or_same"] += 1
        elif not slots_kept(call["arguments"], text):
            stats["slot_lost"] += 1
        elif lexicon.held_out_hit(text):
            stats["held_out"] += 1
        elif dataset.too_close(text, suite):
            stats["near_suite"] += 1
        else:
            candidates.append((example, call, style, text))
    verdicts = teacher.ask([SAME.format(a=e["messages"][0]["content"], b=t)
                            for e, _c, _s, t in candidates],
                           max_new_tokens=4, temperature=0.0, seed=seed)
    for (example, call, style, text), verdict in zip(candidates, verdicts):
        if not fold(verdict).startswith(("yes", "evet")):
            stats["not_same"] += 1
            continue
        kept.append({"id": f"{example['id']}-t-{style}", "family": example["family"],
                     "register": "teacher-" + style, "ctx": example["ctx"],
                     "messages": [{"role": "user", "content": text}, example["messages"][1]]})
    stats["kept"] = len(kept)
    return {"examples": kept, "stats": stats}


_FORMAL = re.compile(r"\bsiz(e|in|i|den|inle)?\b|[ıiuü]n[ıiuü]z")
_FOREIGN = re.compile(r"[^\x00-\u024f\u2010-\u203a\U0001F300-\U0001FAFF]")


def reply_ok(reply: str, english: bool) -> bool:
    """Is a teacher reply usable as the assistant's own voice?

    The first batch showed what to refuse: an English reply to a Turkish
    greeting, three Chinese characters, "I'm Feast", and a mix of formal
    "siz" and informal "sen" - the app says "sen".
    """
    from .evaluate import reply_language

    if not 4 < len(reply) < 200 or "id:" in reply or "<" in reply or "Feast" in reply:
        return False
    if _FOREIGN.search(reply) or lexicon.held_out_hit(reply):
        return False
    if english:
        return reply_language("what is the time", reply)
    lowered = reply.replace("İ", "i").replace("I", "ı").lower()
    return reply_language("saat kaç", reply) and not _FORMAL.search(lowered)


def smalltalk(teacher: Teacher, per_prompt: int, seed: int) -> dict:
    """Assistant replies for turns that need no tool, keyed by the user's sentence.

    Keyed by sentence because a reply only fits what it answers: a refusal
    written for "take a photo" is wrong for "call a taxi".  What the
    assistant *can* do is not asked of the teacher at all - it overclaimed
    ("manage your saved items", "check your messages").
    """
    from . import generate

    tasks = {
        "greet": "Greet them back and invite them to say what they feel like eating.",
        "thanks": "Accept the thanks warmly.",
        "how": "Answer kindly and offer help.",
        "bye": "Say goodbye warmly.",
        "refuse": "You cannot do this in this app. Say so politely in one sentence and "
                  "mention what you can help with instead (finding places and dishes, saved "
                  "items, friends).",
    }
    address = " Address the user informally (Turkish 'sen', never 'siz')."
    jobs = []
    for kind, (asks, _replies, _replies_en) in zip(("greet", "thanks", "capable", "how", "bye"),
                                                   generate.TALK):
        if kind == "capable":
            continue
        for register, sentences in asks.items():
            for sentence in sentences:
                jobs += [(kind, register, sentence)] * per_prompt
    for row in lexicon.OOS:
        for register, sentence in zip(("clean", "casual", "hybrid"), row):
            jobs += [("refuse", register, sentence)] * per_prompt
    for sentence in generate.OOS_EN:
        jobs += [("refuse", "en", sentence)] * per_prompt
    replies = teacher.ask(
        [REPLY.format(text=s, task=tasks[k] + ("" if r == "en" else address))
         for k, r, s in jobs], max_new_tokens=60, temperature=0.9, seed=seed)
    pool: dict[str, dict] = {}
    for (kind, register, sentence), reply in zip(jobs, replies):
        entry = pool.setdefault(sentence, {"kind": kind, "replies": []})
        if reply_ok(reply, register == "en") and reply not in entry["replies"]:
            entry["replies"].append(reply)
    return pool


def blind(teacher: Teacher, seed: int, only: set | None = None) -> list[dict]:
    """The blind suite's user turns, written from the cards' descriptions.

    ``only`` limits it to ``{(card id, turn index)}`` - the turns a review
    sent back.
    """
    from . import blind_cards

    jobs = []
    for card in blind_cards.CARDS:
        for index, turn in enumerate(card["turns"]):
            if only is None or (card["id"], index) in only:
                jobs.append((card["id"], index, turn))
    prompts = [WRITE.format(want=t["want"], style=STYLES[t["style"]],
                            keep=("These must appear in it: " + ", ".join(t["keep"]) + "\n")
                            if t.get("keep") else "") for _c, _i, t in jobs]
    texts = teacher.ask(prompts, max_new_tokens=80, temperature=0.9, seed=seed)
    return [{"case": case, "turn": index, "style": turn["style"], "want": turn["want"],
             "text": text} for (case, index, turn), text in zip(jobs, texts)]


def main(args) -> int:
    paths.ensure_dirs()
    out = Path(args.out) if args.out else paths.DATA_DIR / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.model)
    if "blind" in args.tasks:
        only = {(part.rsplit(":", 1)[0], int(part.rsplit(":", 1)[1]))
                for part in args.only.split(",") if part} or None
        rows = blind(teacher, args.seed, only)
        name = "blind_redrafts.json" if only else "blind_drafts.json"
        (out / name).write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                encoding="utf-8")
        print(f"blind: {len(rows)} user turns drafted -> {out / name}")
    if "smalltalk" in args.tasks:
        pool = smalltalk(teacher, args.per_prompt, args.seed)
        (out / "smalltalk.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
        print(f"smalltalk: {sum(len(e['replies']) for e in pool.values())} replies kept "
              f"for {len(pool)} sentences")
    if "paraphrase" in args.tasks:
        source = dataset.read((Path(args.data) if args.data else paths.DATA_DIR / "v2")
                              / "train.jsonl")
        rng = random.Random(args.seed)
        rng.shuffle(source)
        result = paraphrase(teacher, source[:args.sample], args.per_example, args.seed)
        dataset.write_rows(result["examples"], out / "paraphrases.jsonl")
        (out / "paraphrase_stats.json").write_text(json.dumps(result["stats"], indent=1),
                                                   encoding="utf-8")
        print(f"paraphrase: {result['stats']}")
    return 0
