"""Build the training set, and prove it does not contain the test.

feast AI's suites are only worth something while they are held out, and a
generator that writes thousands of sentences can break that by accident.
Three guards run on every build and their counts go into ``stats.json``:

``pools``     no slot value mentions a dish, place, person or vibe that any
              suite tests (:func:`mustachio.lexicon.check_pools`) - the build
              refuses to continue otherwise;
``entity``    no generated user message mentions one either (a template
              could spell one out) - such a conversation is dropped;
``sentence``  no user message is a suite sentence or close to one: equal
              after folding, or sharing 60% of its words (Jaccard on folded
              word sets) - dropped as well.  Short confirmations ("evet",
              "tamam gönder") are where this bites.

Every gold call is also run through feast AI's own validator, so a template
that writes an argument the tool does not have fails here and not on the
phone.  The dev split is drawn from the same generator with another seed and
shares no user message with train: it measures fit, not generalisation -
that is what the three suites are for.
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from feast_ai.agent.suite import SUITES
from feast_ai.agent.text import fold
from feast_ai.agent.tools import BY_NAME, validate

from . import generate, lexicon, paths
from .hybrid import HYBRID

_WORD = re.compile(r"[a-z0-9]+")
SIMILARITY = 0.6


def _words(text: str) -> frozenset:
    return frozenset(_WORD.findall(fold(text)))


def suite_sentences() -> list[tuple[str, frozenset]]:
    from .blind import BLIND

    cases = list(HYBRID) + list(BLIND) + [case for suite in SUITES.values() for case in suite]
    texts = {turn.text for case in cases for turn in case.turns}
    texts |= {content for case in cases for role, content in case.history if role == "user"}
    return [(text, _words(text)) for text in sorted(texts)]


def too_close(text: str, suite: list[tuple[str, frozenset]]) -> str:
    """The suite sentence ``text`` repeats or nearly repeats, or ``""``."""
    words = _words(text)
    if not words:
        return ""
    for sentence, other in suite:
        if words == other:
            return sentence
        union = len(words | other)
        if union >= 3 and len(words & other) / union >= SIMILARITY:
            return sentence
    return ""


def _gold_issues(example: generate.Example) -> list[str]:
    issues = []
    for message in example.messages:
        for call in message.get("tool_calls") or ():
            function = call["function"]
            if function["name"] not in BY_NAME:
                issues.append(f"unknown tool {function['name']}")
                continue
            issues += validate(function["name"], function["arguments"]).issues
    return issues


def _first_request(example: generate.Example) -> str:
    """The first user message that asks for something (after a seeded result list)."""
    users = [m["content"] for m in example.messages if m["role"] == "user"]
    seeded = (len(example.messages) > 2 and example.messages[1]["role"] == "assistant"
              and not example.messages[1].get("supervise"))
    return users[1] if seeded and len(users) > 1 else users[0]


def build(size: int, seed: int, suite: list[tuple[str, frozenset]],
          exclude: set | None = None) -> tuple[list[generate.Example], dict]:
    rng = random.Random(seed)
    seen: set = set(exclude or ())
    out: list[generate.Example] = []
    dropped: collections.Counter = collections.Counter()
    attempts = 0
    while len(out) < size and attempts < size * 40:
        attempts += 1
        example = generate.one(rng)
        users = tuple(m["content"] for m in example.messages if m["role"] == "user")
        key = (users, json.dumps(example.messages[-1], ensure_ascii=False, sort_keys=True))
        if key in seen:
            dropped["duplicate"] += 1
            continue
        # The opening request must be new; "evet" after it cannot be.
        if exclude is not None and (_first_request(example),) in exclude:
            dropped["shared with train"] += 1
            continue
        if any(lexicon.held_out_hit(u) for u in users):
            dropped["entity"] += 1
            continue
        if any(too_close(u, suite) for u in users):
            dropped["sentence"] += 1
            continue
        issues = _gold_issues(example)
        if issues:
            raise ValueError(f"{example.family}: gold call is invalid: {issues}: "
                             f"{example.messages}")
        seen.add(key)
        example.id = f"{seed}-{len(out):05d}"
        out.append(example)
    return out, dict(dropped)


def _stats(examples: list[generate.Example]) -> dict:
    family = collections.Counter(e.family for e in examples)
    register = collections.Counter(e.register for e in examples)
    tools: collections.Counter = collections.Counter()
    supervised = answers = 0
    for example in examples:
        for index, message in enumerate(example.messages):
            if message["role"] != "assistant" or not message.get("supervise"):
                continue
            supervised += 1
            answers += index > 0 and example.messages[index - 1]["role"] == "tool"
            calls = message.get("tool_calls")
            tools[calls[0]["function"]["name"] if calls else "(text)"] += 1
    return {"conversations": len(examples), "supervised_turns": supervised,
            "result_answers": answers,
            "executed": sum(any(m["role"] == "tool" for m in e.messages) for e in examples),
            "by_family": dict(family.most_common()), "by_register": dict(register.most_common()),
            "by_target": dict(tools.most_common())}


def write(examples: list[generate.Example], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(json.dumps({"id": example.id, "family": example.family,
                                 "register": example.register, "ctx": example.ctx,
                                 "messages": example.messages}, ensure_ascii=False) + "\n")


def write_rows(rows: list[dict], path: Path) -> None:
    """Conversations that are already plain dicts (the teacher's)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def show(example: generate.Example | dict) -> str:
    data = example if isinstance(example, dict) else example.__dict__
    lines = [f"[{data['family']} / {data['register']}]  {data['ctx']['now']} | "
             f"{data['ctx']['label']}"]
    for message in data["messages"]:
        if message["role"] == "user":
            lines.append(f"  user: {message['content']}")
            continue
        if message["role"] == "tool":
            result = json.dumps(message["content"][0]["response"], ensure_ascii=False)
            lines.append(f"  tool: {result[:160]}{'...' if len(result) > 160 else ''}")
            continue
        mark = "" if message.get("supervise") else "   (history, not a target)"
        calls = message.get("tool_calls")
        if calls:
            function = calls[0]["function"]
            lines.append(f"  call: {function['name']} "
                         f"{json.dumps(function['arguments'], ensure_ascii=False)}{mark}")
        else:
            lines.append(f"  text: {message['content']}{mark}".replace("\n", "\n        "))
    return "\n".join(lines)


def _append_paraphrases(train_path: Path, suite: list, seen_requests: set) -> dict:
    """Add the teacher's rewrites (``data/teacher/paraphrases.jsonl``) to the training file.

    They were filtered when they were written; they are checked again here
    because the suites may have grown since (the blind suite did).
    """
    source = paths.DATA_DIR / "teacher" / "paraphrases.jsonl"
    if not source.exists():
        return {"available": 0, "added": 0}
    rows = read(source)
    added, dropped = [], collections.Counter()
    for row in rows:
        text = row["messages"][0]["content"]
        if (text,) in seen_requests:
            dropped["duplicate"] += 1
        elif lexicon.held_out_hit(text):
            dropped["entity"] += 1
        elif too_close(text, suite):
            dropped["sentence"] += 1
        else:
            seen_requests.add((text,))
            added.append(row)
    with open(train_path, "a", encoding="utf-8", newline="\n") as fh:
        for row in added:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"available": len(rows), "added": len(added), "dropped": dict(dropped)}


def main(args) -> int:
    paths.ensure_dirs()
    violations = lexicon.check_pools()
    if violations:
        print("The slot pools contain held-out entities - fix mustachio/lexicon.py:")
        print("\n".join("  " + v for v in violations))
        return 1
    suite = suite_sentences()
    train, dropped_train = build(args.size, args.seed, suite)
    train_requests = {(_first_request(e),) for e in train}
    dev, dropped_dev = build(args.dev, args.seed + 1000, suite, exclude=train_requests)

    out = Path(args.out) if args.out else paths.DATA_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    write(train, out / "train.jsonl")
    write(dev, out / "dev.jsonl")
    teacher = {"smalltalk_sentences": len(generate.smalltalk_pool()),
               "paraphrases": _append_paraphrases(out / "train.jsonl", suite, train_requests)}
    stats = {"seed": args.seed, "train": _stats(train), "dev": _stats(dev), "teacher": teacher,
             "guards": {"suite_sentences": len(suite), "held_out_entities": len(lexicon.HELD_OUT),
                        "pool_violations": 0, "dropped_train": dropped_train,
                        "dropped_dev": dropped_dev, "similarity_threshold": SIMILARITY}}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    print(f"dataset: {out}")
    for name in ("train", "dev"):
        s = stats[name]
        print(f"  {name}: {s['conversations']} conversations, {s['supervised_turns']} targets")
    print(f"  registers: {stats['train']['by_register']}")
    print(f"  targets:   {stats['train']['by_target']}")
    print(f"  teacher: {teacher}")
    print(f"  guards: {len(suite)} suite sentences, {len(lexicon.HELD_OUT)} held-out entities; "
          f"dropped from train {dropped_train}, from dev {dropped_dev}")
    if args.show:
        rng = random.Random(args.seed)
        for example in rng.sample(train, min(args.show, len(train))):
            print("\n" + show(example))
    return 0
