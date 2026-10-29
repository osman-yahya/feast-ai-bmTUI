"""Write a complete dataset directory: world, personas, histories, facts, questions.

Each persona's written history is cached under ``<out>/_raw/`` so a re-run
(after a crash, a rate limit, or a change to the question builders) does not
pay for prose that already passed :func:`writer.check`.  ``--force`` redoes it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import schema
from . import plan as P
from . import world as W
from . import writer
from .llm import DEFAULT_MODEL, GenError, Usage


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _dump_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def export_prompts(out: Path, *, seed: int = 7, personas: list[str] | None = None) -> list[Path]:
    """Write each persona's writer prompt to ``<out>/_prompts/``, for writing
    the histories by hand (or in a Claude Code session) instead of the API.
    The result goes to ``<out>/_raw/<id>.json`` and is checked by ``--offline``."""
    world = W.build(seed)
    paths = []
    for spec in P.SPECS:
        if personas and spec.id not in personas:
            continue
        plan = P.build_plan(spec, world, seed)
        path = out / "_prompts" / f"{spec.id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(writer.SYSTEM + "\n\n" + writer._prompt(plan) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def build(out: Path, *, seed: int = 7, personas: list[str] | None = None, dry_run: bool = False,
          force: bool = False, offline: bool = False, model: str = DEFAULT_MODEL,
          log=print) -> schema.Report:
    world = W.build(seed)
    names = [r["name"] for r in world["restaurants"]]
    _dump(out / "world.json", world)
    specs = [s for s in P.SPECS if not personas or s.id in personas]
    questions: list[dict] = []
    usage = Usage()

    for spec in specs:
        plan = P.build_plan(spec, world, seed)
        raw_path = out / "_raw" / f"{spec.id}.json"
        cached = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
        problems = writer.check(plan, cached, names) if cached is not None else ["no _raw file"]
        if dry_run:
            data = writer.dry_run(plan)
        elif not problems and not force:
            data = cached
            log(f"{spec.id}: cached history reused")
        elif offline:
            raise GenError(f"{spec.id}: {raw_path} not usable offline:\n  - "
                           + "\n  - ".join(problems[:20]))
        else:
            log(f"{spec.id}: writing {len(plan.sessions)} sessions with {model}...")
            data, used = writer.write(plan, names, model=model)
            usage.add(used)
            _dump(raw_path, data)
            log(f"{spec.id}: ok ({used.input_tokens} in / {used.output_tokens} out tokens)")

        sessions, beat_turn = writer.to_sessions(plan, data)
        _dump(out / "personas" / f"{spec.id}.json", plan.persona)
        _dump_lines(out / "histories" / f"{spec.id}.jsonl", sessions)
        facts = [{**f, "source_turn_id": beat_turn[b.id]}
                 for s in plan.sessions for b in s.beats for f in b.facts]
        _dump(out / "facts_gold" / f"{spec.id}.json", facts)
        for q in plan.questions:
            questions.append({**q, "evidence_turn_ids": [beat_turn[b] for b in q.get("evidence_turn_ids", [])]})

    _dump_lines(out / "questions.jsonl", questions)
    _dump(out / "MANIFEST.json", {
        "seed": seed, "model": "dry-run" if dry_run else "offline (hand-written)" if offline else model, "asked_at": P.ASKED_AT.isoformat(),
        "personas": [s.id for s in specs], "questions": len(questions),
        "generation_tokens": {"input": usage.input_tokens, "output": usage.output_tokens},
    })
    return schema.validate(schema.load(out))
