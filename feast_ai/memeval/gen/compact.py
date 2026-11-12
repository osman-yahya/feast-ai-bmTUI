"""Hand-writing a long history in a compact text form instead of JSON.

The v0.3 histories are written in a Claude Code session, not through the API.
A 40-session plan as writer-prompt JSON is ~45 KB per persona; this module
prints it as one line per beat and reads the prose back from the same shape::

    ## 4  25 mayıs pazartesi 19:10  ~11
    U [060] 'Baklavacı Güllüoğlu Usta' Meyve Salatası fiyatı nedir?
    A [061] Meyve Salatası 120 TL.
    U [] (a filler turn that carries no beat)

``[060]`` is beat ``b_<persona>_060``; several ids go comma-separated.  The
parsed result is the writer's JSON shape, saved to ``<data>/_raw/<id>.json``
and checked exactly as an API-written history would be (:func:`writer.check`
plus :func:`plan_long.audit`).

    python -m feast_ai.memeval.gen.compact outline p_elif
    python -m feast_ai.memeval.gen.compact parse p_elif     # _draft/p_elif.txt -> _raw/p_elif.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import plan_long as PL
from . import world as W
from . import writer
from .plan import DAY_TR, MONTH_TR

DATA = Path(__file__).parent.parent / "data" / "v0.3"
SEED = 7
LINE = re.compile(r"^([UA])\s*\[([\d,\s]*)\]\s?(.*)$")
BOILERPLATE = (" Kendisi hakkında bilgi VERMEZ, oraya gittiğini/yediğini SÖYLEMEZ.", " Değerlendirme YAPMAZ.",
               " Değerlendirme \"\"YAPMAZ.", "; kişisel bilgi VERMEZ", ", konuyu uygulamaya bağlamaya zorlamaz, "
               "öneri YAPMAZ", "Kullanıcı ", "Asistan ", " restoranının", " restoranında", " restoranında")


def _plan(pid: str):
    world = W.with_lookalikes(W.build(SEED), SEED)
    spec = next(s for s in PL.SPECS if s.id == pid)
    return PL.build_plan(spec, world, SEED), [r["name"] for r in world["restaurants"]]


def outline(pid: str) -> str:
    plan, _ = _plan(pid)
    prefix = f"b_{pid}_"
    out = [f"# {plan.spec.name} - {plan.spec.voice}"]
    for s in plan.sessions:
        d = s.started_at
        out.append(f"\n## {s.index}  {d.day} {MONTH_TR[d.month]} {DAY_TR[d.weekday()]} {d:%H:%M}  ~{s.target_turns}")
        for b in s.beats:
            text = b.instruction
            if b.id in plan.noise:
                # Everyday beats share their rules (no personal facts, no visits,
                # no verdicts); the outline keeps only what differs.
                for rule in BOILERPLATE:
                    text = text.replace(rule, "")
            mark = "·" if b.id in plan.noise else "*"
            out.append(f"{'U' if b.role == 'user' else 'A'}{mark}[{b.id.removeprefix(prefix)}] {text.strip()}")
    return "\n".join(out) + "\n"


def parse(pid: str, text: str) -> dict:
    sessions: list[dict] = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.rstrip()
        if not line or line.startswith("#") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            sessions.append({"session_index": int(line[3:].split()[0]), "turns": []})
            continue
        m = LINE.match(line)
        if not m or not sessions:
            raise ValueError(f"line {n}: not 'U [ids] text' / 'A [ids] text': {line[:80]}")
        role, ids, body = m.groups()
        if not body.strip():
            raise ValueError(f"line {n}: empty turn")
        beats = [f"b_{pid}_{int(x):03d}" for x in re.split(r"[,\s]+", ids.strip()) if x]
        sessions[-1]["turns"].append({"role": "user" if role == "U" else "assistant",
                                      "text": body.strip(), "beats": beats})
    return {"sessions": sessions}


def check(pid: str, data: dict) -> list[str]:
    plan, names = _plan(pid)
    return writer.check(plan, data, names) + PL.audit(plan, data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("outline", "parse"))
    ap.add_argument("persona")
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)
    data_dir = Path(args.data)
    if args.command == "outline":
        sys.stdout.write(outline(args.persona))
        return 0
    draft = (data_dir / "_draft" / f"{args.persona}.txt").read_text(encoding="utf-8")
    data = parse(args.persona, draft)
    problems = check(args.persona, data)
    turns = sum(len(s["turns"]) for s in data["sessions"])
    if problems:
        print(f"{args.persona}: {len(problems)} problem(s), {turns} turns")
        for p in problems[:60]:
            print("  -", p)
        return 1
    raw = data_dir / "_raw" / f"{args.persona}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.persona}: ok, {len(data['sessions'])} sessions, {turns} turns -> {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
