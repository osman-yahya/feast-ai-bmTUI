"""Score a model on the held-out suites, in the format the phone will use.

feast AI's ``bench`` measures a *pipeline* on a phone budget.  This measures
a *model*: the same suites (``standard``, ``kanka``, this package's
``hybrid`` and ``blind``), the same judge (:func:`feast_ai.agent.suite.score`
and the tool validator), the native prompt, one pass, greedy decoding.

Two backends, because they answer different questions:

``torch``   bf16 weights (+ a LoRA adapter) on the GPU.  Minutes per run, so
            it can follow every training run.  It says what the *weights*
            learned.
``bundle``  a ``.litertlm`` on LiteRT-LM's CPU backend - the quantised
            artefact that ships, rendered and parsed by the runtime itself.
            It says what the *phone* will get.  The gap between the two is
            the price of quantisation.

``--guard`` scores the answers after the app-side checks of
:func:`feast_ai.agent.native.guard`; without it the model is judged alone.
Both are worth a row: the guard must not be allowed to hide what the model
did, and every answer it changed is counted (``guard_changed``).

Differences from ``bench`` that matter when comparing numbers:

* a call in the native syntax is the expected format here, not a slip;
* the context is a ``[ctx]`` line in the user turn, not a CONTEXT block;
* a multi-turn case replays the model's *own* earlier replies, as ``bench``
  does, so a wrong first turn can cost the second one too;
* text replies get one more check, :func:`reply_language`: a Turkish (or
  hybrid) message must be answered in Turkish.  It is reported separately
  and does not change "exact";
* ``ungrounded`` counts answers with a searched word, place or person that
  nothing the user wrote supports - a stricter hallucination count than the
  validator's, which only knows the schema.
"""

from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path

from feast_ai.agent import native
from feast_ai.agent.calls import CallCheck, check_reply, render_call
from feast_ai.agent.context import DEFAULT_TAGS
from feast_ai.agent.suite import SUITES, Case, ordered, score
from feast_ai.agent.text import detect_lang, fold
from feast_ai.agent.tools import ALL_TOOL_NAMES

from . import paths, schema, template
from .hybrid import HYBRID

#: Frozen like ``bench`` freezes its clock: one prompt per run.
EVAL_CTX = schema.Ctx(now="2026-11-10 Tue 19:30", label="Moda, Kadıköy, İstanbul", is_gps=True)
_EN_WORDS = {"the", "you", "your", "is", "are", "what", "can", "help", "with", "that", "for",
             "and", "to", "of", "in", "it", "do", "would", "like", "i", "there", "here"}
_TR_WORDS = {"bir", "mi", "mı", "mu", "mü", "ne", "ve", "bu", "için", "var", "yok", "evet",
             "hayır", "tamam", "sana", "senin", "bunu", "şu", "an", "saat", "bugün", "rica",
             "merhaba", "selam", "nasıl", "göndereyim", "yardımcı"}
_TOKEN = re.compile(r"[a-zçğıöşü']+")


def suite_names() -> list[str]:
    from . import blind

    return list(SUITES) + ["hybrid"] + (["blind"] if blind.BLIND else [])


def cases_of(suite: str, limit: int = 0) -> list[Case]:
    if suite == "hybrid":
        cases = list(HYBRID)
    elif suite == "blind":
        from . import blind

        cases = list(blind.BLIND)
    else:
        return ordered(limit, suite=suite)
    return cases[:limit] if limit else cases


def reply_language(user: str, reply: str) -> bool:
    """True if a text reply is in the language the user wrote in.

    A sentence with one Turkish function word or suffix-bearing word is
    Turkish for this purpose (``detect_lang`` lets Turkish win ties), so
    "kanka yarın kadıköye lets go" expects a Turkish reply.
    """
    want = detect_lang(user)
    words = set(_TOKEN.findall(reply.replace("İ", "i").replace("I", "ı").lower()))
    turkish = len(words & _TR_WORDS) + (2 if re.search(r"[çğışöü]", reply.lower()) else 0)
    english = len(words & _EN_WORDS)
    got = "en" if english > turkish else "tr"
    return got == want


def judge(reply: str) -> CallCheck:
    """A generated turn -> feast AI's verdict, with native syntax as the norm."""
    kind, name, args = template.parse_reply(reply)
    if kind == "text":
        return CallCheck()
    if kind == "error":
        check = CallCheck(status="parse_error", format="native")
        check.issues.append(name)
        return check
    check = check_reply(render_call(name, args), ALL_TOOL_NAMES, DEFAULT_TAGS)
    check.format = "native"
    return check


# --------------------------------------------------------------------------
# Backends: both take the conversation so far and return the raw reply
# --------------------------------------------------------------------------
class TorchBackend:
    name = "torch"

    def __init__(self, base: Path, adapter: Path | None, prompt: schema.Prompt) -> None:
        from . import model as M

        model, tokenizer = M.load(base, adapter)
        self.runner = M.Runner(model, tokenizer)
        self.tools = prompt.declarations()
        self.static_tokens = self.runner.set_prefix(
            template.render([prompt.system_message()], self.tools, add_generation_prompt=False))

    def reply(self, messages: list[dict]) -> tuple[str, dict]:
        started = time.perf_counter()
        raw, fed, new = self.runner.complete(template.render(messages, self.tools), 160)
        return raw, {"tokens_fed": fed, "tokens_out": new,
                     "seconds": round(time.perf_counter() - started, 2)}


class _Declared:
    """A tool as LiteRT-LM wants one: a schema, and nothing to execute here."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec

    def get_tool_description(self) -> dict:
        return self.spec

    def execute(self, param):  # pragma: no cover - calls are judged, not run
        return {}


class BundleBackend:
    """The ``.litertlm`` on LiteRT-LM's CPU backend, driven as an app would.

    Each turn opens a conversation with the history so far and sends the new
    message; the runtime renders the template, reuses the prefilled static
    prompt from its cache and parses the call itself.
    """

    name = "bundle"

    def __init__(self, bundle: Path, prompt: schema.Prompt, threads: int = 8) -> None:
        import litert_lm

        self.lm = litert_lm
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        tool_base = type("Declared", (_Declared, litert_lm.Tool), {})
        self.tools = [tool_base(d) for d in prompt.declarations()]
        self.engine = litert_lm.Engine(str(bundle),
                                       backend=litert_lm.Backend.CPU(thread_count=threads),
                                       max_num_tokens=4096, enable_benchmark=True)
        self.sampler = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=7)
        prefix = template.render([prompt.system_message()], prompt.declarations(),
                                 add_generation_prompt=False)
        self.static_tokens = len(self.engine.tokenize(prefix[len(template.BOS):])) + 1

    def reply(self, messages: list[dict]) -> tuple[str, dict]:
        started = time.perf_counter()
        conversation = self.engine.create_conversation(
            messages=messages[:-1], tools=self.tools, automatic_tool_calling=False,
            sampler_config=self.sampler, max_output_tokens=160)
        try:
            answer = conversation.send_message(messages[-1])
        except RuntimeError as exc:
            # A malformed call is an exception in LiteRT-LM; here it is an
            # answer to judge (it will count as a parse error).
            from feast_ai.engine.litert_worker import recover_raw

            raw = recover_raw(str(exc))
            if raw is None:
                raise
            return raw, {"seconds": round(time.perf_counter() - started, 2),
                         "tokens_fed": 0, "tokens_out": 0, "runtime_parse_error": True}
        seconds = time.perf_counter() - started
        stats = {"seconds": round(seconds, 2), "tokens_fed": 0, "tokens_out": 0}
        try:
            info = conversation.get_benchmark_info()
            stats.update(tokens_fed=int(info.last_prefill_token_count),
                         tokens_out=int(info.last_decode_token_count),
                         prefill_tps=round(info.last_prefill_tokens_per_second, 1),
                         decode_tps=round(info.last_decode_tokens_per_second, 1),
                         ttft=round(info.time_to_first_token_in_second, 2))
        except RuntimeError:
            pass
        calls = answer.get("tool_calls") or []
        if calls:
            function = calls[0]["function"]
            return template.call_text(function["name"], function.get("arguments") or {}), stats
        parts = answer.get("content") or []
        if isinstance(parts, str):
            return parts, stats
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text"), stats


# --------------------------------------------------------------------------
def run_suite(backend, suite: str, prompt: schema.Prompt, guarded: bool = False,
              limit: int = 0, show: bool = False) -> list[dict]:
    rows: list[dict] = []
    for case in cases_of(suite, limit):
        messages = [prompt.system_message()]
        user_texts: list[str] = []
        assistant_texts: list[str] = []
        for role, content in case.history:
            if role == "user":
                messages.append({"role": "user", "content": schema.user_content(content, EVAL_CTX)})
                user_texts.append(content)
            else:
                messages.append({"role": role, "content": content})
                assistant_texts.append(content)
        for index, turn in enumerate(case.turns):
            messages.append({"role": "user", "content": schema.user_content(turn.text, EVAL_CTX)})
            user_texts.append(turn.text)
            reply, stats = backend.reply(messages)
            raw_check = judge(reply)
            text = template.clean(reply) if not raw_check.is_call else ""
            outcome = native.guard(raw_check, user_texts, assistant_texts, text,
                                   context_line=EVAL_CTX.line())
            check = outcome.check if guarded else raw_check
            shown = (outcome.text or text) if guarded else text
            if guarded and outcome.text:
                check = CallCheck()
            result = score(turn, check)
            row = {
                "suite": suite, "case": case.id, "turn": index, "user": turn.text,
                "expected": turn.tool or "(text)", "reply": template.clean(reply),
                "final": (render_call(check.name, check.args) if check.is_call else shown),
                "status": check.status, "called": check.name, "args": check.args,
                "issues": check.issues, "exact": result.exact, "tool_ok": result.tool_ok,
                "forbidden": result.forbidden, "skipped_tool": result.skipped_tool,
                "needless_call": result.needless_call, "missing": result.missing_args,
                "hallucinated": check.hallucinated,
                "ungrounded": outcome.ungrounded, "guard_notes": outcome.notes,
                "echoed_ctx": "[ctx]" in shown,
                "language_ok": reply_language(turn.text, shown) if shown else None,
                **stats,
            }
            rows.append(row)
            if show:
                mark = "ok " if result.exact else "BAD"
                note = f"   guard: {'; '.join(outcome.notes)}" if outcome.notes else ""
                print(f"  {mark} [{case.id}] {turn.text}\n        -> {row['final'][:150]}"
                      + (f"   (missing {result.missing_args})" if result.missing_args else "")
                      + note, flush=True)
            if check.is_call:
                messages.append(template.assistant_call(check.name, check.args))
                assistant_texts.append(render_call(check.name, check.args))
            else:
                messages.append({"role": "assistant", "content": shown})
                assistant_texts.append(shown)
    return rows


def run_dev(backend, prompt: schema.Prompt, examples: list[dict], limit: int = 0,
            seed: int = 7, show: bool = False) -> list[dict]:
    """Every supervised dev turn, with the gold history before it.

    The suites say how a model does on unseen requests; this says whether an
    artefact still *does what it was trained to do* - which is the question
    when choosing between quantisation recipes, and one that must not be
    answered on the suites.  A call counts if its name and arguments equal
    the gold call exactly; a text target counts if the answer is text.

    A text target that answers a tool result is also checked for what it
    states (``faithful``): every number and name in it must be in the
    result, the call or the user's message
    (:func:`feast_ai.agent.replies.unsupported`).  That is the model's own
    hallucination rate on the second hop, before the app's check replaces
    anything.
    """
    import random

    turns = [(example, index) for example in examples
             for index, message in enumerate(example["messages"])
             if message["role"] == "assistant" and message.get("supervise")]
    random.Random(seed).shuffle(turns)
    rows: list[dict] = []
    for example, index in turns[:limit] if limit else turns:
        ctx = schema.Ctx(**example["ctx"])
        messages = [prompt.system_message()]
        for message in example["messages"][:index]:
            if message["role"] == "user":
                messages.append({"role": "user",
                                 "content": schema.user_content(message["content"], ctx)})
            else:
                messages.append({k: v for k, v in message.items() if k != "supervise"})
        gold = example["messages"][index]
        before = example["messages"][index - 1]
        user = [m["content"] for m in example["messages"][:index] if m["role"] == "user"][-1]
        reply, stats = backend.reply(messages)
        kind, name, args = template.parse_reply(reply)
        faithful = None
        if before["role"] == "tool":
            import json

            from feast_ai.agent import replies

            call = example["messages"][index - 2]["tool_calls"][0]["function"]
            sources = [json.dumps(before["content"][0]["response"], ensure_ascii=False),
                       json.dumps(call["arguments"], ensure_ascii=False), user, ctx.line()]
            faithful = kind == "text" and bool(name.strip()) and not replies.unsupported(
                name, sources)
        if gold.get("tool_calls"):
            function = gold["tool_calls"][0]["function"]
            expected = template.call_text(function["name"], function["arguments"])
            exact = kind == "call" and name == function["name"] and args == function["arguments"]
            tool_ok = kind == "call" and name == function["name"]
        else:
            expected, exact = "(text)", kind == "text"
            tool_ok = exact
        rows.append({"suite": "dev", "case": example["id"], "turn": index,
                     "user": user, "faithful": faithful,
                     "same_as_gold": faithful is not None and kind == "text"
                     and name.strip() == str(gold.get("content", "")).strip(),
                     "register": example["register"], "family": example["family"],
                     "expected": expected, "reply": template.clean(reply), "exact": exact,
                     "tool_ok": tool_ok, "parse_error": kind == "error",
                     "doubled_quote": bool(template.DOUBLED_QUOTE.search(reply)), **stats})
        if show and not exact:
            print(f"  BAD [{example['register']}] {rows[-1]['user']}\n      want {expected}\n"
                  f"      got  {rows[-1]['reply'][:160]}", flush=True)
    return rows


def summarise_dev(rows: list[dict]) -> dict:
    n = len(rows) or 1
    by_register: dict = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        by_register[row["register"]][0] += row["exact"]
        by_register[row["register"]][1] += 1
    timed = sorted(r["seconds"] for r in rows)
    answers = [r for r in rows if r.get("faithful") is not None]
    calls = [r for r in rows if r["expected"] != "(text)"]
    return {"turns": len(rows), "exact": sum(r["exact"] for r in rows),
            "calls": len(calls),
            "calls_exact_pct": round(100 * sum(r["exact"] for r in calls) / (len(calls) or 1), 1),
            "result_answers": len(answers),
            "faithful_pct": round(100 * sum(bool(r["faithful"]) for r in answers)
                                  / (len(answers) or 1), 1),
            "unfaithful": [{"user": r["user"], "reply": r["reply"]} for r in answers
                           if not r["faithful"]][:12],
            "exact_pct": round(100 * sum(r["exact"] for r in rows) / n, 1),
            "right_tool": sum(r["tool_ok"] for r in rows),
            "parse_errors": sum(r["parse_error"] for r in rows),
            "doubled_quotes": sum(r["doubled_quote"] for r in rows),
            "runtime_parse_errors": sum(bool(r.get("runtime_parse_error")) for r in rows),
            "by_register": {k: round(100 * a / b, 1) for k, (a, b) in sorted(by_register.items())},
            "reply_median_s": timed[len(timed) // 2] if timed else 0,
            "decode_tps": round(sum(r.get("decode_tps", 0) for r in rows) / n, 1)}


def summarise(rows: list[dict]) -> dict:
    def one(part: list[dict]) -> dict:
        n = len(part) or 1
        texts = [r for r in part if r["language_ok"] is not None]
        timed = [r for r in part if r.get("ttft") is not None]
        out = {
            "turns": len(part),
            "exact": sum(r["exact"] for r in part),
            "exact_pct": round(100 * sum(r["exact"] for r in part) / n, 1),
            "right_tool": sum(r["tool_ok"] for r in part),
            "hallucinated": sum(r["hallucinated"] for r in part),
            "ungrounded": sum(bool(r["ungrounded"]) for r in part),
            "unsafe_sends": sum(r["forbidden"] for r in part),
            "text_where_tool": sum(r["skipped_tool"] and r["expected"] != "(text)"
                                   for r in part),
            "tool_where_text": sum(r["needless_call"] for r in part),
            "wrong_language": sum(not r["language_ok"] for r in texts),
            "echoed_ctx": sum(r["echoed_ctx"] for r in part),
            "guard_changed": sum(bool(r["guard_notes"]) for r in part),
            "runtime_parse_err": sum(bool(r.get("runtime_parse_error")) for r in part),
            "tokens_out_avg": round(sum(r["tokens_out"] for r in part) / n, 1),
        }
        if timed:
            ttfts = sorted(r["ttft"] for r in timed)
            totals = sorted(r["seconds"] for r in timed)
            out["ttft_median_s"] = ttfts[len(ttfts) // 2]
            out["reply_median_s"] = totals[len(totals) // 2]
            out["decode_tps"] = round(sum(r["decode_tps"] for r in timed) / len(timed), 1)
        return out
    by_suite = collections.defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)
    out = {suite: one(part) for suite, part in by_suite.items()}
    out["all"] = one(rows)
    return out


def render_summary(summary: dict, title: str) -> str:
    keys = [k for k in summary["all"]]
    suites = list(summary)
    lines = [title, f"{'':<18}" + "".join(f"{s:>10}" for s in suites)]
    for key in keys:
        lines.append(f"{key:<18}" + "".join(f"{summary[s].get(key, ''):>10}" for s in suites))
    return "\n".join(lines)


def prompt_of(adapter: Path | None, bundle: Path | None, name: str) -> schema.Prompt:
    """The prompt a model was trained with, unless one is named."""
    if name:
        return schema.prompt(name)
    for folder in (adapter, bundle.parent if bundle else None):
        if folder and (folder / "prompt.json").exists():
            return schema.prompt(json.loads((folder / "prompt.json").read_text())["prompt"])
    return schema.prompt("v0" if adapter and "v0" in str(adapter) else "")


def main(args) -> int:
    paths.ensure_dirs()
    adapter = Path(args.adapter) if args.adapter else None
    bundle = Path(args.bundle) if args.bundle else None
    prompt = prompt_of(adapter, bundle, args.prompt)
    if bundle:
        backend = BundleBackend(bundle, prompt, args.threads)
        default_name = bundle.stem
    else:
        base = Path(args.base) if args.base else paths.BUILD_DIR / "base-text"
        backend = TorchBackend(base, adapter, prompt)
        default_name = adapter.parent.name if adapter else "base"
    name = args.name or default_name + ("-guard" if args.guard else "")

    rows: list[dict] = []
    started = time.perf_counter()
    if args.suites.strip() == "dev":
        from . import dataset

        data = Path(args.data) if args.data else paths.DATA_DIR / "v2"
        rows = run_dev(backend, prompt, dataset.read(data / "dev.jsonl"), args.limit,
                       show=args.show)
        summary = summarise_dev(rows)
        report = {"name": name, "backend": backend.name, "prompt": prompt.name, "data": str(data),
                  "bundle": str(bundle) if bundle else None,
                  "adapter": str(adapter) if adapter else None,
                  "seconds": round(time.perf_counter() - started, 1), "summary": summary,
                  "rows": rows}
        out = Path(args.out) if args.out else paths.RUNS_DIR / "eval" / f"{fold(name)}-dev.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name}  dev ({backend.name}, prompt {prompt.name}): "
              + json.dumps(summary, ensure_ascii=False))
        print(f"report: {out}")
        return 0
    suites = [s.strip() for s in args.suites.split(",") if s.strip()] or suite_names()
    for suite in suites:
        if suite not in suite_names():
            print(f"unknown suite {suite!r} (have: {', '.join(suite_names())})")
            return 1
        if args.show:
            print(f"--- {suite} ---")
        rows += run_suite(backend, suite, prompt, args.guard, args.limit, args.show)
    summary = summarise(rows)
    report = {"name": name, "backend": backend.name, "prompt": prompt.name,
              "guard": bool(args.guard), "adapter": str(adapter) if adapter else None,
              "bundle": str(bundle) if bundle else None,
              "static_prompt_tokens": backend.static_tokens, "ctx": EVAL_CTX.__dict__,
              "seconds": round(time.perf_counter() - started, 1),
              "summary": summary, "rows": rows}
    out = Path(args.out) if args.out else paths.RUNS_DIR / "eval" / f"{fold(name)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(render_summary(summary, f"{name}  ({backend.name}, prompt {prompt.name}, "
                                  f"{backend.static_tokens} static tokens"
                                  f"{', guarded' if args.guard else ''}, "
                                  f"{report['seconds']} s)"))
    print(f"report: {out}")
    return 0
