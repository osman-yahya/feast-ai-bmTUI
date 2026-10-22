"""Render bench runs as a side-by-side chat comparison page.

The Markdown report is for the thesis tables; this page is for reading the
conversations.  Every user message is shown once, with each pipeline's answer
next to the others - the reply or the tool call, a verdict, the router's
decision and what the turn cost - so a reader can see *why* the numbers
differ, not just that they do.  Strip plots put every message's time to first
token and reply time on one axis per pipeline, so the spread is visible, not
just the average.

Everything is rendered here, statically: the page is complete at rest and
needs no network beyond fonts.  The only script switches between runs and
applies the "disagree / misses" filters.  ``standalone=True`` wraps it in a
document skeleton for opening from disk; without it the output is the bare
page an Artifact publish wraps itself.

Pipeline colours are the first slots of a categorical palette validated for
colour-vision deficiency in both themes; every coloured mark also carries a
text label, so identity never rests on colour alone.
"""

from __future__ import annotations

import html
import json
import statistics
import time

from .calls import HALLUCINATIONS, extract
from .suite import SUITES

#: id -> (label, what it does, colour class). Order is the display order.
MODE_INFO = {
    "direct": ("Direct", "the model sees all 15 tools, every message", "c-direct"),
    "hierarchical": ("Routed + shortcuts", "router first; it may answer alone", "c-hier"),
    "routed": ("Routed prompt", "router picks the system prompt, the model always answers",
               "c-routed"),
    "contexts": ("Routed context", "routed prompt restored from a saved KV snapshot",
                 "c-contexts"),
    "active": ("Active prompting", "picks a tool by name, then reads its manual", "c-hier"),
    "active2": ("Active, constrained", "grammar-constrained pick, manual, prefilled call",
                "c-contexts"),
    "classes": ("Classes", "router picks one of 13 class prompts with tool manuals",
                "c-routed"),
    "classes@decision-2.0-kai-0.6b": ("Classes (Decision 2.0)",
                                      "Decision 2.0 picks the class prompt", "c-contexts"),
}

ROUTER_NAMES = {
    "modernbert-tr-embed": "ModernBERT-TR router",
    "decision-2.0-kai-0.6b": "Decision 2.0 Kai-0.6B router",
}

#: (key, label, unit, better, only_if_nonzero)
METRICS = (
    ("exact_pct", "Exact answers", "pct", +1, False),
    ("hallucinated", "Hallucinated tool calls", "int", -1, False),
    ("unsafe_sends", "Sent a message without confirmation", "int", -1, False),
    ("skipped_tool", "Replied in text where a tool was needed", "int", -1, False),
    ("router_turns", "Answered without the SLM", "int", +1, True),
    ("_ttft_med", "Time to first token, median", "s", -1, False),
    ("ttft_avg", "Time to first token, average", "s", -1, False),
    ("latency_avg", "Average reply time", "s", -1, False),
    ("latency_p95", "Slowest 5% of replies", "s", -1, False),
    ("decode_tps_avg", "Decode speed", "tps", +1, False),
    ("prefilled_tokens_avg", "Prompt tokens processed per message", "int", -1, False),
    ("router_ms_avg", "Intent routing per message", "ms", -1, True),
    ("restore_ms_avg", "Snapshot restore per message", "ms", -1, True),
    ("peak_rss", "Peak RAM, model + router", "mb", -1, False),
    ("paged_bytes", "Weights re-read under the RAM ceiling", "mb", -1, True),
)

FAILURE_LABELS = {
    "parse_error": "Broken call",
    "unknown_tool": "Invented tool",
    "unoffered_tool": "Tool not offered",
    "invalid_args": "Invalid arguments",
}

ROUTE_KINDS = {
    "direct": "answered by the router",
    "slm": "{n} tool{s}",
    "merged": "near-tie, {n} tools",
    "inherit": "follow-up, {n} tool{s} kept",
    "fallback": "unsure, all {n} tools",
}


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float, unit: str) -> str:
    if unit == "pct":
        return f"{value:.0f}%"
    if unit == "s":
        return f"{value:.2f} s"
    if unit == "ms":
        return f"{value:.0f} ms"
    if unit == "tps":
        return f"{value:.1f} tok/s"
    if unit == "mb":
        return f"{value / 2**20:,.0f} MB"
    return f"{value:,.0f}"


def _info(mode: str) -> tuple[str, str, str]:
    return MODE_INFO.get(mode, (mode, "", "c-direct"))


def _verdict(row: dict) -> tuple[str, str, str]:
    """``(label, tone, detail)`` for one answer."""
    detail = "; ".join(row.get("issues", []) + row.get("missing_args", []))
    if row.get("error"):
        return "Error", "bad", row["error"][:160]
    if row.get("forbidden"):
        return "Sent without confirmation", "bad", "the user had not confirmed yet"
    if row.get("exact"):
        return "Exact", "good", "; ".join(row.get("warnings", []))
    if row.get("status") in HALLUCINATIONS:
        return FAILURE_LABELS[row["status"]], "bad", detail
    if row.get("skipped_tool"):
        return "No tool called", "warn", "a tool was needed here"
    if row.get("needless_call"):
        return "Tool not needed", "warn", "a plain reply was right"
    if row.get("tool_ok"):
        return "Wrong arguments", "warn", detail
    return "Wrong tool", "bad", detail


def _expected(turn) -> str:
    if turn is None:
        return ""
    if not turn.tool:
        text = "a plain reply, no tool"
        if turn.forbid:
            text += " (ask before sending)"
        return _e(text)
    tools = " or ".join(f"<code>{_e(t)}</code>" for t in turn.tool.split("|"))
    if turn.args:
        args = ", ".join(
            f"{_e(k.replace('|', ' or '))} ≈ "
            f"{_e(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}"
            for k, v in turn.args.items()
        )
        return f"{tools} · {args}"
    return tools


def _call_html(name: str, args: dict) -> str:
    parts = ", ".join(
        f'<span class="arg">{_e(k)}</span>: <span class="val">{_e(json.dumps(v, ensure_ascii=False))}</span>'
        for k, v in args.items()
    )
    return f'<div class="call"><span class="fn">{_e(name)}</span>({parts})</div>'


def _lane(row: dict | None, mode: str) -> str:
    label, _, colour = _info(mode)
    name = f'<span class="lane-name"><i class="dot {colour}"></i>{_e(label)}</span>'
    if row is None:
        return f'<div class="lane {colour}"><p class="lane-label">{name}</p><p class="muted">not run</p></div>'
    tone_label, tone, detail = _verdict(row)
    route = ""
    if row.get("route_kind"):
        n = row.get("tools_offered", 0)
        kind = ROUTE_KINDS.get(row["route_kind"], row["route_kind"]).format(n=n, s="" if n == 1 else "s")
        route = (f'<span class="route"><b>{_e(row["route_intent"])}</b> '
                 f'{row["route_score"]:.2f} → {_e(kind)}</span>')
    head = f'<p class="lane-label">{name}{route}</p>'

    args = row.get("call_args")
    if args is None and row.get("call"):
        args = extract(row.get("reply", "")).args
    if row.get("served_by") == "router":
        body = (_call_html(row["call"], args or {}) if row.get("call")
                else f'<div class="bubble bot">{_e(row.get("reply", ""))}</div>')
        body = f'<div class="by-router">{body}<p class="by-router-note">no SLM involved</p></div>'
    elif row.get("call") and row.get("status") != "parse_error":
        body = _call_html(row["call"], args or {})
    else:
        body = f'<div class="bubble bot">{_e(row.get("reply", "") or "(empty reply)")}</div>'

    cost = []
    if row.get("served_by") != "router" and row.get("ttft_s"):
        cost.append(f'first token {row["ttft_s"]:.2f} s')
    cost.append(f'reply {row.get("latency_s", 0):.2f} s')
    if row.get("served_by") != "router" and row.get("prompt_tokens"):
        cost.append(f'{row.get("prefilled_tokens", 0)} of {row["prompt_tokens"]} tok processed')
    if row.get("restore_ms"):
        cost.append(f'snapshot {row["restore_ms"]:.0f} ms')
    stats = "".join(f'<span class="stat">{_e(c)}</span>' for c in cost)
    why = f'<p class="why">{_e(detail)}</p>' if detail and tone != "good" else ""
    return (f'<div class="lane {colour}">{head}{body}'
            f'<div class="lane-foot"><span class="pill {tone}">{_e(tone_label)}</span>{stats}</div>'
            f'{why}</div>')


def _conversation(case, rows: dict, history: list, modes: list[str]) -> tuple[str, bool, bool]:
    turns_html = []
    differs = miss = False
    for index, turn in enumerate(case.turns, 1):
        lane_rows = [rows.get((m, index)) for m in modes]
        verdicts = {_verdict(r)[0] for r in lane_rows if r}
        differs |= len(verdicts) > 1
        miss |= any(r is not None and not r.get("exact") for r in lane_rows)
        lanes = "".join(_lane(r, m) for r, m in zip(lane_rows, modes))
        turns_html.append(
            '<div class="turn">'
            f'<div class="user-row"><div class="bubble user">{_e(turn.text)}</div></div>'
            f'<p class="expect">Expected: {_expected(turn)}</p>'
            f'<div class="lanes">{lanes}</div>'
            '</div>'
        )
    earlier = ""
    if history:
        lines = "".join(
            f'<div class="bubble {"user" if role == "user" else "bot"} past">{_e(text)}</div>'
            for role, text in history
        )
        earlier = f'<div class="earlier"><p class="earlier-label">Earlier in this chat</p>{lines}</div>'
    n = len(case.turns)
    body = (f'<article class="convo" data-differs="{int(differs)}" data-miss="{int(miss)}">'
            f'<header class="convo-head"><h3>{_e(case.title or case.id)}</h3>'
            f'<span class="convo-meta">{n} message{"s" if n > 1 else ""}</span></header>'
            f'{earlier}{"".join(turns_html)}</article>')
    return body, differs, miss


# --------------------------------------------------------------------------
# Scorecard and charts
# --------------------------------------------------------------------------
def _derived(results: dict, rows: list[dict]) -> dict:
    """Add the medians the bench summary does not carry."""
    out = {m: dict(v) for m, v in results.items()}
    for mode in out:
        ttfts = [r["ttft_s"] for r in rows if r["mode"] == mode and r.get("ttft_s")]
        out[mode]["_ttft_med"] = statistics.median(ttfts) if ttfts else 0.0
    return out


def _scorecard(results: dict, modes: list[str], model_bytes: int) -> str:
    head = "".join(f'<th scope="col"><i class="dot {_info(m)[2]}"></i>{_e(_info(m)[0])}</th>'
                   for m in modes)
    body = []
    for key, label, unit, better, only_nonzero in METRICS:
        values = [float(results.get(m, {}).get(key, 0) or 0) for m in modes]
        if only_nonzero and not any(values):
            continue
        if key == "paged_bytes" and max(values) <= max(64 * 2**20, model_bytes):
            continue  # mmap reading the weights in once is not ceiling pressure
        present = [v for v in values if v or not only_nonzero]
        top = max(values) or 1.0
        cells = []
        for mode, value in zip(modes, values):
            shown = "—" if (only_nonzero and not value) else _fmt(value, unit)
            bar = "" if shown == "—" else (
                f'<span class="bar {_info(mode)[2]}" style="width:{100 * value / top:.1f}%"></span>')
            cells.append(f'<td><span class="num">{shown}</span>{bar}</td>')
        best = '<span class="even">—</span>'
        ranked = sorted(present, reverse=better > 0)
        close = len(ranked) > 1 and abs(ranked[0] - ranked[1]) <= 0.02 * max(abs(ranked[0]), 1e-9)
        if close and max(present) != min(present):
            best = '<span class="even">close</span>'
        elif len(present) > 1 and max(present) != min(present):
            pick = max(present) if better > 0 else min(present)
            winners = [m for m, v in zip(modes, values) if v == pick and (v or not only_nonzero)]
            if len(winners) == 1:
                best = (f'<span class="win"><i class="dot {_info(winners[0])[2]}"></i>'
                        f'{_e(_info(winners[0])[0])}</span>')
        elif len(present) > 1:
            best = '<span class="even">same</span>'
        body.append(f'<tr><th scope="row">{_e(label)}</th>{"".join(cells)}<td>{best}</td></tr>')
    return ('<div class="table-wrap"><table class="score"><thead><tr><th scope="col">Per run</th>'
            f'{head}<th scope="col">Best</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def _ticks(top: float) -> list[float]:
    for step in (0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50):
        if top / step <= 6:
            break
    count = int(top / step) + 1
    return [round(i * step, 2) for i in range(count + 1)]


def _strip(rows: list[dict], modes: list[str], key: str, title: str, caption: str,
           titles: dict) -> str:
    """One row per pipeline, one dot per message, a median tick per row."""
    data = {m: [r for r in rows if r["mode"] == m and r.get(key) and r.get("served_by") != "router"]
            for m in modes}
    peak = max((r[key] for v in data.values() for r in v), default=1.0)
    ticks = _ticks(peak)
    hi = ticks[-1]
    left, right, top_pad, row_h = 150, 24, 18, 54
    width = 720
    plot = width - left - right
    height = top_pad + row_h * len(modes) + 34

    def x(v: float) -> float:
        return left + plot * v / hi

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{_e(title)}">']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top_pad - 6}" '
                     f'y2="{height - 28}"/>')
        parts.append(f'<text class="tick" x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">'
                     f'{t:g} s</text>')
    for i, mode in enumerate(modes):
        label, _, colour = _info(mode)
        cy = top_pad + row_h * i + row_h / 2
        values = [r[key] for r in data[mode]]
        parts.append(f'<text class="rowlabel" x="{left - 14}" y="{cy + 4:.1f}" text-anchor="end">'
                     f'{_e(label)}</text>')
        for j, r in enumerate(data[mode]):
            jitter = ((j * 7) % 5 - 2) * 4.0  # deterministic, keeps overlaps visible
            tip = (f"{label} · {titles.get(r['case'], r['case'])} · "
                   f"{r['text'][:48]} · {r[key]:.2f} s")
            parts.append(f'<circle class="pt {colour}" cx="{x(r[key]):.1f}" cy="{cy + jitter:.1f}" r="4.5">'
                         f'<title>{_e(tip)}</title></circle>')
        if values:
            med = statistics.median(values)
            parts.append(f'<line class="median" x1="{x(med):.1f}" x2="{x(med):.1f}" '
                         f'y1="{cy - 17:.1f}" y2="{cy + 17:.1f}"/>')
            anchor = "end" if x(med) > width - 120 else "start"
            dx = -6 if anchor == "end" else 6
            parts.append(f'<text class="medlabel" x="{x(med) + dx:.1f}" y="{cy - 20:.1f}" '
                         f'text-anchor="{anchor}">median {med:.2f} s</text>')
    parts.append("</svg>")
    return (f'<figure class="fig"><figcaption><b>{_e(title)}</b> {_e(caption)}</figcaption>'
            f'<div class="chart-wrap">{"".join(parts)}</div></figure>')


def _bars(results: dict, modes: list[str], key: str, title: str, caption: str, unit: str) -> str:
    values = [float(results.get(m, {}).get(key, 0) or 0) for m in modes]
    hi = max(values) or 1.0
    left, right, row_h, top_pad = 150, 110, 38, 10
    width = 720
    plot = width - left - right
    height = top_pad + row_h * len(modes) + 6
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">']
    for i, (mode, value) in enumerate(zip(modes, values)):
        label, _, colour = _info(mode)
        y = top_pad + row_h * i
        w = max(2.0, plot * value / hi)
        parts.append(f'<text class="rowlabel" x="{left - 14}" y="{y + 19}" text-anchor="end">{_e(label)}</text>')
        parts.append(f'<rect class="barfill {colour}" x="{left}" y="{y + 7}" width="{w:.1f}" height="18" rx="4">'
                     f'<title>{_e(label)} · {_fmt(value, unit)}</title></rect>')
        parts.append(f'<text class="value" x="{left + w + 8:.1f}" y="{y + 20}">{_fmt(value, unit)}</text>')
    parts.append("</svg>")
    return (f'<figure class="fig"><figcaption><b>{_e(title)}</b> {_e(caption)}</figcaption>'
            f'<div class="chart-wrap">{"".join(parts)}</div></figure>')


def _charts(results: dict, rows: list[dict], modes: list[str], titles: dict) -> str:
    return (
        '<section class="charts">'
        + _strip(rows, modes, "ttft_s", "Time to first token",
                 "per message the model answered, routing and snapshot restore included. "
                 "Each dot is one message; hover for which.", titles)
        + _strip(rows, modes, "latency_s", "Full reply time",
                 "from sending the message to the last token.", titles)
        + _bars(results, modes, "decode_tps_avg", "Decode speed",
                "tokens per second once the reply has started, averaged over messages.", "tps")
        + '</section>'
    )


# --------------------------------------------------------------------------
def _ceiling(meta: dict) -> str:
    limit = meta.get("ram_limit_bytes", 0)
    return f"RAM ceiling {limit / 2**20:,.0f} MB" if limit else "no RAM ceiling"


def _run_section(payload: dict, index: int) -> tuple[str, str]:
    meta, rows = payload["meta"], payload["rows"]
    modes = [m for m in MODE_INFO if m in payload["results"]] + \
            [m for m in payload["results"] if m not in MODE_INFO]
    results = _derived(payload["results"], rows)
    suite_name = meta.get("suite", "standard")
    titles = meta.get("case_titles", {})
    by_case: dict[str, dict] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[(row["mode"], row["turn"])] = row
    histories = meta.get("case_history", {})
    convos, n_diff, n_miss = [], 0, 0
    for case in SUITES.get(suite_name, ()):
        if case.id not in by_case:
            continue
        body, differs, miss = _conversation(case, by_case[case.id], histories.get(case.id, []), modes)
        convos.append(body)
        n_diff += differs
        n_miss += miss
    when = time.strftime("%d %b %Y, %H:%M", time.localtime(meta.get("started_at", 0)))
    cache = "warm prompt cache" if meta.get("prompt_cache", True) else "cold prefill"
    if meta.get("cache_ram_mb"):
        cache += f", {meta['cache_ram_mb']} MB of saved prompts"
    facts = [
        f"{meta.get('model_name', meta.get('model_id'))} ({meta.get('quant', '?')})",
        f"{meta.get('threads')} cores"
        + (f" / {meta['nominal_ram_gb']:g} GB" if meta.get("nominal_ram_gb") else ""),
        _ceiling(meta),
        cache,
        ROUTER_NAMES.get(meta.get("router_model"), f"router {meta.get('router_model')}")
        + (f" ({meta['router_rss'] / 2**20:.0f} MB)" if meta.get("router_rss") else ""),
        f"{meta.get('turns')} messages in {len(convos)} chats",
        when,
    ]
    ctx = meta.get("contexts")
    snapshot_note = ""
    if ctx:
        facts.insert(4, f"{ctx['snapshots']} KV snapshots, {ctx['bytes'] / 2**20:.0f} MB on storage")
        snapshot_note = (
            f'<p class="note">Routed context restores one of {ctx["snapshots"]} precomputed '
            f'snapshots ({ctx["avg_bytes"] / 2**20:.1f} MB each, '
            f'{ctx["bytes"] / 2**20:.0f} MB in total) before each message, so only the clock and '
            'location block, the chat history and the new message are processed. They are built '
            'once, before the measured pass, as a phone would at install time. All pipelines in '
            'this run share one server with a full sliding-window cache (llama.cpp --swa-full), '
            'which restored snapshots need.</p>'
        )
    label = (ROUTER_NAMES.get(meta.get("router_model"), meta.get("router_model", "router"))
             + f" · {meta.get('threads')} cores"
             + (f" / {meta['nominal_ram_gb']:g} GB" if meta.get("nominal_ram_gb") else ""))
    heads = "".join(f'<span><i class="dot {_info(m)[2]}"></i>{_e(_info(m)[0])}</span>' for m in modes)
    section = (
        f'<section class="run" style="--n:{len(modes)}" id="run-{index}" data-label="{_e(label)}"'
        f'{"" if index == 0 else " hidden"}>'
        f'<p class="facts">{" · ".join(_e(f) for f in facts)}</p>'
        f'{_scorecard(results, modes, meta.get("model_bytes", 0))}'
        f'<p class="note">{meta.get("turns", 0)} messages per pipeline, so one message moves '
        f'"Exact" by {100 / max(1, meta.get("turns", 1)):.1f} points. A gap of a few messages is '
        'within run-to-run noise: repeat the run, or use the 60-turn standard suite, before '
        'calling a winner on accuracy.</p>'
        f'{snapshot_note}'
        f'{_charts(results, rows, modes, titles)}'
        '<div class="filters" role="group" aria-label="Show">'
        f'<button type="button" class="chip" data-filter="all" aria-pressed="true">All chats <b>{len(convos)}</b></button>'
        f'<button type="button" class="chip" data-filter="differs" aria-pressed="false">Pipelines disagree <b>{n_diff}</b></button>'
        f'<button type="button" class="chip" data-filter="miss" aria-pressed="false">Any miss <b>{n_miss}</b></button>'
        '</div>'
        f'<div class="lane-heads" aria-hidden="true">{heads}</div>'
        f'<div class="convos">{"".join(convos)}</div>'
        '<p class="empty" hidden>No chats match this filter.</p>'
        '</section>'
    )
    return section, label


STYLE = """
/* Layout: summary table, then the charts, then one column of chats; each user
   message spans the width and every pipeline answers in its own lane below. */
:root {
  --paper: #f3f5f2; --surface: #fcfcfb; --ink: #17201c; --muted: #5a6862; --rule: #d9e0db;
  --c-direct: #eb6834; --c-routed: #2a78d6; --c-contexts: #1baf7a; --c-hier: #6250d6;
  --s-direct: #fcefe8; --s-routed: #e8f0fb; --s-contexts: #e4f5ee; --s-hier: #eeebfb;
  --user: #233456; --user-ink: #f5f7fb;
  --good: #1d7a3b; --good-soft: #e3f4e7; --warn: #8f5d00; --warn-soft: #fbf0d9;
  --bad: #b3261e; --bad-soft: #fbe5e3;
  --display: "Schibsted Grotesk", "Segoe UI", system-ui, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #101513; --surface: #1a1a19; --ink: #e9eeeb; --muted: #a2afa8; --rule: #2c3631;
  --c-direct: #d95926; --c-routed: #3987e5; --c-contexts: #199e70; --c-hier: #9085e9;
  --s-direct: #2a1a12; --s-routed: #13202f; --s-contexts: #10261e; --s-hier: #1e1a33;
  --user: #2f4370; --user-ink: #f2f5fb;
  --good: #63d184; --good-soft: #13281a; --warn: #e9b44c; --warn-soft: #2c2310;
  --bad: #ff8478; --bad-soft: #32161a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #101513; --surface: #1a1a19; --ink: #e9eeeb; --muted: #a2afa8; --rule: #2c3631;
  --c-direct: #d95926; --c-routed: #3987e5; --c-contexts: #199e70; --c-hier: #9085e9;
  --s-direct: #2a1a12; --s-routed: #13202f; --s-contexts: #10261e; --s-hier: #1e1a33;
  --user: #2f4370; --user-ink: #f2f5fb;
  --good: #63d184; --good-soft: #13281a; --warn: #e9b44c; --warn-soft: #2c2310;
  --bad: #ff8478; --bad-soft: #32161a; color-scheme: dark; }
body { background: var(--paper); color: var(--ink); font: 15px/1.55 var(--body); }
.page { max-width: 1240px; margin: 0 auto; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }
.masthead { display: grid; gap: 12px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0; }
h1 { font: 800 clamp(34px, 6vw, 56px)/1.02 var(--display); margin: 0; letter-spacing: -.02em; text-wrap: balance; }
.lede { max-width: 70ch; margin: 0; color: var(--muted); font-size: 16px; }
.pipes { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
.pipes li { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.pipes b { font: 700 15px/1.3 var(--display); }
.pipes span { color: var(--muted); font-size: 14px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 7px; background: var(--c-direct); flex: none; }
.dot.c-direct { background: var(--c-direct); } .dot.c-routed { background: var(--c-routed); }
.dot.c-contexts { background: var(--c-contexts); } .dot.c-hier { background: var(--c-hier); }
.runs { display: flex; flex-wrap: wrap; gap: 8px; }
.runs button, .chip { font: 500 13px/1 var(--body); color: var(--ink); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 999px; padding: 9px 14px; cursor: pointer; }
.runs button[aria-pressed="true"], .chip[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.chip b { font-family: var(--mono); font-weight: 500; margin-left: 4px; }
button:focus-visible { outline: 2px solid var(--c-routed); outline-offset: 2px; }
.run { display: grid; gap: 22px; }
.facts { margin: 0; font: 13px/1.5 var(--mono); color: var(--muted); }
.note { margin: 0; max-width: 78ch; font-size: 13px; color: var(--muted); }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; }
table.score { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 620px; }
.score th, .score td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: middle; }
.score tbody tr:last-child th, .score tbody tr:last-child td { border-bottom: 0; }
.score thead th { font: 500 11px/1.2 var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--muted); white-space: nowrap; }
.score tbody th { font-weight: 500; width: 30%; }
.num { display: block; font: 500 15px/1.2 var(--mono); }
.bar { display: block; height: 4px; border-radius: 2px; margin-top: 6px; min-width: 2px; background: var(--c-direct); }
.bar.c-direct { background: var(--c-direct); } .bar.c-routed { background: var(--c-routed); }
.bar.c-contexts { background: var(--c-contexts); } .bar.c-hier { background: var(--c-hier); }
.win { font: 600 13px/1 var(--body); white-space: nowrap; }
.even { color: var(--muted); font-size: 13px; }
.charts { display: grid; gap: 18px; }
.fig { margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; display: grid; gap: 8px; }
.fig figcaption { font-size: 13.5px; color: var(--muted); max-width: 80ch; }
.fig figcaption b { color: var(--ink); font: 700 15px/1.3 var(--display); margin-right: 4px; }
.chart-wrap { overflow-x: auto; }
.chart { display: block; width: 100%; min-width: 560px; height: auto; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .tick { fill: var(--muted); font: 12px var(--mono); }
.chart .medlabel, .chart .value { fill: var(--ink); font: 12px var(--mono); }
.chart .rowlabel { fill: var(--ink); font: 600 13px var(--body); }
.chart .pt { stroke: var(--surface); stroke-width: 2; fill: var(--c-direct); fill-opacity: .9; }
.chart .pt.c-direct, .chart .barfill.c-direct { fill: var(--c-direct); }
.chart .pt.c-routed, .chart .barfill.c-routed { fill: var(--c-routed); }
.chart .pt.c-contexts, .chart .barfill.c-contexts { fill: var(--c-contexts); }
.chart .pt.c-hier, .chart .barfill.c-hier { fill: var(--c-hier); }
.chart .pt:hover { stroke: var(--ink); }
.chart .median { stroke: var(--ink); stroke-width: 2; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; }
.lane-heads { display: grid; grid-template-columns: repeat(var(--n, 2), minmax(0, 1fr)); gap: 14px; font: 700 15px/1.2 var(--display);
  position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: var(--paper); padding-block: 10px; border-bottom: 1px solid var(--rule); }
.convos { display: grid; gap: 20px; }
.convo { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 18px; display: grid; gap: 18px; }
.convo-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.convo-head h3 { margin: 0; font: 700 18px/1.25 var(--display); text-wrap: balance; }
.convo-meta { font: 12px/1 var(--mono); color: var(--muted); }
.earlier { display: grid; gap: 6px; padding: 12px; border-radius: 10px; border: 1px dashed var(--rule); }
.earlier-label { margin: 0; font: 500 11px/1 var(--mono); text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }
.turn { display: grid; gap: 8px; }
.turn + .turn { border-top: 1px solid var(--rule); padding-top: 16px; }
.user-row { display: flex; justify-content: flex-end; }
.bubble { padding: 9px 13px; border-radius: 16px; max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; min-width: 0; }
.bubble.user { background: var(--user); color: var(--user-ink); border-bottom-right-radius: 4px; max-width: min(560px, 92%); font-size: 15.5px; }
.bubble.bot { background: var(--paper); border: 1px solid var(--rule); border-bottom-left-radius: 4px; }
.bubble.past { font-size: 13px; opacity: .8; justify-self: start; }
.bubble.user.past { justify-self: end; }
.expect { margin: 0; text-align: right; font-size: 12.5px; color: var(--muted); }
.expect code { font: 12px var(--mono); }
.lanes { display: grid; grid-template-columns: repeat(var(--n, 2), minmax(0, 1fr)); gap: 14px; }
.lane { display: grid; gap: 8px; align-content: start; min-width: 0; padding: 12px; border-radius: 10px; background: var(--s-direct); }
.lane.c-direct { background: var(--s-direct); } .lane.c-routed { background: var(--s-routed); }
.lane.c-contexts { background: var(--s-contexts); } .lane.c-hier { background: var(--s-hier); }
.lane-label { margin: 0; display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; font-size: 12.5px; color: var(--muted); }
.lane-name { font: 700 12px/1 var(--display); text-transform: uppercase; letter-spacing: .05em; color: var(--ink); display: none; }
.route b { color: var(--ink); font-weight: 600; }
.call { font: 13px/1.5 var(--mono); background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 9px 11px; overflow-wrap: anywhere; min-width: 0; }
.call::before { content: "tool call"; display: block; font: 500 10px/1 var(--mono); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
.fn { font-weight: 600; } .arg { color: var(--muted); }
.by-router { display: grid; gap: 4px; }
.by-router-note { margin: 0; font: 500 11px/1 var(--mono); color: var(--muted); letter-spacing: .04em; }
.lane-foot { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.pill { font: 600 12px/1 var(--body); padding: 5px 9px; border-radius: 999px; }
.pill.good { background: var(--good-soft); color: var(--good); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }
.pill.bad { background: var(--bad-soft); color: var(--bad); }
.stat { font: 12px/1.3 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.why { margin: 0; font-size: 12.5px; color: var(--muted); overflow-wrap: anywhere; }
.muted, .empty { color: var(--muted); }
.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px 24px; margin: 0; font-size: 13.5px; color: var(--muted); }
.legend div { display: grid; gap: 4px; } .legend dt { color: var(--ink); font-weight: 600; } .legend dd { margin: 0; }
footer { font: 12px/1.6 var(--mono); color: var(--muted); border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 900px) {
  .lanes { grid-template-columns: 1fr; }
  .lane-heads { display: none; }
  .lane-name { display: inline-flex; align-items: center; }
}
@media (prefers-reduced-motion: no-preference) { .chip, .runs button { transition: background .15s, color .15s; } }
"""

SCRIPT = """
(function () {
  const runs = Array.from(document.querySelectorAll('.run'));
  const picker = document.querySelector('.runs');
  function show(i) {
    runs.forEach((r, j) => { r.hidden = j !== i; });
    if (picker) picker.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
  }
  if (picker) picker.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) show(Number(b.dataset.run));
  });
  runs.forEach((run) => {
    const empty = run.querySelector('.empty');
    run.querySelector('.filters').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      run.querySelectorAll('.filters button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      let shown = 0;
      run.querySelectorAll('.convo').forEach((c) => {
        const f = b.dataset.filter;
        const on = f === 'all' || c.dataset[f] === '1';
        c.hidden = !on; shown += on ? 1 : 0;
      });
      empty.hidden = shown > 0;
    });
  });
})();
"""

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&"
         "family=IBM+Plex+Sans:wght@400;500;600&family=Schibsted+Grotesk:wght@600;700;800&display=swap")


def render(runs: list[dict], standalone: bool = False, title: str = "") -> str:
    """One page for one or more bench payloads (the JSON ``bench`` writes)."""
    sections, labels = [], []
    for index, payload in enumerate(runs):
        section, label = _run_section(payload, index)
        sections.append(section)
        labels.append(label)
    modes_seen = [m for m in MODE_INFO if any(m in r["results"] for r in runs)]
    kanka = all(r["meta"].get("suite") == "kanka" for r in runs)
    if not title:
        title = ("Routed Context Test" if "contexts" in modes_seen
                 else "Kanka Testi" if kanka else "Agent Pipelines")
    picker = ""
    if len(runs) > 1:
        buttons = "".join(
            f'<button type="button" data-run="{i}" aria-pressed="{str(i == 0).lower()}">{_e(lab)}</button>'
            for i, lab in enumerate(labels)
        )
        picker = f'<div class="runs" role="group" aria-label="Model">{buttons}</div>'
    what = ("casual Turkish chats, the way people text a friend"
            if kanka else "labelled Turkish requests")
    pipes = "".join(
        f'<li><b><i class="dot {_info(m)[2]}"></i>{_e(_info(m)[0])}</b><span>{_e(_info(m)[1])}</span></li>'
        for m in modes_seen
    )
    routers = []
    for r in runs:
        name = ROUTER_NAMES.get(r["meta"].get("router_model"), r["meta"].get("router_model", ""))
        if name and name not in routers:
            routers.append(name)
    router_text = " or the ".join(routers) if routers else "intent router"
    legend_router = ('<div><dt>Answered by the router</dt><dd>The encoder handled it alone in about '
                     '25 ms. The SLM never ran.</dd></div>' if "hierarchical" in modes_seen else "")
    page = f"""<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{STYLE}</style>
<div class="page">
  <header class="masthead">
    <p class="eyebrow">feast AI · on-device agent benchmark</p>
    <h1>{title}</h1>
    <p class="lede">The same {what}, sent to the same model on the same simulated phone, through
    {len(modes_seen)} pipelines. Wherever a pipeline routes, the {_e(router_text)} reads each message
    first{" (switch runs above)" if len(runs) > 1 else ""}.</p>
    <ul class="pipes">{pipes}</ul>
    {picker}
  </header>
  <dl class="legend">
    <div><dt>Exact</dt><dd>Right tool and right arguments, or a plain reply where one was right.</dd></div>
    <div><dt>Hallucinated call</dt><dd>An invented tool or argument, a tool the prompt did not offer, or an unparseable call.</dd></div>
    <div><dt>No tool called</dt><dd>It answered in text where it had to look something up. This is where prices and venues get made up.</dd></div>
    {legend_router}
  </dl>
  {"".join(sections)}
  <footer>Generated by <code>python -m feast_ai bench &lt;model&gt; --suite {_e(runs[0]["meta"].get("suite", "standard"))}
  --modes {",".join(modes_seen)}</code>. Greedy decoding, thinking off, the same prompt text in every
  pipeline except the tool list. Tools are not executed, so each answer is judged on the first step only.</footer>
</div>
<script>{SCRIPT}</script>
"""
    if not standalone:
        return page
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>[hidden]{display:none!important}body{margin:0}</style>'
            f'</head><body>\n{page}</body></html>\n')
