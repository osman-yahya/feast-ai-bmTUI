"""GBNF grammars for the ``active2`` pipeline: the tool registry as decoding rules.

``active`` asked the model, in words, to name a tool and then to write a call,
and the first Gemma 4 E2B run shows how words fail: 6 times out of 91 the
second pass repeated ``<use>tool</use>`` instead of calling, once it wrote
two calls in one ``<tool_call>``, and the first pass sometimes wrote a whole
call before the stop string cut it.  None of that is a choice the model needs
to be free to make.  llama.cpp's grammar sampler (``grammar`` on
``/v1/chat/completions``, verified on b10144) masks every token that would
leave the grammar, so here the same :mod:`feast_ai.agent.tools` data that
renders the prompt and validates calls also becomes the grammar:

``select_grammar``  pass A: ``<use>`` plus one of the 15 exact tool names, or
                    a plain-text answer.  The text branch may not start with
                    ``<`` or ``{`` and never contains them, so a pick is the
                    only way to reach a tool - no invented names, no call
                    without its manual.  The grammar ends right after the
                    name: the closing ``</use>`` would be 3 more decode
                    tokens (~180 ms at 16 tok/s) that tell the app nothing.
``args_grammar``    pass B: the picked tool's arguments object - only its
                    parameter names, typed values, enum members and tags from
                    the context list; required keys first, then optional keys
                    in any order; exactly one of an ``exactly_one`` group.
                    An id parameter is limited to the ids the app has already
                    shown in this chat (an input-dependent grammar, Geng et
                    al. 2023), so an invented id cannot be written.
``confirm_grammar`` pass B for ``send_message_to_friends`` after a proposal:
                    the full call or plain text (the user may cancel).
``TEXT_GRAMMAR``    pass B for ``send_message_to_friends`` when nothing was
                    proposed yet: plain text only - confirm-first enforced in
                    code, not left to the model.

What a grammar does *not* do: choose for the model.  Greedy decoding still
picks the most likely allowed token, so the tool, the values and the
text-or-call decision remain the model's; the grammar only removes the
outcomes the app could not act on.  That also means the judge's
hallucination count for ``active2`` measures the grammar as much as the
model - reported as such.

``limit`` is left out of every grammar, as it is left out of the manuals: a
default-only knob the model would spend ~5 decode tokens on.

``json_schema`` is not used: on b10144 it fails with HTTP 400 ("Unexpected
empty grammar stack after accepting piece: <") as soon as the request ends
with a prefilled assistant message, while a raw ``grammar`` constrains only
the continuation - measured before this module was written.

What it costs: decode speed is unchanged (17.0 tok/s for ``active2`` vs 17.0
for ``active`` over the 91-turn run), but the arguments grammar makes
llama.cpp's sampler allocate fresh memory on every constrained token - ~3.4
MB of demand-zero page faults per token, identical with the RAM ceiling off
(48,302 vs 48,434 faults over 5 calls) and with zero bytes read from the
model file. The bench's "page-ins" figure counts every fault, so ``active2``
shows 1.7-3.2 GB there against 0.1-0.3 GB for the other pipelines; that is
allocator churn, not weights re-read under the ceiling. The select grammar's
text branch costs nothing measurable (3,899 vs 3,921 faults).
"""

from __future__ import annotations

import json
import re

from .tools import BY_NAME, TOOLS, Param, ToolSpec

#: Parameters no grammar offers (see the module docstring).
SKIP = ("limit",)

#: Id parameters and the id prefix the app uses for that kind of result.
ID_KIND = {"restaurant_id": "r", "attach_restaurant_id": "r", "menu_item_id": "m"}

_ID = re.compile(r"\b([rm]_\d+)\b")

# JSON scalars, as in llama.cpp's own json-schema-to-grammar output; the single
# optional space matches how models write JSON ({"a": 1, "b": 2}).
_BASE = (
    'ws ::= " "?\n'
    'string ::= "\\"" char* "\\""\n'
    'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])\n'
    'integer ::= "-"? [0-9] [0-9]{0,6}\n'
    'number ::= integer ("." [0-9] [0-9]?)?\n'
    'boolean ::= "true" | "false"\n'
    'hhmm ::= "\\"" ([01]? [0-9] | "2" [0-3]) ":" [0-5] [0-9] "\\""\n'
)

#: A plain-text reply: no ``<`` and no ``{`` anywhere, so no call syntax.
TEXT_RULE = "text ::= [^<{] [^<{]*\n"
TEXT_GRAMMAR = "root ::= text\n" + TEXT_RULE


def _lit(text: str) -> str:
    """``text`` as a GBNF string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _rule(name: str) -> str:
    return name.replace("_", "-")


def known_ids(texts: list[str]) -> dict[str, tuple[str, ...]]:
    """Ids the app has shown, by prefix (``"r"`` venues, ``"m"`` dishes).

    Read from the assistant's result messages only, never from its own
    earlier calls - an id the model once invented must not become "known".
    """
    out: dict[str, list[str]] = {"r": [], "m": []}
    for text in texts:
        if "<tool_call>" in text:
            continue
        for found in _ID.findall(text):
            bucket = out[found[0]]
            if found not in bucket:
                bucket.append(found)
    return {k: tuple(v) for k, v in out.items()}


def _value(param: Param, tags: tuple[str, ...], ids: dict[str, tuple[str, ...]],
           only_true: bool = False) -> str:
    if param.enum:
        return "(" + " | ".join(_lit(json.dumps(v)) for v in param.enum) + ")"
    kind = param.type
    if kind == "bool":
        # In an exactly-one pair "false" would mean "not given" - and leave
        # the pair empty - so only the value that selects it is offered.
        return '"true"' if only_true else "boolean"
    if kind == "int":
        return "integer"
    if kind == "num":
        return "number"
    if kind == "hhmm":
        return "hhmm"
    if kind == "str[]":
        if param.choices_from == "tags" and tags:
            item = "(" + " | ".join(_lit(json.dumps(t, ensure_ascii=False)) for t in tags) + ")"
        else:
            item = "string"
        return f'"[" ws {item} (ws "," ws {item})* ws "]"'
    shown = ids.get(ID_KIND.get(param.name, ""), ())
    if shown:
        return "(" + " | ".join(_lit(json.dumps(i)) for i in shown) + ")"
    return "string"


def _object_rules(spec: ToolSpec, tags: tuple[str, ...],
                  ids: dict[str, tuple[str, ...]]) -> tuple[str, list[str]]:
    """``(expression, rules)`` for the tool's arguments object."""
    prefix = _rule(spec.name)
    params = [p for p in spec.params if p.name not in SKIP]
    grouped = {n for group in spec.exactly_one for n in group}
    rules: list[str] = []

    def kv(param: Param) -> str:
        name = f"{prefix}-{_rule(param.name)}"
        value = _value(param, tags, ids, only_true=param.name in grouped)
        rules.append(f'{name} ::= {_lit(json.dumps(param.name))} ws ":" ws {value}')
        return name

    head = [kv(p) for p in params if p.required]
    for group in spec.exactly_one:
        head.append("(" + " | ".join(kv(spec.param(n)) for n in group) + ")")
    optional = [kv(p) for p in params if not p.required and p.name not in grouped]
    if optional:
        rules.append(f"{prefix}-opt ::= " + " | ".join(optional))
    more = f' (ws "," ws {prefix}-opt)*' if optional else ""
    if head:
        body = ' ws "," ws '.join(head) + more
    elif optional:
        body = f"({prefix}-opt{more})?"
    else:
        body = ""
    return f'"{{" ws {body} ws "}}"' if body else '"{" ws "}"', rules


def args_grammar(name: str, tags: tuple[str, ...] = (),
                 ids: dict[str, tuple[str, ...]] | None = None) -> str:
    """Pass B: the arguments object of ``name``, then the call's own ``}``.

    It starts with an optional space and ends with the outer brace on
    purpose: the prefill stops at ``"arguments":`` and the model writes
    ``␣{`` and ``}}`` as the multi-character tokens it would naturally pick
    (Gemma's vocabulary has `` {"`` and ``"}}``). A prefill ending in a space,
    or a grammar ending at the inner brace, masks those tokens and forces
    unnatural splits - the first smoke test wrote ``{ "item_type": "any" }``
    for a sentence naming a friend, and padded every object with spaces.
    """
    expression, rules = _object_rules(BY_NAME[name], tags, ids or {})
    return f'root ::= " "? {expression} "}}"\n' + "\n".join(rules) + "\n" + _BASE


def call_open(name: str) -> str:
    """The call up to its arguments - what pass B prefills for the model.

    No trailing space: Anthropic's prefill guide forbids it for the same
    token-boundary reason (a lone space token is not how text continues).
    """
    return '<tool_call>{"name": "' + name + '", "arguments":'


#: What the app appends after the grammar's last ``}``.
CALL_CLOSE = "</tool_call>"


def confirm_grammar(name: str, tags: tuple[str, ...] = (),
                    ids: dict[str, tuple[str, ...]] | None = None) -> str:
    """Pass B after a proposal: the whole call, or plain text."""
    expression, rules = _object_rules(BY_NAME[name], tags, ids or {})
    return ("root ::= call | text\n"
            f"call ::= {_lit(call_open(name) + ' ')} {expression} {_lit('}' + CALL_CLOSE)}\n"
            + "\n".join(rules) + "\n" + TEXT_RULE + _BASE)


def select_grammar(names: tuple[str, ...] | list[str] | None = None) -> str:
    """Pass A: ``<use>`` + an exact tool name, a bare exact tool name, or text.

    The bare name is admitted because Gemma 4 E2B writes it - 2 of 23 dev
    sentences answered ``search_chat_memory`` / ``send_message_to_friends``
    with no tag - and it names the tool exactly as clearly; without this
    branch it would go down the text branch and be judged a broken call.
    """
    wanted = [t.name for t in TOOLS if names is None or t.name in names]
    return ("root ::= pick | text\n"
            'pick ::= "<use>"? tool\n'
            "tool ::= " + " | ".join(_lit(n) for n in wanted) + "\n" + TEXT_RULE)


_PICKED = re.compile(r"(?:<use>)?\s*([a-z_]+)\s*(?:</use>)?")


def picked_tool(reply: str) -> str:
    """The tool a pass-A reply picked - the whole reply is the pick - or ""."""
    match = _PICKED.fullmatch(reply.strip())
    return match.group(1) if match and match.group(1) in BY_NAME else ""


__all__ = ["CALL_CLOSE", "TEXT_GRAMMAR", "args_grammar", "call_open", "confirm_grammar",
           "known_ids", "picked_tool", "select_grammar"]
