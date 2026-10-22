"""System prompts for the two agent pipelines being compared.

``direct``        one prompt with every tool - the SLM decides alone.
``hierarchical``  the router picks an intent first; the prompt then lists only
                  that intent's tool group (or none, for small talk).

``active``/``active2`` (index + on-demand manual) and ``classes`` (one prompt
per router class) have their own prompts further down.

Everything except the TOOLS block is identical text in both pipelines - the
preamble, the call format, the rules, the context.  That is deliberate: the
comparison is meant to isolate one variable, how many tools the model is
shown, so a difference in accuracy or prefill cost can be attributed to the
routing rather than to wording.

The call-format line is a template, not a filled-in example: a concrete
example invites small models to copy its values into unrelated calls, and it
would favour whichever tool it happened to show.  The placeholders are spelled
as obvious capitals and the next line says where the real names come from -
measured on the first benchmark: with a lowercase ``{"arg": value}``
placeholder, Qwen3 1.7B wrote ``"arg"`` as the argument name whenever a routed
prompt listed a single tool.

Order: static first, ``CONTEXT`` last - see :mod:`feast_ai.agent.context` for
why that is what lets llama.cpp keep the tools in its prompt cache.
"""

from __future__ import annotations

from .context import AgentContext
from .tools import ALL_TOOL_NAMES, render_tools

PREAMBLE = (
    "You are feast AI, the assistant inside the feast mobile app, running on the "
    "user's phone. You help with restaurants, cafés and dishes near the user, and "
    "with their saved items, posts, friends and chats."
)

CALL_FORMAT = (
    "HOW TO CALL A TOOL\n"
    "Reply with one line and nothing else:\n"
    '<tool_call>{"name": "TOOL", "arguments": {"ARGUMENT": VALUE}}</tool_call>\n'
    "TOOL and every ARGUMENT must be names written in the tool list above. Leave "
    "out arguments you do not need. If no listed tool fits, reply in plain text.\n"
    "Never invent venues, dishes, prices, people or messages - they only come "
    "from tool results. Reply in the user's language."
)

NO_TOOLS = (
    "You cannot use tools in this reply. Answer briefly in plain text, in the "
    "user's language. Never invent venues, dishes, prices or people."
)


def tool_prompt(tools: tuple[str, ...] | list[str], context: AgentContext) -> str:
    return "\n\n".join((
        PREAMBLE,
        "TOOLS  (* = required)\n\n" + render_tools(tools, context.tags),
        CALL_FORMAT,
        context.render(),
    ))


def chat_prompt(context: AgentContext) -> str:
    return "\n\n".join((PREAMBLE, NO_TOOLS, context.render()))


def full_prompt(context: AgentContext) -> str:
    """The ``direct`` pipeline: every tool, every time."""
    return tool_prompt(ALL_TOOL_NAMES, context)


def prompt_for(tools: tuple[str, ...], context: AgentContext) -> str:
    """What a routing decision's tool set turns into."""
    return tool_prompt(tools, context) if tools else chat_prompt(context)


# --------------------------------------------------------------------------
# active: pick a tool by name, then get its manual
# --------------------------------------------------------------------------
SELECT_FORMAT = (
    "HOW TO ANSWER\n"
    "If the user's message needs one of these tools, first reply with only its name, like this:\n"
    "<use>TOOL</use>\n"
    "The app then shows you that tool's manual. Follow it and write the call it asks for.\n"
    "Otherwise answer in plain text, in the user's language. Never invent venues, "
    "dishes, prices, people or messages - they only come from tool results."
)
#: The first pass ends as soon as a tool is named: the name is all the app
#: needs. Gemma 4 sometimes writes a whole call instead of ``<use>``; stopping
#: at ``"arguments"`` saved ~2 s of decode on the first smoke run.
SELECT_STOP = ("</use>", '"arguments"', "</tool_call>")


def index_static() -> str:
    from .manuals import render_index

    return "\n\n".join((PREAMBLE, "TOOLS\n" + render_index(), SELECT_FORMAT))


def index_prompt(context: AgentContext) -> str:
    """The ``active`` pipeline's one fixed prompt: every tool, names only."""
    return index_static() + "\n\n" + context.render()


def manual_message(tool: str, context: AgentContext) -> str:
    """Appended after the user's message once a tool is picked, then dropped.

    It goes at the end so llama.cpp reuses the whole conversation from its
    cache and prefills only the manual.
    """
    from .manuals import render_manual

    # Measured on the first full run: without "do not write <use> again" the
    # model obeyed the first-pass instruction once more and replied with the
    # bare tag on 13 of 31 turns.
    return (
        f"TOOL MANUAL for {tool}. You picked it; do not write <use> again. (* = required)\n\n"
        + render_manual(tool, context.tags)
        + "\n\nNow reply with the call, one line: <tool_call>{\"name\": \"" + tool
        + "\", \"arguments\": {...}}</tool_call>, using only the arguments above. If the "
        "tool does not fit after all, or its rules say not to call yet, answer in plain text."
    )


# --------------------------------------------------------------------------
# active2: the same two passes, constrained (see feast_ai.agent.grammar)
# --------------------------------------------------------------------------
#: No "plain text" wording here or in the cue list: on the dev set Gemma once
#: began a reply with "plain text: ..." copied from an earlier wording.
SELECT2_FORMAT = (
    "HOW TO ANSWER\n"
    "If a tool fits, reply only <use>TOOL</use> with its exact name. The app then shows you "
    "that tool's manual.\n"
    "A short follow-up that only adds a condition (a price limit, a distance, open now) to "
    "your last tool call keeps that tool: pick it again, do not ask what they mean.\n"
    "Otherwise answer briefly in the user's language; the time, date and location are in "
    "CONTEXT. Never invent venues, dishes, prices, people or messages."
)


def index2_static() -> str:
    from .manuals import render_cues2, render_index2

    return "\n\n".join((PREAMBLE, "TOOLS\n" + render_index2(), render_cues2(), SELECT2_FORMAT))


def index2_prompt(context: AgentContext) -> str:
    """``active2``'s one fixed prompt - static, so it stays in the KV cache."""
    return index2_static() + "\n\n" + context.render()


def manual2_message(tool: str, context: AgentContext, step: str = "") -> str:
    """``active2``'s pass-B system turn: the compact manual, dropped afterwards.

    No "do not write <use> again" and no call template: the call's opening
    is prefilled as the assistant's turn, so the tag echo ``active`` fought
    with words cannot happen (except for the confirm step of
    ``send_message_to_friends``, which must stay free to answer in text).
    """
    from .manuals import render_manual_compact

    return "TOOL MANUAL\n" + render_manual_compact(tool, context.tags, step)


# --------------------------------------------------------------------------
# classes: one prompt per class, with the manuals of its tools
# --------------------------------------------------------------------------
def class_static(ids: tuple[str, ...], tools: tuple[str, ...], tags: tuple[str, ...]) -> str:
    """Everything before CONTEXT for a class (or a merged near-tie of classes)."""
    from .classes import BY_ID
    from .manuals import render_manuals

    specs = [BY_ID[i] for i in ids if i in BY_ID]
    task = " ".join(dict.fromkeys(s.task for s in specs if s.task))
    rules = [r for s in specs for r in s.rules]
    parts = [PREAMBLE, "TASK: " + task if task else ""]
    if tools:
        parts += ["TOOLS  (* = required)\n\n" + render_manuals(tools, tags), CALL_FORMAT]
    else:
        parts.append(NO_TOOLS)
    if rules:
        parts.append("RULES\n" + "\n".join(f"- {r}" for r in dict.fromkeys(rules)))
    return "\n\n".join(p for p in parts if p)


def class_prompt(ids: tuple[str, ...], tools: tuple[str, ...], context: AgentContext) -> str:
    return class_static(ids, tools, context.tags) + "\n\n" + context.render()


def static_part(tools: tuple[str, ...], context: AgentContext) -> str:
    """Everything before CONTEXT - the part worth a precomputed KV snapshot.

    The clock and location change all the time, so a snapshot that included
    them would be stale within a minute. This text is a string prefix of
    :func:`prompt_for`, so llama.cpp's prefix match reuses the restored state
    and prefills only CONTEXT, history and the new message.
    """
    if tools:
        return "\n\n".join((
            PREAMBLE, "TOOLS  (* = required)\n\n" + render_tools(tools, context.tags), CALL_FORMAT,
        ))
    return "\n\n".join((PREAMBLE, NO_TOOLS))
