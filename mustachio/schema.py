"""The prompt a model is trained and served with - by name, so a run can say which.

The prompt itself is defined in :mod:`feast_ai.agent.native`, because the app
serves it; this module only names the versions the workspace has trained
with, so an adapter can be evaluated with the prompt it learned:

``v0``  the first run: 193 tokens of system text and the ``full``
        declarations (``active2``'s descriptions and a hint per argument),
        2,087 static tokens.  Right for judging an untrained model, and what
        ``runs/v0`` was trained on;
``v1``  what a trained model needs: a shorter system text and ``terse``
        declarations.  Every static token is prefilled once per session on a
        phone CPU, so the prompt is part of the latency budget;
``v2``  what the app serves: ``v1`` plus the persona (Mustachio) and the
        rules for the sentence said after a tool result.

Why the context is a ``[ctx]`` line on the user's turn and not part of the
system text is explained in :mod:`feast_ai.agent.native`.
"""

from __future__ import annotations

from dataclasses import dataclass

from feast_ai.agent import native
from feast_ai.agent.context import DEFAULT_TAGS, AgentContext

SYSTEM_V0 = (
    "You are feast AI, the assistant inside the feast mobile app, running on the user's "
    "phone. You help with restaurants, cafés and dishes near the user, and with their saved "
    "items, posts, friends and chats.\n"
    "Call a tool when one fits; otherwise answer briefly in plain text. Never invent venues, "
    "dishes, prices, people or messages - they only come from tool results. Reply in the "
    "user's language; Turkish with English words mixed in is still Turkish.\n"
    "Every user message starts with a [ctx] line: the time, the active search location and "
    "whether that is the phone's GPS position. Never call a tool to learn them.\n"
    "send_message_to_friends is the only tool that changes anything. Never call it in the "
    "turn the user asks: first say who gets it and the exact text and ask \"Göndereyim mi?\". "
    "Call it only after the user confirms."
)


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    style: str  # declaration style, see feast_ai.agent.native.STYLES

    def system_message(self) -> dict:
        return {"role": "system", "content": self.system}

    def declarations(self, tags: tuple[str, ...] = DEFAULT_TAGS) -> list[dict]:
        return native.declarations(tags, self.style)


PROMPTS = {
    "v0": Prompt("v0", SYSTEM_V0, "full"),
    "v1": Prompt("v1", native.SYSTEM_V1, "terse"),
    "v2": Prompt("v2", native.SYSTEM, "terse"),
}
DEFAULT = "v2"


def prompt(name: str = "") -> Prompt:
    return PROMPTS[name or DEFAULT]


@dataclass(frozen=True)
class Ctx:
    """The volatile phone state one message is sent with."""

    now: str = "2026-11-10 Tue 19:30"
    label: str = "Moda, Kadıköy, İstanbul"
    is_gps: bool = True

    def line(self) -> str:
        return native.ctx_line(AgentContext(label=self.label, is_gps=self.is_gps, now=self.now))


def user_content(text: str, ctx: Ctx) -> str:
    return f"{ctx.line()}\n{text}"
