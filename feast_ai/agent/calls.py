"""Find the tool call in an SLM reply and judge it.

The system prompt asks for exactly one ``<tool_call>{json}</tool_call>`` line.
Small models drift from that in predictable ways - a ```json fence, a bare
object, ``name(arg=...)`` Python style, arguments without their "arguments"
wrapper, or the model's own trained syntax (Gemma 4 writes
``name{key:<|"|>value<|"|>}``) - and those are *format* slips, not
hallucinations: the app could still act on them.  So they are accepted, but
the format actually used is recorded, which keeps protocol adherence and tool
choice as separate measurements.

What counts as a hallucination (:attr:`CallCheck.hallucinated`):

``parse_error``     it tried to call a tool and the payload is not parseable;
``unknown_tool``    the tool does not exist at all;
``unoffered_tool``  the tool exists but was not in *this* prompt - under
                    routing, the model reached for something it was not shown;
``invalid_args``    invented argument names, enum values outside the list,
                    a missing required argument, both or neither of an
                    exactly-one pair.

Nothing here executes ``eval``: Python-style calls go through
:func:`ast.literal_eval`, which only ever builds literals.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from .tools import BY_NAME, validate

_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.S)
_PYCALL = re.compile(r"^\s*`?([a-z][a-z0-9_]*)\((.*)\)`?\s*$", re.S)
_NATIVE = re.compile(r"^(?:call:)?([a-z][a-z0-9_]*)\s*[>(]?\s*(\{.*\})\s*\)?$", re.S)
_USE_ARGS = re.compile(r"^<use>\s*([a-z][a-z0-9_]*)\s*</use>\s*(.*)$", re.S)

HALLUCINATIONS = ("parse_error", "unknown_tool", "unoffered_tool", "invalid_args")


@dataclass
class CallCheck:
    status: str = "text"  # "text" | "ok" | one of HALLUCINATIONS
    name: str = ""
    args: dict = field(default_factory=dict)
    format: str = ""  # "tagged" | "fenced" | "bare" | "pycall" | "bare-name"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Positional values from a Python-style call, mapped once the tool is known.
    positional: list = field(default_factory=list)

    @property
    def is_call(self) -> bool:
        return self.status != "text"

    @property
    def hallucinated(self) -> bool:
        return self.status in HALLUCINATIONS

    def label(self) -> str:
        if self.status == "text":
            return "text reply"
        if self.status == "ok":
            return f"{self.name} ok" + (f" ({len(self.warnings)} warn)" if self.warnings else "")
        return f"{self.name or '?'} {self.status.replace('_', ' ')}"


def _decode_object(text: str) -> dict | None:
    """The first JSON object in ``text``, tolerating prose around it."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalise(payload: dict, check: CallCheck) -> None:
    name = payload.get("name")
    args = payload.get("arguments")
    if args is None:
        for alias in ("parameters", "args", "input"):
            if alias in payload:
                args = payload[alias]
                check.warnings.append(f"arguments given as '{alias}'")
                break
    if isinstance(args, str):
        # Some templates double-encode: "arguments": "{\"query\": ...}".
        try:
            args = json.loads(args)
            check.warnings.append("arguments were a JSON string")
        except json.JSONDecodeError:
            pass
    if args is None and len(payload) > 1:
        # {"name": "get_directions", "restaurant_id": "r_481"} - unambiguous,
        # just missing the "arguments" wrapper. Gemma 4 E2B does this often.
        args = {k: v for k, v in payload.items() if k != "name"}
        check.warnings.append("arguments at the top level")
        check.format += "/flat"
    check.name = str(name or "")
    if check.name.startswith("functions."):
        check.name = check.name[len("functions."):]
    check.args = args if isinstance(args, dict) else {}
    if args is not None and not isinstance(args, dict):
        check.issues.append("arguments is not an object")


def _native_object(text: str) -> dict | None:
    """Gemma 4's own call syntax, ``{query:<|"|>burger<|"|>,limit:5}``.

    ``<|"|>`` is Gemma 4's string-delimiter token; with a hand-written prompt
    the model falls back to the format it was trained on. Swap the token for
    quotes and quote the bare keys, and it is plain JSON.
    """
    raw = text.replace('<|"|>', '"')
    for candidate in (raw, re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', raw)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_native(text: str, check: CallCheck) -> bool:
    """``name{...}``, ``name({...})``, ``call:name{...}`` - a name glued to
    an argument object. Unambiguous, so a format slip rather than a defect."""
    match = _NATIVE.match(text.strip())
    if not match:
        return False
    args = _native_object(match.group(2))
    if args is None:
        return False
    check.name, check.args = match.group(1), args
    check.format = (check.format + "/native") if check.format else "native"
    check.warnings.append("model-native call syntax")
    return True


def _parse_pycall(text: str, check: CallCheck) -> bool:
    match = _PYCALL.match(text.strip())
    if not match or "_" not in match.group(1):
        return False
    try:
        node = ast.parse(text.strip().strip("`"), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    args: dict = {}
    try:
        for keyword in node.keywords:
            if keyword.arg is None:
                return False
            args[keyword.arg] = ast.literal_eval(keyword.value)
    except ValueError:
        check.issues.append("non-literal argument")
    try:
        check.positional = [ast.literal_eval(arg) for arg in node.args]
    except ValueError:
        check.issues.append("non-literal argument")
    check.name, check.args, check.format = node.func.id, args, "pycall"
    return True


def extract(reply: str) -> CallCheck:
    """Locate and decode the call, without judging it against a tool set."""
    check = CallCheck()
    text = reply.strip()
    if not text:
        return check

    picked = _USE_ARGS.match(text)
    if picked:
        # ``<use>name</use>{args}`` - the active pipeline's pick tag followed
        # by the arguments: the tool is named, so it is a format slip.
        args = _decode_object(picked.group(2))
        if args is not None and "name" not in args:
            check.name, check.args, check.format = picked.group(1), args, "use+args"
            check.warnings.append("pick tag followed by bare arguments")
            return check
        text = picked.group(2).strip()

    tagged = _TAGGED.search(text)
    if tagged:
        check.format = "tagged"
        body = tagged.group(1).strip().strip("`")
        if body.startswith("json"):
            body = body[4:]
        payload = _decode_object(body)
        if payload is not None and "name" in payload:
            _normalise(payload, check)
            return check
        if _parse_native(body, check):
            return check
        check.status = "parse_error"
        check.issues.append("<tool_call> body is not a call")
        return check

    fenced = _FENCED.search(text)
    if fenced:
        payload = _decode_object(fenced.group(1))
        if payload is not None and "name" in payload:
            check.format = "fenced"
            _normalise(payload, check)
            return check

    if text.startswith("{") or '"name"' in text:
        payload = _decode_object(text)
        if payload is not None and "name" in payload:
            check.format = "bare"
            _normalise(payload, check)
            return check
        if text.startswith("{"):
            check.format = "bare"
            check.status = "parse_error"
            check.issues.append("JSON object without a parseable tool name")
            return check

    if _parse_pycall(text, check) or _parse_native(text, check):
        return check
    bare = text.strip("`'\" .!")
    if bare in BY_NAME:
        # Just the tool's name: it meant to call, and botched the format.
        check.name, check.format, check.status = bare, "bare-name", "parse_error"
        check.issues.append("bare tool name, no call payload")
        return check
    return check  # plain text


def check_reply(reply: str, offered: tuple[str, ...] | list[str],
                tags: tuple[str, ...] = ()) -> CallCheck:
    """Extract the call and judge it against the tools this prompt offered."""
    check = extract(reply)
    if check.status == "parse_error" or not check.format:
        return check
    if not check.name:
        check.status = "parse_error"
        check.issues.append("no tool name")
        return check
    if check.name not in BY_NAME:
        check.status = "unknown_tool"
        return check
    if check.name not in offered:
        check.status = "unoffered_tool"
        return check
    if check.positional:
        _map_positional(check)
    result = validate(check.name, check.args, tags)
    check.issues += result.issues
    check.warnings += result.warnings
    check.status = "invalid_args" if check.issues else "ok"
    if len(_TAGGED.findall(reply)) > 1:
        check.warnings.append("more than one call; only the first was judged")
    return check


def _map_positional(check: CallCheck) -> None:
    """``open_screen("settings")`` is unambiguous when the values line up with
    the required parameters one-to-one, so an app could act on it: map them
    and warn. Anything looser stays an issue."""
    required = [n for n in BY_NAME[check.name].required if n not in check.args]
    if len(check.positional) <= len(required):
        for name, value in zip(required, check.positional):
            check.args[name] = value
        check.warnings.append("positional arguments mapped to required parameters")
    else:
        check.issues.append("positional arguments")


def render_call(name: str, args: dict) -> str:
    """The canonical wire form, used for direct calls and history."""
    payload = {"name": name, "arguments": args}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
