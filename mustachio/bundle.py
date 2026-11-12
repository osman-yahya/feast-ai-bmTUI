"""A text-only ``.litertlm``: the bundle the app would actually download.

A LiteRT-LM bundle is a list of sections - metadata, tokenizer and one
TFLite model per job.  Google's Gemma 4 E2B bundle has twelve; six of them
exist only for images and audio (``vision_encoder``, ``vision_adapter``,
``end_of_vision``, ``audio_encoder_hw``, ``audio_adapter``,
``end_of_audio``).  ``litert-lm unpack`` writes each section to a file plus a
``model.toml`` manifest, and ``litert-lm pack`` builds a bundle from one, so
a text-only bundle is the same manifest without those six blocks.  The
decoder, both embedders, the tokenizer and the MTP drafter are copied byte
for byte, so the text path cannot change.

Two things to know before shipping one:

* the runtime loads the encoders on demand, so this saves download and flash
  size, not RAM (measured: see the workspace README);
* an app that opens a text-only bundle must not ask for a vision or audio
  backend - the engine then fails with ``TF_LITE_VISION_ENCODER not found``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: ``model_type`` values that only serve image or audio input.
MULTIMODAL = re.compile(r"^(vision_|audio_|end_of_vision|end_of_audio)")
_SECTION = "[[section]]"
_MODEL_TYPE = re.compile(r'^model_type\s*=\s*"([^"]+)"', re.M)
_DATA_PATH = re.compile(r'^data_path\s*=\s*"([^"]+)"', re.M)


@dataclass
class Section:
    kind: str       # model_type, or the section_type when it has none
    path: str
    bytes: int
    kept: bool


def _tool(name: str) -> str:
    """The LiteRT-LM CLI that sits next to this interpreter."""
    exe = Path(sys.executable).parent / (name + (".exe" if sys.platform == "win32" else ""))
    return str(exe) if exe.exists() else name


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {result.stderr.strip()[-600:]}")


def unpack(bundle: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([_tool("litert-lm"), "unpack", str(bundle), "--output-dir", str(out_dir),
          "--allow-overwrite"])
    return out_dir / "model.toml"


def read_sections(manifest: Path) -> tuple[str, list[tuple[str, Section]]]:
    """``(preamble, [(toml block, Section)])`` of an unpacked bundle."""
    text = manifest.read_text(encoding="utf-8")
    head, *blocks = text.split(_SECTION)
    out = []
    for block in blocks:
        kind = _MODEL_TYPE.search(block)
        path = _DATA_PATH.search(block)
        name = kind.group(1) if kind else (path.group(1).split(".")[0] if path else "?")
        file = manifest.parent / path.group(1) if path else None
        size = file.stat().st_size if file and file.exists() else 0
        keep = not (kind and MULTIMODAL.match(kind.group(1)))
        out.append((block, Section(name, path.group(1) if path else "", size, keep)))
    return head, out


def text_only(bundle: Path, out: Path, work: Path) -> list[Section]:
    """Repack ``bundle`` without its vision and audio sections."""
    manifest = unpack(bundle, work)
    head, blocks = read_sections(manifest)
    kept = "".join(_SECTION + block for block, section in blocks if section.kept)
    slim = work / "model.text.toml"
    slim.write_text(head + kept, encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([_tool("litert-lm"), "pack", str(slim), "--output", str(out), "--allow-overwrite"])
    return [section for _block, section in blocks]


def render_sections(sections: list[Section], before: int, after: int) -> str:
    lines = [f"{'section':<24}{'MB':>10}  kept"]
    for section in sorted(sections, key=lambda s: -s.bytes):
        lines.append(f"{section.kind:<24}{section.bytes / 1e6:>10.1f}  "
                     f"{'yes' if section.kept else 'no'}")
    lines.append(f"{'bundle':<24}{before / 1e6:>10.1f}  ->  {after / 1e6:.1f} MB "
                 f"({(before - after) / before:.1%} smaller)")
    return "\n".join(lines)


def cleanup(work: Path) -> None:
    """Unpacked sections are a second copy of the bundle - drop them."""
    shutil.rmtree(work, ignore_errors=True)
