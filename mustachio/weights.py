"""What is inside a Gemma 4 E2B checkpoint, and a text-only copy of it.

The premise this module was written to test: "drop the vision and audio
encoders and the model gets much smaller".  Measured on the three artefacts
that matter (sizes in MB = 10^6 bytes):

=============================  ========  =============  ==============
artefact                       total     vision+audio   vocab tables*
=============================  ========  =============  ==============
bf16 checkpoint (training)     10,209    952  (9.3%)    5,503 (53.9%)
QAT "mobile" checkpoint        2,458     340 (13.8%)    1,414 (57.6%)
official ``.litertlm`` bundle  2,588     332 (12.8%)    1,388 (53.6%)
=============================  ========  =============  ==============

\\* token embeddings + per-layer embeddings (+ the untied 2-bit ``lm_head``
in the mobile checkpoint): everything with one row per vocabulary entry.

So the encoders are an eighth of the file, and LiteRT-LM already loads them
on demand - removing them saves download size, not RAM.  More than half of
every artefact is indexed by the 262,144-token vocabulary, which is why
trimming the vocabulary to Turkish + English is the bigger lever (see the
workspace README).

A safetensors file is an 8-byte header length, a JSON header and raw tensor
bytes, so both jobs here are done without loading a single tensor: the
breakdown reads only the header (over HTTP range requests for a repo that is
not downloaded), and the text-only copy streams byte ranges into a new file
under new names.  ``Gemma4ForCausalLM`` wants ``model.layers...`` where the
multimodal checkpoint has ``model.language_model.layers...``; nothing else
changes, which :func:`verify_text_only` checks by comparing logits.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

LANG_PREFIX = "model.language_model."
#: Components indexed by token id - the part a smaller vocabulary shrinks.
VOCAB_PARTS = ("token embeddings", "per-layer embeddings", "lm_head")
_COPY_CHUNK = 64 * 1024 * 1024


@dataclass
class Part:
    name: str
    bytes: int = 0
    params: int = 0
    tensors: int = 0


def read_header(source: str | Path) -> dict:
    """The JSON header of a local file, or of ``owner/repo`` on Hugging Face.

    For a repo only the header is fetched (two range requests), so a 10 GB
    checkpoint can be measured before deciding to download it.
    """
    path = Path(source)
    if path.is_file():
        with open(path, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(size))
    import httpx

    url = f"https://huggingface.co/{source}/resolve/main/model.safetensors"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        head = client.get(url, headers={"Range": "bytes=0-7"})
        head.raise_for_status()
        size = struct.unpack("<Q", head.content[:8])[0]
        body = client.get(url, headers={"Range": f"bytes=8-{8 + size - 1}"})
        body.raise_for_status()
        return json.loads(body.content)


def component(key: str) -> str:
    """Which part of the model a tensor belongs to."""
    if key.startswith("lm_head"):
        return "lm_head"
    second = (key.split(".") + [""])[1]
    if "vision" in second:
        return "vision encoder"
    if "audio" in second:
        return "audio encoder"
    rest = key[len(LANG_PREFIX):] if key.startswith(LANG_PREFIX) else key.split(".", 1)[-1]
    if rest.startswith("embed_tokens_per_layer"):
        return "per-layer embeddings"
    if rest.startswith("embed_tokens"):
        return "token embeddings"
    return "decoder layers"


def breakdown(header: dict) -> list[Part]:
    parts: dict[str, Part] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        part = parts.setdefault(component(key), Part(component(key)))
        start, end = info["data_offsets"]
        count = 1
        for dim in info["shape"]:
            count *= dim
        part.bytes += end - start
        part.params += count
        part.tensors += 1
    return sorted(parts.values(), key=lambda p: -p.bytes)


def render_breakdown(parts: list[Part], title: str) -> str:
    total = sum(p.bytes for p in parts) or 1
    lines = [title, f"{'component':<22}{'MB':>10}{'share':>8}{'tensors':>9}"]
    for part in parts:
        lines.append(f"{part.name:<22}{part.bytes / 1e6:>10.1f}{part.bytes / total:>8.1%}"
                     f"{part.tensors:>9}")
    encoders = sum(p.bytes for p in parts if "encoder" in p.name)
    vocab = sum(p.bytes for p in parts if p.name in VOCAB_PARTS)
    lines.append(f"{'total':<22}{total / 1e6:>10.1f}")
    lines.append(f"vision + audio: {encoders / 1e6:.1f} MB ({encoders / total:.1%})  |  "
                 f"vocabulary-indexed: {vocab / 1e6:.1f} MB ({vocab / total:.1%})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Text-only copy
# --------------------------------------------------------------------------
def strip_text_only(src: Path, dst: Path, chat_template: Path | None = None) -> dict:
    """Write ``dst`` as a ``Gemma4ForCausalLM`` checkpoint with ``src``'s text weights.

    Tensor bytes are copied unchanged; only names lose ``language_model.``.
    ``chat_template`` replaces the checkpoint's template - pass the on-device
    one, so ``apply_chat_template`` renders what LiteRT-LM renders.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    weights = src / "model.safetensors"
    with open(weights, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    data_start = 8 + size
    metadata = header.pop("__metadata__", None) or {"format": "pt"}

    kept = sorted(((k, v) for k, v in header.items() if k.startswith(LANG_PREFIX)),
                  key=lambda kv: kv[1]["data_offsets"][0])
    dropped = sum(v["data_offsets"][1] - v["data_offsets"][0]
                  for k, v in header.items() if not k.startswith(LANG_PREFIX))
    new_header: dict = {"__metadata__": metadata}
    offset = 0
    for key, info in kept:
        length = info["data_offsets"][1] - info["data_offsets"][0]
        new_header["model." + key[len(LANG_PREFIX):]] = {
            "dtype": info["dtype"], "shape": info["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # the format wants an 8-byte aligned header

    out = dst / "model.safetensors"
    with open(weights, "rb") as fin, open(out, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for _key, info in kept:
            start, end = info["data_offsets"]
            fin.seek(data_start + start)
            left = end - start
            while left:
                chunk = fin.read(min(_COPY_CHUNK, left))
                if not chunk:
                    raise OSError(f"{weights} ended early")
                fout.write(chunk)
                left -= len(chunk)

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text = dict(config["text_config"])
    text["architectures"] = ["Gemma4ForCausalLM"]
    text["model_type"] = "gemma4_text"
    text["eos_token_id"] = config.get("eos_token_id", text.get("eos_token_id"))
    text["transformers_version"] = config.get("transformers_version")
    (dst / "config.json").write_text(json.dumps(text, indent=2) + "\n", encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    template = chat_template if chat_template and chat_template.exists() \
        else src / "chat_template.jinja"
    if template.exists():
        shutil.copyfile(template, dst / "chat_template.jinja")
    return {
        "source_bytes": weights.stat().st_size,
        "text_only_bytes": out.stat().st_size,
        "dropped_bytes": dropped,
        "tensors_kept": len(kept),
        "tensors_dropped": len(header) - len(kept),
    }


#: Config keys that only describe the encoders or their placeholder tokens.
MULTIMODAL_KEYS = ("vision_config", "audio_config", "image_token_id", "audio_token_id",
                   "video_token_id", "boi_token_id", "eoi_token_id", "boa_token_id",
                   "eoa_token_id", "eoa_token_index", "vision_soft_tokens_per_image")


def rename_tensors(src: Path, dst: Path, rename) -> int:
    """Copy a safetensors file, giving every tensor the name ``rename(name)``.

    Streams the bytes; nothing is loaded.  Returns the size written.
    """
    with open(src, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    data_start = 8 + size
    metadata = header.pop("__metadata__", None) or {"format": "pt"}
    ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header: dict = {"__metadata__": metadata}
    for key, info in ordered:
        new_header[rename(key)] = info  # offsets are unchanged: same bytes, same order
    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        fin.seek(data_start)
        while True:
            chunk = fin.read(_COPY_CHUNK)
            if not chunk:
                break
            fout.write(chunk)
    return dst.stat().st_size


def to_export_layout(text_dir: Path, original_config: Path, dst: Path) -> dict:
    """A text-only checkpoint in the layout ``litert-torch`` exports Gemma 4 from.

    The converter's Gemma 4 path (per-layer embedder, tool-call metadata) is
    keyed on ``model_type: gemma4`` and reaches the decoder as
    ``model.language_model``; a ``gemma4_text`` checkpoint takes its generic
    path instead.  So the weights go back under ``model.language_model.``
    and the config is the original one without its encoders.
    """
    text_dir, dst = Path(text_dir), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    weights = sorted(text_dir.glob("*.safetensors"))
    if len(weights) != 1:
        raise ValueError(f"expected one safetensors file in {text_dir}, found {len(weights)}")
    size = rename_tensors(
        weights[0], dst / "model.safetensors",
        lambda key: LANG_PREFIX + key[len("model."):] if key.startswith("model.") else key)
    config = json.loads(Path(original_config).read_text(encoding="utf-8"))
    for key in MULTIMODAL_KEYS:
        config.pop(key, None)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        if (text_dir / name).exists():
            shutil.copyfile(text_dir / name, dst / name)
    return {"export_layout_bytes": size, "dropped_config_keys": list(MULTIMODAL_KEYS)}


PROBES = (
    "Kanka yarın Kadıköy'e lets go, iyi bi kahveci var mı oralarda?",
    "The capital of Türkiye is",
)


def verify_text_only(src: Path, dst: Path, device: str = "cpu") -> dict:
    """Same text in, same logits out: the multimodal model against its text-only copy.

    The two are loaded one after the other (both do not fit in memory at
    once) and compared on next-token logits.  A difference of exactly 0 is
    the expected result - the same weights run through the same layers - and
    it is what "smaller, with no change in intelligence" means here.
    """
    import torch
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(dst)
    batches = [tokenizer(p, return_tensors="pt") for p in PROBES]

    def logits_of(cls, path: Path) -> list:
        model = cls.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        out = []
        with torch.no_grad():
            for batch in batches:
                ids = batch["input_ids"].to(device)
                out.append(model(input_ids=ids).logits[0, -1].float().cpu())
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
        return out

    full = logits_of(Gemma4ForConditionalGeneration, Path(src))
    text = logits_of(Gemma4ForCausalLM, Path(dst))
    diffs = [float((a - b).abs().max()) for a, b in zip(full, text)]
    same_top = [int(a.argmax()) == int(b.argmax()) for a, b in zip(full, text)]
    return {
        "max_abs_logit_diff": max(diffs),
        "same_next_token": all(same_top),
        "next_tokens": [tokenizer.decode([int(a.argmax())]) for a in text],
        "probes": len(PROBES),
    }
