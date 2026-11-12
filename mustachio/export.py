"""From a LoRA adapter to a ``.litertlm`` a phone - and the TUI - can load.

Three steps, the second of which only runs on Linux:

1. :func:`merge` folds the adapter into the text-only weights and writes
   them in the layout the converter wants (``model.language_model.*`` with
   the original config minus its encoders - see
   :func:`mustachio.weights.to_export_layout`);
2. :func:`convert` runs ``litert-torch export_hf``: it traces the decoder,
   the token embedder and the per-layer embedder, quantises them and bundles
   them with the tokenizer and the chat template that is in the checkpoint
   directory (the on-device one);
3. the bundle is checked the way Google's was: loaded by LiteRT-LM and
   scored on the suites (``eval --bundle``).

What this path gives up, compared with Google's own bundle: quantisation
happens *after* training (a recipe such as ``dynamic_wi4_afp32``), where
Google's decoder was trained into a mixed 2/4/8-bit scheme
(``quant_method: gemma``) that the public converter does not produce.  The
other way to ship - Google's bundle plus a LoRA file loaded at run time - is
described in the workspace README; it is not built.

``litert-torch`` does not install on Windows (pip finds no
``litert-converter`` wheel), so step 2 is run on a Linux machine; the
command it runs is printed either way.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import paths, weights

#: The quantisation recipe.  ``litert-torch``'s stock ``dynamic_wi4_afp32``
#: (channel-wise min-max int4 on every weight) is not usable for this model:
#: on 200 dev turns the v0 adapter fell from 96% to 65.0% exact, with 21
#: calls LiteRT-LM could not parse (a doubled quote token, garbled queries).
#: ``ai_edge_quantizer`` carries Gemma 4 recipes in a form the CLI rejects by
#: name; flattened to JSON (``assets/recipes/``) it takes them as a file:
#:
#: ``gemma4_mixed48_b32``  int4 block-wise (32) for embeddings and linear
#:                         layers, int8 for the per-layer embedding
#:                         projection - 95.0% exact on the same turns, no
#:                         parse error, 2,860 MB;
#: ``gemma4_mixed48``      the same, channel-wise: 2,571 MB;
#: ``gemma4_mixed48_hr``   Hadamard-rotated: 2,657 MB, 520 s to convert.
#:
#: The choice is made on the dev split (``eval --suites dev``), never on the
#: suites.  The prefill and cache sizes are what the app needs.
RECIPE = "gemma4_mixed48_b32"
RECIPES_DIR = paths.ASSETS / "recipes"
CACHE_LENGTH = 4096
#: Prefill signatures.  The runtime reads a prompt in blocks of these sizes
#: and pads the last one, so a single 256-token signature made every request
#: cost 256 tokens of reading however short the message: measured 1.9 s to
#: the first token for a 21-token tool result at 4 threads, against 0.5 s on
#: Google's bundle, which carries ``prefill_128`` and ``prefill_1024``.  The
#: same two are exported here.
PREFILL_LENGTHS = (128, 1024)


def recipe_argument(recipe: str) -> str:
    """A recipe name the converter knows, or the path of the JSON for ours."""
    shipped = RECIPES_DIR / f"{recipe}.json"
    return str(shipped) if shipped.exists() else recipe


def merge(base: Path, adapter: Path, out: Path, original_config: Path) -> dict:
    """Fold a LoRA adapter into the text-only weights, in the converter's layout."""
    import torch
    from peft import PeftModel
    from transformers import Gemma4ForCausalLM

    model = Gemma4ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        scratch_dir = Path(scratch)
        model.save_pretrained(scratch_dir, safe_serialization=True)
        del model
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "generation_config.json"):
            if (base / name).exists():
                shutil.copyfile(base / name, scratch_dir / name)
        report = weights.to_export_layout(scratch_dir, original_config, out)
    report.update(base=str(base), adapter=str(adapter))
    prompt = adapter / "prompt.json"
    if prompt.exists():
        shutil.copyfile(prompt, out / "prompt.json")
    (out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def convert_command(merged: Path, out_dir: Path, recipe: str = RECIPE,
                    cache_length: int = CACHE_LENGTH, converter: str = "") -> list[str]:
    exe = (converter or shutil.which("litert-torch")
           or str(Path(sys.executable).parent / "litert-torch"))
    return [exe, "export_hf", str(merged), str(out_dir),
            "--task=text_generation", "--bundle_litert_lm=True",
            f"--quantization_recipe={recipe_argument(recipe)}",
            f"--cache_length={cache_length}",
            "--prefill_lengths=" + ",".join(str(n) for n in PREFILL_LENGTHS),
            "--externalize_embedder=True", "--single_token_embedder=True",
            "--use_jinja_template=True"]


def convert(merged: Path, out_dir: Path, recipe: str = RECIPE,
            cache_length: int = CACHE_LENGTH, converter: str = "") -> dict:
    """Run the converter; returns the bundle path, its size and the time taken."""
    command = convert_command(merged, out_dir, recipe, cache_length, converter)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Traced and quantised on the CPU: that is the path that was verified,
    # and it leaves the GPU to whatever is training.
    result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    bundles = sorted(out_dir.glob("*.litertlm"))
    if result.returncode != 0 or not bundles:
        raise RuntimeError(f"litert-torch failed (exit {result.returncode}); command: "
                           + " ".join(command))
    return {"bundle": str(bundles[0]), "bytes": bundles[0].stat().st_size,
            "seconds": round(time.monotonic() - started, 1), "recipe": recipe,
            "cache_length": cache_length, "command": command}


def main(args) -> int:
    paths.ensure_dirs()
    base = Path(args.base) if args.base else paths.BUILD_DIR / "base-text"
    adapter = Path(args.adapter)
    name = args.name or adapter.parent.name
    merged = paths.BUILD_DIR / f"merged-{name}"
    original = paths.repo_dir(paths.BASE_REPO) / "config.json"
    if not (adapter / "adapter_config.json").exists():
        print(f"no adapter at {adapter}")
        return 1
    if not original.exists():
        print(f"the original config is needed for the export layout: {original}")
        return 1
    if not (merged / "model.safetensors").exists() or args.force:
        report = merge(base, adapter, merged, original)
        print(f"merged checkpoint: {merged} ({report['export_layout_bytes'] / 1e6:,.0f} MB)")
    out_dir = paths.BUILD_DIR / f"litert-{name}"
    command = convert_command(merged, out_dir, args.recipe, args.cache_length, args.converter)
    if args.no_convert or not Path(command[0]).exists():
        print("Convert on Linux with litert-torch installed:\n  " + " ".join(command))
        return 0
    result = convert(merged, out_dir, args.recipe, args.cache_length, args.converter)
    (out_dir / "convert_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    info = {"recipe": args.recipe}
    try:  # which prompt the adapter was trained with travels with the bundle
        info.update(json.loads((merged / "prompt.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    (out_dir / "prompt.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"bundle: {result['bundle']} ({result['bytes'] / 1e6:,.0f} MB, "
          f"{result['seconds']:.0f} s, {result['recipe']})")
    return 0
