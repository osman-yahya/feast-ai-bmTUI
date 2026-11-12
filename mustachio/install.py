"""Put an exported bundle where feast AI finds it.

``python -m mustachio install <bundle.litertlm>`` copies the bundle to
``~/.feast-ai/models/<id>/`` and registers it as a user model with
``engine="litert"``, so it shows up in the TUI's model list and in
``python -m feast_ai models`` like any other.  ``--activate`` also sets the
agent mode to ``native`` (the pipeline the model was trained for) and makes
it the last-used model.

The bundle says which prompt it was trained with (``prompt.json`` next to
it, written by ``export``).  The app serves one prompt -
:data:`feast_ai.agent.native.SYSTEM`, ``v2`` here - so a bundle trained with
another is installed with a warning: it will run, and answer worse than it
scored.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import paths, schema

DEFAULT_ID = "mustachio-e2b"


def main(args) -> int:
    from feast_ai import catalog
    from feast_ai.config import Settings

    bundle = Path(args.bundle) if args.bundle else \
        paths.BUILD_DIR / "litert-final" / "model.litertlm"
    if not bundle.is_file():
        print(f"no bundle at {bundle}")
        return 1
    info: dict = {}
    try:
        info = json.loads((bundle.parent / "prompt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    trained_with = info.get("prompt", "")
    if trained_with and trained_with != schema.DEFAULT:
        print(f"warning: this bundle was trained with prompt {trained_with}; the app serves "
              f"{schema.DEFAULT}")

    model_id = args.id or DEFAULT_ID
    target = catalog.MODELS_DIR / model_id / f"{model_id}.litertlm"
    target.parent.mkdir(parents=True, exist_ok=True)
    if bundle.resolve() != target.resolve():
        if not (target.exists() and target.stat().st_size == bundle.stat().st_size
                and target.stat().st_mtime >= bundle.stat().st_mtime):
            for stale in target.parent.glob("*.xnnpack_cache"):
                stale.unlink()  # built for the previous weights
            print(f"copying {bundle.stat().st_size / 1e6:,.0f} MB -> {target}")
            shutil.copyfile(bundle, target)
    recipe = str(info.get("recipe") or args.precision or "int4")
    spec = catalog.ModelSpec(
        id=model_id,
        name=args.label or "Mustachio (Gemma 4 E2B, feast tools)",
        repo="local/mustachio", filename=target.name, params="2.3B",
        approx_mb=int(target.stat().st_size / 1e6), context=4096, vendor="feast",
        license="Apache-2.0", tr=5, tier="custom", rank=0,
        why="Text-only Gemma 4 E2B fine-tuned on the 15 feast tools (mustachio workspace); "
            "run it with agent mode 'native'.",
        tags=("litert", "tools", "turkish", "fine-tuned"),
        engine="litert", onnx_precision=recipe)
    catalog.add_custom(spec)
    print(f"registered {model_id} ({spec.approx_mb:,} MB, {recipe})")
    if args.activate:
        settings = Settings.load()
        settings.last_model = model_id
        settings.agent_mode = "native"
        settings.save()
        print("settings: last model set, agent mode native")
    print("run:  python -m feast_ai   (pick the model; Ctrl+R cycles the agent mode)")
    return 0
