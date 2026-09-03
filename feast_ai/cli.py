"""Command line entry point for feast AI."""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import APP_NAME, __version__


def _cmd_models() -> int:
    from .catalog import CATALOG
    from .paths import human_bytes

    print(f"{APP_NAME} catalog - {len(CATALOG)} edge-optimised models\n")
    header = f"{'#':>2}  {'MODEL':<32} {'PARAMS':<9} {'SIZE':<8} {'TR':<6} {'STATUS':<9} REPO"
    print(header)
    print("-" * len(header))
    for spec in CATALOG:
        status = "local" if spec.downloaded else "remote"
        size = human_bytes(spec.local_size) if spec.downloaded else spec.size_label
        print(
            f"{spec.rank:>2}  {spec.name:<32} {spec.params:<9} {size:<8} "
            f"{spec.tr_stars:<6} {status:<9} {spec.repo}"
        )
    print("\nTR column: Turkish capability, 1-5. Entries at 5 are Turkish-native fine-tunes.")
    return 0


def _cmd_pull(model_id: str) -> int:
    from .catalog import get
    from .download import download
    from .paths import human_bytes

    spec = get(model_id)
    if spec is None:
        print(f"Unknown model id: {model_id}. Run 'feast-ai models' to list them.", file=sys.stderr)
        return 2
    if spec.downloaded:
        print(f"{spec.name} already present at {spec.local_path}")
        return 0

    def on_progress(progress) -> None:
        total = f"/{human_bytes(progress.total)}" if progress.total else ""
        print(
            f"\r  {human_bytes(progress.downloaded)}{total}  "
            f"{human_bytes(progress.speed_bps)}/s   ",
            end="",
            flush=True,
        )

    print(f"Downloading {spec.name} ({spec.size_label}) from {spec.repo}")
    if spec.is_onnx:
        from .download import download_many

        asyncio.run(download_many(
            spec.onnx_urls, spec.local_path,
            total_bytes=spec.approx_mb * 1024 * 1024, on_progress=on_progress,
        ))
    else:
        asyncio.run(download(spec.url, spec.local_path, on_progress=on_progress))
    print(f"\nSaved to {spec.local_path}")
    return 0


def _cmd_add(source: str, name: str | None, quant: str) -> int:
    from . import catalog as cat
    from .paths import human_bytes

    try:
        kind, value = cat.parse_source(source)
    except cat.LookupError_ as exc:
        print(exc, file=sys.stderr)
        return 2

    if kind == "url":
        from pathlib import Path as _Path

        filename = _Path(value.split("?")[0]).name
        size = cat.remote_size(value) or 0
        spec = cat.build_custom_spec(
            name=name or _Path(filename).stem,
            repo="",
            filename=filename,
            size_bytes=size,
            source_url=value,
        )
    else:
        try:
            files = cat.list_repo_files(value)
        except cat.LookupError_ as exc:
            print(exc, file=sys.stderr)
            return 2
        chosen = next((f for f in files if f.quant == quant.upper()), None)
        if chosen is None:
            print(f"No {quant} build in {value}. Available:", file=sys.stderr)
            for f in files:
                print(f"  {f.quant:<10} {human_bytes(f.size):>10}  {f.filename}", file=sys.stderr)
            return 2
        spec = cat.build_custom_spec(
            name=name or value.split("/")[-1].replace("-GGUF", ""),
            repo=value,
            filename=chosen.filename,
            size_bytes=chosen.size,
        )

    cat.add_custom(spec)
    print(f"Added '{spec.name}' as id '{spec.id}' ({spec.size_label})")
    print(f"Download it with:  feast-ai pull {spec.id}")
    return 0


def _cmd_runtime() -> int:
    from .engine import ensure_runtime
    from .paths import human_bytes

    def on_progress(progress) -> None:
        total = f"/{human_bytes(progress.total)}" if progress.total else ""
        print(f"\r  {human_bytes(progress.downloaded)}{total}   ", end="", flush=True)

    info = asyncio.run(ensure_runtime(on_progress=on_progress, on_status=print))
    print(f"\nllama-server: {info.server}  (build {info.build})")
    return 0


def _cmd_doctor() -> int:
    from .catalog import CATALOG
    from .config import HOST_CORES, Settings
    from .engine import runtime_info
    from .metrics.sampler import host_snapshot
    from .paths import HOME, human_bytes

    host = host_snapshot()
    settings = Settings.load()
    runtime = runtime_info()

    print(f"{APP_NAME} v{__version__} - environment check\n")
    print(f"  python           {sys.version.split()[0]} ({sys.platform})")
    print(f"  cpu              {host.cpu_name or 'unknown'}")
    print(f"  cores            {host.logical_cores} logical / {host.physical_cores} physical")
    print(f"  ram              {human_bytes(host.total_ram)}")
    print(f"  data dir         {HOME}")
    engine = (
        f"{runtime.server}  (build {runtime.build})"
        if runtime.available
        else "NOT INSTALLED"
    )
    print(f"  llama-server     {engine}")
    from .engine import onnx_available, onnx_version

    print(f"  onnx runtime     "
          f"{'onnxruntime-genai ' + onnx_version() if onnx_available() else 'NOT INSTALLED'}")
    print(f"  profile          {settings.device.name}")
    print(f"  cpu budget       {settings.cpu_budget_cores}/{HOST_CORES} cores"
          f"{' (pinned)' if settings.pin_affinity else ''}")
    print(f"  ram ceiling      {_ram_cap_status(settings)}")
    downloaded = [m for m in CATALOG if m.downloaded]
    print(f"  models on disk   {len(downloaded)}/{len(CATALOG)}")
    for spec in downloaded:
        tag = "  [onnx]" if spec.is_onnx else ""
        print(f"                   - {spec.name}  {human_bytes(spec.local_size)}{tag}")
    if not runtime.available:
        print("\n  Fix: feast-ai runtime")
    if not downloaded:
        print("  Fix: feast-ai pull gemma-3-270m")
    return 0


def _ram_cap_status(settings) -> str:
    """Report the ceiling *and* whether this OS can actually enforce it.

    The check runs against this very process, so it exercises the same syscall
    the engine will use rather than guessing from the platform name.
    """
    import os

    from .engine.memcap import apply_cap, release_cap
    from .paths import human_bytes

    if not settings.ram_limit_enabled:
        return "disabled (model held fully resident)"

    limit = settings.ram_limit_bytes
    if not limit:
        return "enabled, but no device RAM figure to derive it from"

    probe = apply_cap(os.getpid(), 512 * 1024 * 1024)
    release_cap(os.getpid())
    if probe.applied:
        return f"{human_bytes(limit)} via {probe.method}"
    return f"{human_bytes(limit)} requested - NOT SUPPORTED HERE: {probe.detail}"


def _use_utf8() -> None:
    """Model names contain non-ASCII punctuation; cp1252 consoles mangle it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    parser = argparse.ArgumentParser(
        prog="feast-ai",
        description=f"{APP_NAME} - chat with edge-optimised LLMs on a CPU inference engine.",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="launch the TUI (default)")
    sub.add_parser("models", help="list the model catalog")
    sub.add_parser("runtime", help="download the llama.cpp CPU runtime")
    sub.add_parser("doctor", help="print environment diagnostics")
    pull = sub.add_parser("pull", help="download a model's weights")
    pull.add_argument("model_id")
    add = sub.add_parser("add", help="add a model by HF repo id or direct .gguf URL")
    add.add_argument("source", help="owner/name, a huggingface.co URL, or a .gguf link")
    add.add_argument("--name", help="display name for the catalog")
    add.add_argument("--quant", default="Q4_K_M", help="quantisation to pick (default Q4_K_M)")

    args = parser.parse_args(argv)

    if args.command == "models":
        return _cmd_models()
    if args.command == "pull":
        return _cmd_pull(args.model_id)
    if args.command == "add":
        return _cmd_add(args.source, args.name, args.quant)
    if args.command == "runtime":
        return _cmd_runtime()
    if args.command == "doctor":
        return _cmd_doctor()

    from .ui import FeastApp

    FeastApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
