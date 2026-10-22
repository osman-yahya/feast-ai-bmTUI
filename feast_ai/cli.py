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

    from .encoders import ENCODERS

    print(f"\nEncoders - Cosmos ModernBERT-TR ({len(ENCODERS)}), used by the intent router\n")
    header = f"    {'ENCODER':<32} {'TASK':<15} {'ROLE':<10} {'STATUS':<9} REPO"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for enc in ENCODERS:
        status = "local" if enc.downloaded else "remote"
        print(f"    {enc.id:<32} {enc.task:<15} {enc.role:<10} {status:<9} {enc.repo}")
    print("\nPull an encoder like a model:  feast-ai pull modernbert-tr-embed")
    return 0


def _cmd_pull(model_id: str) -> int:
    from .catalog import get
    from .download import download
    from .paths import human_bytes

    from . import encoders

    encoder = encoders.get(model_id)
    if encoder is not None:
        return _pull_encoder(encoder)

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


def _pull_encoder(encoder) -> int:
    from .download import download_many
    from .paths import human_bytes

    if encoder.downloaded:
        print(f"{encoder.name} already present at {encoder.local_path}")
        return 0

    def on_progress(progress) -> None:
        total = f"/{human_bytes(progress.total)}" if progress.total else ""
        print(f"\r  {human_bytes(progress.downloaded)}{total}   ", end="", flush=True)

    print(f"Downloading {encoder.name} (int8 ONNX, ~{encoder.approx_mb} MB) from {encoder.repo}")
    asyncio.run(download_many(
        encoder.urls, encoder.local_path,
        total_bytes=encoder.approx_mb * 1024 * 1024, on_progress=on_progress,
    ))
    print(f"\nSaved to {encoder.local_path}")
    if encoder.role != "router":
        print(f"Note: feast AI does not use this encoder yet - {encoder.role_note}")
    return 0


def _cmd_bench(args) -> int:
    from .agent.bench import run_bench
    from .engine.server import EngineError

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    from .agent.bench import MODES

    unknown = [m for m in modes if m.split("@")[0] not in MODES]
    if unknown:
        print(f"Unknown mode(s): {unknown}. Choose from: {', '.join(MODES)}.", file=sys.stderr)
        return 2
    if not args.router_only and not args.model_id:
        print("Give a model id, or --router-only.", file=sys.stderr)
        return 2
    try:
        md_path, json_path = asyncio.run(run_bench(
            model_id=args.model_id or "",
            modes=modes,
            router_only=args.router_only,
            limit=args.limit,
            cache=not args.no_cache,
            thinking=args.thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            profile=args.profile,
            cache_ram=args.cache_ram,
            suite=args.suite,
            cores=args.cores,
            ram_gb=args.ram_gb,
            router_model=args.router,
        ))
    except EngineError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(f"\nReport: {md_path}\nData:   {json_path}")
    if not args.router_only:
        print(f"Page:   {md_path.with_name(md_path.stem + '.html')}")
    return 0


def _cmd_compare(paths: list[str], out: str | None) -> int:
    """Merge bench runs into one side-by-side comparison page."""
    import json
    from pathlib import Path

    from .agent.page import render
    from .paths import SESSIONS_DIR

    runs = []
    for raw in paths:
        try:
            runs.append(json.loads(Path(raw).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Cannot read {raw}: {exc}", file=sys.stderr)
            return 2
    runs = [r for r in runs if r.get("meta", {}).get("model_id")]
    if not runs:
        print("No SLM bench runs given (router-only runs have no chats).", file=sys.stderr)
        return 2
    target = Path(out) if out else SESSIONS_DIR / "compare.html"
    target.write_text(render(runs, standalone=True), encoding="utf-8")
    print(f"Page: {target}")
    return 0


def _cmd_route(texts: list[str]) -> int:
    """Route sentences and print each decision - for tuning exemplars."""
    from .agent import start_router
    from .agent.context import AgentContext
    from .config import Settings
    from .engine.server import EngineError

    if not texts:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    async def run() -> None:
        settings = Settings.load()
        server, router = await start_router(settings)
        context = AgentContext.load()
        previous = None
        try:
            print(f"router up in {server.report.load_seconds:.1f}s on "
                  f"{settings.cpu_budget_cores} core(s)\n")
            for text in texts:
                decision = await router.route(text, context, previous)
                previous = decision
                print(f"> {text}\n  {decision.summary()}\n  {decision.reason}"
                      f" · nearest: {decision.nearest!r} · encoder {decision.encoder_ms:.1f} ms")
                if decision.direct is not None:
                    result = decision.direct
                    print(f"  -> {result.tool}({result.args})" if result.is_call
                          else f"  -> {result.text}")
                else:
                    print(f"  -> SLM with: {', '.join(decision.tools) or 'no tools'}")
                print("  top: " + ", ".join(f"{k} {v:.2f}" for k, v in decision.top))
        finally:
            await router.aclose()
            await server.stop()

    try:
        asyncio.run(run())
    except EngineError as exc:
        print(exc, file=sys.stderr)
        return 1
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
    from . import encoders
    from .engine.encoder_server import INSTALL_HINT, missing_packages

    router = encoders.get(settings.router_model)
    missing = missing_packages()
    if missing:
        router_state = f"needs {', '.join(missing)} ({INSTALL_HINT})"
    elif router is None or not router.downloaded:
        router_state = f"{settings.router_model} NOT DOWNLOADED"
    else:
        router_state = f"{router.name}  {human_bytes(router.local_size)}"
    print(f"  intent router    {router_state}")
    print(f"  agent mode       {settings.agent_mode}")
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
    if router is not None and not router.downloaded:
        print(f"  Fix: feast-ai pull {router.id}   (for agent_mode hierarchical)")
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
    bench = sub.add_parser(
        "bench", help="compare the direct and hierarchical agent pipelines on a model"
    )
    bench.add_argument("model_id", nargs="?", help="catalog id of the SLM")
    bench.add_argument("--modes", default="direct,hierarchical",
                       help="comma-separated: direct, hierarchical, routed (router picks the "
                            "prompt, SLM always), contexts (routed + per-intent KV snapshots), "
                            "active (pick a tool by name, then its manual), active2 (active "
                            "with grammar-constrained pick and arguments, prefilled call), "
                            "classes (router picks a class prompt)")
    bench.add_argument("--router-only", action="store_true",
                       help="measure the intent router alone, no SLM")
    bench.add_argument("--limit", type=int, default=0, help="only the first N cases")
    bench.add_argument("--no-cache", action="store_true",
                       help="disable llama.cpp's prompt cache (cold prefill every turn)")
    bench.add_argument("--thinking", default="off", choices=["auto", "off", "on"],
                       help="reasoning mode (default off - agent calls want speed)")
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--temperature", type=float, default=0.0,
                       help="default 0: greedy, so pipelines differ only by prompt")
    bench.add_argument("--profile", default="",
                       help="device profile for this run only (budget, midrange, ...)")
    bench.add_argument("--router", default="",
                       choices=["", "modernbert-tr-embed", "decision-2.0-kai-0.6b"],
                       help="intent router for this run (default: the setting)")
    bench.add_argument("--cores", type=int, default=0,
                       help="CPU cores for this run (overrides the profile)")
    bench.add_argument("--ram-gb", type=float, default=0.0,
                       help="device RAM for this run; enforces the RAM ceiling "
                            "(ram_budget_pct of it, 70%% by default)")
    bench.add_argument("--suite", default="standard", choices=["standard", "kanka"],
                       help="standard labelled set, or kanka: casual end-user chats")
    bench.add_argument("--cache-ram", type=int, default=None,
                       help="llama-server cross-prompt KV cache in MB for this run "
                            "(default: the setting, 0 = off)")
    compare = sub.add_parser("compare", help="merge bench runs into one comparison page")
    compare.add_argument("runs", nargs="+", help="bench-*.json files")
    compare.add_argument("-o", "--out", help="output .html (default sessions/compare.html)")
    route = sub.add_parser("route", help="show how the intent router serves sentences")
    route.add_argument("texts", nargs="*", help="sentences (or one per line on stdin)")

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
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "route":
        return _cmd_route(args.texts)
    if args.command == "compare":
        return _cmd_compare(args.runs, args.out)

    from .ui import FeastApp

    FeastApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
