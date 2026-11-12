"""``python -m mustachio <command>`` - run from the git root, like feast AI.

Commands import their dependencies lazily: ``inspect`` and ``data`` need
nothing beyond feast AI's own requirements, while ``train`` and ``eval``
need torch, transformers and peft (``mustachio/requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _use_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _cmd_inspect(args: argparse.Namespace) -> int:
    from . import weights

    source = args.source
    local = Path(source)
    if local.is_dir():
        source = local / "model.safetensors"
    header = weights.read_header(source)
    print(weights.render_breakdown(weights.breakdown(header), f"{args.source}"))
    return 0


def _cmd_strip(args: argparse.Namespace) -> int:
    from . import paths, weights

    paths.ensure_dirs()
    src = Path(args.src) if args.src else paths.repo_dir(paths.BASE_REPO)
    dst = Path(args.out) if args.out else paths.BUILD_DIR / "base-text"
    if not (src / "model.safetensors").exists():
        print(f"no checkpoint at {src} - download {paths.BASE_REPO} there first")
        return 1
    report = weights.strip_text_only(src, dst, paths.CHAT_TEMPLATE)
    print(f"text-only checkpoint: {dst}")
    print(f"  {report['source_bytes'] / 1e6:,.0f} MB -> {report['text_only_bytes'] / 1e6:,.0f} MB "
          f"({report['tensors_dropped']} vision/audio tensors dropped, "
          f"{report['tensors_kept']} kept)")
    if args.verify:
        check = weights.verify_text_only(src, dst, args.device)
        print(f"  logits vs the multimodal model on {check['probes']} prompts: "
              f"max |diff| = {check['max_abs_logit_diff']:.6g}, "
              f"same next token = {check['same_next_token']} {check['next_tokens']}")
        report.update(check)
        if check["max_abs_logit_diff"] != 0:
            print("  WARNING: the text-only copy does not reproduce the original exactly")
    (dst / "strip_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    from . import bundle, paths

    paths.ensure_dirs()
    src = Path(args.src) if args.src else paths.repo_dir(paths.LITERT_REPO) / paths.LITERT_FILE
    out = Path(args.out) if args.out else paths.BUILD_DIR / (src.stem + "-text.litertlm")
    if not src.exists():
        print(f"no bundle at {src}")
        return 1
    work = paths.BUILD_DIR / (src.stem + "-unpacked")
    sections = bundle.text_only(src, out, work)
    print(bundle.render_sections(sections, src.stat().st_size, out.stat().st_size))
    print(f"text-only bundle: {out}")
    if not args.keep_work:
        bundle.cleanup(work)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from . import dataset

    return dataset.main(args)


def _cmd_train(args: argparse.Namespace) -> int:
    from . import train

    return train.main(args)


def _cmd_eval(args: argparse.Namespace) -> int:
    from . import evaluate

    return evaluate.main(args)


def _cmd_teacher(args: argparse.Namespace) -> int:
    from . import teacher

    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    return teacher.main(args)


def _cmd_export(args: argparse.Namespace) -> int:
    from . import export

    return export.main(args)


def _cmd_install(args: argparse.Namespace) -> int:
    from . import install

    return install.main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mustachio",
        description="Slim, train and evaluate the on-device model for feast mobile.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="size of each component of a checkpoint")
    p.add_argument("source", help="a local directory / .safetensors file, or owner/repo on "
                                  "Hugging Face (only the header is fetched)")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("strip", help="text-only copy of the training checkpoint")
    p.add_argument("--src", help="multimodal checkpoint directory (default: the QAT base)")
    p.add_argument("--out", help="output directory (default: build/base-text)")
    p.add_argument("--verify", action="store_true",
                   help="compare logits with the multimodal model (loads both, slow)")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_cmd_strip)

    p = sub.add_parser("bundle", help="text-only copy of a .litertlm bundle")
    p.add_argument("--src", help=".litertlm file (default: Google's Gemma 4 E2B bundle)")
    p.add_argument("--out", help="output .litertlm")
    p.add_argument("--keep-work", action="store_true", help="keep the unpacked sections")
    p.set_defaults(func=_cmd_bundle)

    p = sub.add_parser("data", help="build the training set and print its statistics")
    p.add_argument("--size", type=int, default=8000, help="training conversations")
    p.add_argument("--dev", type=int, default=400, help="in-distribution dev conversations")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--name", default="v1", help="dataset name under data/")
    p.add_argument("--out", help="output directory (default: data/<name>)")
    p.add_argument("--show", type=int, default=0, help="print N random conversations")
    p.set_defaults(func=_cmd_data)

    p = sub.add_parser("train", help="LoRA fine-tune on the generated data")
    p.add_argument("--data", help="dataset directory (default: data/v1)")
    p.add_argument("--base", help="text-only checkpoint (default: build/base-text)")
    p.add_argument("--run", default="v1", help="run name under runs/")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=16, help="most targets per step")
    p.add_argument("--batch-tokens", type=int, default=560,
                   help="most padded tokens per step (what fits a 12 GB card)")
    p.add_argument("--prompt", default="", help="prompt version (default: the newest, v1)")
    p.add_argument("--live-share", type=float, default=0.3,
                   help="share of call-in-history samples rendered as a live conversation")
    p.add_argument("--save-every", type=int, default=300,
                   help="save the adapter and score dev every N steps")
    p.add_argument("--max-len", type=int, default=640,
                   help="longest sample after the shared prefix, in tokens")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="score a model on the held-out suites")
    p.add_argument("--base", help="text-only checkpoint (default: build/base-text)")
    p.add_argument("--adapter", help="LoRA adapter directory (omit for the base model)")
    p.add_argument("--bundle", help="a .litertlm file: score the quantised model on LiteRT-LM")
    p.add_argument("--prompt", default="",
                   help="prompt version (default: the one the model was trained with)")
    p.add_argument("--guard", action="store_true",
                   help="score the answers after the app-side guard")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for --bundle")
    p.add_argument("--suites", default="",
                   help="comma-separated (default: all), or 'dev' for the dev split")
    p.add_argument("--data", help="dataset directory for --suites dev (default: data/v2)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", help="report path (default: runs/eval/<name>.json)")
    p.add_argument("--name", default="", help="label for the report")
    p.add_argument("--show", action="store_true", help="print every turn")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("teacher", help="a larger model writes phrasings (GPU, 24 GB)")
    p.add_argument("--tasks", default="blind", help="comma-separated: blind, smalltalk, paraphrase")
    p.add_argument("--model", default="google/gemma-4-12B-it")
    p.add_argument("--data", help="dataset to paraphrase (default: data/v2)")
    p.add_argument("--sample", type=int, default=4000, help="conversations offered for rewriting")
    p.add_argument("--per-example", type=int, default=2, help="rewrites asked per conversation")
    p.add_argument("--per-prompt", type=int, default=6, help="replies asked per small-talk line")
    p.add_argument("--only", default="", help="blind: redraft these turns only (card:turn,...)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="output directory (default: data/teacher)")
    p.set_defaults(func=_cmd_teacher)

    p = sub.add_parser("export", help="merge an adapter and convert it to .litertlm")
    p.add_argument("adapter", help="LoRA adapter directory (runs/<name>/adapter)")
    p.add_argument("--base", help="text-only checkpoint (default: build/base-text)")
    p.add_argument("--name", default="", help="names build/merged-<name> and build/litert-<name>")
    p.add_argument("--recipe", default="gemma4_mixed48_b32",
                   help="quantisation recipe: a name in assets/recipes/ or one litert-torch "
                        "knows")
    p.add_argument("--cache-length", type=int, default=4096,
                   help="KV cache the bundle is built for")
    p.add_argument("--converter", default="", help="path to litert-torch (Linux only)")
    p.add_argument("--no-convert", action="store_true", help="merge only, print the command")
    p.add_argument("--force", action="store_true", help="merge again even if it exists")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("install", help="register a .litertlm bundle with the feast AI TUI")
    p.add_argument("bundle", nargs="?", help="bundle path (default: build/litert-final)")
    p.add_argument("--id", default="", help="model id in the catalog (default: mustachio-e2b)")
    p.add_argument("--label", default="", help="name shown in the model list")
    p.add_argument("--precision", default="", help="quantisation label if the bundle has none")
    p.add_argument("--activate", action="store_true",
                   help="also set agent mode 'native' and make it the last-used model")
    p.set_defaults(func=_cmd_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
