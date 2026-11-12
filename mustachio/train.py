"""LoRA fine-tuning of the text-only Gemma 4 E2B on one 12 GB GPU.

Every training sequence starts with the same 2,087 tokens - the system text
and the 15 tool declarations - followed by 40-400 tokens that differ.  A
plain trainer would push those 2,087 tokens through forward and backward for
every example: 9,600 targets x 2,200 tokens, about 21 M tokens an epoch, to
learn from the last few hundred of each.

So the prefix is run **once per step**, without gradient, and its KV cache
is shared by the whole batch (:func:`prefix_cache`).  The batch then costs
its own tokens only.  The forward pass is exact - the cache is recomputed
with the current LoRA weights every step, and :func:`check_cache_exact`
compares it with a full-sequence forward before training starts.  What is
approximated is the gradient: it flows through the tokens of each example
but not through the prefix's keys and values.  That is the same thing the
phone does at inference time - it prefills the tool block once and decodes
against it - so nothing is trained that will not be used.

Three more choices, each for a measured or structural reason:

* **one sample per supervised turn**, not one per conversation.  The
  template renders a past call differently (``...<tool_call|><turn|>``) from
  how the model must *generate* it (``...<tool_call|><|tool_response>``), so
  a turn cannot be both history and target in one sequence;
* **loss only on what the model writes** - calls and short protocol replies.
  Result lists in a history are the app's text and are masked, or the model
  would learn to invent venues;
* **logits only at supervised positions** (:func:`mustachio.model.
  supervised_logits` explains the 262,144-wide output layer).

LoRA follows Google's own LiteRT-LM example for this model (rank 16, alpha
32, all attention and MLP projections of the text decoder), so the merged
weights are the kind of checkpoint its exporter is known to take.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from . import dataset, paths, schema, template

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class Sample:
    ids: list[int]      # everything after the shared prefix: turns, then the target
    n_context: int      # how many of them are context (not supervised)
    family: str
    register: str


def conversation_messages(example: dict, upto: int, prompt: schema.Prompt) -> list[dict]:
    """The system turn plus ``example``'s first ``upto`` messages, as the template wants them."""
    ctx = schema.Ctx(**example["ctx"])
    out = [prompt.system_message()]
    for message in example["messages"][:upto]:
        if message["role"] == "user":
            out.append({"role": "user", "content": schema.user_content(message["content"], ctx)})
        else:
            out.append({k: v for k, v in message.items() if k != "supervise"})
    return out


def build_samples(examples: list[dict], tokenizer, prefix: str, tools: list[dict],
                  max_len: int, prompt: schema.Prompt,
                  live_share: float = 0.0, seed: int = 0) -> tuple[list[Sample], int]:
    """One sample per supervised turn.

    ``live_share`` of the samples whose history holds a call are rendered the
    way a *live* LiteRT-LM conversation holds it (see
    :func:`mustachio.template.live_history`), the rest the way a replayed
    history renders - so the model is at home in both.
    """
    rng = random.Random(seed)
    samples, skipped = [], 0
    for example in examples:
        for index, message in enumerate(example["messages"]):
            if message["role"] != "assistant" or not message.get("supervise"):
                continue
            text = template.render(conversation_messages(example, index, prompt), tools)
            if not text.startswith(prefix):
                raise ValueError("a prompt does not start with the static prefix")
            rest = text[len(prefix):]
            if template.CALL_CLOSE in rest and rng.random() < live_share:
                rest = template.live_history(rest)
            context = tokenizer(rest, add_special_tokens=False)["input_ids"]
            target = tokenizer(template.target_text(message), add_special_tokens=False)["input_ids"]
            if len(context) + len(target) > max_len:
                skipped += 1
                continue
            samples.append(Sample(context + target, len(context), example["family"],
                                  example["register"]))
    return samples, skipped


def batches(samples: list[Sample], size: int, max_tokens: int,
            rng: random.Random) -> list[list[Sample]]:
    """Shuffled batches of similar length, capped by padded tokens.

    The cap is what fits the card, not the count: every token of a sample
    attends to the 2,087-token prefix in all 35 layers, and with the MLP
    activations kept for the backward pass that measured ~7 MB of GPU memory
    per padded token (16 short samples, 1,056 tokens: 12.7 GB peak on a
    12 GB card, i.e. swapping).  560 tokens a step peaks near 9 GB.
    """
    order = list(samples)
    rng.shuffle(order)
    out: list[list[Sample]] = []
    for start in range(0, len(order), size * 50):
        chunk = sorted(order[start:start + size * 50], key=lambda s: len(s.ids))
        batch: list[Sample] = []
        for sample in chunk:  # ascending, so the newest sample is the longest
            if batch and (len(batch) >= size or (len(batch) + 1) * len(sample.ids) > max_tokens):
                out.append(batch)
                batch = []
            batch.append(sample)
        if batch:
            out.append(batch)
    rng.shuffle(out)
    return out


def collate(batch: list[Sample], pad: int, device):
    import torch

    width = max(len(s.ids) for s in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, sample in enumerate(batch):
        ids[row, :len(sample.ids)] = torch.tensor(sample.ids)
        labels[row, sample.n_context:len(sample.ids)] = torch.tensor(sample.ids[sample.n_context:])
    return ids.to(device), labels.to(device)


def prefix_cache(model, prefix_ids):
    """The KV cache of the static prompt under the current weights (no gradient)."""
    import torch

    from .model import causal

    core = causal(model)
    was_training = model.training
    model.eval()  # the phone prefills without dropout
    with torch.no_grad():
        out = core.model(input_ids=prefix_ids, use_cache=True)
    model.train(was_training)
    return out.past_key_values


def forward_loss(model, cache, ids, labels):
    """Mean cross-entropy over supervised tokens; also per-sequence correctness."""
    import copy

    import torch
    import torch.nn.functional as F

    from .model import causal

    core = causal(model)
    past = copy.deepcopy(cache)
    past.batch_repeat_interleave(ids.shape[0])
    hidden = core.model(input_ids=ids, past_key_values=past, use_cache=True).last_hidden_state
    # Position t predicts token t+1: shift, then keep supervised targets only.
    keep = labels[:, 1:] != -100
    logits = core.lm_head(hidden[:, :-1][keep]).float()
    cap = core.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    gold = labels[:, 1:][keep]
    loss = F.cross_entropy(logits, gold)
    with torch.no_grad():
        wrong = torch.zeros_like(keep, dtype=torch.long)
        wrong[keep] = (logits.argmax(-1) != gold).long()
        sequence_ok = wrong.sum(1) == 0
    return loss, sequence_ok


def check_cache_exact(model, tokenizer, prefix: str, samples: list[Sample]) -> float:
    """Largest loss difference between the cached and the full-sequence forward."""
    import torch

    from .model import main_device, supervised_logits

    device = main_device(model)
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prefix_ids = prefix_ids.to(device)
    cache = prefix_cache(model, prefix_ids)
    worst = 0.0
    model.eval()
    with torch.no_grad():
        for sample in samples:
            ids, labels = collate([sample], tokenizer.pad_token_id, device)
            cached, _ = forward_loss(model, cache, ids, labels)
            full_ids = torch.cat([prefix_ids, ids], dim=1)
            mask = torch.zeros_like(full_ids, dtype=torch.bool)
            start = prefix_ids.shape[1] + sample.n_context
            mask[0, start - 1:full_ids.shape[1] - 1] = True
            logits = supervised_logits(model, full_ids, torch.ones_like(full_ids), mask)
            full = torch.nn.functional.cross_entropy(logits, full_ids[0, start:])
            worst = max(worst, abs(float(cached) - float(full)))
    return worst


def evaluate_dev(model, tokenizer, prefix_ids, samples: list[Sample], batch: int) -> dict:
    """Teacher-forced dev loss, and the share of targets predicted token for token."""
    import torch

    from .model import main_device

    device = main_device(model)
    model.eval()
    cache = prefix_cache(model, prefix_ids)
    losses, right, total = [], 0, 0
    by_register: dict = {}
    ordered = sorted(samples, key=lambda s: len(s.ids))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        for start in range(0, len(ordered), batch):
            part = ordered[start:start + batch]
            ids, labels = collate(part, tokenizer.pad_token_id, device)
            loss, ok = forward_loss(model, cache, ids, labels)
            losses.append(float(loss) * len(part))
            for sample, good in zip(part, ok.tolist()):
                hit = by_register.setdefault(sample.register, [0, 0])
                hit[0] += int(good)
                hit[1] += 1
            right += int(ok.sum())
            total += len(part)
    return {"loss": round(sum(losses) / max(total, 1), 4),
            "target_exact_pct": round(100 * right / max(total, 1), 1),
            "by_register": {k: round(100 * a / b, 1) for k, (a, b) in sorted(by_register.items())},
            "targets": total}


def main(args) -> int:
    import torch
    from peft import LoraConfig, get_peft_model

    from . import model as M

    paths.ensure_dirs()
    data = Path(args.data) if args.data else paths.DATA_DIR / "v1"
    base = Path(args.base) if args.base else paths.BUILD_DIR / "base-text"
    run = paths.RUNS_DIR / args.run
    run.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = M.load(base)
    device = M.main_device(model)
    prompt = schema.prompt(args.prompt)
    tools = prompt.declarations()
    prefix = template.render([prompt.system_message()], tools, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prefix_ids = prefix_ids.to(device)

    train, skipped = build_samples(dataset.read(data / "train.jsonl"), tokenizer, prefix, tools,
                                   args.max_len, prompt, args.live_share, args.seed)
    dev, _ = build_samples(dataset.read(data / "dev.jsonl"), tokenizer, prefix, tools,
                           args.max_len, prompt)
    lengths = sorted(len(s.ids) for s in train)
    print(f"prompt {prompt.name}: {prefix_ids.shape[1]} static tokens")
    print(f"samples: {len(train)} train, {len(dev)} dev ({skipped} over {args.max_len} tokens "
          f"skipped); prefix {prefix_ids.shape[1]} tokens; per-sample tokens median "
          f"{lengths[len(lengths) // 2]}, p95 {lengths[int(len(lengths) * 0.95)]}, "
          f"max {lengths[-1]}")

    gap = check_cache_exact(model, tokenizer, prefix, rng.sample(train, 4))
    print(f"cached vs full-sequence forward: max loss difference {gap:.4f}")
    if gap > 0.05:
        print("the shared-prefix forward does not reproduce the full forward - not training")
        return 1

    config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                        target_modules=list(TARGETS), task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA rank {args.rank}: {trainable / 1e6:.1f} M trainable parameters")

    before = evaluate_dev(model, tokenizer, prefix_ids, dev, 8)
    print(f"dev before training: {before}")

    steps_per_epoch = len(batches(train, args.batch, args.batch_tokens, random.Random(0)))
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * 0.05))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    log: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    started, step, window = time.perf_counter(), 0, []
    model.train()
    print(f"{total_steps} steps ({steps_per_epoch} per epoch, at most {args.batch} targets "
          f"or {args.batch_tokens} padded tokens each)", flush=True)
    while step < total_steps:
        for batch in batches(train, args.batch, args.batch_tokens, rng):
            if step >= total_steps:
                break
            cache = prefix_cache(model, prefix_ids)
            ids, labels = collate(batch, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ok = forward_loss(model, cache, ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window.append(float(loss.detach()))
            step += 1
            if step % 25 == 0 or step == total_steps:
                elapsed = time.perf_counter() - started
                entry = {"step": step, "loss": round(sum(window) / len(window), 4),
                         "lr": round(scheduler.get_last_lr()[0], 7),
                         "seconds": round(elapsed, 1)}
                log.append(entry)
                window = []
                print(f"step {step}/{total_steps}  loss {entry['loss']:.4f}  "
                      f"{elapsed / step:.2f} s/step  "
                      f"VRAM peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if step % args.save_every == 0 and step < total_steps:
                # A long run on a desktop GPU gets interrupted: keep what it has learned.
                model.save_pretrained(run / "adapter")
                (run / "adapter" / "prompt.json").write_text(
                    json.dumps({"prompt": prompt.name}), encoding="utf-8")
                checkpoint = evaluate_dev(model, tokenizer, prefix_ids, dev, 8)
                log.append({"step": step, "dev": checkpoint})
                print(f"dev after step {step}: {checkpoint}", flush=True)
                model.train()

    seconds = time.perf_counter() - started
    after = evaluate_dev(model, tokenizer, prefix_ids, dev, 8)
    print(f"dev after training: {after}")
    model.save_pretrained(run / "adapter")
    # The adapter only means something with the prompt it was trained on.
    (run / "adapter" / "prompt.json").write_text(json.dumps({"prompt": prompt.name}),
                                                 encoding="utf-8")
    report = {
        "run": args.run, "base": str(base), "data": str(data), "seed": args.seed,
        "prompt": prompt.name, "live_share": args.live_share,
        "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.05,
                 "targets": list(TARGETS), "trainable_parameters": trainable},
        "batch": args.batch, "batch_tokens": args.batch_tokens, "lr": args.lr, "steps": step,
        "epochs": args.epochs,
        "train_samples": len(train), "dev_samples": len(dev),
        "prefix_tokens": int(prefix_ids.shape[1]), "cache_check_max_loss_diff": gap,
        "seconds": round(seconds, 1), "seconds_per_step": round(seconds / max(step, 1), 3),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "dev_before": before, "dev_after": after, "log": log,
    }
    (run / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"adapter: {run / 'adapter'}  ({seconds / 60:.1f} min, "
          f"VRAM peak {report['vram_peak_gb']} GB)")
    return 0
