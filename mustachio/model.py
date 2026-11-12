"""Load and run the text-only Gemma 4 E2B on one 12 GB GPU.

Two facts about this model shape everything here.

**The per-layer embedding table stays in system RAM.**  In bf16 the text
model is 9.3 GB, of which 4.7 GB is that table: one 8,960-wide row per
token, read by lookup and never multiplied.  With it on the CPU the card
holds 4.6 GB of weights (measured peak while decoding: 5.2 GB on an RTX
5070); the rows looked up for a whole prompt are a few tens of MB.

**Logits are only ever computed where they are needed.**  The output layer
is 262,144 wide, so logits at every position of the 2,100-token tool prompt
would be 550 M values - 1.1 GB in bf16, before the soft-cap copies them and
before gradients.  :class:`Runner` asks for the last position only
(``logits_to_keep=1``) and prefills the static prompt once, then replays
its KV cache for every question - which is also what the phone does
(measured here: 2,087 tokens prefilled in 0.6 s, then ~20 tokens/s greedy).
Training does the same from the other side: the loss gathers the supervised
positions first and runs the output layer on those alone
(:func:`supervised_logits`).
"""

from __future__ import annotations

import copy
from pathlib import Path

PLE = "embed_tokens_per_layer"
#: From this much GPU memory on, the whole 9.3 GB model lives on the card.
ROOMY = 20e9
STOP_TOKENS = ("<turn|>", "<|tool_response>")


def place(model, gpu: str = "cuda") -> None:
    """Move everything but the per-layer embedding table to the GPU.

    ``device_map={"...embed_tokens_per_layer": "cpu"}`` looks like the same
    thing and is not: accelerate *executes* an offloaded module on the main
    device, so it copied the 4.7 GB table to the card and back on every
    forward - measured at 0.7 s per generated token and a 9.35 GB peak.  Two
    hooks do what was meant: ids go to the CPU for the lookup, the rows that
    were looked up go to the GPU.
    """
    text = model.model
    for name, child in text.named_children():
        if name != PLE:
            child.to(gpu)
    for param in text._parameters.values():
        if param is not None:
            param.data = param.data.to(gpu)
    for name, buffer in list(text._buffers.items()):
        if buffer is not None:
            text._buffers[name] = buffer.to(gpu)
    model.lm_head.to(gpu)
    table = getattr(text, PLE)
    table.register_forward_pre_hook(lambda _m, args: (args[0].to("cpu"), *args[1:]))
    table.register_forward_hook(lambda _m, _args, out: out.to(gpu))


def load(base: Path, adapter: Path | None = None):
    """``(model, tokenizer)`` in bf16; a LoRA adapter is applied if given."""
    import torch
    from transformers import AutoTokenizer, Gemma4ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = Gemma4ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).total_memory >= ROOMY:
            model.to("cuda")  # a 24 GB+ card holds the table too
        else:
            place(model)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def causal(model):
    """The ``Gemma4ForCausalLM`` under an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def main_device(model):
    """Where input ids go: the device of the token embedding."""
    return causal(model).get_input_embeddings().weight.device


def supervised_logits(model, input_ids, attention_mask, positions):
    """Logits at ``positions`` (a boolean mask over ``input_ids``) and nowhere else.

    Returns ``[n_supervised, vocab]`` in float32, soft-capped the way the
    model's own forward does it.
    """
    import torch

    core = causal(model)
    hidden = core.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    logits = core.lm_head(hidden[positions]).float()
    cap = core.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


class Runner:
    """Greedy decoding with a prefilled, reusable static prefix."""

    def __init__(self, model, tokenizer) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.device = main_device(model)
        self.stops = {tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS}
        self.stops.add(tokenizer.eos_token_id)
        self._prefix = ""
        self._prefix_cache = None
        self.prefix_tokens = 0

    def _ids(self, text: str):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    def set_prefix(self, text: str) -> int:
        """Prefill ``text`` once; later prompts that start with it reuse the cache."""
        import torch

        ids = self._ids(text)
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        self._prefix, self._prefix_cache = text, out.past_key_values
        self.prefix_tokens = int(ids.shape[1])
        return self.prefix_tokens

    def complete(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, int, int]:
        """``(raw reply with special tokens, prompt tokens fed now, new tokens)``."""
        import torch

        if self._prefix and prompt.startswith(self._prefix):
            past = copy.deepcopy(self._prefix_cache)
            ids = self._ids(prompt[len(self._prefix):])
        else:
            past, ids = None, self._ids(prompt)
        fed = int(ids.shape[1])
        new: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(input_ids=ids, past_key_values=past, use_cache=True,
                                 logits_to_keep=1)
                past = out.past_key_values
                token = int(out.logits[0, -1].argmax())
                new.append(token)
                if token in self.stops:
                    break
                ids = torch.tensor([[token]], device=self.device)
        return self.tokenizer.decode(new, skip_special_tokens=False), fed, len(new)
