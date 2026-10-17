"""The Cosmos ModernBERT-TR encoder family (YTU COSMOS group).

These are not chat models.  They are 149M-parameter Turkish encoders (22
layers, hidden 768, 8,192-token context, cased WordPiece vocabulary of 50,008)
that read a sentence and return a vector, a label or per-token tags in a single
forward pass - no decoding loop.  That makes them cheap enough to run *before*
the SLM on every message, which is what the intent router does.

Every repo below was verified against the live Hugging Face API in October
2026: public, Apache-2.0, with int8 ONNX exports (``onnx/model_quantized.onnx``,
~150 MB) beside the 595 MB fp32 weights.  Benchmark figures are quoted from
the family's write-up (cosmos-ytu.github.io/modernbert-tr).

Why the embedding model routes, not the intent classifier
---------------------------------------------------------
``modernbert-tr-massive-intent`` looks like the obvious router, and it is fast
(~7 ms per query here, 4 cores, int8).  But its 60 classes are MASSIVE's voice
assistant taxonomy - alarms, IoT lights, music - with no notion of saved
items, posts, friends or an active search location.  Measured on feast
queries it is *confidently* wrong where it matters: "Ayarlari ac" ->
``music_settings`` (0.65), "Uskudar'a tasi konumu" -> ``transport_query``
(0.70), "Bana gonderilen restoranlar" -> ``takeaway_order``; restaurant search,
menu search and saved-item search all collapse into
``recommendation_locations``.  Using it would mean retraining.

``modernbert-tr-embed`` needs no training: embed a few example utterances per
feast intent once, then classify a query by nearest neighbour.  With 3-4
exemplars per intent it routed 15/16 held-out feast queries correctly
(~20 ms/query, 4 cores).  The query instruction prefix matters - without it
the same probe scored 10/16, because it spreads the similarity distribution
and widens the margin between intents.  Applying it to the exemplars too
(symmetric) beat the asymmetric query/document split, 15/16 vs 13/16.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paths import ENCODERS_DIR

HF_ENDPOINT = "https://huggingface.co"

#: The instruction the embedding model was trained with for queries, verbatim
#: from its ``config_sentence_transformers.json``.
EMBED_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)

_TOKENIZER = ("tokenizer.json", "tokenizer_config.json", "config.json")


@dataclass(frozen=True)
class EncoderSpec:
    """One encoder in the family."""

    id: str
    name: str
    task: str  # "embed" | "classify" | "token-classify" | "rerank" | "fill-mask"
    purpose: str
    reported: str
    #: How feast AI uses it: "router", or "candidate" with the reason it is
    #: not wired yet.
    role: str
    role_note: str
    #: ``(path in repo, local relative path)``.  The ONNX file is flattened to
    #: the model directory so the worker always finds ``model_quantized.onnx``.
    files: tuple[tuple[str, str], ...]
    approx_mb: int = 150
    params: str = "149M"
    license: str = "Apache-2.0"
    onnx_file: str = "model_quantized.onnx"
    query_prefix: str = ""

    @property
    def repo(self) -> str:
        return f"ytu-ce-cosmos/{self.id}"

    @property
    def page_url(self) -> str:
        return f"{HF_ENDPOINT}/{self.repo}"

    @property
    def local_path(self):
        return ENCODERS_DIR / self.id

    @property
    def downloaded(self) -> bool:
        return all((self.local_path / rel).exists() for _, rel in self.files)

    @property
    def local_size(self) -> int:
        return sum(
            (self.local_path / rel).stat().st_size
            for _, rel in self.files
            if (self.local_path / rel).exists()
        )

    @property
    def urls(self) -> list[tuple[str, str]]:
        return [
            (f"{HF_ENDPOINT}/{self.repo}/resolve/main/{src}?download=true", rel)
            for src, rel in self.files
        ]


def _files(*extra: str) -> tuple[tuple[str, str], ...]:
    onnx = (("onnx/model_quantized.onnx", "model_quantized.onnx"),)
    return onnx + tuple((name, name) for name in _TOKENIZER + extra)


ENCODERS: list[EncoderSpec] = [
    EncoderSpec(
        id="modernbert-tr-embed",
        name="ModernBERT-TR Embed",
        task="embed",
        purpose="Dense retrieval / sentence embeddings. 768-d, mean pooling, "
                "L2-normalised; queries take a fixed instruction prefix.",
        reported="MTEB(Turkish) mean 68.14 (Turkish E5 Large 67.17, 8B teacher "
                 "68.59). Retrieval 59.42, Classification 76.51, STS 77.62.",
        role="router",
        role_note="The intent router: nearest-neighbour over feast exemplars. "
                  "Training-free, so new intents are a list of sentences.",
        files=_files("config_sentence_transformers.json", "special_tokens_map.json",
                     "1_Pooling/config.json"),
        query_prefix=EMBED_QUERY_PREFIX,
    ),
    EncoderSpec(
        id="modernbert-tr-massive-intent",
        name="ModernBERT-TR MASSIVE Intent",
        task="classify",
        purpose="Intent classification, 60 MASSIVE classes.",
        reported="Intent accuracy 88.21 +/- 0.16, macro-F1 85.43 +/- 0.14 "
                 "(five seeds).",
        role="candidate",
        role_note="Measured, not wired: MASSIVE's taxonomy has no saved items, "
                  "posts or active location, and it is confidently wrong on "
                  "feast queries. Good only as a coarse out-of-domain signal "
                  "(weather 0.89, datetime 0.90, greet 0.89 were right).",
        files=_files(),
    ),
    EncoderSpec(
        id="modernbert-tr-massive-slot",
        name="ModernBERT-TR MASSIVE Slot",
        task="token-classify",
        purpose="Slot filling: 111 BIO tags over 55 slot types (place_name, "
                "person, food_type, time, date, meal_type, ...).",
        reported="Slot entity-F1 75.27 +/- 0.31, frame exact-match "
                 "67.21 +/- 0.34.",
        role="candidate",
        role_note="Next step: extract place_name / person / food_type so more "
                  "intents can be served as direct tool calls without the SLM.",
        files=_files(),
    ),
    EncoderSpec(
        id="modernbert-tr-guardrail",
        name="ModernBERT-TR Guardrail",
        task="classify",
        purpose="Safety of Turkish user prompts: 1 overall unsafe score + 10 "
                "hazard categories (sigmoid, no default thresholds).",
        reported="Binary F1 93.0, macro AUPRC 91.7, weighted F1 88.6.",
        role="candidate",
        role_note="Pre-filter in front of send_message_to_friends - its "
                  "INJECTION_JAILBREAK head matters once the agent can act.",
        files=_files(),
    ),
    EncoderSpec(
        id="modernbert-tr-reranker",
        name="ModernBERT-TR Reranker",
        task="rerank",
        purpose="Cross-encoder reranking of (query, document) pairs.",
        reported="+8.38 mean NDCG@10 over the embedder's top 100 across six "
                 "tasks (SquadTR 75.93 -> 90.12).",
        role="candidate",
        role_note="On-device ranking for search_posts / search_chat_memory "
                  "over cached captions.",
        files=_files("special_tokens_map.json"),
    ),
    EncoderSpec(
        id="modernbert-tr-absa-polarity",
        name="ModernBERT-TR ABSA Polarity",
        task="classify",
        purpose="Aspect polarity: (sentence, aspect) -> negative/neutral/positive.",
        reported="Macro-F1 72.74 +/- 2.10 (five seeds); neutral F1 ~36.",
        role="candidate",
        role_note="Could answer 'is the service good here?' from post captions "
                  "without the SLM reading them all.",
        files=_files(),
    ),
    EncoderSpec(
        id="modernbert-tr-absa-extraction",
        name="ModernBERT-TR ABSA Extraction",
        task="token-classify",
        purpose="Aspect-term extraction (BIO: B-ASP, I-ASP).",
        reported="Exact span F1 64.09 +/- 0.30; 64.48 ABSA-TR, 69.46 M-ABSA.",
        role="candidate",
        role_note="Pairs with polarity to summarise what posts praise or "
                  "complain about per venue.",
        files=_files(),
    ),
    EncoderSpec(
        id="modernbert-tr-pii-ner",
        name="ModernBERT-TR PII NER",
        task="token-classify",
        purpose="PII recognition: 25 KVKK-oriented entity types, 51 BIO tags.",
        reported="F2 80.85 +/- 0.19; precision 72.1, recall 83.4.",
        role="candidate",
        role_note="Scrub chat memory and logs before anything leaves the "
                  "phone (KVKK).",
        files=_files("special_tokens_map.json"),
    ),
    EncoderSpec(
        id="modernbert-tr-base",
        name="ModernBERT-TR Base",
        task="fill-mask",
        purpose="Base masked-LM encoder the rest are fine-tuned from.",
        reported="Above BERTurk and TabiBERT on TabiBench; mmBERT leads.",
        role="candidate",
        role_note="Starting point for a feast-specific intent head once real "
                  "app queries exist to train on.",
        files=_files("special_tokens_map.json"),
    ),
]

BY_ID: dict[str, EncoderSpec] = {e.id: e for e in ENCODERS}

#: Encoders the router can actually run on.
ROUTER_MODELS = [e.id for e in ENCODERS if e.role == "router"]


def get(encoder_id: str) -> EncoderSpec | None:
    return BY_ID.get(encoder_id)
