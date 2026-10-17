"""The feast mobile agent: tools, prompts, the intent router and its benchmark.

Two pipelines are compared (``Settings.agent_mode``):

``direct``        the SLM sees every tool and decides alone;
``hierarchical``  a ModernBERT-TR encoder routes the message first, then the
                  SLM sees only the relevant tools - or is skipped entirely.

``off`` keeps feast AI's plain chat behaviour, so earlier benchmarks still
mean what they meant.
"""

from __future__ import annotations

from ..config import Settings
from ..engine.memcap import MB

#: Never hand the router more than this share of a RAM ceiling: the model is
#: the thing being measured, and must keep most of the budget.
MAX_ROUTER_SHARE = 0.5


def router_slice_bytes(settings: Settings, share_mb: int = 0) -> int:
    """The router's share of the RAM ceiling (0 when there is no ceiling)."""
    total = settings.ram_limit_bytes
    if not total:
        return 0
    return min((share_mb or settings.router_ram_mb) * MB, int(total * MAX_ROUTER_SHARE))


async def start_router(settings: Settings, slm_server=None, on_status=None,
                       taxonomy: str = "intents"):
    """Launch the encoder worker and return ``(EncoderServer, Router)``.

    Under a RAM ceiling both processes live on the same simulated phone, so
    they split one budget: the router is capped at its slice and a running
    SLM is re-capped to what is left.  Without that the pair could hold up to
    twice the device's memory - exactly the kind of quiet overrun the harness
    exists to prevent.  The SLM is re-capped live; no reload.

    ``taxonomy="classes"`` routes to the 13 action classes of
    :mod:`feast_ai.agent.classes` instead of the 19 intents.
    """
    from .. import encoders
    from ..engine import decision_server as ds
    from ..engine.encoder_server import EncoderServer
    from ..engine.server import EngineError
    from .router import Router

    if taxonomy == "classes":
        from .classes import BY_ID as labels, exemplar_bank
    else:
        from .intents import exemplar_bank
        labels = None

    if ds.is_decision(settings.router_model):
        # Decision 2.0 (vLLM Semantic Router): a 0.6B decoder with a decision
        # head, on its own llama.cpp build. Same pinning, a larger RAM share.
        from .decision import DecisionRouter

        server = ds.DecisionServer(settings, settings.router_model)
        share = router_slice_bytes(settings, ds.DECISION_RAM_MB)
        if share:
            server.ram_limit_override = share
        await server.start(ds.model_path(settings.router_model), alias=settings.router_model,
                           on_status=on_status)
        if share and slm_server is not None:
            slm_server.set_ram_limit(settings.ram_limit_bytes - share)
        router = DecisionRouter(server.base_url, settings, labels)
        await router.warmup()
        return server, router

    spec = encoders.get(settings.router_model)
    if spec is None or spec.role != "router":
        raise EngineError(f"'{settings.router_model}' is not a router model.")
    server = EncoderServer(settings, spec, exemplar_bank())
    share = router_slice_bytes(settings)
    if share:
        server.ram_limit_override = share
    await server.start(spec.local_path, alias=spec.id, on_status=on_status)
    if share and slm_server is not None:
        slm_server.set_ram_limit(settings.ram_limit_bytes - share)
    router = Router(server.base_url, settings, labels)
    await router.warmup()
    return server, router
