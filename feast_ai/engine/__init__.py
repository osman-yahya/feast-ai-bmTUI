"""CPU inference engines: llama.cpp, ONNX Runtime GenAI and LiteRT-LM.

Every backend is driven through the same OpenAI-compatible HTTP surface and
the same :class:`~feast_ai.engine.server.BaseServer` supervision, so a model
run on any of them is measured identically.  :func:`make_server` is the only
place that has to know which is which.
"""

from .binaries import RuntimeInfo, ensure_runtime, find_server, runtime_info
from .client import ChatClient, StreamEvent
from .litert_server import LiteRtServer
from .litert_server import available as litert_available
from .litert_server import version as litert_version
from .onnx_server import OnnxServer
from .onnx_server import available as onnx_available
from .onnx_server import version as onnx_version
from .server import BaseServer, EngineError, LlamaServer

#: ``id -> (label, description)`` for the engines the UI can offer.
ENGINES = {
    "llama.cpp": (
        "llama.cpp",
        "GGUF weights, hand-written CPU kernels. The default.",
    ),
    "onnx": (
        "ONNX Runtime GenAI",
        "ONNX weights, Microsoft's edge runtime. Needs onnxruntime-genai.",
    ),
    "litert": (
        "LiteRT-LM",
        "A .litertlm bundle on Google's edge runtime - what the phone runs. Needs litert-lm.",
    ),
}


def make_server(engine: str, settings, server_bin=None):
    """Build the backend a model asks for.

    ``engine`` comes from :attr:`feast_ai.catalog.ModelSpec.engine`, so the
    model itself decides - there is no global switch to get out of sync with
    the weights actually loaded.
    """
    if engine == "onnx":
        return OnnxServer(settings)
    if engine == "litert":
        return LiteRtServer(settings)
    if server_bin is None:
        raise EngineError("llama.cpp runtime is not installed yet.")
    return LlamaServer(server_bin, settings)


__all__ = [
    "RuntimeInfo",
    "ensure_runtime",
    "find_server",
    "runtime_info",
    "ChatClient",
    "StreamEvent",
    "BaseServer",
    "EngineError",
    "LlamaServer",
    "OnnxServer",
    "onnx_available",
    "onnx_version",
    "LiteRtServer",
    "litert_available",
    "litert_version",
    "ENGINES",
    "make_server",
]
