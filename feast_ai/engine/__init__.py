"""CPU inference engines: llama.cpp and ONNX Runtime GenAI.

Both backends are driven through the same OpenAI-compatible HTTP surface and
the same :class:`~feast_ai.engine.server.BaseServer` supervision, so a model
run on either one is measured identically.  :func:`make_server` is the only
place that has to know which is which.
"""

from .binaries import RuntimeInfo, ensure_runtime, find_server, runtime_info
from .client import ChatClient, StreamEvent
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
}


def make_server(engine: str, settings, server_bin=None):
    """Build the backend a model asks for.

    ``engine`` comes from :attr:`feast_ai.catalog.ModelSpec.engine`, so the
    model itself decides - there is no global switch to get out of sync with
    the weights actually loaded.
    """
    if engine == "onnx":
        return OnnxServer(settings)
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
    "ENGINES",
    "make_server",
]
