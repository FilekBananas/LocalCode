"""Local image generation (text-to-image) for LocalCode.

Two backends, mirroring the GGUF/MLX split of the language side (see
``docs/IMAGES.md``):

============  ==========================================  ==========================
format        runtime                                     models
============  ==========================================  ==========================
``mlx``       ``mflux`` >= 0.20 (Apple Silicon)            FLUX.1/FLUX.2, Qwen-Image
``gguf``      ``stable-diffusion.cpp`` (Metal) via         SD 1.5, SDXL, SD 3.5,
              ``stable_diffusion_cpp`` >= 0.4              FLUX GGUF
============  ==========================================  ==========================

Both run in their own **worker subprocess** (``python -m
localcode.images.mflux_worker`` / ``…sdcpp_worker``) that speaks a tiny HTTP
protocol, exactly like the language engines do with llama-server: unloading a
model then reliably returns its memory to the OS and a crash in a backend
cannot take the app down.

Public API
----------
``base``      :class:`ImageParams`, :class:`ImageProgress`, :class:`ImageResult`,
              :class:`ImageEngine` (ABC), :class:`ImageEngineError`
``runtime``   :class:`~localcode.images.runtime.ImageRuntime` — loaded image
              models, the shared RAM budget, serialised generation, progress
              events and instance DTOs for ``/api/runtime``
``memory``    memory estimates for image models (weights + a measured floor)

The heavy third-party packages are optional (``pip install localcode[images]``);
nothing in this package imports them at module import time, so the rest of
LocalCode keeps working without them and the user gets a Polish explanation
when an image model is actually requested.
"""

from __future__ import annotations

from localcode.images.base import (
    IMAGE_FAMILIES,
    ImageEngine,
    ImageEngineError,
    ImageEngineStartError,
    ImageEngineUnavailableError,
    ImageGenerationCancelled,
    ImageParams,
    ImageProgress,
    ImageResult,
)

__all__ = [
    "IMAGE_FAMILIES",
    "ImageEngine",
    "ImageEngineError",
    "ImageEngineStartError",
    "ImageEngineUnavailableError",
    "ImageGenerationCancelled",
    "ImageParams",
    "ImageProgress",
    "ImageResult",
    "ImageRuntime",
]


def __getattr__(name: str) -> object:
    # Lazy so that importing the package never pulls in aiohttp/mflux/sdcpp.
    if name == "ImageRuntime":
        from localcode.images.runtime import ImageRuntime

        return ImageRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
