"""Video: turning a text prompt into a short clip.

LocalCode targets one model today — **LTX-Video 2B** — run through
``diffusers`` on PyTorch's MPS backend (there is no small MLX engine for it;
see ``docs/VIDEO.md``). Generation is slow (minutes per second of footage on
32 GB Apple Silicon) and runs in its own worker subprocess
(``python -m localcode.video.worker``) so progress can be reported per step
and a crash cannot take the server down with it — the same reasoning as the
language and image engines, not the one-shot-in-a-thread shape of
:mod:`localcode.speech`.

The optional dependencies live in the ``video`` extra
(``pip install "localcode[video]"``); without them every entry point explains
what to install instead of failing with an ImportError.
"""

from __future__ import annotations

from localcode.video.base import (
    VideoError,
    VideoParams,
    VideoProgress,
    VideoResult,
)
from localcode.video.runtime import VideoRuntime

__all__ = [
    "VideoError",
    "VideoParams",
    "VideoProgress",
    "VideoResult",
    "VideoRuntime",
]
