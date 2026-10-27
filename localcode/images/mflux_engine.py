"""Engine for MLX diffusion models: drives ``localcode.images.mflux_worker``.

``MfluxEngine`` is the client half of the worker protocol described in
``images/base.py``; everything model-specific lives in the worker.  The engine
only knows how to turn a :class:`~localcode.hub.types.LocalModel` into a command
line:

* ``--model`` is the model directory on disk (mflux accepts a local path and
  never touches the Hugging Face cache for it);
* ``--family`` is ``ModelMeta.image_family`` when the library detected it, so
  the worker does not have to re-infer it from the name;
* ``--base-model`` (optional, from the model's load defaults) pins the mflux
  registry entry for checkpoints whose name does not carry a family alias.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from localcode.hub.types import LocalModel
from localcode.images.base import (
    ImageEngineError,
    WorkerImageEngine,
    backend_missing_message,
)

__all__ = ["MfluxEngine", "mflux_available"]


def mflux_available(python_path: str | None = None) -> bool:
    """Is ``mflux`` importable by the interpreter that will run the worker?

    Only answerable for this interpreter; a custom ``python_path`` is trusted
    (the worker reports the real answer through its log).
    """
    if python_path and python_path != sys.executable:
        return True
    try:
        return importlib.util.find_spec("mflux") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken sys.path
        return False


class MfluxEngine(WorkerImageEngine):
    """One MLX image model running in a ``mflux_worker`` subprocess."""

    backend = "mflux"
    format = "mlx"

    def __init__(
        self,
        instance_id: str,
        model: LocalModel,
        *,
        settings: Any | None = None,
        python_path: str | None = None,
        base_model: str | None = None,
        quantize: int | None = None,
        default_steps: int | None = None,
        embed_metadata: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instance_id, model, settings=settings, python_path=python_path, env=env)
        self.base_model = base_model
        self.quantize = quantize
        self.default_steps = default_steps
        self.embed_metadata = embed_metadata

    # ------------------------------------------------------------------ command
    def _worker_module(self) -> str:
        return "localcode.images.mflux_worker"

    def _worker_args(self) -> list[str]:
        meta = self.model.meta
        family = getattr(meta, "image_family", None)
        args = ["--model", str(self.model.path), "--name", self.model.display_name]
        if family and family != "unknown":
            args += ["--family", str(family)]
        if self.base_model:
            args += ["--base-model", str(self.base_model)]
        if self.quantize:
            args += ["--quantize", str(int(self.quantize))]
        if self.default_steps:
            args += ["--steps", str(int(self.default_steps))]
        if not self.embed_metadata:
            args += ["--no-metadata"]
        return args

    async def start(self) -> None:
        if not mflux_available(self._python):
            raise ImageEngineError(backend_missing_message("mflux"))
        await super().start()

    def size_multiple(self) -> int:
        """mflux resolves dimensions on a 16-pixel grid (``DIMENSION_STEP_PIXELS``)."""
        return 16
