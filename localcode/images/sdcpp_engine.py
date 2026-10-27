"""Engine for GGUF diffusion models: drives ``localcode.images.sdcpp_worker``.

Turns a :class:`~localcode.hub.types.LocalModel` into a stable-diffusion.cpp
command line.  Two layouts exist and both come straight from the library scan:

* **self-contained checkpoint** (SD 1.5, SDXL): one ``*.gguf`` with UNet, text
  encoder and VAE → ``--model <file>``;
* **split setup** (FLUX, SD 3.5): a diffusion model plus companion files
  recorded in ``ModelMeta.image_components`` (``clip_l``, ``clip_g``, ``t5xxl``,
  ``vae``) → ``--diffusion-model <file> --clip-l … --t5xxl … --vae …``.

Per-model overrides (a VAE the user picked, flash attention, thread count) come
from the library's load defaults via ``components``/``extra_args``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

from localcode.hub.types import LocalModel
from localcode.images.base import (
    ImageEngineError,
    WorkerImageEngine,
    backend_missing_message,
)

__all__ = ["SdcppEngine", "sdcpp_available"]

#: ``ModelMeta.image_components`` keys understood by the worker.
COMPONENT_FLAGS: dict[str, str] = {
    "clip_l": "--clip-l",
    "clip_g": "--clip-g",
    "t5xxl": "--t5xxl",
    "vae": "--vae",
    "taesd": "--taesd",
}


def sdcpp_available(python_path: str | None = None) -> bool:
    """Is ``stable_diffusion_cpp`` importable by the worker's interpreter?"""
    if python_path and python_path != sys.executable:
        return True
    try:
        return importlib.util.find_spec("stable_diffusion_cpp") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken sys.path
        return False


class SdcppEngine(WorkerImageEngine):
    """One GGUF image model running in a ``sdcpp_worker`` subprocess."""

    backend = "sdcpp"
    format = "gguf"

    def __init__(
        self,
        instance_id: str,
        model: LocalModel,
        *,
        settings: Any | None = None,
        python_path: str | None = None,
        components: Mapping[str, str] | None = None,
        default_steps: int | None = None,
        threads: int | None = None,
        flash_attn: bool = False,
        embed_metadata: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instance_id, model, settings=settings, python_path=python_path, env=env)
        self.components = dict(components or getattr(model.meta, "image_components", None) or {})
        self.default_steps = default_steps
        self.threads = threads
        self.flash_attn = flash_attn
        self.embed_metadata = embed_metadata

    # ------------------------------------------------------------------ command
    def _worker_module(self) -> str:
        return "localcode.images.sdcpp_worker"

    def _worker_args(self) -> list[str]:
        meta = self.model.meta
        family = str(getattr(meta, "image_family", None) or "unknown")
        components = {k: v for k, v in self.components.items() if v}
        diffusion = components.pop("unet", None) or components.pop("diffusion_model", None)
        args: list[str] = ["--family", family, "--name", self.model.display_name]
        if diffusion:
            args += ["--diffusion-model", str(Path(diffusion).expanduser())]
        else:
            args += ["--model", str(self.model.path)]
        for key, flag in COMPONENT_FLAGS.items():
            value = components.get(key)
            if value:
                args += [flag, str(Path(value).expanduser())]
        if self.default_steps:
            args += ["--steps", str(int(self.default_steps))]
        if self.threads:
            args += ["--threads", str(int(self.threads))]
        if self.flash_attn:
            args += ["--flash-attn"]
        if not self.embed_metadata:
            args += ["--no-metadata"]
        return args

    async def start(self) -> None:
        if not sdcpp_available(self._python):
            raise ImageEngineError(backend_missing_message("sdcpp"))
        await super().start()

    def size_multiple(self) -> int:
        """stable-diffusion.cpp works on a 64-pixel grid (8× latent downsampling)."""
        return 64
