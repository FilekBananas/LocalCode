"""mflux worker: one MLX diffusion model behind the LocalCode image protocol.

Usage::

    python -m localcode.images.mflux_worker --model <dir-or-repo> --port <p> \\
        [--base-model schnell|dev|klein-4b|qwen-image|…] [--family flux|flux2|…] \\
        [--quantize 4|6|8] [--steps N] [--no-metadata] [--parent-pid PID]

What was verified against the installed mflux (0.20.0, read from its source)
----------------------------------------------------------------------------
* mflux is CLI-first but every command is a thin wrapper around a model class:
  ``Flux1`` (FLUX.1 dev/schnell/Krea), ``Flux2Klein`` (FLUX.2 klein family),
  ``QwenImage``/``QwenImage21``, ``Krea2``, ``ZImage``, ``LensImage``, ``FIBO``,
  ``BooguImage``, ``ErnieImage``, ``Ideogram4``.  They share the constructor
  ``(model_config, quantize=None, model_path=None, lora_paths=…, lora_scales=…)``
  and the method ``generate_image(seed, prompt, num_inference_steps, height,
  width, guidance, image_path, image_strength, scheduler, …) -> GeneratedImage``.
* ``ModelConfig.from_name(model_name, base_model)`` resolves a name to a config:
  an exact alias/repo id wins, otherwise the *longest alias that appears in the
  name* decides (so ``Runpod/FLUX.2-klein-4B-mflux-4bit`` → ``klein-4b`` and
  ``dhairyashil/FLUX.1-schnell-mflux-4bit`` → ``schnell``).
  ``ConfigResolution.resolve_key()`` gives the canonical registry key, which is
  what this worker maps to a model class.
* ``model_path`` may be a local directory (``PathResolution`` takes it as-is) —
  that is how LocalCode loads models from its own library without ever going
  through the HF cache.
* Progress: the model owns a ``CallbackRegistry`` (``model.callbacks``); an
  object with ``call_before_loop`` / ``call_in_loop`` / ``call_after_loop`` is
  called once per denoising step with ``t`` (0-based step index) and the
  ``Config`` (``config.num_inference_steps``).  No faked numbers are needed.
  An exception raised inside ``call_in_loop`` propagates out of
  ``generate_image`` (only ``KeyboardInterrupt`` is caught there), which is how
  cancellation stops the loop between steps.
* ``GeneratedImage.save(path, export_json_metadata=False, overwrite=True)``
  writes the PNG exactly at ``path`` and embeds the generation metadata.
* Memory: ``mx.get_active_memory()`` / ``mx.get_peak_memory()``.

Everything user-facing is Polish; code and comments are English.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from localcode.images.base import (
    CancelledJob,
    ImageParams,
    ImageWorkerServer,
    WorkerDefaults,
    backend_missing_message,
    family_defaults,
)

log = logging.getLogger("localcode.images.mflux_worker")

# Canonical mflux registry key (prefix) -> module and class implementing txt2img.
# Checked against mflux 0.20.0; unknown keys fall back to Flux1, which covers
# every FLUX.1 derivative.
_MODEL_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("flux2-", "mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein"),
    ("qwen-image-2.1", "mflux.models.qwen21.variants.txt2img.qwen_image_21", "QwenImage21"),
    ("qwen-image", "mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage"),
    ("krea-2", "mflux.models.krea2.variants.txt2img.krea2", "Krea2"),
    ("z-image", "mflux.models.z_image.variants.z_image", "ZImage"),
    ("lens-", "mflux.models.lens.variants.txt2img.lens_image", "LensImage"),
    ("fibo", "mflux.models.fibo.variants.txt2img.fibo", "FIBO"),
    ("boogu-", "mflux.models.boogu.variants.txt2img.boogu_image", "BooguImage"),
    ("ernie-image", "mflux.models.ernie_image.variants.txt2img.ernie_image", "ErnieImage"),
    ("ideogram-4", "mflux.models.ideogram4.variants.txt2img.ideogram4", "Ideogram4"),
)

# Registry key -> LocalCode image family (``ModelMeta.image_family``).
_FAMILY_BY_KEY: tuple[tuple[str, str], ...] = (
    ("flux2-", "flux2"),
    ("qwen-image", "qwen-image"),
    ("dev", "flux"),
    ("schnell", "flux"),
    ("krea-dev", "flux"),
)


def family_for_key(key: str | None) -> str:
    """LocalCode image family of an mflux registry key (``flux2-klein-4b`` → ``flux2``)."""
    key = (key or "").lower()
    if not key:
        return "unknown"
    for prefix, family in _FAMILY_BY_KEY:
        if key.startswith(prefix):
            return family
    return "unknown"


class _ProgressCallback:
    """mflux callback that forwards denoising steps and honours cancellation."""

    def __init__(self, server: "MfluxWorker") -> None:
        self.server = server

    def call_before_loop(self, seed: int, prompt: str, latents: Any, config: Any,
                         canny_image: Any = None, depth_image: Any = None,
                         control_images: Any = None) -> None:
        self.server.emit_progress(step=0, steps=_total_steps(config), message="Generowanie…")

    def call_in_loop(self, t: int, seed: int, prompt: str, latents: Any, config: Any,
                     time_steps: Any = None) -> None:
        if self.server.cancelled():
            raise CancelledJob("Generowanie obrazu zostało przerwane.")
        self.server.emit_progress(step=int(t) + 1, steps=_total_steps(config))

    def call_after_loop(self, seed: int, prompt: str, latents: Any, config: Any) -> None:
        total = _total_steps(config)
        self.server.emit_progress(step=total, steps=total, message="Dekodowanie obrazu…")


def _total_steps(config: Any) -> int:
    for attr in ("num_inference_steps", "steps"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 0


class MfluxWorker(ImageWorkerServer):
    """Serves one mflux model (see the module docstring)."""

    backend = "mflux"
    hard_cancel = False          # cancellation works between denoising steps

    def __init__(
        self,
        model: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        parent_pid: int | None = None,
        base_model: str | None = None,
        family: str | None = None,
        quantize: int | None = None,
        default_steps: int | None = None,
        embed_metadata: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(host, port, parent_pid=parent_pid)
        self.model_path = model
        self.base_model = base_model or None
        self.requested_family = (family or "").lower() or None
        self.quantize = quantize
        self.default_steps = default_steps
        self.embed_metadata = embed_metadata
        self.name = name or Path(model).name
        self._model: Any = None
        self._model_config: Any = None
        self._registry_key: str | None = None
        self._family = self.requested_family or "unknown"
        self._index = 0
        self._emit: Any = None
        self._count = 1

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        try:
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.common.resolution.config_resolution import ConfigResolution
        except ImportError as exc:  # pragma: no cover - the engine reports this
            raise RuntimeError(backend_missing_message("mflux")) from exc

        name = self._resolution_name()
        self._model_config = ModelConfig.from_name(model_name=name, base_model=self.base_model)
        try:
            self._registry_key = ConfigResolution.resolve_key(model_name=name, base_model=self.base_model)
        except Exception:  # noqa: BLE001 - key is only used to pick the class
            self._registry_key = None
        if not self.requested_family:
            self._family = family_for_key(self._registry_key)
        cls = self._model_class(self._registry_key)
        log.info("loading %s with %s (config %s)", self.model_path, cls.__name__, self._registry_key)
        self._model = cls(
            model_config=self._model_config,
            quantize=self.quantize,
            model_path=self.model_path,
        )
        registry = getattr(self._model, "callbacks", None)
        if registry is not None and hasattr(registry, "register"):
            registry.register(_ProgressCallback(self))
        if not self.embed_metadata:
            try:
                from mflux.utils.image_util import ImageUtil

                ImageUtil.embed_metadata_enabled = False
            except ImportError:  # pragma: no cover
                pass

    def _resolution_name(self) -> str:
        """The name ``ModelConfig`` resolves the geometry from.

        A repo id (``org/repo``) or an alias is passed through; a filesystem path
        is reduced to its basename, which is what carries the family alias
        (``…/Runpod/FLUX.2-klein-4B-mflux-4bit`` → ``FLUX.2-klein-4B-mflux-4bit``).
        """
        if self.base_model and not self.model_path:
            return self.base_model
        raw = str(self.model_path)
        p = Path(raw)
        if p.exists():
            parts = [x for x in p.resolve().parts if x not in ("/", "")]
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
            return p.name
        return raw

    @staticmethod
    def _model_class(registry_key: str | None) -> Any:
        import importlib

        key = (registry_key or "").lower()
        for prefix, module_name, class_name in _MODEL_CLASSES:
            if key.startswith(prefix):
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
        module = importlib.import_module("mflux.models.flux.variants.txt2img.flux")
        return getattr(module, "Flux1")

    # -------------------------------------------------------------------- props
    def props_payload(self) -> dict[str, Any]:
        defaults = family_defaults(self._family)
        steps = self.default_steps or self._default_steps()
        supports_guidance = bool(getattr(self._model_config, "supports_guidance", False))
        return {
            "backend": self.backend,
            "modelPath": str(self.model_path),
            "family": self._family,
            "registryKey": self._registry_key,
            "quantization": getattr(self._model_config, "quantization_level", None),
            "defaults": WorkerDefaults(
                steps=steps,
                guidance=defaults["guidance"] if supports_guidance else None,
                sizes=list(defaults["sizes"]),
                size_multiple=16,
            ).to_dict(),
            "supports": {
                "negativePrompt": self._supports_negative_prompt(),
                "guidance": supports_guidance,
                "img2img": True,
                "seed": True,
                "batch": True,
            },
        }

    def _default_steps(self) -> int:
        try:
            from mflux.cli.defaults.defaults import model_inference_steps

            return int(model_inference_steps(self._registry_key or self._resolution_name()))
        except Exception:  # noqa: BLE001 - fall back to the family table
            return int(family_defaults(self._family)["steps"])

    def _supports_negative_prompt(self) -> bool:
        """FLUX.1/FLUX.2 are distilled and have no negative branch."""
        if self._family in ("flux", "flux2"):
            return False
        try:
            import inspect

            sig = inspect.signature(self._model.generate_image)
            return "negative_prompt" in sig.parameters
        except (TypeError, ValueError, AttributeError):  # pragma: no cover
            return False

    def memory_payload(self) -> dict[str, Any]:
        try:
            import mlx.core as mx

            return {"bytes": int(mx.get_active_memory()), "peakBytes": int(mx.get_peak_memory())}
        except Exception:  # noqa: BLE001 - reported as unknown
            return {}

    def unload(self) -> None:
        self._model = None
        try:
            import gc

            import mlx.core as mx

            gc.collect()
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- progress
    def emit_progress(self, *, step: int, steps: int, message: str = "") -> None:
        emit = self._emit
        if emit is None:
            return
        emit({"type": "progress", "index": self._index, "step": int(step),
              "steps": int(steps), "count": self._count, "message": message})

    # -------------------------------------------------------------- generation
    def run_job(self, params: ImageParams, outputs: list[str], emit: Any) -> None:
        if self._model is None:  # pragma: no cover - guarded by /health
            raise RuntimeError("Model obrazów nie jest wczytany.")
        self._emit = emit
        self._count = max(1, params.count)
        steps = params.steps or self._default_steps()
        guidance = params.guidance
        supports_guidance = bool(getattr(self._model_config, "supports_guidance", False))
        kwargs: dict[str, Any] = {}
        if guidance is not None and supports_guidance:
            kwargs["guidance"] = float(guidance)
        elif self._family == "flux2":
            # Distilled FLUX.2 checkpoints only accept guidance 1.0.
            kwargs["guidance"] = 1.0
        if params.init_image:
            kwargs["image_path"] = params.init_image
            kwargs["image_strength"] = float(params.strength)
        if params.negative_prompt and self._supports_negative_prompt():
            kwargs["negative_prompt"] = params.negative_prompt

        seeds = params.seeds()
        for index, (seed, output) in enumerate(zip(seeds, outputs, strict=False)):
            if self.cancelled():
                raise CancelledJob()
            self._index = index
            started = time.perf_counter()
            image = self._model.generate_image(
                seed=int(seed),
                prompt=params.prompt,
                num_inference_steps=int(steps),
                width=int(params.width),
                height=int(params.height),
                **kwargs,
            )
            if self.cancelled():
                raise CancelledJob()
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path=str(path), export_json_metadata=False, overwrite=True)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            pil = getattr(image, "image", None)
            size = getattr(pil, "size", None) or (params.width, params.height)
            emit({
                "type": "image",
                "index": index,
                "path": str(path),
                "seed": int(seed),
                "width": int(size[0]),
                "height": int(size[1]),
                "steps": int(steps),
                "guidance": kwargs.get("guidance"),
                "elapsedMs": elapsed_ms,
            })


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m localcode.images.mflux_worker",
                                description="Worker generujący obrazy przez mflux (MLX).")
    p.add_argument("--model", required=True, help="katalog modelu lub repo Hugging Face")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--parent-pid", type=int, default=None)
    p.add_argument("--base-model", default=None, help="alias modelu bazowego mflux (np. schnell, klein-4b)")
    p.add_argument("--family", default=None, help="rodzina obrazowa LocalCode (flux, flux2, qwen-image…)")
    p.add_argument("--quantize", type=int, default=None, help="kwantyzacja w locie (3/4/5/6/8)")
    p.add_argument("--steps", type=int, default=None, help="domyślna liczba kroków")
    p.add_argument("--name", default=None, help="nazwa modelu do logów")
    p.add_argument("--no-metadata", action="store_true", help="nie zapisuj metadanych w PNG")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    worker = MfluxWorker(
        args.model,
        host=args.host,
        port=args.port,
        parent_pid=args.parent_pid,
        base_model=args.base_model,
        family=args.family,
        quantize=args.quantize,
        default_steps=args.steps,
        embed_metadata=not args.no_metadata,
        name=args.name,
    )
    return worker.main()


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
