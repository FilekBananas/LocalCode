"""stable-diffusion.cpp worker: one GGUF diffusion model behind the image protocol.

Usage::

    python -m localcode.images.sdcpp_worker --model <checkpoint.gguf> --port <p> \\
        [--diffusion-model <unet.gguf>] [--clip-l …] [--clip-g …] [--t5xxl …] \\
        [--vae …] [--family sd1|sdxl|sd3|flux] [--steps N] [--threads N] \\
        [--flash-attn] [--parent-pid PID]

A self-contained checkpoint (SD 1.5 / SDXL: UNet + CLIP + VAE in one file) is
passed as ``--model``; a split FLUX/SD3 setup passes the diffusion model plus its
companions (``--diffusion-model`` + ``--clip-l``/``--clip-g``/``--t5xxl``/``--vae``),
which is what ``ModelMeta.image_components`` records at scan time.

What was verified against the installed ``stable_diffusion_cpp`` (0.4.7)
------------------------------------------------------------------------
* ``StableDiffusion(model_path=…, diffusion_model_path=…, clip_l_path=…,
  clip_g_path=…, t5xxl_path=…, vae_path=…, n_threads=-1, flash_attn=False,
  diffusion_flash_attn=False, vae_decode_only=True, verbose=False)`` — the
  wheel builds ``libstable-diffusion.dylib`` against Metal on this machine
  (``otool -L`` shows Metal/MetalKit; the log prints ``ggml_metal_init: found
  device: Apple M5``), so no extra flag is needed for GPU use.
* ``generate_image(prompt, negative_prompt, width, height, cfg_scale, guidance,
  sample_steps, seed, strength, init_image, batch_count, sample_method,
  progress_callback=…) -> list[PIL.Image.Image]``.
* ``progress_callback(step, steps, time)`` really fires once per denoising step
  (verified: ``[(0, 2, 0.0), (1, 2, 0.93), (2, 2, 0.92)]`` for a 2-step run);
  ``step == 0`` is the "started" tick.
* Inpainting checkpoints (FLUX.1 Fill, SD "inpainting") condition every step on
  the VAE-encoded source image and its mask.  Asked for plain text-to-image
  they segfault inside ``generate_image`` (no source image to encode — and
  with ``vae_decode_only=True`` no encoder either), killing the worker with
  exit code -11.  Such a model gets a neutral grey source image and the
  binding's default all-white mask ("repaint everything") at strength 1.0,
  which is text-to-image by another name.
* ``vae_decode_only=True`` makes the binding refuse any ``init_image``
  (``ValueError``), so the encoder is always loaded: img2img is advertised in
  ``/props``, and the encoder costs well under 100 MB next to the model.
* There is **no** abort callback in the binding, so a running generation cannot
  be stopped from Python: ``hard_cancel`` is True and the engine kills the
  process instead (the runtime then reloads the model on the next request).
* Images come back as RGBA ``PIL.Image``; they are saved as PNG with the
  generation parameters in the PNG text chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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

log = logging.getLogger("localcode.images.sdcpp_worker")

#: Sampler that suits each family best (stable-diffusion.cpp names).
_SAMPLER_BY_FAMILY = {
    "sd1": "euler_a",
    "sdxl": "euler_a",
    "sd3": "euler",
    "flux": "euler",
    "flux2": "euler",
    "unknown": "default",
}

#: Inpainting checkpoints: "FLUX.1-Fill-dev", "sd-v1-5-inpainting"…
_INPAINT_NAME = re.compile(r"(?<![a-z])(?:fill|inpaint(?:ing)?)(?![a-z])", re.IGNORECASE)


def is_inpaint_model(*names: str | None) -> bool:
    """True when the name or path says the checkpoint is an inpainting model."""
    return any(_INPAINT_NAME.search(Path(n).name if os.sep in n else n) for n in names if n)


class SdcppWorker(ImageWorkerServer):
    """Serves one stable-diffusion.cpp model (see the module docstring)."""

    backend = "sdcpp"
    hard_cancel = True           # the C library cannot abort a running generation

    def __init__(
        self,
        *,
        model: str | None = None,
        diffusion_model: str | None = None,
        clip_l: str | None = None,
        clip_g: str | None = None,
        t5xxl: str | None = None,
        vae: str | None = None,
        taesd: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        parent_pid: int | None = None,
        family: str | None = None,
        default_steps: int | None = None,
        threads: int = -1,
        flash_attn: bool = False,
        embed_metadata: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(host, port, parent_pid=parent_pid)
        self.model = model or ""
        self.diffusion_model = diffusion_model or ""
        self.clip_l = clip_l or ""
        self.clip_g = clip_g or ""
        self.t5xxl = t5xxl or ""
        self.vae = vae or ""
        self.taesd = taesd or ""
        self.family = (family or "unknown").lower()
        self.default_steps = default_steps
        self.threads = threads
        self.flash_attn = flash_attn
        self.embed_metadata = embed_metadata
        self.name = name or Path(self.model or self.diffusion_model).name
        self.inpaint = is_inpaint_model(self.name, self.model, self.diffusion_model)
        self._sd: Any = None
        self._index = 0
        self._count = 1
        self._emit: Any = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        try:
            from stable_diffusion_cpp import StableDiffusion
        except ImportError as exc:  # pragma: no cover - the engine reports this
            raise RuntimeError(backend_missing_message("sdcpp")) from exc

        for label, path in (("model", self.model), ("diffusion-model", self.diffusion_model),
                            ("clip_l", self.clip_l), ("clip_g", self.clip_g),
                            ("t5xxl", self.t5xxl), ("vae", self.vae)):
            if path and not Path(path).expanduser().exists():
                raise FileNotFoundError(f"Brak pliku modelu ({label}): {path}")
        if not self.model and not self.diffusion_model:
            raise ValueError("Nie podano pliku modelu (--model albo --diffusion-model).")

        log.info("loading %s (family %s%s)", self.model or self.diffusion_model, self.family,
                 ", inpainting" if self.inpaint else "")
        self._sd = StableDiffusion(
            model_path=_expand(self.model),
            diffusion_model_path=_expand(self.diffusion_model),
            clip_l_path=_expand(self.clip_l),
            clip_g_path=_expand(self.clip_g),
            t5xxl_path=_expand(self.t5xxl),
            vae_path=_expand(self.vae),
            taesd_path=_expand(self.taesd),
            n_threads=int(self.threads),
            flash_attn=bool(self.flash_attn),
            diffusion_flash_attn=bool(self.flash_attn),
            vae_decode_only=False,         # img2img and inpainting models encode a source image
            verbose=False,
        )

    def unload(self) -> None:
        sd, self._sd = self._sd, None
        if sd is not None:
            try:
                sd.close()
            except Exception:  # noqa: BLE001 - best effort
                log.debug("closing the stable-diffusion context failed", exc_info=True)

    # -------------------------------------------------------------------- props
    def props_payload(self) -> dict[str, Any]:
        defaults = family_defaults(self.family)
        steps = self.default_steps or int(defaults["steps"])
        return {
            "backend": self.backend,
            "modelPath": self.model or self.diffusion_model,
            "family": self.family,
            "components": {
                "diffusionModel": self.diffusion_model or None,
                "clipL": self.clip_l or None,
                "clipG": self.clip_g or None,
                "t5xxl": self.t5xxl or None,
                "vae": self.vae or None,
            },
            "defaults": WorkerDefaults(
                steps=steps,
                guidance=defaults["guidance"],
                sizes=list(defaults["sizes"]),
                size_multiple=64,
            ).to_dict(),
            "supports": {
                "negativePrompt": self.family in ("sd1", "sdxl", "sd3"),
                "guidance": True,
                "img2img": True,
                "seed": True,
                "batch": True,
            },
        }

    def memory_payload(self) -> dict[str, Any]:
        from localcode.runtime import memory as rt_memory

        return {"bytes": rt_memory.process_memory_bytes(os.getpid()) or 0}

    # ---------------------------------------------------------------- progress
    def _progress(self, step: int, steps: int, seconds: float) -> None:
        emit = self._emit
        if emit is None:
            return
        emit({"type": "progress", "index": self._index, "step": int(step),
              "steps": int(steps), "count": self._count})

    # -------------------------------------------------------------- generation
    def run_job(self, params: ImageParams, outputs: list[str], emit: Any) -> None:
        if self._sd is None:  # pragma: no cover - guarded by /health
            raise RuntimeError("Model obrazów nie jest wczytany.")
        self._emit = emit
        self._count = max(1, params.count)
        defaults = family_defaults(self.family)
        steps = int(params.steps or self.default_steps or defaults["steps"])
        guidance = params.guidance if params.guidance is not None else defaults["guidance"]
        cfg_scale = float(guidance if guidance is not None else 7.0)
        sampler = _SAMPLER_BY_FAMILY.get(self.family, "default")
        init_image: Any = params.init_image or None
        strength = float(params.strength) if init_image else 0.75
        if self.inpaint and init_image is None:
            # Text-to-image on an inpainting model: repaint a blank canvas
            # (the binding's default mask is all white).  See the module docstring.
            init_image = _blank_canvas(int(params.width), int(params.height))
            strength = 1.0

        for index, (seed, output) in enumerate(zip(params.seeds(), outputs, strict=False)):
            if self.cancelled():
                raise CancelledJob()
            self._index = index
            started = time.perf_counter()
            images = self._sd.generate_image(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or "",
                width=int(params.width),
                height=int(params.height),
                cfg_scale=cfg_scale,
                guidance=float(guidance) if guidance is not None else 3.5,
                sample_steps=steps,
                sample_method=sampler,
                seed=int(seed),
                batch_count=1,
                init_image=init_image,
                strength=strength,
                progress_callback=self._progress,
            )
            if not images:
                raise RuntimeError("Backend nie zwrócił żadnego obrazu.")
            image = images[0]
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save(image, path, params, seed=int(seed), steps=steps, guidance=cfg_scale)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            emit({
                "type": "image",
                "index": index,
                "path": str(path),
                "seed": int(seed),
                "width": int(image.width),
                "height": int(image.height),
                "steps": steps,
                "guidance": cfg_scale,
                "elapsedMs": elapsed_ms,
            })
            if self.cancelled():
                raise CancelledJob()

    def _save(self, image: Any, path: Path, params: ImageParams, *, seed: int,
              steps: int, guidance: float) -> None:
        """Write the PNG, with the generation parameters in its text chunks."""
        info = None
        if self.embed_metadata:
            try:
                from PIL import PngImagePlugin

                info = PngImagePlugin.PngInfo()
                info.add_text("parameters", _parameters_text(params, seed=seed, steps=steps,
                                                             guidance=guidance, model=self.name))
                info.add_text("localcode", json.dumps({
                    "backend": self.backend, "model": self.name, "family": self.family,
                    "prompt": params.prompt, "negativePrompt": params.negative_prompt,
                    "seed": seed, "steps": steps, "guidance": guidance,
                    "width": params.width, "height": params.height,
                }, ensure_ascii=False))
            except Exception:  # noqa: BLE001 - metadata is optional
                info = None
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        image.save(str(path), format="PNG", pnginfo=info)


def _parameters_text(params: ImageParams, *, seed: int, steps: int, guidance: float, model: str) -> str:
    """AUTOMATIC1111-style ``parameters`` chunk (what image viewers show)."""
    lines = [params.prompt]
    if params.negative_prompt:
        lines.append(f"Negative prompt: {params.negative_prompt}")
    lines.append(
        f"Steps: {steps}, CFG scale: {guidance}, Seed: {seed}, "
        f"Size: {params.width}x{params.height}, Model: {model}"
    )
    return "\n".join(lines)


def _blank_canvas(width: int, height: int) -> Any:
    """A neutral grey RGB image: the source an inpainting model repaints from scratch."""
    from PIL import Image

    return Image.new("RGB", (max(1, width), max(1, height)), (127, 127, 127))


def _expand(path: str) -> str:
    return str(Path(path).expanduser()) if path else ""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m localcode.images.sdcpp_worker",
                                description="Worker generujący obrazy przez stable-diffusion.cpp.")
    p.add_argument("--model", default=None, help="samodzielny checkpoint (SD 1.5 / SDXL)")
    p.add_argument("--diffusion-model", default=None, help="sam model dyfuzji (FLUX / SD3)")
    p.add_argument("--clip-l", default=None)
    p.add_argument("--clip-g", default=None)
    p.add_argument("--t5xxl", default=None)
    p.add_argument("--vae", default=None)
    p.add_argument("--taesd", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--parent-pid", type=int, default=None)
    p.add_argument("--family", default="unknown", help="sd1 | sdxl | sd3 | flux")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--flash-attn", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--no-metadata", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = SdcppWorker(
        model=args.model,
        diffusion_model=args.diffusion_model,
        clip_l=args.clip_l,
        clip_g=args.clip_g,
        t5xxl=args.t5xxl,
        vae=args.vae,
        taesd=args.taesd,
        host=args.host,
        port=args.port,
        parent_pid=args.parent_pid,
        family=args.family,
        default_steps=args.steps,
        threads=args.threads,
        flash_attn=args.flash_attn,
        embed_metadata=not args.no_metadata,
        name=args.name,
    )
    return worker.main()


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
